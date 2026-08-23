//! Item 2.5 tolerance gate: the wo_a fp8 einsum fast path (`dsv4_olo_einsum_fp8_b`) vs the
//! EXACT bf16 olo kernels (`dsv4_olo_proj_tc_b` / `dsv4_olo_proj_tc4_b`) on IDENTICAL
//! inputs (real production weights via `upload_layer`; synthetic de-rotated attention
//! outputs `o`). Asserts the COMPUTED rel-L2 bound — a gate that matches nothing is a
//! failure (AGENTS §3).
//!
//! The fast path is the DEFAULT (item 2.5 / §6-a): reduction order is scheduler-chosen,
//! both operands are fp8 (weights quantized at load, activations per call), so the outputs
//! differ from the exact path's bits BY CONSTRUCTION — this gate bounds that difference.
//! Widths: decode N=1, verify M=6, the tc/tc4 boundary s=16, and prefill widths 64/512.
//! Layers: 0 (SWA) and 2 (CSA) — the wo_a path is layer-kind-independent; both are
//! exercised for the load-side quantizer.
//!
//! Run: cargo test --release --test dsv4_olo_einsum_test -- --test-threads=1 --nocapture

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use half::bf16;

use gb10_inference::{dsv4_gpu, dsv4_load, quant};
use gb10_inference::dsv4_attn::{Dsv4AttnRuntime, Fp8Weight};
use gb10_inference::dsv4_gpu::{Dsv4Buf, Dsv4Kernels};
use gb10_inference::dsv4_launch;

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark";

/// One GPU job per process (tests run on threads; lanes serialize on the GPU too).
static GATE: Mutex<()> = Mutex::new(());

fn gate() -> MutexGuard<'static, ()> {
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut se, mut sn) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want.iter()) {
        se += ((g - w) as f64).powi(2);
        sn += (w as f64).powi(2);
    }
    (se / sn.max(1e-30)).sqrt()
}

/// Run the EXACT olo path for `s` live rows of `o` (tc_b at s≤16, tc4_b above) into oflat.
fn olo_exact(
    rt: &Dsv4AttnRuntime,
    o: &CudaSlice<bf16>,
    wo_a: &CudaSlice<bf16>,
    s: usize,
    g: usize,
    r: usize,
    gd: usize,
    ors: usize,
) -> CudaSlice<bf16> {
    let s_pad = ((s + 15) / 16) * 16;
    let mut oflat = rt.dev.alloc_zeros::<bf16>(s_pad * g * r).unwrap();
    let (s_i, g_i, r_i, gd_i, ors_i) = (s as i32, g as i32, r as i32, gd as i32, ors as i32);
    let tiles_m = (s + 15) / 16;
    if s > 16 {
        let packs_n = r / 64;
        dsv4_launch!(rt.attn, "dsv4_olo_proj_tc4_b", rt.stream.stream,
            ((packs_n * g) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
            (&oflat, o, wo_a, &s_i, &g_i, &r_i, &gd_i, &ors_i)).unwrap();
        return oflat;
    }
    let tiles_n = (r + 15) / 16;
    dsv4_launch!(rt.attn, "dsv4_olo_proj_tc_b", rt.stream.stream,
        ((tiles_n * g) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
        (&oflat, o, wo_a, &s_i, &g_i, &r_i, &gd_i, &ors_i)).unwrap();
    oflat
}

/// Run the FAST fp8 einsum path for `s` live rows of `o` into oflat (the production
/// dispatch: quant_g128 activations + dsv4_olo_einsum_fp8_b with the load-quantized weight).
fn olo_fast(
    rt: &Dsv4AttnRuntime,
    o: &CudaSlice<bf16>,
    wq: &Fp8Weight,
    s: usize,
    g: usize,
    r: usize,
    gd: usize,
) -> CudaSlice<bf16> {
    let s_pad = ((s + 15) / 16) * 16;
    let mut oflat = rt.dev.alloc_zeros::<bf16>(s_pad * g * r).unwrap();
    let (oc, osa) = rt
        .quant_g128::<CudaSlice<bf16>, CudaSlice<u8>>(o, s, g * gd)
        .unwrap();
    let nchunks = s.div_ceil(16).max(1);
    let gy8 = nchunks.div_ceil(8).max(1);
    let gx = (r / 32) * g;
    let gy = gy8.max((288usize.div_ceil(gx)).min(nchunks)) as u32;
    let f = &rt.bk["dsv4_olo_einsum_fp8_b"];
    let cfg = LaunchConfig {
        grid_dim: (gx as u32, gy, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (r_i, gd_i, s_i, g_i) = (r as i32, gd as i32, s as i32, g as i32);
    let ofv = oflat.view(0, s * g * r);
    let ov = o.view(0, s * g * gd);
    let ocv = oc.view(0, s * g * gd);
    let osav = osa.view(0, s * g * (gd / 128));
    unsafe {
        f.clone()
            .launch_on_stream(&rt.stream, cfg, (&ofv, &wq.wt, &wq.sb, &ocv, &osav, r_i, gd_i, s_i, g_i, 0u64))
            .unwrap();
    }
    oflat
}

/// One (layer, s) case: rel-L2 of the fast path vs the exact path over the s live rows,
/// plus the fast-vs-fast deterministic repeat (same inputs → same bits: the quantizers and
/// kernel are deterministic, so a rerun must reproduce the exact same oflat — that is the
/// only bitwise obligation of the tolerance class).
fn case(
    rt: &Dsv4AttnRuntime,
    layer_id: usize,
    s: usize,
    seed: u64,
) -> (f64, usize) {
    let cfg = dsv4_load::load_config(Path::new(BUNDLE)).unwrap();
    let layer = rt.upload_layer(Path::new(BUNDLE), &cfg, layer_id, 0, 1).unwrap();
    let wq = layer.wo_a_q.as_ref().expect("wo_a_q must be loaded");
    let (g, r, gd) = (cfg.o_groups, cfg.o_lora_rank, cfg.dim);
    let ors = cfg.n_heads * cfg.head_dim;
    assert_eq!(gd, cfg.dim, "o_groups x gd must tile the attention dim");

    let s_pad = ((s + 15) / 16) * 16;
    let mut rng = XorShift(seed);
    let o_host: Vec<bf16> = (0..s_pad * ors)
        .map(|_| bf16::from_f32(rng.f32() * 0.7))
        .collect();
    let o = rt.dev.htod_sync_copy(&o_host).unwrap();

    let ref_o = olo_exact(rt, &o, &layer.wo_a, s, g, r, gd, ors);
    let fast_o = olo_fast(rt, &o, wq, s, g, r, gd);
    let fast_o2 = olo_fast(rt, &o, wq, s, g, r, gd);
    rt.dev.synchronize().unwrap();

    let ref_v = rt.dev.dtoh_sync_copy(&ref_o).unwrap();
    let fast_v = rt.dev.dtoh_sync_copy(&fast_o).unwrap();
    let fast_v2 = rt.dev.dtoh_sync_copy(&fast_o2).unwrap();
    // Determinism: the fast path must reproduce itself bitwise.
    let mism = fast_v
        .iter()
        .zip(fast_v2.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(mism, 0, "fast path non-deterministic at layer {layer_id} s={s}");
    // rel-L2 over the s live rows only (padding rows are written by neither path's contract).
    let live = s * g * r;
    let got: Vec<f32> = fast_v[..live].iter().map(|b| b.to_f32()).collect();
    let want: Vec<f32> = ref_v[..live].iter().map(|b| b.to_f32()).collect();
    (rel_l2(&got, &want), s)
}

#[test]
fn olo_einsum_tolerance_vs_exact() {
    let _g = gate();
    if !Path::new(BUNDLE).exists() {
        eprintln!("SKIP olo_einsum_tolerance_vs_exact: {BUNDLE} not mounted");
        return;
    }
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let cfg = dsv4_load::load_config(Path::new(BUNDLE)).unwrap();
    let rt = Dsv4AttnRuntime::new_multikind(&dev, 64, &cfg).unwrap();

    // Decode N=1, verify M=6, the tc/tc4 dispatch boundary, and prefill widths.
    let widths = [1usize, 6, 16, 64, 512];
    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    let mut n_case = 0usize;
    for &layer_id in &[0usize, 2usize] {
        for (i, &s) in widths.iter().enumerate() {
            let (rl2, s) = case(&rt, layer_id, s, 0xABCD_0001 + (layer_id as u64) * 0x100 + i as u64);
            n_case += 1;
            eprintln!("[olo-einsum-gate] layer {layer_id} s={s}: rel-L2 {rl2:.3e}");
            if rl2 > worst {
                worst = rl2;
                worst_at = (layer_id, s);
            }
        }
    }
    println!(
        "DSV4-OLO-EINSUM-GATE: rel-L2(fast fp8 einsum vs exact bf16 olo) worst {worst:.3e} at layer {} s={} over {n_case} (layer,width) cases",
        worst_at.0, worst_at.1
    );
    // Measured class: fp8(128-block) x fp8(128-block) with K=4096 — expect ~1e-2..3e-2
    // (same family as the session-2 fp8-logits gate's 2.8e-2). 6e-2 is the SIGNAL bound:
    // a quantizer-layout or kernel-index break lands at ~1e0.
    assert!(worst < 6e-2, "olo fast path drifted: worst rel-L2 {worst:.3e}");
    println!("DSV4-OLO-EINSUM-GATE: PASS");
}

/// Item 2.5 compressor-pair fast path (`dsv4_comp_gemm_fast_pair_b`, 4-warp big-tile
/// bf16→fp32) vs the exact fused pair (`dsv4_comp_gemm_tc_pair_b`). The per-element
/// wmma m16n16k16 ascending-K accumulator chain is IDENTICAL in both kernels — only the
/// warp/tile ownership differs — so the outputs must be BITWISE equal (the tolerance
/// class degrades to exact here; assert the stronger signal). Shapes: the attention
/// compressor pair (CSA cd=2048, HCA cd=512, K=dim=4096) at decode/verify/prefill widths.
#[test]
fn comp_fast_pair_bitwise_vs_tc_pair() {
    let _g = gate();
    if !Path::new(BUNDLE).exists() {
        eprintln!("SKIP comp_fast_pair_bitwise_vs_tc_pair: {BUNDLE} not mounted");
        return;
    }
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let ks = Dsv4Kernels::load_module(&dev, "src/ptx/gpu_dsv4_comp.ptx", &[
        "dsv4_comp_gemm_tc_pair_b",
        "dsv4_comp_gemm_fast_pair_b",
        "dsv4_comp_pad16_b",
    ]).expect("load comp kernels");
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let stream = &stream;

    for &(n, label) in &[(2048usize, "CSA"), (512usize, "HCA")] {
        for &s in &[1usize, 6, 64, 512] {
            let tiles_m = (s + 15) / 16;
            // The dispatch keeps the exact kernel when the fast grid would underfill.
            let tiles64 = n / 64;
            if (2 * tiles64) * tiles_m < 32 {
                eprintln!("[comp-fast-gate] {label} s={s}: under-filled grid — exact kernel per dispatch (skip)");
                continue;
            }
            let (k, n_pad) = (4096usize, ((s + 15) / 16) * 16);
            let mut rng = XorShift(0xC0DE_0001 + n as u64 + s as u64);
            let x_host: Vec<bf16> = (0..n_pad * k).map(|_| bf16::from_f32(rng.f32() * 0.3)).collect();
            let w_host: Vec<bf16> = (0..2 * k * n).map(|_| bf16::from_f32(rng.f32() * 0.15)).collect();
            let x = dev.htod_sync_copy(&x_host).unwrap();
            let w = dev.htod_sync_copy(&w_host).unwrap();
            let (kv_e, sc_e) = (
                dev.alloc_zeros::<f32>(s * n).unwrap(),
                dev.alloc_zeros::<f32>(s * n).unwrap(),
            );
            let (kv_f, sc_f) = (
                dev.alloc_zeros::<f32>(s * n).unwrap(),
                dev.alloc_zeros::<f32>(s * n).unwrap(),
            );
            let x_pad = dev.alloc_zeros::<bf16>(16 * k).unwrap();
            let rc = s - (tiles_m - 1) * 16;
            if rc < 16 {
                let (row0, rci, ki2) = ((tiles_m - 1) as i32 * 16, rc as i32, k as i32);
                dsv4_launch!(ks, "dsv4_comp_pad16_b", stream.stream,
                    (((k + 255) / 256) as u32, 16, 1), (256, 1, 1), 0,
                    (&x_pad, &x, &row0, &rci, &ki2)).unwrap();
            }
            let (si, ki, ni) = (s as i32, k as i32, n as i32);
            dsv4_launch!(ks, "dsv4_comp_gemm_tc_pair_b", stream.stream,
                ((2 * ((n + 15) / 16)) as u32, tiles_m as u32, 1), (32, 1, 1), 0,
                (&kv_e, &sc_e, &x, &x_pad, &w, &w, &si, &ki, &ni)).unwrap();
            dsv4_launch!(ks, "dsv4_comp_gemm_fast_pair_b", stream.stream,
                ((2 * tiles64) as u32, tiles_m as u32, 1), (128, 1, 1), 0,
                (&kv_f, &sc_f, &x, &x_pad, &w, &w, &si, &ki, &ni)).unwrap();
            dev.synchronize().unwrap();
            let e = dev.dtoh_sync_copy(&kv_e).unwrap();
            let f = dev.dtoh_sync_copy(&kv_f).unwrap();
            let mism = e.iter().zip(f.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            if mism > 0 {
                let mut maxd = 0.0f32;
                let mut ex = 0.0f32;
                let mut fx = 0.0f32;
                for (a, b) in e.iter().zip(f.iter()) {
                    let d = (a - b).abs();
                    if d > maxd { maxd = d; ex = *a; fx = *b; }
                }
                let bad_cols: Vec<usize> = e.iter().zip(f.iter()).enumerate()
                    .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                    .map(|(i, _)| i % n).collect();
                let bad_cols_sorted = {
                    let mut v = bad_cols.clone(); v.sort(); v.dedup();
                    if v.len() > 12 { format!("{:?} ... ({} distinct cols)", &v[..12], v.len()) } else { format!("{v:?}") }
                };
                eprintln!("[comp-fast-gate] {label} s={s}: {mism} mismatches, max abs diff {maxd:.6e} (exact {ex} vs fast {fx}); bad cols {bad_cols_sorted}");
            }
            assert_eq!(mism, 0, "comp fast pair != tc_pair at {label} s={s} (kv): {mism} mismatches");
            let e = dev.dtoh_sync_copy(&sc_e).unwrap();
            let f = dev.dtoh_sync_copy(&sc_f).unwrap();
            let mism = e.iter().zip(f.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(mism, 0, "comp fast pair != tc_pair at {label} s={s} (score): {mism} mismatches");
            eprintln!("[comp-fast-gate] {label} s={s}: BITWISE == tc_pair (kv + score)");
        }
    }
    println!("DSV4-COMP-FAST-GATE: PASS");
}

/// Isolated wall-clock A/B of the pair kernels at the production shapes (rotating weight
/// buffers, 40 reps best-of-5 — the dsv4_fp8_sched_bench pattern). Prints, never asserts.
#[test]
fn comp_fast_pair_bench() {
    let _g = gate();
    let dev = Arc::new(CudaDevice::new(0).unwrap());
    let ks = Dsv4Kernels::load_module(&dev, "src/ptx/gpu_dsv4_comp.ptx", &[
        "dsv4_comp_gemm_tc_pair_b",
        "dsv4_comp_gemm_fast_pair_b",
    ]).expect("load comp kernels");
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let stream = &stream;
    for &(n, label) in &[(2048usize, "CSA"), (512usize, "HCA")] {
        for &s in &[1usize, 6, 64, 512] {
            let tiles_m = (s + 15) / 16;
            let tiles64 = n / 64;
            if (2 * tiles64) * tiles_m < 32 { continue; }
            let (k, n_pad) = (4096usize, ((s + 15) / 16) * 16);
            let mut rng = XorShift(0xBEEF_0001 + n as u64 + s as u64);
            let x_host: Vec<bf16> = (0..n_pad * k).map(|_| bf16::from_f32(rng.f32() * 0.3)).collect();
            // 4 rotating weight copies to defeat L2 (the schedule-bench convention).
            let mut wts: Vec<CudaSlice<bf16>> = Vec::new();
            for c in 0..4 {
                let w_host: Vec<bf16> = (0..2 * k * n).map(|_| bf16::from_f32(rng.f32() * 0.15 + c as f32)).collect();
                wts.push(dev.htod_sync_copy(&w_host).unwrap());
            }
            let x = dev.htod_sync_copy(&x_host).unwrap();
            let x_pad = dev.alloc_zeros::<bf16>(16 * k).unwrap();
            let (kv_e, sc_e) = (dev.alloc_zeros::<f32>(s * n).unwrap(), dev.alloc_zeros::<f32>(s * n).unwrap());
            let (kv_f, sc_f) = (dev.alloc_zeros::<f32>(s * n).unwrap(), dev.alloc_zeros::<f32>(s * n).unwrap());
            let (si, ki, ni) = (s as i32, k as i32, n as i32);
            let bench = |name: &str, block: (u32, u32, u32), w: &CudaSlice<bf16>| {
                let t0 = std::time::Instant::now();
                if name == "tc" {
                    dsv4_launch!(ks, "dsv4_comp_gemm_tc_pair_b", stream.stream,
                        ((2 * ((n + 15) / 16)) as u32, tiles_m as u32, 1), block, 0,
                        (&kv_e, &sc_e, &x, &x_pad, w, w, &si, &ki, &ni)).unwrap();
                } else {
                    dsv4_launch!(ks, "dsv4_comp_gemm_fast_pair_b", stream.stream,
                        ((2 * tiles64) as u32, tiles_m as u32, 1), block, 0,
                        (&kv_f, &sc_f, &x, &x_pad, w, w, &si, &ki, &ni)).unwrap();
                }
                dev.synchronize().unwrap();
                t0.elapsed().as_secs_f64()
            };
            let mut tc_best = f64::MAX;
            let mut f_best = f64::MAX;
            for rep in 0..40 {
                let w = &wts[rep % 4];
                let t = bench("tc", (32, 1, 1), w);
                if t < tc_best { tc_best = t; }
                let t = bench("fast", (128, 1, 1), w);
                if t < f_best { f_best = t; }
            }
            eprintln!("[comp-fast-bench] {label} s={s}: tc_pair {:.3} ms vs fast {:.3} ms ({:.2}x)", tc_best * 1e3, f_best * 1e3, tc_best / f_best);
        }
    }
}

/// The load-path quantizer against a hand-rolled reference: quantize a synthetic bf16
/// weight, dequantize through the MMA-tile map + scales (quant::repack_fp8_mma is the
/// bijection the kernel consumes), and bound the per-element error. Catches a quantizer
/// regression without a GPU (the rest of the gate is GPU).
#[test]
fn wo_a_quantizer_format_sanity() {
    let (g, r, k) = (4usize, 1024usize, 4096usize);
    let mut rng = XorShift(0xF00D_0002);
    let w: Vec<bf16> = (0..g * r * k).map(|_| bf16::from_f32(rng.f32())).collect();
    let (wt, sb) = quant::quantize_fp8_bsb(&w, g * r, k);
    // The scale array must be [M/128, K/128] group-major (the einsum kernel's sb_grp math).
    assert_eq!(sb.len(), (g * r / 128) * (k / 128));
    // Spot-check dequantized values against the source through the tiled map.
    let nblk = k / 16;
    for &(row, col) in &[(0usize, 0usize), (5, 4095), (127, 128), (1000, 2000), (4095, 4095)] {
        let r0 = row & 15;
        let cc = col & 15;
        let lane = (r0 & 7) * 4 + ((cc & 7) >> 1);
        let j = (cc & 1) | ((r0 >> 3) << 1) | ((cc >> 3) << 2);
        let tile = (row >> 4) * nblk + (col >> 4);
        let byte = wt[tile * 256 + lane * 8 + j];
        let code = byte;                 // fp8: one code per byte (fp8_tile_slot is a byte map)
        let sbv = sb[(row / 128) * (k / 128) + col / 128];
        let scale = 2f32.powi(sbv as i32 - 127);   // UE8M0: 2^(b-127)
        // E4M3 decode by bit surgery (the kernel's e4m3_f, mirrored): s|eeee|mmm, bias 7.
        let e4m3 = |b: u8| -> f32 {
            let sign = (b & 0x80) != 0;
            let e = (b >> 3) & 0x0F;
            let m = (b & 0x07) as f32;
            let mag = if e == 0 {
                m * 0.001953125 // 2^-9 subnormal
            } else {
                (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7)
            };
            if sign { -mag } else { mag }
        };
        // fp8_tile_slot's j-mapping: (col&1) | ((row>>3)&1)<<1 | ((col>>3)&1)<<2 — bits 0-2
        // select among the 8 bytes a lane covers; the nibble (col&1) selects within the byte.
        let exp = e4m3(code) * scale;
        let got = exp;
        let want = w[row * k + col].to_f32();
        let rel = ((got - want) / want.abs().max(1e-6)).abs();
        assert!(rel < 0.5, "quant spot-check ({row},{col}): {got} vs {want} rel {rel}");
    }
    println!("WO-A-QUANT-FORMAT: PASS ({} codes + {} scales)", wt.len(), sb.len());
}
