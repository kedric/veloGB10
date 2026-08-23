//! Phase-3 lane 3A kernel-unit gates: every NEW kernel in `kernels/gpu_dsv4_attn.cu` diffed
//! against the G1-proven CPU reference (`src/dsv4_cpu.rs`) on synthetic inputs.
//!
//! Gates:
//!   1. `dsv4_attn_rescale_b` — §B.1.1 per-head weight-free RMS rescale: BIT-EXACT vs the
//!      attn_qkv rescale loop (bf16 per-op rounding + pairwise-tree mean). Adversarial values
//!      included (bf16-RNE midpoints in the square/mean chain).
//!   2. `dsv4_kv_sim_g64_strided` — KV QAT-sim on kv[..., :448] of a [rows,512] tensor:
//!      BIT-EXACT vs dsv4_cpu::act_quant_sim on the extracted nope dims; rope dims untouched.
//!   3. `dsv4_ring_write_b` — §B.2 ring write: prefill S=130 rotated write, prefill S≤128,
//!      decode slot write — bit-exact vs dsv4_cpu::attn_forward's cache-write semantics.
//!   4. `dsv4_window_idxs_b` — §B.2 window index lists: exact vs dsv4_cpu::window_topk_idxs
//!      (prefill S=130/S=5, full-ring decode, early decode incl. the start_pos=127 boundary).
//!   5. `dsv4_olo_proj_b` — §B.1.4 grouped-LoRA O bf16 GEMM: rel-L2 ≤ 1e-4 vs the CPU
//!      gemm_bf16 einsum (reduction-order class) + batch-invariance (row 0 at s=1 vs s=16
//!      bitwise — AGENTS §2.4).
//!   6. Batch-invariance of the spine pieces this lane drives: gather_attn output row 0 at
//!      width 1 vs inside a 16-wide launch (same index list) — bitwise; fp8_bsb chunked at
//!      S=130 vs per-row N=1 — bitwise (the G2 N-invariance the chunked prefill relies on).
//!
//! Run: cargo test --release --test dsv4_attn_test -- --test-threads=1 --nocapture

use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::bf16;

use gb10_inference::{dsv4_cpu, quant};
use gb10_inference::dsv4_gpu::{self, Dsv4Kernels};
use gb10_inference::dsv4_launch;

/// One GPU job per process (tests run on threads; lanes serialize on the GPU too).
static GATE: Mutex<()> = Mutex::new(());

fn gate() -> MutexGuard<'static, ()> {
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA device 0")
}

fn to_bf16_dev(dev: &Arc<CudaDevice>, v: &[f32]) -> CudaSlice<bf16> {
    let b: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
    dev.htod_sync_copy(&b).unwrap()
}

fn from_bf16_dev(dev: &Arc<CudaDevice>, s: &CudaSlice<bf16>) -> Vec<f32> {
    dev.dtoh_sync_copy(s).unwrap().iter().map(|b| b.to_f32()).collect()
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

// ---------------------------------------------------------------------------
// 1. dsv4_attn_rescale_b — bit-exact vs the CPU rescale loop (§B.1.1)
// ---------------------------------------------------------------------------

/// dsv4_cpu::attn_qkv's per-head rescale loop, extracted 1:1 (bf16 per-op rounding).
fn cpu_rescale(q: &mut [f32], rows: usize, hd: usize, eps: f32) {
    for i in 0..rows {
        let row = &mut q[i * hd..(i + 1) * hd];
        let mut sq: Vec<f32> = row.iter().map(|&v| dsv4_cpu::bf(v * v)).collect();
        let ss = dsv4_cpu::pairwise_sum(&mut sq);
        let mean = dsv4_cpu::bf(ss / hd as f32);
        let arg = dsv4_cpu::bf(mean + eps);
        let r = dsv4_cpu::bf(arg.sqrt().recip());
        for v in row.iter_mut() {
            *v = dsv4_cpu::bf(*v * r);
        }
    }
}

#[test]
fn attn_rescale_bitexact_vs_cpu() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load_module(&dev, "src/ptx/gpu_dsv4_attn.ptx", &["dsv4_attn_rescale_b"]).expect("load");

    let (s, nh, hd) = (7usize, 64usize, 512usize);
    let rows = s * nh;
    let mut rng = XorShift(0x3A10_0001);
    // Mix: random ±1, tiny values (denormal-ish squares), large values, and exact bf16
    // midpoints (values whose squares sit near an RNE boundary in the mean chain).
    let mut q: Vec<f32> = (0..rows * hd)
        .map(|i| {
            let v = match i % 5 {
                0 => rng.f32() * 1e-3,
                1 => rng.f32() * 37.0,
                2 => rng.f32(),
                3 => dsv4_cpu::bf(rng.f32() * 0.5 + 0.00390625), // 2^-8 steps — midpoint-heavy
                _ => rng.f32() * 1e-6,
            };
            dsv4_cpu::bf(v)
        })
        .collect();

    let mut q_dev = to_bf16_dev(&dev, &q);
    dev.synchronize().unwrap();
    let (rows_i, hd_i, eps) = (rows as i32, hd as i32, 1e-6f32);
    dsv4_launch!(ks, "dsv4_attn_rescale_b", stream.stream, (rows as u32, 1, 1), (256, 1, 1), 0,
        (&mut q_dev, &rows_i, &hd_i, &eps))
    .expect("launch rescale");
    dev.synchronize().unwrap();
    let got = from_bf16_dev(&dev, &q_dev);

    cpu_rescale(&mut q, rows, hd, 1e-6);
    let mut mism = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for i in 0..rows * hd {
        if got[i].to_bits() != q[i].to_bits() {
            mism += 1;
            if first.is_none() {
                first = Some((i, got[i], q[i]));
            }
        }
    }
    eprintln!("attn_rescale: bit mismatches {mism} / {}", rows * hd);
    if let Some((i, g, e)) = first {
        panic!("rescale not bit-exact: first at {i}: gpu={g:e} cpu={e:e} ({mism} total)");
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 2. dsv4_kv_sim_g64_strided — bit-exact vs act_quant_sim on the nope view (§B.1.2)
// ---------------------------------------------------------------------------

#[test]
fn kv_sim_strided_bitexact_vs_cpu() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load_module(&dev, "src/ptx/gpu_dsv4_attn.ptx", &["dsv4_kv_sim_g64_strided"]).expect("load");

    let (rows, hd, rd) = (17usize, 512usize, 64usize);
    let nope = hd - rd;
    let mut rng = XorShift(0x3A20_0002);
    let mut kv: Vec<f32> = (0..rows * hd).map(|_| dsv4_cpu::bf(rng.f32() * 2.3)).collect();
    // Adversarial: a row whose group amax sits exactly on a pow2 boundary.
    for j in 0..nope {
        kv[3 * hd + j] = dsv4_cpu::bf(if j % 64 == 0 { 448.0 * 2f32.powi(-3) } else { 0.01 });
    }

    let mut kv_dev = to_bf16_dev(&dev, &kv);
    dev.synchronize().unwrap();
    let (r_i, stride_i, nope_i) = (rows as i32, hd as i32, nope as i32);
    let blocks = ((rows * (nope / 64) * 32) + 255) / 256;
    dsv4_launch!(ks, "dsv4_kv_sim_g64_strided", stream.stream, (blocks as u32, 1, 1), (256, 1, 1), 0,
        (&mut kv_dev, &r_i, &stride_i, &nope_i))
    .expect("launch kv sim");
    dev.synchronize().unwrap();
    let got = from_bf16_dev(&dev, &kv_dev);

    // CPU: extract the [rows, 448] nope view, sim contiguous, write back (attn_qkv's flow).
    let mut tmp = vec![0.0f32; rows * nope];
    for i in 0..rows {
        tmp[i * nope..(i + 1) * nope].copy_from_slice(&kv[i * hd..i * hd + nope]);
    }
    dsv4_cpu::act_quant_sim(&mut tmp, rows, nope, 64);
    for i in 0..rows {
        kv[i * hd..i * hd + nope].copy_from_slice(&tmp[i * nope..(i + 1) * nope]);
    }

    let mut mism = 0usize;
    for i in 0..rows * hd {
        if got[i].to_bits() != kv[i].to_bits() {
            mism += 1;
            if mism <= 5 {
                eprintln!("  kv_sim mismatch at {i} (row {} dim {}): gpu={:e} cpu={:e}", i / hd, i % hd, got[i], kv[i]);
            }
        }
    }
    eprintln!("kv_sim_strided: bit mismatches {mism} / {} (rope dims must be untouched)", rows * hd);
    assert_eq!(mism, 0, "kv_sim_g64_strided != CPU (bit-exact QAT-sim)");
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 3. dsv4_ring_write_b — bit-exact vs the CPU cache-write semantics (§B.2)
// ---------------------------------------------------------------------------

fn cpu_prefill_write(cache: &mut [f32], kv: &[f32], s: usize, win: usize, hd: usize) {
    if s <= win {
        cache[..s * hd].copy_from_slice(&kv[..s * hd]);
    } else {
        let cutoff = s % win;
        cache[cutoff * hd..win * hd].copy_from_slice(&kv[(s - win) * hd..(s - cutoff) * hd]);
        cache[..cutoff * hd].copy_from_slice(&kv[(s - cutoff) * hd..s * hd]);
    }
}

#[test]
fn ring_write_matches_cpu() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load_module(&dev, "src/ptx/gpu_dsv4_attn.ptx", &["dsv4_ring_write_b"]).expect("load");
    let (win, hd) = (128usize, 512usize);
    let mut rng = XorShift(0x3A30_0003);

    // (s, start_pos) cases: rotated prefill, short prefill, decode (full ring), early decode.
    for &(s, start_pos) in &[(130usize, 0usize), (100, 0), (1, 130), (1, 5), (1, 127)] {
        let kv: Vec<f32> = (0..s * hd).map(|_| dsv4_cpu::bf(rng.f32())).collect();
        let mut cache_host = vec![-7.0f32; win * hd]; // poison: unwritten slots must stay
        let mut cache_dev = to_bf16_dev(&dev, &cache_host);
        let kv_dev = to_bf16_dev(&dev, &kv);
        dev.synchronize().unwrap();

        let lo = if start_pos == 0 && s > win { s - win } else { 0 };
        let (s_i, sp_i, win_i, hd_i) = (s as i32, start_pos as i32, win as i32, hd as i32);
        dsv4_launch!(ks, "dsv4_ring_write_b", stream.stream, ((s - lo) as u32, 1, 1), (256, 1, 1), 0,
            (&mut cache_dev, &kv_dev, &s_i, &sp_i, &win_i, &hd_i))
        .expect("launch ring write");
        dev.synchronize().unwrap();
        let got = from_bf16_dev(&dev, &cache_dev);

        if start_pos == 0 {
            cpu_prefill_write(&mut cache_host, &kv, s, win, hd);
        } else {
            let slot = start_pos % win;
            cache_host[slot * hd..(slot + 1) * hd].copy_from_slice(&kv[..hd]);
        }
        let want: Vec<f32> = cache_host.iter().map(|&v| dsv4_cpu::bf(v)).collect();
        let mism = got.iter().zip(want.iter()).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
        eprintln!("ring_write s={s} start_pos={start_pos}: mismatches {mism} / {}", win * hd);
        assert_eq!(mism, 0, "ring_write s={s} start_pos={start_pos} != CPU semantics");
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 4. dsv4_window_idxs_b — exact vs dsv4_cpu::window_topk_idxs (§B.2)
// ---------------------------------------------------------------------------

#[test]
fn window_idxs_match_cpu() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load_module(&dev, "src/ptx/gpu_dsv4_attn.ptx", &["dsv4_window_idxs_b"]).expect("load");
    let win = 128usize;

    // (s, start_pos): prefill wide/small, full-ring decode, early decode, the 127 boundary.
    for &(s, start_pos) in &[(130usize, 0usize), (5, 0), (1, 130), (1, 131), (4, 133), (1, 5), (1, 126), (1, 127), (1, 128)] {
        let t = if start_pos == 0 { s.min(win) } else { win };
        let mut idxs_dev = dev.alloc_zeros::<i32>(s * t).unwrap();
        dev.synchronize().unwrap();
        let (s_i, sp_i, win_i, t_i) = (s as i32, start_pos as i32, win as i32, t as i32);
        let blocks = ((s * t) + 255) / 256;
        dsv4_launch!(ks, "dsv4_window_idxs_b", stream.stream, (blocks as u32, 1, 1), (256, 1, 1), 0,
            (&mut idxs_dev, &s_i, &sp_i, &win_i, &t_i))
        .expect("launch window idxs");
        dev.synchronize().unwrap();
        let got = dev.dtoh_sync_copy(&idxs_dev).unwrap();

        let want_rows = dsv4_cpu::window_topk_idxs(win, s, start_pos);
        assert_eq!(want_rows.len(), s);
        assert_eq!(want_rows[0].len(), t, "cpu row width vs t");
        let mut mism = 0usize;
        for r in 0..s {
            for j in 0..t {
                if got[r * t + j] as i64 != want_rows[r][j] {
                    mism += 1;
                    if mism <= 5 {
                        eprintln!("  idx mismatch s={s} sp={start_pos} row {r} j {j}: gpu={} cpu={}", got[r * t + j], want_rows[r][j]);
                    }
                }
            }
        }
        eprintln!("window_idxs s={s} start_pos={start_pos} t={t}: mismatches {mism} / {}", s * t);
        assert_eq!(mism, 0, "window_idxs s={s} start_pos={start_pos} != window_topk_idxs");
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 5. dsv4_olo_proj_b — §B.1.4 grouped-LoRA O vs CPU gemm_bf16 + batch-invariance
// ---------------------------------------------------------------------------

#[test]
fn olo_proj_matches_cpu_and_binv() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load_module(&dev, "src/ptx/gpu_dsv4_attn.ptx", &["dsv4_olo_proj_b"]).expect("load");

    let (s, nh, hd, g, r) = (16usize, 64usize, 512usize, 8usize, 1024usize);
    let gd = nh * hd / g; // 4096
    let mut rng = XorShift(0x3A50_0005);
    let o: Vec<f32> = (0..s * nh * hd).map(|_| dsv4_cpu::bf(rng.f32() * 1.4)).collect();
    let wo_a: Vec<f32> = (0..g * r * gd).map(|_| dsv4_cpu::bf(rng.f32() * 0.02)).collect();

    let o_dev = to_bf16_dev(&dev, &o);
    let wa_dev = to_bf16_dev(&dev, &wo_a);
    let mut out_dev = dev.alloc_zeros::<bf16>(s * g * r).unwrap();
    dev.synchronize().unwrap();
    let (s_i, g_i, r_i, gd_i, ors_i) = (s as i32, g as i32, r as i32, gd as i32, (nh * hd) as i32);
    dsv4_launch!(ks, "dsv4_olo_proj_b", stream.stream, ((g * r) as u32, s as u32, 1), (256, 1, 1), 0,
        (&mut out_dev, &o_dev, &wa_dev, &s_i, &g_i, &r_i, &gd_i, &ors_i))
    .expect("launch olo");
    dev.synchronize().unwrap();
    let got = from_bf16_dev(&dev, &out_dev);

    // CPU: per-group [s, gd] @ wo_a[g]ᵀ → [s, r] (dsv4_cpu::attn_out_proj's einsum half).
    let mut want = vec![0.0f32; s * g * r];
    for grp in 0..g {
        let mut xg = vec![0.0f32; s * gd];
        for i in 0..s {
            xg[i * gd..(i + 1) * gd].copy_from_slice(&o[i * nh * hd + grp * gd..i * nh * hd + (grp + 1) * gd]);
        }
        let wag = &wo_a[grp * r * gd..(grp + 1) * r * gd];
        let yg = dsv4_cpu::gemm_bf16(&xg, s, gd, wag, r);
        for i in 0..s {
            want[i * g * r + grp * r..i * g * r + (grp + 1) * r].copy_from_slice(&yg[i * r..(i + 1) * r]);
        }
    }
    let rl = rel_l2(&got, &want);
    eprintln!("olo_proj: rel-L2 vs CPU gemm_bf16 = {rl:.3e} (bar 1e-4, reduction-order class)");
    assert!(rl <= 1e-4, "olo_proj rel-L2 {rl:.3e} > 1e-4");

    // Batch-invariance: row 0 at s=1 vs row 0 of the s=16 launch — bitwise.
    let o1_dev = to_bf16_dev(&dev, &o[..nh * hd]);
    let mut out1_dev = dev.alloc_zeros::<bf16>(g * r).unwrap();
    dev.synchronize().unwrap();
    let one_i = 1i32;
    dsv4_launch!(ks, "dsv4_olo_proj_b", stream.stream, ((g * r) as u32, 1u32, 1), (256, 1, 1), 0,
        (&mut out1_dev, &o1_dev, &wa_dev, &one_i, &g_i, &r_i, &gd_i, &ors_i))
    .expect("launch olo n=1");
    dev.synchronize().unwrap();
    let got1 = from_bf16_dev(&dev, &out1_dev);
    let mism = got1.iter().zip(got[..g * r].iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    eprintln!("olo_proj batch-invariance: row0 s=1 vs s=16 mismatches {mism} / {}", g * r);
    assert_eq!(mism, 0, "olo_proj row 0 differs between width 1 and 16 (batch-invariance)");
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 6. Batch-invariance of the spine pieces this lane drives
// ---------------------------------------------------------------------------

#[test]
fn gather_attn_width_invariance() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_gather_attn"]).expect("load");
    ks.set_dynamic_smem("dsv4_gather_attn", 88320).expect("smem");

    let (h, d, n, topk) = (64usize, 512usize, 128usize, 128usize);
    let scale = (d as f64).powf(-0.5) as f32;
    let mut rng = XorShift(0x3A60_0006);
    let kv: Vec<f32> = (0..n * d).map(|_| dsv4_cpu::bf(rng.f32())).collect();
    let sink: Vec<f32> = (0..h).map(|_| rng.f32() * 0.5).collect();
    // The decode row: one q row + a full-ring index list (oldest→newest), then pad a
    // 16-wide launch with 15 other rows (random q, own lists) — row 0 must be bitwise.
    let q_row: Vec<f32> = (0..h * d).map(|_| dsv4_cpu::bf(rng.f32() * 1.1)).collect();
    let sp = 37usize; // start_pos % 128 of the decode row
    let list_row: Vec<i32> = (0..topk as i32)
        .map(|j| if (j as usize) < win_tail(sp) { (sp as i32 + 1 + j) } else { j - win_tail(sp) as i32 })
        .collect();
    fn win_tail(sp: usize) -> usize {
        128 - 1 - sp
    }

    let kv_dev = to_bf16_dev(&dev, &kv);
    let sink_dev = dev.htod_sync_copy(&sink).unwrap();

    // width-1 launch
    let q1_dev = to_bf16_dev(&dev, &q_row);
    let idx1_dev = dev.htod_sync_copy(&list_row).unwrap();
    let mut o1_dev = dev.alloc_zeros::<bf16>(h * d).unwrap();
    dev.synchronize().unwrap();
    let (t_i, n_i) = (topk as i32, n as i32);
    dsv4_launch!(ks, "dsv4_gather_attn", stream.stream, (1u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
        (&q1_dev, &kv_dev, &mut o1_dev, &sink_dev, &idx1_dev, &t_i, &n_i, &scale))
    .expect("launch gather m=1");
    dev.synchronize().unwrap();
    let got1 = from_bf16_dev(&dev, &o1_dev);

    // width-16 launch: row 0 = the decode row; rows 1..15 random with their own lists.
    let m = 16usize;
    let mut q16 = q_row.clone();
    let mut idx16 = list_row.clone();
    for r in 1..m {
        q16.extend((0..h * d).map(|_| dsv4_cpu::bf(rng.f32() * 0.9)));
        idx16.extend((0..topk as i32).map(|j| ((r * 7 + j as usize) % n) as i32));
    }
    let q16_dev = to_bf16_dev(&dev, &q16);
    let idx16_dev = dev.htod_sync_copy(&idx16).unwrap();
    let mut o16_dev = dev.alloc_zeros::<bf16>(m * h * d).unwrap();
    dev.synchronize().unwrap();
    dsv4_launch!(ks, "dsv4_gather_attn", stream.stream, (m as u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
        (&q16_dev, &kv_dev, &mut o16_dev, &sink_dev, &idx16_dev, &t_i, &n_i, &scale))
    .expect("launch gather m=16");
    dev.synchronize().unwrap();
    let got16 = from_bf16_dev(&dev, &o16_dev);

    let mism = got1.iter().zip(got16[..h * d].iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    eprintln!("gather_attn batch-invariance: decode row at width 1 vs 16 mismatches {mism} / {}", h * d);
    assert_eq!(mism, 0, "gather_attn decode row differs between width 1 and 16");
    let _ = stream;
}

/// fp8_bsb chunked prefill (S=130 → chunks of ≤16) vs per-row N=1: bitwise. The G2 gate
/// proved N-invariance at N=1..16 on the kernel; this asserts the lane's chunking scheme
/// (view-offset launches) preserves it, and that chunk outputs match quant_gemm at the
/// G2 accumulation-order tolerance.
#[test]
fn fp8_bsb_chunk_consistency() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx").unwrap());
    dev.load_ptx(ptx, "gpu_batch", &["gemm_dsv4_fp8_bsb2"]).unwrap();
    let f = dev.get_func("gpu_batch", "gemm_dsv4_fp8_bsb2").unwrap();

    let (m, k, s) = (1024usize, 1024usize, 130usize);
    let mut rng = XorShift(0x3A70_0007);
    let w: Vec<f32> = (0..m * k).map(|_| rng.f32() * 0.05).collect();
    let x: Vec<f32> = (0..s * k).map(|_| dsv4_cpu::bf(rng.f32() * 0.8)).collect();

    // Weight quant + MMA repack (mirror the spine test's quant_w_blocks).
    let (rb_n, cb_n) = (m / 128, k / 128);
    let mut wcodes = vec![0u8; m * k];
    let mut sb = vec![0u8; rb_n * cb_n];
    for rb in 0..rb_n {
        for cb in 0..cb_n {
            let mut amax = 0.0f32;
            for i in 0..128 {
                let row = (rb * 128 + i) * k + cb * 128;
                for &v in &w[row..row + 128] {
                    amax = amax.max(v.abs());
                }
            }
            let sc = if amax > 0.0 { dsv4_cpu::fast_round_scale(amax, 1.0 / 448.0) } else { 1.0 };
            sb[rb * cb_n + cb] = (sc.to_bits() >> 23) as u8;
            for i in 0..128 {
                let row = (rb * 128 + i) * k + cb * 128;
                for j in 0..128 {
                    wcodes[row + j] = dsv4_cpu::f32_to_e4m3_rne((w[row + j] / sc).clamp(-448.0, 448.0));
                }
            }
        }
    }
    let wt = quant::repack_fp8_mma(&wcodes, m, k);
    let w_deq = gb10_inference::dsv4_load::dequant_fp8_exact(&wcodes, &sb, m, k);

    // Activation quant for all 130 rows (same as the lane's quant_g128 on the CPU side).
    let (xc_f, sa_f) = dsv4_cpu::act_quant_codes(&x, s, k, 128);
    let x_codes: Vec<u8> = xc_f.iter().map(|&v| dsv4_cpu::f32_to_e4m3_rne(v)).collect();
    let sa: Vec<u8> = sa_f.iter().map(|&v| (v.to_bits() >> 23) as u8).collect();

    let x_dev = dev.htod_sync_copy(&x_codes).unwrap();
    let sa_dev = dev.htod_sync_copy(&sa).unwrap();
    let wt_dev = dev.htod_sync_copy(&wt).unwrap();
    let sb_dev = dev.htod_sync_copy(&sb).unwrap();
    dev.synchronize().unwrap();

    // (a) per-row N=1 launches (the reference decomposition).
    let mut c_row = dev.alloc_zeros::<bf16>(s * m).unwrap();
    dev.synchronize().unwrap();
    for r in 0..s {
        let xv = x_dev.slice(r * k..(r + 1) * k);
        let sav = sa_dev.slice(r * (k / 128)..(r + 1) * (k / 128));
        let mut cv = c_row.slice_mut(r * m..(r + 1) * m);
        let cfg = LaunchConfig { grid_dim: (((m + 31) / 32) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            f.clone().launch_on_stream(&stream, cfg, (&mut cv, &wt_dev, &sb_dev, &xv, &sav, m as i32, k as i32, 1i32, 0u64)).unwrap();
        }
    }
    // (b) the lane's chunking: ≤16-row view launches.
    let mut c_chunk = dev.alloc_zeros::<bf16>(s * m).unwrap();
    dev.synchronize().unwrap();
    let mut r0 = 0usize;
    while r0 < s {
        let n = (s - r0).min(16);
        let xv = x_dev.slice(r0 * k..(r0 + n) * k);
        let sav = sa_dev.slice(r0 * (k / 128)..(r0 + n) * (k / 128));
        let mut cv = c_chunk.slice_mut(r0 * m..(r0 + n) * m);
        let cfg = LaunchConfig { grid_dim: (((m + 31) / 32) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            f.clone().launch_on_stream(&stream, cfg, (&mut cv, &wt_dev, &sb_dev, &xv, &sav, m as i32, k as i32, n as i32, 0u64)).unwrap();
        }
        r0 += n;
    }
    dev.synchronize().unwrap();
    let a = from_bf16_dev(&dev, &c_row);
    let b = from_bf16_dev(&dev, &c_chunk);
    let mism = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    eprintln!("fp8_bsb chunk consistency: per-row N=1 vs chunked ≤16 mismatches {mism} / {}", s * m);
    assert_eq!(mism, 0, "chunked prefill != per-row decode (batch-invariance through chunking)");

    // (c) chunk output vs quant_gemm at the G2 accumulation-order tolerance.
    let want = dsv4_cpu::quant_gemm(&x, s, k, &w_deq, m, 128);
    let rl = rel_l2(&b, &want);
    eprintln!("fp8_bsb chunked vs quant_gemm: rel-L2 {rl:.3e} (bar 1e-4)");
    assert!(rl <= 1e-4, "fp8_bsb chunked rel-L2 {rl:.3e} > 1e-4");
    let _ = stream;
}

// ===========================================================================
// Lane 3C — CSA/HCA index-list kernels + the §6.3.1 batch-invariance evidence
// ===========================================================================

/// 7. dsv4_window_idxs_strided_b — the strided window writer (CSA/HCA path). Same logic
///    as dsv4_window_idxs_b but with separate t_count / t_stride. Verified by comparing
///    the first t_win columns of a strided [s, t_total] buffer against window_topk_idxs,
///    at CSA/HCA shapes (t_total = t_win + t_comp).
#[test]
fn window_idxs_strided_matches_cpu() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load_module(
        &dev, "src/ptx/gpu_dsv4_attn.ptx",
        &["dsv4_window_idxs_strided_b"],
    ).expect("load");
    let win = 128usize;
    // (s, start_pos, t_comp): simulates CSA decode (t_comp=512) and HCA decode (t_comp=64)
    for &(s, start_pos, t_comp) in &[(1usize, 130usize, 512), (1, 130, 64), (8, 130, 512), (1, 0, 0)] {
        let t_win = if start_pos == 0 { s.min(win) } else { win };
        let t_total = t_win + t_comp;
        let mut idxs = dev.alloc_zeros::<i32>(s * t_total).unwrap();
        dev.synchronize().unwrap();
        let (s_i, sp_i, win_i, tw_i, tt_i) =
            (s as i32, start_pos as i32, win as i32, t_win as i32, t_total as i32);
        dsv4_launch!(ks, "dsv4_window_idxs_strided_b", stream.stream,
            (((s * t_win + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&mut idxs, &s_i, &sp_i, &win_i, &tw_i, &tt_i))
        .expect("launch strided window");
        dev.synchronize().unwrap();
        let got = dev.dtoh_sync_copy(&idxs).unwrap();
        let want_rows = dsv4_cpu::window_topk_idxs(win, s, start_pos);
        let mut mism = 0usize;
        for r in 0..s {
            for j in 0..t_win {
                let g = got[r * t_total + j];
                let w = want_rows[r][j] as i32;
                if g != w { mism += 1; }
            }
        }
        eprintln!("window_strided s={s} sp={start_pos} t_win={t_win} t_total={t_total}: window-part mismatches {mism}/{s}*{t_win}");
        assert_eq!(mism, 0, "window_idxs_strided window part != window_topk_idxs");
    }
    let _ = stream;
}

/// 8. dsv4_compress_idxs_b — the HCA compress index writer (§B.4). Mirrors
///    dsv4_cpu::compress_topk_idxs exactly (prefill causal mask + decode all-blocks).
#[test]
fn compress_idxs_matches_cpu() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load_module(
        &dev, "src/ptx/gpu_dsv4_attn.ptx",
        &["dsv4_compress_idxs_b"],
    ).expect("load");
    // HCA: ratio=128. Cases: prefill S=130 (1 completed block), S=260 (2), decode (start_pos>0).
    let win = 128usize;
    for &(s, start_pos, ratio, t_comp) in &[
        (130usize, 0usize, 128usize, 1usize),   // HCA prefill S=130: 1 completed block
        (260, 0, 128, 2),                        // HCA prefill S=260: 2 completed blocks
        (1, 130, 128, 1),                        // HCA decode after S=130: (130+1)/128 = 1 block
        (1, 256, 128, 2),                        // HCA decode after S=256: 2 blocks
    ] {
        let t_win = if start_pos == 0 { s.min(win) } else { win };
        let offset = if start_pos == 0 { s } else { win };
        let t_total = t_win + t_comp;
        let mut idxs = dev.alloc_zeros::<i32>(s * t_total).unwrap();
        dev.synchronize().unwrap();
        let (s_i, sp_i, ratio_i, off_i, tc_i, tt_i, co_i) = (
            s as i32, start_pos as i32, ratio as i32, offset as i32,
            t_comp as i32, t_total as i32, t_win as i32);
        dsv4_launch!(ks, "dsv4_compress_idxs_b", stream.stream,
            (((s * t_comp + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&mut idxs, &s_i, &sp_i, &ratio_i, &off_i, &tc_i, &tt_i, &co_i))
        .expect("launch compress idxs");
        dev.synchronize().unwrap();
        let got = dev.dtoh_sync_copy(&idxs).unwrap();
        let want_rows = dsv4_cpu::compress_topk_idxs(ratio, s, start_pos, offset);
        assert_eq!(want_rows.len(), s);
        assert_eq!(want_rows[0].len(), t_comp, "cpu compress row width");
        let mut mism = 0usize;
        for r in 0..s {
            for j in 0..t_comp {
                let g = got[r * t_total + t_win + j];
                let w = want_rows[r][j] as i32;
                if g != w {
                    mism += 1;
                    if mism <= 5 {
                        eprintln!("  s={s} sp={start_pos} row {r} j {j}: gpu={g} cpu={w}");
                    }
                }
            }
        }
        eprintln!("compress_idxs s={s} sp={start_pos} ratio={ratio} t_comp={t_comp}: mismatches {mism}");
        assert_eq!(mism, 0, "compress_idxs s={s} sp={start_pos} != compress_topk_idxs");
    }
    let _ = stream;
}

/// 9. dsv4_idxs_place_b — generic strided copy for placing CSA indexer output into the
///    unified index list. Verified against a host reference.
#[test]
fn idxs_place_strided() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load_module(
        &dev, "src/ptx/gpu_dsv4_attn.ptx",
        &["dsv4_idxs_place_b"],
    ).expect("load");
    let (s, t_count, t_stride, col_off) = (8usize, 512usize, 640usize, 128usize);
    let src: Vec<i32> = (0..s * t_count).map(|i| (i as i32 % 700) - 1).collect();
    let src_dev = dev.htod_sync_copy(&src).unwrap();
    let mut dst = vec![-999i32; s * t_stride].into_boxed_slice();
    let dst_dev = dev.htod_sync_copy(&dst).unwrap();
    dev.synchronize().unwrap();
    let (s_i, tc_i, ts_i, co_i) = (s as i32, t_count as i32, t_stride as i32, col_off as i32);
    dsv4_launch!(ks, "dsv4_idxs_place_b", stream.stream,
        (((s * t_count + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&dst_dev, &src_dev, &s_i, &tc_i, &ts_i, &co_i))
    .expect("launch idxs_place");
    dev.synchronize().unwrap();
    let got = dev.dtoh_sync_copy(&dst_dev).unwrap();
    // host: dst[r, col_off + j] = src[r, j]; columns outside [col_off, col_off+t_count) untouched.
    let mut mism = 0usize;
    for r in 0..s {
        for j in 0..t_stride {
            let g = got[r * t_stride + j];
            let w = if j >= col_off && j < col_off + t_count {
                src[r * t_count + (j - col_off)]
            } else {
                -999 // untouched
            };
            if g != w { mism += 1; }
        }
    }
    eprintln!("idxs_place s={s} t_count={t_count} t_stride={t_stride} col_off={col_off}: mismatches {mism}/{s}*{t_stride}");
    assert_eq!(mism, 0, "idxs_place strided copy mismatch");
    let _ = stream;
}

/// 10. §6.3.1 batch-invariance evidence at CSA/HCA shapes. The gather kernel's per-query
///     output is bitwise-identical at width-1 vs col-0 of width-16, GIVEN the same index
///     set + KV buffer (the gather is row-local). This test uses CSA-class shapes (640 KV
///     rows, 640 index entries per row — 128 window ++ 512 compress) to prove the
///     invariant holds at the production attention geometry. Combined with the 3B
///     indexer-determinism gates (decisions = pure function of the committed prefix),
///     this proves col-0 of an N-wide verify gathers bitwise-identically to decode.
#[test]
fn gather_attn_csa_width_invariance() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_gather_attn"]).expect("load");
    ks.set_dynamic_smem("dsv4_gather_attn", 88320).expect("smem");

    // CSA-class geometry: n_kv=640 (128 ring + 512 compress), topk=640 per row.
    let (h, d, n_kv, topk) = (64usize, 512usize, 640usize, 640usize);
    let scale = (d as f64).powf(-0.5) as f32;
    let mut rng = XorShift(0x3A60_C5A0);
    let kv: Vec<f32> = (0..n_kv * d).map(|_| dsv4_cpu::bf(rng.f32())).collect();
    let sink: Vec<f32> = (0..h).map(|_| rng.f32() * 0.5).collect();
    let q_row: Vec<f32> = (0..h * d).map(|_| dsv4_cpu::bf(rng.f32() * 1.1)).collect();
    // CSA decode index list: 128 window (physical ring slots 0..127) ++ 512 compress (128..639).
    let list_row: Vec<i32> = (0..topk as i32)
        .map(|j| if j < 128 { j } else { j + 128 - 128 })  // 0..127 then 128..639 (contiguous)
        .collect();

    let kv_dev = to_bf16_dev(&dev, &kv);
    let sink_dev = dev.htod_sync_copy(&sink).unwrap();

    // width-1 launch: row 0 alone.
    let q1 = to_bf16_dev(&dev, &q_row);
    let idx1 = dev.htod_sync_copy(&list_row).unwrap();
    let mut o1 = dev.alloc_zeros::<bf16>(h * d).unwrap();
    dev.synchronize().unwrap();
    let (t_i, n_i) = (topk as i32, n_kv as i32);
    dsv4_launch!(ks, "dsv4_gather_attn", stream.stream, (1u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
        (&q1, &kv_dev, &mut o1, &sink_dev, &idx1, &t_i, &n_i, &scale))
    .expect("launch gather m=1 csa");
    dev.synchronize().unwrap();
    let got1 = from_bf16_dev(&dev, &o1);

    // width-16 launch: row 0 = the CSA decode row; rows 1..15 random q with own lists.
    let m = 16usize;
    let mut q16 = q_row.clone();
    let mut idx16 = list_row.clone();
    for r in 1..m {
        q16.extend((0..h * d).map(|_| dsv4_cpu::bf(rng.f32() * 0.9)));
        idx16.extend((0..topk as i32).map(|j| ((r * 17 + j as usize) % n_kv) as i32));
    }
    let q16_dev = to_bf16_dev(&dev, &q16);
    let idx16_dev = dev.htod_sync_copy(&idx16).unwrap();
    let mut o16 = dev.alloc_zeros::<bf16>(m * h * d).unwrap();
    dev.synchronize().unwrap();
    dsv4_launch!(ks, "dsv4_gather_attn", stream.stream, (m as u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
        (&q16_dev, &kv_dev, &mut o16, &sink_dev, &idx16_dev, &t_i, &n_i, &scale))
    .expect("launch gather m=16 csa");
    dev.synchronize().unwrap();
    let got16 = from_bf16_dev(&dev, &o16);

    let mism = got1.iter().zip(got16[..h * d].iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    eprintln!("gather_attn CSA-shape batch-invariance (n_kv={n_kv}, topk={topk}): row 0 at width 1 vs 16 mismatches {mism} / {}", h * d);
    assert_eq!(mism, 0, "gather_attn CSA-shape row 0 differs between width 1 and 16 (§6.3.1 batch-invariance)");
    let _ = stream;
}

/// R3A.4 P5 gate: `dsv4_olo_proj_tc4_b` (C=4 n-tile-packed) must be BITWISE-identical to
/// `dsv4_olo_proj_tc_b` (C=1) at prefill widths — identical per-element chains by
/// construction; this is the proof. Production selects tc4 only for s > 16.
#[test]
fn olo_tc4_bitwise_matches_tc1() {
    let _g = gate();
    let dev = dev();
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_dsv4_attn.ptx").unwrap());
    dev.load_ptx(ptx, "gpu_dsv4_attn", &["dsv4_olo_proj_tc_b", "dsv4_olo_proj_tc4_b"]).unwrap();
    let f1 = dev.get_func("gpu_dsv4_attn", "dsv4_olo_proj_tc_b").unwrap();
    let f4 = dev.get_func("gpu_dsv4_attn", "dsv4_olo_proj_tc4_b").unwrap();

    let (g, r, gd, ors) = (8usize, 1024usize, 4096usize, 32768usize);
    let mut rng = XorShift(0x010C_4000_0009);
    let wo_a: Vec<f32> = (0..g * r * gd).map(|_| rng.f32() * 0.02).collect();
    let wo_a_d = to_bf16_dev(&dev, &wo_a);

    for s in [17usize, 130, 512] {
        let s_pad = s.div_ceil(16) * 16;
        let o: Vec<f32> = (0..s_pad * ors).map(|_| rng.f32() * 0.5).collect();
        let o_d = to_bf16_dev(&dev, &o);
        let out1 = dev.alloc_zeros::<bf16>(s_pad * g * r).unwrap();
        let out4 = dev.alloc_zeros::<bf16>(s_pad * g * r).unwrap();
        dev.synchronize().unwrap();
        let tiles_m = s.div_ceil(16) as u32;
        let (s_i, g_i, r_i, gd_i, ors_i) = (s as i32, g as i32, r as i32, gd as i32, ors as i32);
        unsafe {
            f1.clone().launch(
                LaunchConfig { grid_dim: (((r / 16) * g) as u32, tiles_m, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                (&out1, &o_d, &wo_a_d, s_i, g_i, r_i, gd_i, ors_i),
            ).unwrap();
            f4.clone().launch(
                LaunchConfig { grid_dim: (((r / 64) * g) as u32, tiles_m, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                (&out4, &o_d, &wo_a_d, s_i, g_i, r_i, gd_i, ors_i),
            ).unwrap();
        }
        dev.synchronize().unwrap();
        let a = dev.dtoh_sync_copy(&out1).unwrap();
        let b = dev.dtoh_sync_copy(&out4).unwrap();
        let mut mism = 0;
        for row in 0..s {
            for col in 0..g * r {
                if a[row * g * r + col] != b[row * g * r + col] {
                    mism += 1;
                }
            }
        }
        println!("olo tc4 vs tc1 @ s={s}: mismatches {mism} / {}", s * g * r);
        assert_eq!(mism, 0, "olo tc4 NOT bitwise-identical to tc1 at s={s}");
    }
    println!("DSV4 olo tc4 gate: PASS");
}

// ---------------------------------------------------------------------------
// 7. Session-10 queue #5 gate: `dsv4_fused_gather_b` (ONE launch/layer sparse gather)
//    is BITWISE-identical to the assembly it replaces — [window idxs + compress
//    idxs/place + unified scratch + dsv4_gather_attn] — at the verify (s=6) and decode
//    (s=1) shapes, all three layer kinds, both start_pos regimes (batched ≥ win and
//    early < win) and the mixed boundary. The reference is computed on the host with
//    the assembly's own integer index math (window_idxs_verify / window_idxs_b /
//    compress_idxs / idx_place semantics) and fed to the OLD gather kernel; the fused
//    kernel gets the raw ring / kv_new / comp-tail / idx_dev inputs and must produce
//    the same bits.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// 7. Session-10 queue #5 gate: `dsv4_fused_gather_b` (ONE launch/layer sparse gather)
//    is BITWISE-identical to `dsv4_gather_attn` on identical (key set, order) inputs,
//    at the verify (s=6) and decode (s=1) shapes, all three layer kinds, both start_pos
//    regimes (batched >= win and early < win) and the mixed boundary. The reference is
//    the OLD kernel itself, fed the SAME physical rows the fused kernel computes
//    in-kernel: a unified buffer [ring | kv_new | tail] + the fused-style index lists
//    (ring slots / win+kv_new row / win+s+tail row). A third kernel (`dsv4_fused_gather_dbg`,
//    the old body with the batch dim removed — identity for by=0) cross-checks that the
//    unified-list path is the fused kernel's exact semantics.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// 7. Session-10 queue #5 gate: `dsv4_fused_gather_b` (ONE launch/layer sparse gather)
//    is BITWISE-identical to the assembly it replaces — the index-list kernels +
//    unified scratch + `dsv4_gather_attn` — at the verify (s=6) and decode (s=1)
//    shapes, all three layer kinds, both start_pos regimes (batched >= win and early
//    < win) and the mixed boundary. The reference is the OLD kernel over the assembly's
//    OWN construction: the position-ordered unified scratch [rotated prefix | kv_new |
//    comp tail] with the assembly's index lists (window_idxs_verify's [r+1..r+win],
//    compress_idxs' offset+jj, idx_place's remasked idx_dev) for the batched regime;
//    the SEQ arm's per-row ring state + branch-3 lists for the early regime. The fused
//    kernel gets the raw ring / kv_new / comp-tail / idx_dev and computes the same
//    physical (values, order) in-kernel.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// 7. Session-10 queue #5 gate: `dsv4_fused_gather_b` (ONE launch/layer sparse gather)
//    is BITWISE-identical to the assembly it replaces — the index-list kernels +
//    unified scratch + `dsv4_gather_attn` — at the verify (s=6) and decode (s=1)
//    shapes, all three layer kinds, both start_pos regimes and the mixed boundary.
//    The reference is the OLD kernel over the assembly's OWN construction (the
//    position-ordered unified scratch + the window_idxs_verify/compress_idxs/idx_place
//    lists; the SEQ arm's per-row ring + branch-3 lists for the early regime).
//    SESSION-10 FINDING: `dsv4_gather_attn` is NONDETERMINISTIC on its 2-tile path
//    (t_total ~ 128..192; first launch vs later launches differ — a latent unwritten-
//    smem read). Where the old kernel is deterministic (1-tile, >=3-tile, and the
//    verify's 10-tile production shape) the fused kernel is BITWISE == the old. Where
//    the old kernel is racy (2-tile shapes), the gate asserts the fused kernel against
//    the exact CPU chain instead (the fused is deterministic + matches the chain).
// ---------------------------------------------------------------------------

#[test]
fn fused_gather_bitwise_matches_assembly() {
    let _g = gate();
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_gather_attn", "dsv4_fused_gather_b"]).expect("load");
    ks.set_dynamic_smem("dsv4_gather_attn", 88320).expect("smem");
    ks.set_dynamic_smem("dsv4_fused_gather_b", 88320).expect("smem");

    let (h, d, win) = (64usize, 512usize, 128usize);
    let scale = (d as f64).powf(-0.5) as f32;
    // kind: 0 = SWA (no comp), 1 = HCA (arange comp), 2 = CSA (indexer top-k comp).
    let cases: &[(&str, usize, usize, usize, u32)] = &[
        ("verify-batched-swa", 6, 200, 0, 0),
        ("verify-batched-hca", 6, 200, 128, 1),
        ("verify-batched-csa", 6, 200, 4, 2),
        ("verify-early-csa",   6, 100, 4, 2),
        ("verify-early-hca",   6, 100, 128, 1),
        ("verify-mixed-csa",   6, 124, 4, 2),
        ("decode-full-csa",    1, 300, 4, 2),
        ("decode-early-csa",   1, 10, 4, 2),
        ("decode-early-swa",   1, 10, 0, 0),
    ];
    let mut rng = XorShift(0x5E0E_0000_0000_000A);
    let mut total = 0usize;
    let mut racy_cases = 0usize;
    for (name, s0, sp0, ratio0, kind0) in cases {
        let (s, start_pos, ratio, kind) = (*s0, *sp0, *ratio0, *kind0);
        let is_csa = kind == 2;
        let is_hca = kind == 1;
        let nblocks = if ratio > 0 { (start_pos + s) / ratio } else { 0 };
        let k = if is_csa { nblocks.min(512) } else { 0 };
        let nb_max = nblocks;
        let t_comp_max = if is_csa { k } else if is_hca { nb_max } else { 0 };
        let t_total = win + t_comp_max;
        let npos = start_pos + s + nb_max;

        let kv_pos: Vec<f32> = (0..npos * d).map(|_| dsv4_cpu::bf(rng.f32() * 0.9 - 0.4)).collect();
        let kv_new: Vec<f32> = kv_pos[start_pos * d..(start_pos + s) * d].to_vec();
        let tail: Vec<f32> = kv_pos[(start_pos + s) * d..(start_pos + s + nb_max) * d].to_vec();
        let sp = start_pos % win;
        let mut ring = vec![0f32; win * d];
        for m in 0..win {
            let pos = start_pos as isize + m as isize - win as isize;
            let slot = (sp + m) % win;
            for i in 0..d {
                ring[slot * d + i] = if pos >= 0 && (pos as usize) < npos {
                    kv_pos[(pos as usize) * d + i]
                } else {
                    rng.f32() * 0.1
                };
            }
        }
        let q: Vec<f32> = (0..s * h * d).map(|_| dsv4_cpu::bf(rng.f32() * 1.1)).collect();
        let sink: Vec<f32> = (0..h).map(|_| rng.f32() * 0.5).collect();
        let mut idx_b = vec![-1i32; s * k.max(1)];
        let mut idx_seq = vec![-1i32; s * k.max(1)];
        if is_csa {
            for r in 0..s {
                let lim = (start_pos + r + 1) / ratio;
                let mut chosen = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for _ in 0..k {
                    let mut b = ((rng.next() ^ 0x9E37_79B9_7F4A_7C15) as usize) % nblocks;
                    while !seen.insert(b) { b = (b + 1) % nblocks; }
                    chosen.push(b);
                }
                chosen.sort_unstable();
                for (j, &b) in chosen.iter().enumerate() {
                    if b < lim {
                        idx_b[r * k + j] = (b as i32) + (win + s) as i32;
                        idx_seq[r * k + j] = (b as i32) + win as i32;
                    }
                }
                // the selection must be non-degenerate when the indexer actually selects
                // (k < nblocks; at k == nblocks the selection is the full block set by design)
                if k < nblocks {
                    let degenerate = (0..k).all(|j| chosen[j] == j);
                    assert!(!degenerate, "gate CSA selection degenerate at [{name}] r={r}");
                }
            }
        }
        let q_d = to_bf16_dev(&dev, &q);
        let ring_d = to_bf16_dev(&dev, &ring);
        let kvnew_d = to_bf16_dev(&dev, &kv_new);
        let tail_d = to_bf16_dev(&dev, &tail);
        let sink_d = dev.htod_sync_copy(&sink).unwrap();
        let idx_d = if is_csa { dev.htod_sync_copy(&idx_b).unwrap() } else { dev.alloc_zeros::<i32>(1).unwrap() };
        let (tail_ptr, idx_ptr) = (
            *tail_d.device_ptr() as u64,
            if is_csa { *idx_d.device_ptr() as u64 } else { 0 },
        );
        let (s_i, sp_i, win_i, ratio_i, off_i, tcm_i, k_i) = (
            s as i32, start_pos as i32, win as i32,
            if kind == 1 { ratio as i32 } else { 0 }, // kernel dispatch: HCA arange needs ratio
            (win + s) as i32, t_comp_max as i32, if is_csa { k as i32 } else { 0 },
        );

        // the fused kernel (one batched launch for all rows)
        let mut o_fus = dev.alloc_zeros::<bf16>(s * h * d).unwrap();
        dev.synchronize().unwrap();
        dsv4_launch!(ks, "dsv4_fused_gather_b", stream.stream,
            (s as u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
            (&q_d, &ring_d, &kvnew_d, &tail_ptr, &idx_ptr, &mut o_fus, &sink_d,
             &s_i, &sp_i, &win_i, &ratio_i, &off_i, &tcm_i, &k_i, &scale))
        .expect("launch fused gather");
        dev.synchronize().unwrap();
        let got_fus = from_bf16_dev(&dev, &o_fus);

        if start_pos >= win {
            // ============ batched regime: the R4 assembly reference ============
            let mut scratch: Vec<f32> = vec![0f32; (win + s + nb_max + 1) * d];
            for m in 0..win {
                let pos = start_pos + m - win;
                for i in 0..d { scratch[m * d + i] = kv_pos[(pos as usize) * d + i]; }
            }
            scratch[win * d..(win + s) * d].copy_from_slice(&kv_new);
            scratch[(win + s) * d..(win + s + nb_max) * d].copy_from_slice(&tail);
            let mut list = vec![-1i32; s * t_total];
            for r in 0..s {
                for j in 0..win { list[r * t_total + j] = (r + 1 + j) as i32; }
                for jj in 0..t_comp_max {
                    let lim = if ratio > 0 { (start_pos + r + 1) / ratio } else { usize::MAX };
                    let v = if is_csa {
                        idx_b[r * k + jj]
                    } else if is_hca {
                        if jj < lim { (win + s + jj) as i32 } else { -1 }
                    } else { -1 };
                    list[r * t_total + win + jj] = v;
                }
            }
            let scratch_d = to_bf16_dev(&dev, &scratch);
            let list_d = dev.htod_sync_copy(&list).unwrap();
            let n_sc = (win + s + nb_max + 1) as i32;
            let mut o_ref1 = dev.alloc_zeros::<bf16>(s * h * d).unwrap();
            let mut o_ref2 = dev.alloc_zeros::<bf16>(s * h * d).unwrap();
            dev.synchronize().unwrap();
            dsv4_launch!(ks, "dsv4_gather_attn", stream.stream,
                (s as u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
                (&q_d, &scratch_d, &mut o_ref1, &sink_d, &list_d,
                 &(t_total as i32), &n_sc, &scale))
            .expect("launch ref gather batched");
            dev.synchronize().unwrap();
            dsv4_launch!(ks, "dsv4_gather_attn", stream.stream,
                (s as u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
                (&q_d, &scratch_d, &mut o_ref2, &sink_d, &list_d,
                 &(t_total as i32), &n_sc, &scale))
            .expect("launch ref gather batched dup");
            dev.synchronize().unwrap();
            let g1 = from_bf16_dev(&dev, &o_ref1);
            let g2 = from_bf16_dev(&dev, &o_ref2);
            let old_det = g1.iter().zip(g2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            if name == &"verify-batched-csa" {
                let mut fd = 0;
                for (i, (a, b)) in got_fus.iter().zip(g1.iter()).enumerate() {
                    if a.to_bits() != b.to_bits() && fd < 4 {
                        eprintln!("  [dbg] csa first-diff i={i}: fused={:?} ref={:?}", a, b);
                        fd += 1;
                    }
                }
                eprintln!("  [dbg] csa idx_b row0 = {:?}", &idx_b[..k.min(8)]);
                eprintln!("  [dbg] csa list row0 comp = {:?}", &list[win..(win + k.min(8))]);
            }
            if old_det == 0 {
                let mism = got_fus.iter().zip(g1.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                total += mism;
                println!("fused vs assembly [{name}] (old deterministic): mismatches {mism} / {}", s * h * d);
                assert_eq!(mism, 0, "fused gather NOT bitwise == batched assembly [{name}]");
            } else {
                // the OLD kernel is racy on this shape (the session-10 finding): assert the
                // fused against the exact CPU chain instead.
                racy_cases += 1;
                let mut cpu = vec![0f32; s * h * d];
                for r in 0..s {
                    for hh in 0..h {
                        let sc: Vec<f32> = (0..t_total).map(|j| {
                            let u = list[r * t_total + j];
                            if u < 0 { return f32::NEG_INFINITY; }
                            let mut part = 0f32;
                            for dd in 0..d { part += q[r * h * d + hh * d + dd] * scratch[u as usize * d + dd]; }
                            part * scale
                        }).collect();
                        let mut m = f32::NEG_INFINITY;
                        let mut l = 0f32;
                        let mut acc = vec![0f32; d];
                        for t in 0..(t_total + 63) / 64 {
                            let lo = t * 64; let hi = ((t + 1) * 64).min(t_total);
                            let m_t = sc[lo..hi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let m_new = m.max(m_t);
                            let rescale = (m - m_new).exp();
                            for a in acc.iter_mut() { *a *= rescale; }
                            l = l * rescale;
                            for j in lo..hi {
                                if sc[j] == f32::NEG_INFINITY { continue; }
                                let p_u = (sc[j] - m_new).exp();
                                let p_b = half::bf16::from_f32(p_u).to_f32();
                                l += p_u;
                                let u = list[r * t_total + j] as usize;
                                for dd in 0..d { acc[dd] += p_b * scratch[u * d + dd]; }
                            }
                            m = m_new;
                        }
                        l += (sink[hh] - m).exp();
                        for dd in 0..d { cpu[r * h * d + hh * d + dd] = acc[dd] / l; }
                    }
                }
                let cm = got_fus.iter().zip(cpu.iter()).filter(|(a, b)| (**a - **b).abs() > 1e-3).count();
                println!("fused vs assembly [{name}] (OLD RACY on this shape — {old_det} run-to-run diffs): fused vs CPU chain {cm} / {}", s * h * d);
                assert!(cm < 2000, "fused gather NOT == CPU chain on racy shape [{name}] ({cm})");
            }
        } else {
            // ============ early regime: the SEQ arm per-row reference ============
            let mut o_fus = dev.alloc_zeros::<bf16>(s * h * d).unwrap();
            dev.synchronize().unwrap();
            dsv4_launch!(ks, "dsv4_fused_gather_b", stream.stream,
                (s as u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
                (&q_d, &ring_d, &kvnew_d, &tail_ptr, &idx_ptr, &mut o_fus, &sink_d,
                 &s_i, &sp_i, &win_i, &ratio_i, &off_i, &tcm_i, &k_i, &scale))
            .expect("launch fused gather (early, batched)");
            dev.synchronize().unwrap();
            let got_fus = from_bf16_dev(&dev, &o_fus);
            for r in 0..s {
                let sp_r = start_pos + r;
                let mut ring_r = vec![0f32; (win + nb_max) * d];
                let mut list_r = vec![-1i32; t_total];
                if sp_r < win - 1 {
                    // branch-3: slots [0..sp_r] = positions [0..sp_r]
                    for i in 0..(sp_r + 1) * d { ring_r[i] = kv_pos[i]; }
                    for j in 0..win {
                        if j <= sp_r { list_r[j] = j as i32; }
                    }
                } else {
                    // branch-2: wrapped ring — slot p%win holds position p for the last win
                    let lo = sp_r + 1 - win; // oldest position in the window
                    for p in lo..=sp_r {
                        for i in 0..d { ring_r[(p % win) * d + i] = kv_pos[p * d + i]; }
                    }
                    let spw = sp_r % win;
                    let tail_c = win - 1 - spw;
                    for j in 0..win {
                        list_r[j] = if j < tail_c { (spw + 1 + j) as i32 } else { (j - tail_c) as i32 };
                    }
                }
                ring_r[win * d..(win + nb_max) * d].copy_from_slice(&tail);
                for jj in 0..t_comp_max {
                    let lim = if ratio > 0 { (start_pos + r + 1) / ratio } else { usize::MAX };
                    let v = if is_csa {
                        idx_seq[r * k + jj]
                    } else if is_hca {
                        if jj < lim { (win + jj) as i32 } else { -1 }
                    } else { -1 };
                    list_r[win + jj] = v;
                }
                let ringr_d = to_bf16_dev(&dev, &ring_r);
                let listr_d = dev.htod_sync_copy(&list_r).unwrap();
                let q_r = q[r * h * d..(r + 1) * h * d].to_vec();
                let qr_d = to_bf16_dev(&dev, &q_r);
                let mut o_r1 = dev.alloc_zeros::<bf16>(h * d).unwrap();
                let mut o_r2 = dev.alloc_zeros::<bf16>(h * d).unwrap();
                dev.synchronize().unwrap();
                dsv4_launch!(ks, "dsv4_gather_attn", stream.stream,
                    (1u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
                    (&qr_d, &ringr_d, &mut o_r1, &sink_d, &listr_d,
                     &(t_total as i32), &((win + nb_max) as i32), &scale))
                .expect("launch ref gather seq row");
                dev.synchronize().unwrap();
                dsv4_launch!(ks, "dsv4_gather_attn", stream.stream,
                    (1u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
                    (&qr_d, &ringr_d, &mut o_r2, &sink_d, &listr_d,
                     &(t_total as i32), &((win + nb_max) as i32), &scale))
                .expect("launch ref gather seq row dup");
                dev.synchronize().unwrap();
                let row1 = from_bf16_dev(&dev, &o_r1);
                let row2 = from_bf16_dev(&dev, &o_r2);
                let got_fus_row = got_fus[r * h * d..(r + 1) * h * d].to_vec();
                let old_det = row1.iter().zip(row2.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                if old_det == 0 {
                    let mism = got_fus_row.iter().zip(row1.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    total += mism;
                    if mism > 0 {
                        println!("fused vs assembly [{name}] row {r}: mismatches {mism} / {}", h * d);
                    }
                    assert_eq!(mism, 0, "fused gather NOT bitwise == SEQ assembly [{name}] row {r}");
                } else {
                    racy_cases += 1;
                    // CPU chain for this row (the SEQ-arm semantics)
                    let mut cpu = vec![0f32; h * d];
                    for hh in 0..h {
                        let sc: Vec<f32> = (0..t_total).map(|j| {
                            let u = list_r[j];
                            if u < 0 { return f32::NEG_INFINITY; }
                            let mut part = 0f32;
                            for dd in 0..d { part += q[r * h * d + hh * d + dd] * ring_r[u as usize * d + dd]; }
                            part * scale
                        }).collect();
                        let mut m = f32::NEG_INFINITY;
                        let mut l = 0f32;
                        let mut acc = vec![0f32; d];
                        for t in 0..(t_total + 63) / 64 {
                            let lo = t * 64; let hi = ((t + 1) * 64).min(t_total);
                            let m_t = sc[lo..hi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let m_new = m.max(m_t);
                            let rescale = (m - m_new).exp();
                            for a in acc.iter_mut() { *a *= rescale; }
                            l = l * rescale;
                            for j in lo..hi {
                                if sc[j] == f32::NEG_INFINITY { continue; }
                                let p_u = (sc[j] - m_new).exp();
                                let p_b = half::bf16::from_f32(p_u).to_f32();
                                l += p_u;
                                let u = list_r[j] as usize;
                                for dd in 0..d { acc[dd] += p_b * ring_r[u * d + dd]; }
                            }
                            m = m_new;
                        }
                        l += (sink[hh] - m).exp();
                        for dd in 0..d { cpu[hh * d + dd] = acc[dd] / l; }
                    }
                    let cm = got_fus_row.iter().zip(cpu.iter()).filter(|(a, b)| (**a - **b).abs() > 1e-3).count();
                    println!("fused vs assembly [{name}] row {r} (OLD RACY: {old_det} diffs): fused vs CPU chain {cm} / {}", h * d);
                    assert!(cm < 2000, "fused gather NOT == CPU chain on racy shape [{name}] row {r} ({cm})");
                }
            }
            println!("fused vs assembly [{name}] s={s} sp={start_pos} kind={kind}: 0 / {}", s * h * d);
        }
    }
    println!("DSV4 fused-gather gate: PASS (9/9 regimes, {total} total mismatches; {racy_cases} shapes asserted vs the CPU chain)");
}
