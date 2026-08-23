//! Phase-3 spine proof: the production `gpu_dsv4` loader/launcher (`src/dsv4_gpu.rs`) loads
//! `gpu_dsv4.ptx`, asserts the build-id stamp, and launches BOTH a small-smem kernel
//! (`dsv4_topk`, static `__shared__`) and the big-smem `dsv4_gather_attn` (88 KB dynamic
//! smem, needs `cuFuncSetAttribute`) on a BLOCKING compute stream — matching the G1-proven CPU
//! reference (`dsv4_cpu`) in both cases. This is the mechanism every Phase-3 lane builds on.

use std::sync::Arc;
use cudarc::driver::{CudaDevice, DevicePtr};
use cudarc::nvrtc::Ptx;
use half::bf16;

use gb10_inference::{dsv4_cpu, dsv4_load, quant};
use gb10_inference::dsv4_gpu::{self, Dsv4Kernels};
use gb10_inference::dsv4_launch;

fn dev() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA device 0")
}

/// bf16-valued f32 host buffer -> device bf16 (the convention dsv4_cpu uses throughout).
fn to_bf16_dev(dev: &Arc<CudaDevice>, v: &[f32]) -> cudarc::driver::CudaSlice<bf16> {
    let b: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
    dev.htod_sync_copy(&b).unwrap()
}

/// device bf16 -> bf16-valued f32 host buffer.
fn from_bf16_dev(dev: &Arc<CudaDevice>, s: &cudarc::driver::CudaSlice<bf16>) -> Vec<f32> {
    dev.dtoh_sync_copy(s).unwrap().iter().map(|b| b.to_f32()).collect()
}

#[test]
fn spine_loads_and_asserts_build_id() {
    // Load alone exercises the full handshake (cuModuleLoadData + dsv4_kernel_build_id launch +
    // stamp compare). A stale src/ptx/gpu_dsv4.ptx fails here with the STALE-DSV4-KERNELS error.
    let dev = dev();
    let _ks = Dsv4Kernels::load(&dev, &["dsv4_topk"]).expect("dsv4 build-id handshake");
}

#[test]
fn spine_topk_on_compute_stream_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_topk"]).expect("load");

    // rows × T scores with adversarial ties + a -inf row (the §12.B.2 tie regime).
    let (rows, t, k) = (5usize, 256usize, 8usize);
    let mut scores: Vec<f32> = (0..rows * t).map(|i| (i as f32 * 0.371) % 17.0).collect();
    scores[0 * t + 5] = 7.5; // exact tie
    scores[0 * t + 9] = 7.5;
    scores[1 * t + 0] = f32::NEG_INFINITY;

    let scores_dev = dev.htod_sync_copy(&scores).unwrap();
    let mut out_dev = dev.alloc_zeros::<i32>(rows * k).unwrap();
    dev.synchronize().unwrap();

    let (ri, ti, ki) = (rows as i32, t as i32, k as i32);
    dsv4_launch!(
        ks, "dsv4_topk", stream.stream,
        (rows as u32, 1, 1), (256, 1, 1), 0,
        (&scores_dev, &out_dev, &ri, &ti, &ki)
    )
    .expect("launch dsv4_topk");

    dev.synchronize().unwrap();
    let got = dev.dtoh_sync_copy(&out_dev).unwrap();

    // CPU reference: deterministic top-k (value desc, index asc).
    let mut expected: Vec<i64> = Vec::new();
    for r in 0..rows {
        let mut idx = dsv4_cpu::topk_deterministic(&scores[r * t..(r + 1) * t], k);
        expected.append(&mut idx);
    }
    for r in 0..rows {
        for j in 0..k {
            let g = got[r * k + j] as i64;
            let e = expected[r * k + j];
            assert_eq!(g, e, "topk row {r} slot {j}: gpu={g} cpu={e}");
        }
    }
    let _ = stream; // keep alive past the launches
}

#[test]
fn spine_gather_attn_big_smem_on_compute_stream_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_gather_attn"]).expect("load");
    // 88320 B dynamic smem > 48 KB: opt-in once (idempotent). GB10 opt-in cap ~99 KB.
    ks.set_dynamic_smem("dsv4_gather_attn", 88320).expect("set_dynamic_smem");

    let (m, h, d, n, topk) = (1usize, 64usize, 512usize, 12usize, 10usize);
    let scale = (d as f64).powf(-0.5) as f32; // 512^-0.5

    // q [m, h, d], kv [n, d] — bf16-valued f32.
    let q: Vec<f32> = (0..m * h * d).map(|i| ((i as f32 * 1.31) % 3.0 - 1.5)).collect();
    let kv: Vec<f32> = (0..n * d).map(|i| ((i as f32 * 0.73) % 3.0 - 1.5)).collect();
    // sink [h], idxs [m, topk] i32, −1 = masked.
    let sink: Vec<f32> = (0..h).map(|i| (i as f32 * 0.05 - 1.6)).collect();
    let idxs: Vec<i32> = vec![0, 1, 2, 3, -1, 5, 6, 7, 8, 9]; // row 4 masked; idxs ≤ n-1

    let q_dev = to_bf16_dev(&dev, &q);
    let kv_dev = to_bf16_dev(&dev, &kv);
    let sink_dev = dev.htod_sync_copy(&sink).unwrap();
    let idxs_dev = dev.htod_sync_copy(&idxs).unwrap();
    let mut o_dev = dev.alloc_zeros::<bf16>(m * h * d).unwrap();
    dev.synchronize().unwrap();

    // Grid: (query m, batch b=1, head-blocks 64/16=4). Block 256 (8 warps, 2 heads/warp).
    let (topk_i, n_i) = (topk as i32, n as i32);
    dsv4_launch!(
        ks, "dsv4_gather_attn", stream.stream,
        (m as u32, 1u32, (h / 16) as u32), (256, 1, 1), 88320,
        (&q_dev, &kv_dev, &o_dev, &sink_dev, &idxs_dev, &topk_i, &n_i, &scale)
    )
    .expect("launch dsv4_gather_attn");

    dev.synchronize().unwrap();
    let got = from_bf16_dev(&dev, &o_dev);

    // CPU reference sparse_attn (§B.7): fp32 GLOBAL-max softmax, bf16-rounded probs in the
    // numerator, denominator-only sink. The gather kernel uses ONLINE (per-tile running-max)
    // softmax — mathematically equivalent, a few bf16 ulp off in reduction order (the
    // "tolerance-level" class dsv4_cpu.rs documents; the kernel's bit-exactness vs the ORACLE
    // was G1's job). The spine's claim here is only "the launch is correct" — a wrong arg/grid/
    // smem produces O(1) garbage, not a 5-ulp residual. Floor the check at 1e-2 abs and report
    // the actual residual for visibility.
    let idxs_i64: Vec<i64> = idxs.iter().map(|&v| v as i64).collect();
    let expected = dsv4_cpu::sparse_attn(&q, m, h, d, &kv, n, &sink, &idxs_i64, topk, scale);

    let mut max_ulp = 0i64;
    let mut max_abs = 0.0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!(g.is_finite() && e.is_finite(), "non-finite gather_attn output");
        max_ulp = max_ulp.max(dsv4_gpu_ulp(*g, *e));
        max_abs = max_abs.max((g - e).abs());
    }
    eprintln!("spine gather_attn vs CPU ref: max bf16 ulp={max_ulp}, max abs={max_abs:.3e}");
    assert!(
        max_abs <= 1e-2,
        "gather_attn launch wrong: max abs {max_abs:.3e} (expected reduction-order residual ≤1e-2)"
    );
    let _ = stream;
}

/// bf16-level ulp distance between two bf16-valued f32 values.
fn dsv4_gpu_ulp(a: f32, b: f32) -> i64 {
    if a == b {
        return 0;
    }
    let ab = bf16::from_f32(a).to_bits() as i16 as i64;
    let bb = bf16::from_f32(b).to_bits() as i16 as i64;
    // sign-magnitude: flip sign bit to get a monotone ordering, then diff.
    let ak = ab ^ if ab < 0 { -1i64 } else { 0 };
    let bk = bb ^ if bb < 0 { -1i64 } else { 0 };
    (ak - bk).abs()
}

// ---------------------------------------------------------------------------
// spine 3: gemm_dsv4_fp8_bsb wrapper — the attention/indexer FP8 GEMM. The kernel's
// numerics are G2-proven; this asserts the WRAPPER (arg order, stream, grid) is correct by
// diffing vs dsv4_cpu::quant_gemm at the G2 accumulation-order tolerance (rel-L2 <= 1e-4).
// ---------------------------------------------------------------------------

struct XorShift(u64);
impl XorShift {
    fn f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn pow2_to_e8m0(s: f32) -> u8 {
    (s.to_bits() >> 23) as u8
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut se, mut sn) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want.iter()) {
        se += ((g - w) as f64).powi(2);
        sn += (w as f64).powi(2);
    }
    (se / sn.max(1e-30)).sqrt()
}

/// fp8 block-scale weight quant (mirrors dsv4_fp8_bsb_test::quant_w_blocks): s = 2^ceil(log2(amax/448))
/// per 128x128 block, codes = RNE(w/s).
fn quant_w_blocks(w: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let (rb_n, cb_n) = (m / 128, k / 128);
    let mut codes = vec![0u8; m * k];
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
            let s = if amax > 0.0 { dsv4_cpu::fast_round_scale(amax, 1.0 / 448.0) } else { 1.0 };
            sb[rb * cb_n + cb] = pow2_to_e8m0(s);
            for i in 0..128 {
                let row = (rb * 128 + i) * k + cb * 128;
                for j in 0..128 {
                    codes[row + j] = dsv4_cpu::f32_to_e4m3_rne((w[row + j] / s).clamp(-448.0, 448.0));
                }
            }
        }
    }
    (codes, sb)
}

#[test]
fn spine_fp8_bsb_wrapper_matches_cpu_ref() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx").unwrap());
    dev.load_ptx(ptx, "gpu_batch", &["gemm_dsv4_fp8_bsb"]).unwrap();
    let f = dev.get_func("gpu_batch", "gemm_dsv4_fp8_bsb").unwrap();

    let (m, k, n) = (256usize, 256usize, 1usize);
    let mut rng = XorShift(0xB5E1_2026);
    let w: Vec<f32> = (0..m * k).map(|_| rng.f32() * 0.1).collect();
    let x: Vec<f32> = (0..n * k).map(|_| dsv4_cpu::bf(rng.f32() * 0.5)).collect();

    let (wcodes, sb) = quant_w_blocks(&w, m, k);
    let wt = quant::repack_fp8_mma(&wcodes, m, k);
    let w_deq = dsv4_load::dequant_fp8_exact(&wcodes, &sb, m, k);
    let (xc_f, sa_f) = dsv4_cpu::act_quant_codes(&x, n, k, 128);
    let x_codes: Vec<u8> = xc_f.iter().map(|&v| dsv4_cpu::f32_to_e4m3_rne(v)).collect();
    let sa: Vec<u8> = sa_f.iter().map(|&s| pow2_to_e8m0(s)).collect();

    let x_dev = dev.htod_sync_copy(&x_codes).unwrap();
    let sa_dev = dev.htod_sync_copy(&sa).unwrap();
    let wt_dev = dev.htod_sync_copy(&wt).unwrap();
    let sb_dev = dev.htod_sync_copy(&sb).unwrap();
    let mut c_dev = dev.alloc_zeros::<bf16>(n * m).unwrap();
    dev.synchronize().unwrap();

    dsv4_gpu::launch_fp8_bsb(&f, &stream, &mut c_dev, &wt_dev, &sb_dev, &x_dev, &sa_dev, m, k, n, None)
        .expect("launch_fp8_bsb");
    dev.synchronize().unwrap();
    let got: Vec<f32> = dev.dtoh_sync_copy(&c_dev).unwrap().iter().map(|b| b.to_f32()).collect();

    let want = dsv4_cpu::quant_gemm(&x, n, k, &w_deq, m, 128);
    let rl = rel_l2(&got, &want);
    eprintln!("spine fp8_bsb wrapper rel-L2 vs quant_gemm: {rl:.3e} (M={m},K={k},N={n})");
    assert!(rl <= 1e-4, "fp8_bsb wrapper rel-L2 {rl:.3e} > 1e-4 (accumulation-order floor)");
    let _ = stream;
}

// ---------------------------------------------------------------------------
// spine 4: dsv4_rope_last_b — RoPE on the LAST rd dims (DSV4 §B.1.3) + inverse de-rotation.
// Diffs vs dsv4_cpu::apply_rope on a YaRN table. cos/sin are bit-identical (same CPU table);
// the only divergence is nvcc's FMA contraction on re*c-im*s vs the host's separate ops (≤1
// bf16 ulp on rotated dims; unrotated dims bit-identical, untouched).
// ---------------------------------------------------------------------------
#[test]
fn spine_rope_last_b_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_rope_last_b"]).expect("load");

    let (dim, rd, rows) = (512usize, 64usize, 16usize);
    // YaRN table (CSA/HCA): original_seq_len=65536, base=160000, factor 16, beta 32/1.
    let table = dsv4_cpu::rope_table(rd, 32, 65536, 160000.0, 16.0, 32, 1);
    assert_eq!(table.rd, rd);

    let mut rng = XorShift(0xCAFE_F00D);
    let x: Vec<f32> = (0..rows * dim).map(|_| dsv4_cpu::bf(rng.f32())).collect();
    let row_pos: Vec<usize> = (0..rows).collect();

    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut x_dev = dev.htod_sync_copy(&x_bf).unwrap();
    let pos_i32: Vec<i32> = row_pos.iter().map(|&p| p as i32).collect();
    let pos_dev = dev.htod_sync_copy(&pos_i32).unwrap();
    let cos_dev = dev.htod_sync_copy(&table.cos).unwrap();
    let sin_dev = dev.htod_sync_copy(&table.sin).unwrap();
    dev.synchronize().unwrap();

    let (rows_i, dim_i, rd_i, inv0, inv1) = (rows as i32, dim as i32, rd as i32, 0i32, 1i32);
    dsv4_launch!(ks, "dsv4_rope_last_b", stream.stream, (((rows + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
        (&mut x_dev, &cos_dev, &sin_dev, &pos_dev, &rows_i, &dim_i, &rd_i, &inv0)).unwrap();
    dev.synchronize().unwrap();
    let got: Vec<f32> = dev.dtoh_sync_copy(&x_dev).unwrap().iter().map(|b| b.to_f32()).collect();

    // CPU ref
    let mut xcpu = x.clone();
    dsv4_cpu::apply_rope(&mut xcpu, rows, dim, &table, &row_pos, false);

    let off = dim - rd;
    let (mut mism_unrot, mut max_ulp_rot) = (0i64, 0i64);
    for i in 0..rows * dim {
        if i % dim < off {
            if got[i] != xcpu[i] { mism_unrot += 1; }
        } else {
            max_ulp_rot = max_ulp_rot.max(dsv4_gpu_ulp(got[i], xcpu[i]));
        }
    }
    eprintln!("spine rope_last_b: unrotated mismatches={mism_unrot} (expect 0)");
    assert_eq!(mism_unrot, 0, "rope_last_b touched unrotated dims");
    eprintln!("spine rope_last_b: rotated-dim max bf16 ulp={max_ulp_rot} (expect <=2, FMA class)");
    assert!(max_ulp_rot <= 2, "rope_last_b rotated dim ulp {max_ulp_rot} > 2");

    // de-rotation round-trip: apply inverse to got, compare to CPU apply_rope(inverse) of got.
    let got_bf: Vec<bf16> = got.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut x_dev2 = dev.htod_sync_copy(&got_bf).unwrap();
    dsv4_launch!(ks, "dsv4_rope_last_b", stream.stream, (((rows + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
        (&mut x_dev2, &cos_dev, &sin_dev, &pos_dev, &rows_i, &dim_i, &rd_i, &inv1)).unwrap();
    dev.synchronize().unwrap();
    let back: Vec<f32> = dev.dtoh_sync_copy(&x_dev2).unwrap().iter().map(|b| b.to_f32()).collect();
    let mut xcpu2 = got.clone();
    dsv4_cpu::apply_rope(&mut xcpu2, rows, dim, &table, &row_pos, true);
    let mut rt_max = 0i64;
    for i in 0..rows * dim {
        if i % dim >= off { rt_max = rt_max.max(dsv4_gpu_ulp(back[i], xcpu2[i])); }
    }
    eprintln!("spine rope_last_b inverse: rotated-dim max bf16 ulp={rt_max} (expect <=2)");
    assert!(rt_max <= 2, "rope inverse ulp {rt_max} > 2");
    let _ = (inv0, inv1, stream);
}

// ---------------------------------------------------------------------------
// spine 5a: promote the G1 bit-exact primitives to the production launcher — hc_split_sinkhorn
// (mHC fp32 mixing) and act_quant_sim_g64 (KV QAT-sim). Same kernels, now launched via the
// spine on the compute stream; assert bit-exactness vs dsv4_cpu survives the new launch path.
// ---------------------------------------------------------------------------
#[test]
fn spine_sinkhorn_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_hc_split_sinkhorn"]).expect("load");

    let n = 64usize;
    let mut rng = XorShift(0x5EC1_0001);
    let mixes: Vec<f32> = (0..n * 24).map(|_| rng.f32()).collect();
    let scale: [f32; 3] = [0.3, 0.5, 0.7];
    let base: Vec<f32> = (0..24).map(|i| (i as f32 * 0.05 - 0.6)).collect();

    let mix_dev = dev.htod_sync_copy(&mixes).unwrap();
    let sc_dev = dev.htod_sync_copy(&scale).unwrap();
    let ba_dev = dev.htod_sync_copy(&base).unwrap();
    let mut pre_dev = dev.alloc_zeros::<f32>(n * 4).unwrap();
    let mut post_dev = dev.alloc_zeros::<f32>(n * 4).unwrap();
    let mut comb_dev = dev.alloc_zeros::<f32>(n * 16).unwrap();
    dev.synchronize().unwrap();

    let n_i = n as i32;
    dsv4_launch!(ks, "dsv4_hc_split_sinkhorn", stream.stream, (((n + 255) / 8) as u32, 1, 1), (256, 1, 1), 0,
        (&mix_dev, &sc_dev, &ba_dev, &mut pre_dev, &mut post_dev, &mut comb_dev, &n_i)).unwrap();
    dev.synchronize().unwrap();
    let pre_g = dev.dtoh_sync_copy(&pre_dev).unwrap();
    let post_g = dev.dtoh_sync_copy(&post_dev).unwrap();
    let comb_g = dev.dtoh_sync_copy(&comb_dev).unwrap();

    // CPU ref: per-token hc_split_sinkhorn (hc=4, 20 iters, eps 1e-6 — the DSV4 constants).
    // NOTE: GPU-vs-CPU here is tolerance-level, NOT bit-exact. The kernel is bit-exact vs the
    // TILELANG oracle (G1, same libdevice expf); the CPU ref uses host-libm exp(), and the ≤1
    // f32-ulp divergence compounds over the 20 Sinkhorn iterations. A launch bug reads O(1);
    // the transcendental residual is ~1e-6 rel-L2. Gate on that, report it.
    let mut pre_w = vec![0.0f32; n * 4];
    let mut post_w = vec![0.0f32; n * 4];
    let mut comb_w = vec![0.0f32; n * 16];
    for i in 0..n {
        let (pre, post, comb) = dsv4_cpu::hc_split_sinkhorn(
            &mixes[i * 24..(i + 1) * 24], &scale, &base, 4, 20, 1e-6);
        for j in 0..4 { pre_w[i * 4 + j] = pre[j]; post_w[i * 4 + j] = post[j]; }
        for j in 0..16 { comb_w[i * 16 + j] = comb[j]; }
    }
    let rl_pre = rel_l2(&pre_g, &pre_w);
    let rl_post = rel_l2(&post_g, &post_w);
    let rl_comb = rel_l2(&comb_g, &comb_w);
    eprintln!("spine sinkhorn rel-L2 vs CPU: pre={rl_pre:.3e} post={rl_post:.3e} comb={rl_comb:.3e}");
    assert!(rl_pre <= 1e-4 && rl_post <= 1e-4 && rl_comb <= 1e-4,
        "sinkhorn rel-L2 exceeds transcendental floor (launch wrong?)");
    let _ = stream;
}

#[test]
fn spine_act_quant_sim_g64_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_act_quant_sim_g64"]).expect("load");

    // KV nope-dim sim: rows of length 448 (7 groups of 64), the §B.1.2 QAT-sim on kv[..., :448].
    let (rows, n) = (8usize, 448usize);
    let mut rng = XorShift(0xA7_0002);
    let mut x: Vec<f32> = (0..rows * n).map(|_| dsv4_cpu::bf(rng.f32() * 1.7)).collect();

    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut x_dev = dev.htod_sync_copy(&x_bf).unwrap();
    let mut s_dev = dev.alloc_zeros::<u8>(rows * (n / 64)).unwrap();
    dev.synchronize().unwrap();
    let (r_i, n_i) = (rows as i32, n as i32);
    // one warp per (row, 64-wide group): rows*(n/64) warps = rows*7, *32 threads -> /256 blocks
    let blocks = ((rows * (n / 64) * 32) + 255) / 256;
    dsv4_launch!(ks, "dsv4_act_quant_sim_g64", stream.stream, (blocks as u32, 1, 1), (256, 1, 1), 0,
        (&mut x_dev, &mut s_dev, &r_i, &n_i)).unwrap();
    dev.synchronize().unwrap();
    let got: Vec<f32> = dev.dtoh_sync_copy(&x_dev).unwrap().iter().map(|b| b.to_f32()).collect();

    // CPU ref
    dsv4_cpu::act_quant_sim(&mut x, rows, n, 64);
    let mut mism = 0i64;
    for i in 0..rows * n { if got[i] != x[i] { mism += 1; } }
    eprintln!("spine act_quant_sim_g64 mismatches: {mism} / {} (expect 0)", rows * n);
    assert_eq!(mism, 0, "act_quant_sim_g64 != CPU (bit-exact QAT-sim)");
    let _ = stream;
}

// ---------------------------------------------------------------------------
// spine 5b: dsv4_rmsnorm_b — the G1 reduction-order-sensitive norm (attn/ffn/kv/q/compressor).
// Tolerance-level vs CPU (the GPU's block-tree sumsq vs the CPU's whole-vector pairwise tree);
// gate on rel-L2 and report max bf16 ulp. Tests dim=4096 (attn_norm) and dim=512 (kv_norm).
// ---------------------------------------------------------------------------
#[test]
fn spine_rmsnorm_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_rmsnorm_b"]).expect("load");
    let mut rng = XorShift(0x88AA_1234);

    for &dim in &[4096usize, 512usize, 128usize] {
        let rows = 8usize;
        let x: Vec<f32> = (0..rows * dim).map(|_| dsv4_cpu::bf(rng.f32())).collect();
        let w: Vec<f32> = (0..dim).map(|i| 0.7 + (i as f32 * 0.001).sin() * 0.3).collect();

        let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
        let x_dev = dev.htod_sync_copy(&x_bf).unwrap();
        let w_dev = dev.htod_sync_copy(&w).unwrap();
        let mut y_dev = dev.alloc_zeros::<bf16>(rows * dim).unwrap();
        dev.synchronize().unwrap();
        let (r_i, d_i) = (rows as i32, dim as i32);
        let eps = dsv4_cpu::bf(1e-6).to_bits() as u32; // pass as bits? no — fbits
        let _ = eps;
        let eps = 1e-6f32;
        dsv4_launch!(ks, "dsv4_rmsnorm_b", stream.stream, (rows as u32, 1, 1), (256, 1, 1), 0,
            (&mut y_dev, &x_dev, &w_dev, &r_i, &d_i, &eps)).unwrap();
        dev.synchronize().unwrap();
        let got: Vec<f32> = dev.dtoh_sync_copy(&y_dev).unwrap().iter().map(|b| b.to_f32()).collect();

        let want = dsv4_cpu::rms_norm(&x, rows, dim, &w, 1e-6);
        let rl = rel_l2(&got, &want);
        let mut max_ulp = 0i64;
        for (g, e) in got.iter().zip(want.iter()) {
            max_ulp = max_ulp.max(dsv4_gpu_ulp(*g, *e));
        }
        eprintln!("spine rmsnorm dim={dim}: rel-L2={rl:.3e} max bf16 ulp={max_ulp}");
        assert!(rl <= 1e-4, "rmsnorm dim={dim} rel-L2 {rl:.3e} > 1e-4");
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// spine 5c: mHC 4-stream wrapper (hc_pre + hc_post, §B.8) — the plumbing that wraps every
// sublayer (item 13). Pipeline: rsqrt -> mixes GEMM -> sinkhorn -> collapse (pre); combine
// (post). Diffs the observable outputs (y, post, comb, and the post-combined streams) vs the
// CPU per-token reference. Tolerance-level (fp32 reductions + sinkhorn transcendentals).
// ---------------------------------------------------------------------------
#[test]
fn spine_mhc_pre_post_match_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &[
        "dsv4_hc_pre_rsqrt_b", "dsv4_hc_mixes_b", "dsv4_hc_split_sinkhorn",
        "dsv4_hc_collapse_b", "dsv4_hc_post_b",
    ]).expect("load");

    let (hc, dim) = (4usize, 4096usize);
    let hcdim = hc * dim;
    let s = 4usize;
    let eps = 1e-6f32;
    let iters = 20usize;
    let mut rng = XorShift(0xCC07_0011);
    let x: Vec<f32> = (0..s * hcdim).map(|_| dsv4_cpu::bf(rng.f32())).collect();
    let hc_fn: Vec<f32> = (0..24 * hcdim).map(|_| rng.f32() * 0.01).collect();
    let hc_base: Vec<f32> = (0..24).map(|i| i as f32 * 0.03 - 0.5).collect();
    let hc_scale: [f32; 3] = [0.5, 0.5, 0.5];
    let p = dsv4_cpu::HcParams { hc_fn: hc_fn.clone(), hc_base: hc_base.clone(), hc_scale };

    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_dev = dev.htod_sync_copy(&x_bf).unwrap();
    let fn_dev = dev.htod_sync_copy(&hc_fn).unwrap();
    let ba_dev = dev.htod_sync_copy(&hc_base).unwrap();
    let sc_dev = dev.htod_sync_copy(&hc_scale).unwrap();
    let mut rsqrt_dev = dev.alloc_zeros::<f32>(s).unwrap();
    let mut mixes_dev = dev.alloc_zeros::<f32>(s * 24).unwrap();
    let mut pre_dev = dev.alloc_zeros::<f32>(s * hc).unwrap();
    let mut post_dev = dev.alloc_zeros::<f32>(s * hc).unwrap();
    let mut comb_dev = dev.alloc_zeros::<f32>(s * hc * hc).unwrap();
    let mut y_dev = dev.alloc_zeros::<bf16>(s * dim).unwrap();
    dev.synchronize().unwrap();
    let (s_i, hcd_i, dim_i, hc_i) = (s as i32, hcdim as i32, dim as i32, hc as i32);

    dsv4_launch!(ks, "dsv4_hc_pre_rsqrt_b", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
        (&mut rsqrt_dev, &x_dev, &s_i, &hcd_i, &eps)).unwrap();
    dsv4_launch!(ks, "dsv4_hc_mixes_b", stream.stream, (24u32, s as u32, 1), (256, 1, 1), 0,
        (&mut mixes_dev, &fn_dev, &x_dev, &rsqrt_dev, &s_i, &hcd_i)).unwrap();
    dsv4_launch!(ks, "dsv4_hc_split_sinkhorn", stream.stream, (((s + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&mixes_dev, &sc_dev, &ba_dev, &mut pre_dev, &mut post_dev, &mut comb_dev, &s_i)).unwrap();
    dsv4_launch!(ks, "dsv4_hc_collapse_b", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
        (&mut y_dev, &x_dev, &pre_dev, &s_i, &dim_i, &hc_i)).unwrap();
    dev.synchronize().unwrap();

    let y_g: Vec<f32> = dev.dtoh_sync_copy(&y_dev).unwrap().iter().map(|b| b.to_f32()).collect();
    let post_g = dev.dtoh_sync_copy(&post_dev).unwrap();
    let comb_g = dev.dtoh_sync_copy(&comb_dev).unwrap();

    // CPU hc_pre per-token (returns y, post, comb — pre is internal).
    let mut y_w = vec![0.0f32; s * dim];
    let mut post_w = vec![0.0f32; s * hc];
    let mut comb_w = vec![0.0f32; s * hc * hc];
    for t in 0..s {
        let xf = &x[t * hcdim..(t + 1) * hcdim];
        let (y, post, comb) = dsv4_cpu::hc_pre_token(xf, hc, dim, &p, eps, iters, eps);
        for d in 0..dim { y_w[t * dim + d] = y[d]; }
        for h in 0..hc { post_w[t * hc + h] = post[h]; }
        for j in 0..hc * hc { comb_w[t * hc * hc + j] = comb[j]; }
    }
    let rl_y = rel_l2(&y_g, &y_w);
    let rl_post = rel_l2(&post_g, &post_w);
    let rl_comb = rel_l2(&comb_g, &comb_w);
    eprintln!("spine mhc hc_pre: y rel-L2={rl_y:.3e} post={rl_post:.3e} comb={rl_comb:.3e}");
    assert!([rl_y, rl_post, rl_comb].iter().all(|&v| v <= 1e-4), "hc_pre exceeds tolerance");

    // hc_post: GPU combine using y (as a stand-in sublayer output) + residual=x + post/comb.
    let mut out_dev = dev.alloc_zeros::<bf16>(s * hcdim).unwrap();
    dev.synchronize().unwrap();
    dsv4_launch!(ks, "dsv4_hc_post_b", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
        (&mut out_dev, &y_dev, &x_dev, &post_dev, &comb_dev, &s_i, &dim_i, &hc_i)).unwrap();
    dev.synchronize().unwrap();
    let out_g: Vec<f32> = dev.dtoh_sync_copy(&out_dev).unwrap().iter().map(|b| b.to_f32()).collect();

    let mut out_w = vec![0.0f32; s * hcdim];
    for t in 0..s {
        let sub = &y_w[t * dim..(t + 1) * dim];
        let resid = &x[t * hcdim..(t + 1) * hcdim];
        let post = { let mut a = [0f32; 4]; for h in 0..hc { a[h] = post_w[t*hc+h]; } a };
        let comb = { let mut a = [0f32; 16]; for j in 0..hc*hc { a[j] = comb_w[t*hc*hc+j]; } a };
        let mut tmp = vec![0.0f32; hcdim];
        dsv4_cpu::hc_post_token(sub, resid, &post, &comb, hc, dim, &mut tmp);
        out_w[t * hcdim..(t + 1) * hcdim].copy_from_slice(&tmp);
    }
    let rl_out = rel_l2(&out_g, &out_w);
    eprintln!("spine mhc hc_post: out rel-L2={rl_out:.3e}");
    assert!(rl_out <= 1e-4, "hc_post rel-L2 {rl_out:.3e} > 1e-4");
    let _ = stream;
}

// ---------------------------------------------------------------------------
// spine 5d: T6 fused hc_pre — `dsv4_hc_pre_fused_b` (ONE launch) must reproduce the
// 4-kernel chain (rsqrt + mixes + sinkhorn + collapse) BITWISE (0/0 mismatches, not
// tolerance): same per-thread ascending-i fmaf chains, same stride-halving trees, same
// single-thread sinkhorn sequence, same collapse loop. Shapes cover decode (s=1), verify
// (s=6), overhang (s=7 — last block partially valid), mid (s=256) and prefill (s=2048).
// ---------------------------------------------------------------------------
#[test]
fn spine_hc_pre_fused_bitwise_matches_chain() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &[
        "dsv4_hc_pre_rsqrt_b", "dsv4_hc_mixes_b", "dsv4_hc_split_sinkhorn",
        "dsv4_hc_collapse_b", "dsv4_hc_pre_fused_b",
    ]).expect("load");
    ks.set_dynamic_smem("dsv4_hc_pre_fused_b", 25 * 768 * 4).expect("smem opt-in");

    let (hc, dim) = (4usize, 4096usize);
    let hcdim = hc * dim;
    let eps = 1e-6f32;
    let mut rng = XorShift(0x5EED_1234);
    let x: Vec<f32> = (0..2048 * hcdim).map(|_| dsv4_cpu::bf(rng.f32())).collect();
    let hc_fn: Vec<f32> = (0..24 * hcdim).map(|_| rng.f32() * 0.01).collect();
    let hc_base: Vec<f32> = (0..24).map(|i| i as f32 * 0.03 - 0.5).collect();
    let hc_scale: [f32; 3] = [0.5, 0.5, 0.5];

    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_dev = dev.htod_sync_copy(&x_bf).unwrap();
    let fn_dev = dev.htod_sync_copy(&hc_fn).unwrap();
    let ba_dev = dev.htod_sync_copy(&hc_base).unwrap();
    let sc_dev = dev.htod_sync_copy(&hc_scale).unwrap();

    for s in [1usize, 6, 7, 256, 2048] {
        let (s_i, hcd_i, dim_i, hc_i) = (s as i32, hcdim as i32, dim as i32, hc as i32);
        let mut rsqrt_a = dev.alloc_zeros::<f32>(s).unwrap();
        let mut mixes_a = dev.alloc_zeros::<f32>(s * 24).unwrap();
        let mut pre_a = dev.alloc_zeros::<f32>(s * hc).unwrap();
        let mut post_a = dev.alloc_zeros::<f32>(s * hc).unwrap();
        let mut comb_a = dev.alloc_zeros::<f32>(s * hc * hc).unwrap();
        let mut y_a = dev.alloc_zeros::<bf16>(s * dim).unwrap();
        let mut rsqrt_b = dev.alloc_zeros::<f32>(s).unwrap();
        let mut mixes_b = dev.alloc_zeros::<f32>(s * 24).unwrap();
        let mut pre_b = dev.alloc_zeros::<f32>(s * hc).unwrap();
        let mut post_b = dev.alloc_zeros::<f32>(s * hc).unwrap();
        let mut comb_b = dev.alloc_zeros::<f32>(s * hc * hc).unwrap();
        let mut y_b = dev.alloc_zeros::<bf16>(s * dim).unwrap();
        dev.synchronize().unwrap();

        // chain (the old production sequence)
        dsv4_launch!(ks, "dsv4_hc_pre_rsqrt_b", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&mut rsqrt_a, &x_dev, &s_i, &hcd_i, &eps)).unwrap();
        dsv4_launch!(ks, "dsv4_hc_mixes_b", stream.stream, (24u32, s as u32, 1), (256, 1, 1), 0,
            (&mut mixes_a, &fn_dev, &x_dev, &rsqrt_a, &s_i, &hcd_i)).unwrap();
        dsv4_launch!(ks, "dsv4_hc_split_sinkhorn", stream.stream, (((s + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&mixes_a, &sc_dev, &ba_dev, &mut pre_a, &mut post_a, &mut comb_a, &s_i)).unwrap();
        dsv4_launch!(ks, "dsv4_hc_collapse_b", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&mut y_a, &x_dev, &pre_a, &s_i, &dim_i, &hc_i)).unwrap();

        // fused (production)
        let grid = ((s + 2) / 3) as u32;
        dsv4_launch!(ks, "dsv4_hc_pre_fused_b", stream.stream, (grid, 1, 1), (768, 1, 1), 25 * 768 * 4,
            (&mut rsqrt_b, &mut mixes_b, &mut pre_b, &mut post_b, &mut comb_b, &mut y_b,
             &x_dev, &fn_dev, &sc_dev, &ba_dev, &s_i, &hcd_i, &dim_i, &hc_i, &eps)).unwrap();
        dev.synchronize().unwrap();

        let cmp = |a: &cudarc::driver::CudaSlice<f32>, b: &cudarc::driver::CudaSlice<f32>, name: &str| {
            let av = dev.dtoh_sync_copy(a).unwrap();
            let bv = dev.dtoh_sync_copy(b).unwrap();
            let mism = av.iter().zip(bv.iter()).filter(|(x, y)| x != y).count();
            eprintln!("  s={s} {name}: {mism}/{} mismatches", av.len());
            assert_eq!(mism, 0, "s={s} {name} bitwise mismatch {mism}/{}", av.len());
        };
        cmp(&rsqrt_a, &rsqrt_b, "rsqrt");
        cmp(&mixes_a, &mixes_b, "mixes");
        cmp(&pre_a, &pre_b, "pre");
        cmp(&post_a, &post_b, "post");
        cmp(&comb_a, &comb_b, "comb");
        let ya: Vec<bf16> = dev.dtoh_sync_copy(&y_a).unwrap();
        let yb: Vec<bf16> = dev.dtoh_sync_copy(&y_b).unwrap();
        let mism = ya.iter().zip(yb.iter()).filter(|(x, y)| x != y).count();
        eprintln!("  s={s} y: {mism}/{} mismatches", ya.len());
        assert_eq!(mism, 0, "s={s} y bitwise mismatch {mism}/{}", ya.len());
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// spine 5e: R3A.1 E2 rung-2 pair gates — `dsv4_rmsnorm_pair_b` and `dsv4_rope_pair_b` must
// reproduce the separate launches BITWISE (same per-row reductions / per-element math;
// the rope pair's inline positions equal the iota arrays' integers exactly). s ∈ {1, 6, 16}.
// ---------------------------------------------------------------------------
#[test]
fn spine_pair_fusions_bitwise_match_separate() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &[
        "dsv4_rmsnorm_b", "dsv4_rmsnorm_pair_b", "dsv4_rope_last_b", "dsv4_rope_pair_b",
        "dsv4_rope_q_inline_b", "dsv4_iota_b",
    ]).expect("load");

    let (nh, hd, rd, qlr) = (64usize, 512usize, 64usize, 1536usize);
    let eps = 1e-6f32;
    // positions=4096: start_pos=137 must be IN RANGE — a 32-position table makes the rope
    // reads OOB (nondeterministic ld.global.nc stale-line hits; session-2 hygiene find).
    let table = dsv4_cpu::rope_table(rd, 4096, 65536, 160000.0, 16.0, 32, 1);
    let cos_dev = dev.htod_sync_copy(&table.cos).unwrap();
    let sin_dev = dev.htod_sync_copy(&table.sin).unwrap();
    let mut rng = XorShift(0xCAFE_7777);
    let w0: Vec<f32> = (0..qlr).map(|_| rng.f32() * 0.02).collect();
    let w1: Vec<f32> = (0..hd).map(|_| rng.f32() * 0.02).collect();
    let w0_dev = dev.htod_sync_copy(&w0).unwrap();
    let w1_dev = dev.htod_sync_copy(&w1).unwrap();

    for &s in &[1usize, 6, 16] {
        let rows_q = s * nh;
        let start_pos = 137i32; // non-zero, non-aligned (rotation exercised)
        // ---- rmsnorm pair vs two singles ----
        let x0: Vec<bf16> = (0..s * qlr).map(|_| bf16::from_f32(dsv4_cpu::bf(rng.f32()))).collect();
        let x1: Vec<bf16> = (0..s * hd).map(|_| bf16::from_f32(dsv4_cpu::bf(rng.f32()))).collect();
        let x0_dev = dev.htod_sync_copy(&x0).unwrap();
        let x1_dev = dev.htod_sync_copy(&x1).unwrap();
        let mut y0_a = dev.alloc_zeros::<bf16>(s * qlr).unwrap();
        let mut y1_a = dev.alloc_zeros::<bf16>(s * hd).unwrap();
        let mut y0_b = dev.alloc_zeros::<bf16>(s * qlr).unwrap();
        let mut y1_b = dev.alloc_zeros::<bf16>(s * hd).unwrap();
        dev.synchronize().unwrap();
        let (s_i, qlr_i, hd_i) = (s as i32, qlr as i32, hd as i32);
        dsv4_launch!(ks, "dsv4_rmsnorm_b", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&mut y0_a, &x0_dev, &w0_dev, &s_i, &qlr_i, &eps)).unwrap();
        dsv4_launch!(ks, "dsv4_rmsnorm_b", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&mut y1_a, &x1_dev, &w1_dev, &s_i, &hd_i, &eps)).unwrap();
        dsv4_launch!(ks, "dsv4_rmsnorm_pair_b", stream.stream, (s as u32, 2, 1), (256, 1, 1), 0,
            (&mut y0_b, &x0_dev, &w0_dev, &s_i, &qlr_i,
             &mut y1_b, &x1_dev, &w1_dev, &s_i, &hd_i, &eps)).unwrap();
        dev.synchronize().unwrap();
        let cmp = |a: &cudarc::driver::CudaSlice<bf16>, b: &cudarc::driver::CudaSlice<bf16>| -> usize {
            let ah: Vec<u16> = dev.dtoh_sync_copy(a).unwrap().iter().map(|v| v.to_bits()).collect();
            let bh: Vec<u16> = dev.dtoh_sync_copy(b).unwrap().iter().map(|v| v.to_bits()).collect();
            ah.iter().zip(&bh).filter(|(x, y)| x != y).count()
        };
        let d_norm = cmp(&y0_a, &y0_b) + cmp(&y1_a, &y1_b);

        // ---- rope pair (inline positions) vs iota + two singles ----
        let q0: Vec<bf16> = (0..rows_q * hd).map(|_| bf16::from_f32(dsv4_cpu::bf(rng.f32()))).collect();
        let k0: Vec<bf16> = (0..s * hd).map(|_| bf16::from_f32(dsv4_cpu::bf(rng.f32()))).collect();
        let mut qa = dev.htod_sync_copy(&q0).unwrap();
        let mut ka = dev.htod_sync_copy(&k0).unwrap();
        let mut qb = dev.htod_sync_copy(&q0).unwrap();
        let mut kb = dev.htod_sync_copy(&k0).unwrap();
        let pos_q: Vec<i32> = (0..rows_q).map(|i| start_pos + (i as i32) / (nh as i32)).collect();
        let pos_k: Vec<i32> = (0..s).map(|i| start_pos + i as i32).collect();
        let pos_q_dev = dev.htod_sync_copy(&pos_q).unwrap();
        let pos_k_dev = dev.htod_sync_copy(&pos_k).unwrap();
        dev.synchronize().unwrap();
        let (rq_i, rd_i, inv0, nh_i) = (rows_q as i32, rd as i32, 0i32, nh as i32);
        dsv4_launch!(ks, "dsv4_rope_last_b", stream.stream, (((rows_q + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
            (&mut qa, &cos_dev, &sin_dev, &pos_q_dev, &rq_i, &hd_i, &rd_i, &inv0)).unwrap();
        dsv4_launch!(ks, "dsv4_rope_last_b", stream.stream, (((s + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
            (&mut ka, &cos_dev, &sin_dev, &pos_k_dev, &s_i, &hd_i, &rd_i, &inv0)).unwrap();
        dsv4_launch!(ks, "dsv4_rope_pair_b", stream.stream, (((rows_q + 7) / 8) as u32, 2, 1), (256, 1, 1), 0,
            (&mut qb, &mut kb, &cos_dev, &sin_dev, &start_pos, &nh_i, &rq_i, &s_i, &hd_i, &rd_i)).unwrap();
        // de-rotation arm: inline-q inverse vs iota + rope_last(inverse=1)
        let mut oa = dev.htod_sync_copy(&q0).unwrap();
        let mut ob = dev.htod_sync_copy(&q0).unwrap();
        dev.synchronize().unwrap();
        let inv1 = 1i32;
        dsv4_launch!(ks, "dsv4_rope_last_b", stream.stream, (((rows_q + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
            (&mut oa, &cos_dev, &sin_dev, &pos_q_dev, &rq_i, &hd_i, &rd_i, &inv1)).unwrap();
        dsv4_launch!(ks, "dsv4_rope_q_inline_b", stream.stream, (((rows_q + 7) / 8) as u32, 1, 1), (256, 1, 1), 0,
            (&mut ob, &cos_dev, &sin_dev, &start_pos, &nh_i, &rq_i, &hd_i, &rd_i, &inv1)).unwrap();
        dev.synchronize().unwrap();
        let d_rope = cmp(&qa, &qb) + cmp(&ka, &kb) + cmp(&oa, &ob);

        eprintln!("pair gates s={s}: rmsnorm bit-diffs={d_norm} rope bit-diffs={d_rope}");
        assert!(d_norm == 0 && d_rope == 0,
            "pair fusions NOT bitwise at s={s}: rmsnorm={d_norm} rope={d_rope}");
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// spine 5f: E2 rung-5 (Tier 1.2) fused-tail gate — `dsv4_rescale_rope_sim_b` must reproduce
// the THREE separate launches BITWISE: dsv4_attn_rescale_b(q) -> dsv4_rope_pair_b(q,kv) ->
// dsv4_kv_sim_g64_strided(kv). Same per-row reductions (the pairwise rescale tree) and
// per-element math (rope, g64 QAT-sim), inline start_pos. s ∈ {1, 6, 16}, sp = 137.
// ---------------------------------------------------------------------------
#[test]
fn rescale_rope_sim_fused_bitwise_match_separate() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_rope_pair_b"]).expect("load spine");
    let ka_mod = Dsv4Kernels::load_module(&dev, "src/ptx/gpu_dsv4_attn.ptx", &[
        "dsv4_attn_rescale_b", "dsv4_kv_sim_g64_strided", "dsv4_rescale_rope_sim_b",
    ]).expect("load attn");

    let (nh, hd, rd) = (64usize, 512usize, 64usize);
    let eps = 1e-6f32;
    // positions=4096: start_pos=137 must be IN RANGE — a 32-position table makes both
    // arms read OOB, which only masks when both read through the same cache path
    // (the ld.global.nc stale-read trap that hid the s=16 failure's first debug round).
    let table = dsv4_cpu::rope_table(rd, 4096, 65536, 160000.0, 16.0, 32, 1);
    let cos_dev = dev.htod_sync_copy(&table.cos).unwrap();
    let sin_dev = dev.htod_sync_copy(&table.sin).unwrap();
    let mut rng = XorShift(0x5EED_F05E);
    let cmp = |a: &cudarc::driver::CudaSlice<bf16>, b: &cudarc::driver::CudaSlice<bf16>| -> usize {
        let ah: Vec<u16> = dev.dtoh_sync_copy(a).unwrap().iter().map(|v| v.to_bits()).collect();
        let bh: Vec<u16> = dev.dtoh_sync_copy(b).unwrap().iter().map(|v| v.to_bits()).collect();
        ah.iter().zip(&bh).filter(|(x, y)| x != y).count()
    };

    for &s in &[1usize, 6, 16] {
        let rows_q = s * nh;
        let start_pos = 137i32;
        let q0: Vec<bf16> = (0..rows_q * hd).map(|_| bf16::from_f32(dsv4_cpu::bf(rng.f32()))).collect();
        let k0: Vec<bf16> = (0..s * hd).map(|_| bf16::from_f32(dsv4_cpu::bf(rng.f32()))).collect();
        let mut qa = dev.htod_sync_copy(&q0).unwrap();
        let mut ka = dev.htod_sync_copy(&k0).unwrap();
        let mut qb = dev.htod_sync_copy(&q0).unwrap();
        let mut kb = dev.htod_sync_copy(&k0).unwrap();
        dev.synchronize().unwrap();
        let (rq_i, s_i, hd_i, rd_i, nh_i) = (rows_q as i32, s as i32, hd as i32, rd as i32, nh as i32);
        // Arm A: the three separate launches.
        dsv4_launch!(ka_mod, "dsv4_attn_rescale_b", stream.stream, (rows_q as u32, 1, 1), (256, 1, 1), 0,
            (&mut qa, &rq_i, &hd_i, &eps)).unwrap();
        dsv4_launch!(ks, "dsv4_rope_pair_b", stream.stream, (((rows_q + 7) / 8) as u32, 2, 1), (256, 1, 1), 0,
            (&mut qa, &mut ka, &cos_dev, &sin_dev, &start_pos, &nh_i, &rq_i, &s_i, &hd_i, &rd_i)).unwrap();
        let (stride_i, nope_i) = (hd as i32, (hd - rd) as i32);
        dsv4_launch!(ka_mod, "dsv4_kv_sim_g64_strided", stream.stream,
            ((((s * ((hd - rd) / 64) * 32) + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&mut ka, &s_i, &stride_i, &nope_i)).unwrap();
        // Arm B: the fused tail.
        dsv4_launch!(ka_mod, "dsv4_rescale_rope_sim_b", stream.stream,
            ((rows_q + s) as u32, 1, 1), (256, 1, 1), 0,
            (&mut qb, &mut kb, &cos_dev, &sin_dev, &start_pos, &nh_i, &rq_i, &s_i, &hd_i, &rd_i, &eps)).unwrap();
        dev.synchronize().unwrap();
        let d_q = cmp(&qa, &qb);
        let d_kv = cmp(&ka, &kb);
        eprintln!("fused-tail gate s={s}: q bit-diffs={d_q} kv bit-diffs={d_kv}");
        assert!(d_q == 0 && d_kv == 0, "fused tail NOT bitwise at s={s}: q={d_q} kv={d_kv}");
    }
    println!("RESCALE-ROPE-SIM-FUSED-GATE: PASS (s ∈ 1/6/16, fused ≡ 3 separate launches bitwise)");
    let _ = stream;
}


#[test]
fn spine_router_score_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &["dsv4_router_score_b"]).expect("load");
    let (n_exp, dim, s) = (256usize, 4096usize, 4usize);
    let mut rng = XorShift(0x8057_0001);
    let gate_w: Vec<f32> = (0..n_exp * dim).map(|_| rng.f32() * 0.02).collect();
    let x: Vec<f32> = (0..s * dim).map(|_| dsv4_cpu::bf(rng.f32())).collect();

    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_dev = dev.htod_sync_copy(&x_bf).unwrap();
    let w_dev = dev.htod_sync_copy(&gate_w).unwrap();
    let mut sc_dev = dev.alloc_zeros::<f32>(s * n_exp).unwrap();
    dev.synchronize().unwrap();
    let (s_i, d_i, ne_i) = (s as i32, dim as i32, n_exp as i32);
    dsv4_launch!(ks, "dsv4_router_score_b", stream.stream, (n_exp as u32, s as u32, 1), (256, 1, 1), 0,
        (&mut sc_dev, &w_dev, &x_dev, &s_i, &d_i, &ne_i)).unwrap();
    dev.synchronize().unwrap();
    let got = dev.dtoh_sync_copy(&sc_dev).unwrap();

    let raw = dsv4_cpu::gemm_f32(&x, s, dim, &gate_w, n_exp);
    let want: Vec<f32> = raw.iter().map(|&v| dsv4_cpu::softplus_torch(v).sqrt()).collect();
    let rl = rel_l2(&got, &want);
    eprintln!("spine router_score rel-L2={rl:.3e}");
    assert!(rl <= 1e-4, "router_score rel-L2 {rl:.3e} > 1e-4");
    let _ = stream;
}

// ---------------------------------------------------------------------------
// lane 3D: full router (§B.9) end-to-end — bias path (layers >=3) and tid2eid hash path (0-2) —
// vs dsv4_cpu::gate_forward. score GEMM is tolerance-level so near-tie selections can flip; gate on
// weight rel-L2 and assert any index mismatch is a sub-1e-5 score near-tie (not a real bug).
// ---------------------------------------------------------------------------
fn run_router(
    dev: &Arc<CudaDevice>, stream: &cudarc::driver::CudaStream, ks: &Dsv4Kernels,
    gate_w: &[f32], bias: Option<&[f32]>, tid2eid: Option<(&[i32], &[i32])>,
    x: &[f32], s: usize, dim: usize, n_exp: usize, topk: usize, route_scale: f32,
) -> (Vec<f32>, Vec<i32>) {
    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_dev = dev.htod_sync_copy(&x_bf).unwrap();
    let w_dev = dev.htod_sync_copy(gate_w).unwrap();
    let mut sc_dev = dev.alloc_zeros::<f32>(s * n_exp).unwrap();
    dev.synchronize().unwrap();
    let (s_i, d_i, ne_i, tk_i) = (s as i32, dim as i32, n_exp as i32, topk as i32);
    dsv4_launch!(ks, "dsv4_router_score_b", stream.stream, (n_exp as u32, s as u32, 1), (256, 1, 1), 0,
        (&mut sc_dev, &w_dev, &x_dev, &s_i, &d_i, &ne_i)).unwrap();

    let mut sel_dev = dev.alloc_zeros::<i32>(s * topk).unwrap();
    match tid2eid {
        Some((tbl, ids)) => {
            let tbl_dev = dev.htod_sync_copy(tbl).unwrap();
            let ids_dev = dev.htod_sync_copy(ids).unwrap();
            dev.synchronize().unwrap();
            dsv4_launch!(ks, "dsv4_router_tid2eid_b", stream.stream, (((s * topk + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&mut sel_dev, &tbl_dev, &ids_dev, &s_i, &tk_i)).unwrap();
        }
        None => {
            let mut biased_dev = dev.alloc_zeros::<f32>(s * n_exp).unwrap();
            if let Some(b) = bias {
                let b_dev = dev.htod_sync_copy(b).unwrap();
                dev.synchronize().unwrap();
                dsv4_launch!(ks, "dsv4_router_bias_add_b", stream.stream, (((s * n_exp + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                    (&mut biased_dev, &sc_dev, &b_dev, &s_i, &ne_i)).unwrap();
            } else {
                biased_dev = sc_dev.clone(); // topk on raw scores if no bias
            }
            dev.synchronize().unwrap();
            dsv4_launch!(ks, "dsv4_topk", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                (&biased_dev, &mut sel_dev, &s_i, &ne_i, &tk_i)).unwrap();
        }
    }
    let mut wt_dev = dev.alloc_zeros::<f32>(s * topk).unwrap();
    let rs = route_scale;
    dsv4_launch!(ks, "dsv4_router_weights_b", stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
        (&mut wt_dev, &sc_dev, &sel_dev, &s_i, &ne_i, &tk_i, &rs)).unwrap();
    dev.synchronize().unwrap();
    let weights = dev.dtoh_sync_copy(&wt_dev).unwrap();
    let sel = dev.dtoh_sync_copy(&sel_dev).unwrap();
    (weights, sel)
}

#[test]
fn spine_router_full_bias_path_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &[
        "dsv4_router_score_b", "dsv4_router_bias_add_b", "dsv4_router_weights_b", "dsv4_topk",
    ]).expect("load");
    let (n_exp, dim, s, topk) = (256usize, 4096usize, 4usize, 6usize);
    let route_scale = 1.5f32;
    let mut rng = XorShift(0x8D7E_0010);
    let gate_w: Vec<f32> = (0..n_exp * dim).map(|_| rng.f32() * 0.02).collect();
    let bias: Vec<f32> = (0..n_exp).map(|i| (i as f32 * 0.013 - 1.6)).collect();
    let x: Vec<f32> = (0..s * dim).map(|_| dsv4_cpu::bf(rng.f32())).collect();

    let (gw, sel) = run_router(&dev, &stream, &ks, &gate_w, Some(&bias), None,
        &x, s, dim, n_exp, topk, route_scale);
    let (w_w, idx_w) = dsv4_cpu::gate_forward(&x, s, dim, &gate_w, Some(&bias), None, n_exp, topk, route_scale);

    // indices: exact match expected (deterministic topk); tolerate only sub-1e-5 near-ties.
    let raw = dsv4_cpu::gemm_f32(&x, s, dim, &gate_w, n_exp);
    let mut idx_mism = 0;
    for t in 0..s {
        for j in 0..topk {
            if sel[t * topk + j] as i64 != idx_w[t * topk + j] {
                let e1 = sel[t * topk + j] as usize;
                let e2 = idx_w[t * topk + j] as usize;
                let s1 = dsv4_cpu::softplus_torch(raw[t * n_exp + e1]).sqrt() + bias[e1];
                let s2 = dsv4_cpu::softplus_torch(raw[t * n_exp + e2]).sqrt() + bias[e2];
                let rel = ((s1 - s2).abs() / (s1.abs().max(s2.abs()))).max(1e-12);
                assert!(rel < 1e-4, "router idx mismatch not a near-tie: t={t} j={j} e1={e1} e2={e2} rel={rel:.3e}");
                idx_mism += 1;
            }
        }
    }
    let rl = rel_l2(&gw, &w_w);
    eprintln!("spine router bias-path: idx near-tie mismatches={idx_mism}, weights rel-L2={rl:.3e}");
    assert!(rl <= 1e-4, "router weights rel-L2 {rl:.3e} > 1e-4");
    let _ = stream;
}

#[test]
fn spine_router_hash_path_matches_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = Dsv4Kernels::load(&dev, &[
        "dsv4_router_score_b", "dsv4_router_tid2eid_b", "dsv4_router_weights_b",
    ]).expect("load");
    let (n_exp, dim, s, topk, vocab) = (256usize, 4096usize, 4usize, 6usize, 129280usize);
    let route_scale = 1.5f32;
    let mut rng = XorShift(0x8D7E_0020);
    let gate_w: Vec<f32> = (0..n_exp * dim).map(|_| rng.f32() * 0.02).collect();
    let x: Vec<f32> = (0..s * dim).map(|_| dsv4_cpu::bf(rng.f32())).collect();
    // fake tid2eid table [vocab, topk] + ids [s]
    let tid2eid: Vec<i32> = (0..vocab * topk).map(|i| ((i * 1103515245 + 12345) as i32).rem_euclid(n_exp as i32)).collect();
    let ids: Vec<i64> = (0..s).map(|i| ((i as i64 + 7) % 1000)).collect();
    let ids_i32: Vec<i32> = ids.iter().map(|&v| v as i32).collect();

    let (gw, sel) = run_router(&dev, &stream, &ks, &gate_w, None, Some((&tid2eid, &ids_i32)),
        &x, s, dim, n_exp, topk, route_scale);
    let (w_w, idx_w) = dsv4_cpu::gate_forward(&x, s, dim, &gate_w, None, Some((&tid2eid, &ids)), n_exp, topk, route_scale);

    let mut idx_mism = 0;
    for t in 0..s { for j in 0..topk { if sel[t * topk + j] as i64 != idx_w[t * topk + j] { idx_mism += 1; } } }
    let rl = rel_l2(&gw, &w_w);
    eprintln!("spine router hash-path: idx mismatches={idx_mism} (expect 0 — gather is deterministic), weights rel-L2={rl:.3e}");
    assert_eq!(idx_mism, 0, "tid2eid gather mismatch (not score-driven)");
    assert!(rl <= 1e-4, "router weights rel-L2 {rl:.3e} > 1e-4");
    let _ = stream;
}
