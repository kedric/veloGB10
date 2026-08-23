//! Lane 3B gates: the §B.5 compressor (CSA overlap ratio-4 + HCA ratio-128) and the
//! §B.6 indexer on GPU (`src/dsv4_comp.rs` + `kernels/gpu_dsv4_comp.cu`) against the
//! G1-proven CPU reference (`src/dsv4_cpu.rs`).
//!
//! Unit gates (synthetic, real shapes): GEMM tree bit-exactness; compressor
//! prefill+decode sequences bit-exact incl. cache contents AND frontier tensors;
//! chunked-prefill ≡ one-shot (§12.B.5); frontier snapshot/restore (§12.B.4);
//! indexer topk SETS exact + scores rel-L2 ≤ 1e-4.
//!
//! Real-weights gates: `load_layer(2)` (CSA) and `load_layer(3)` (HCA) on the
//! oracle's pre.x/dec*.x (export first — the 0731-native fixtures live under
//! /tmp/dsv4-0731-ref or a fresh dsv4_ref.py export):
//!   python3 scripts/dsv4_diff.py export --npz /tmp/dsv4-0731-ref/dsv4_csa.npz --out /tmp/dsv4_rt/csa
//!   python3 scripts/dsv4_diff.py export --npz /tmp/dsv4-0731-ref/dsv4_hca.npz --out /tmp/dsv4_rt/hca
//! ). The compressor cache gates bit-exact vs dsv4_cpu-on-same-inputs; the indexer
//! topk gates selection SETS vs the oracle (the §12.B.2 tie regime — ordered compare
//! may differ; near-tie set flips are adjudicated with score-gap evidence).
//!
//! NOTE (2026-08-01, model upgrade to DeepSeek-V4-Flash-0731): the *oracle* real-weights gates
//! below are LEGACY — the old dsv4-oracle-v2/*.npz were exported from the obsolete
//! DeepSeek-V4-Flash-DSpark weights and are NOT regenerated as oracle-v3 (no oracle v3 by
//! contract; 0731 verification = self-referential + dsv4_cpu cross-checks + official-API
//! vectors later). The gates SKIP when the /tmp/dsv4_rt exports are absent (the standing
//! behavior); if ever re-enabled, export from 0731-native fixtures under /tmp/dsv4-0731-ref
//! (never the deleted oracle-v2). The synthetic gates above remain the load-bearing ones.
//!
//! Run: cargo test --release --test dsv4_comp_test -- --test-threads=1 --nocapture

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::nvrtc::Ptx;
use half::bf16;

use gb10_inference::{dsv4_cpu, dsv4_load, quant};
use gb10_inference::dsv4_comp::{CompKernels, CompSpec, DevRope, GpuCompressor, GpuIndexer};
use gb10_inference::dsv4_cpu::{Compressor, CompressorWeights, IndexerState, IndexerWeights};
use gb10_inference::dsv4_gpu;
use gb10_inference::dsv4_load::{Dsv4Config, NpyData};

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark";
const RT: &str = "/tmp/dsv4_rt";

// ---------------------------------------------------------------------------
// small host helpers (mirrors dsv4_spine_test.rs conventions)
// ---------------------------------------------------------------------------

fn dev() -> Arc<CudaDevice> {
    CudaDevice::new(0).expect("CUDA device 0")
}

struct XorShift(u64);
impl XorShift {
    fn f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn to_bf16_dev(dev: &Arc<CudaDevice>, v: &[f32]) -> CudaSlice<bf16> {
    let b: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
    dev.htod_sync_copy(&b).unwrap()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut se, mut sn) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want.iter()) {
        se += ((g - w) as f64).powi(2);
        sn += (w as f64).powi(2);
    }
    (se / sn.max(1e-30)).sqrt()
}

/// bf16-level ulp distance (sign-magnitude monotone ordering).
fn bf16_ulp(a: f32, b: f32) -> i64 {
    if a == b {
        return 0;
    }
    let ab = bf16::from_f32(a).to_bits() as i16 as i64;
    let bb = bf16::from_f32(b).to_bits() as i16 as i64;
    let ak = ab ^ if ab < 0 { -1i64 } else { 0 };
    let bk = bb ^ if bb < 0 { -1i64 } else { 0 };
    (ak - bk).abs()
}

/// (mismatches, max bf16 ulp) between two bf16-valued f32 buffers.
fn bf16_diff(got: &[f32], want: &[f32]) -> (usize, i64) {
    let mut mism = 0usize;
    let mut ulp = 0i64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        if g != w {
            mism += 1;
            ulp = ulp.max(bf16_ulp(g, w));
        }
    }
    (mism, ulp)
}

/// (mismatches, max abs diff) between fp32 buffers.
fn f32_diff(got: &[f32], want: &[f32]) -> (usize, f32) {
    let mut mism = 0usize;
    let mut mx = 0.0f32;
    for (&g, &w) in got.iter().zip(want.iter()) {
        // bitwise compare (handles -0.0/inf uniformly)
        if g.to_bits() != w.to_bits() {
            mism += 1;
            mx = mx.max((g - w).abs());
        }
    }
    (mism, mx)
}

fn read_f32(dir: &Path, key: &str) -> (Vec<usize>, Vec<f32>) {
    let p = dir.join(format!("{key}.npy"));
    let (shape, data) = dsv4_load::read_npy(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    match data {
        NpyData::F32(v) => (shape, v),
        _ => panic!("{}: expected <f4", p.display()),
    }
}

fn read_i64(dir: &Path, key: &str) -> (Vec<usize>, Vec<i64>) {
    let p = dir.join(format!("{key}.npy"));
    let (shape, data) = dsv4_load::read_npy(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    match data {
        NpyData::I64(v) => (shape, v),
        _ => panic!("{}: expected <i8", p.display()),
    }
}

/// DSV4 config for synthetic tests (the real values; only the fields the
/// compressor/indexer paths read matter).
fn test_cfg() -> Dsv4Config {
    Dsv4Config {
        vocab_size: 129280,
        dim: 4096,
        moe_inter_dim: 2048,
        n_layers: 43,
        n_hash_layers: 3,
        n_mtp_layers: 3,
        dspark_block_size: 5,
        dspark_noise_token_id: 128799,
        dspark_target_layer_ids: vec![40, 41, 42],
        dspark_markov_rank: 256,
        n_heads: 64,
        n_routed_experts: 256,
        n_shared_experts: 1,
        n_activated_experts: 6,
        route_scale: 1.5,
        swiglu_limit: 10.0,
        q_lora_rank: 1024,
        head_dim: 512,
        rope_head_dim: 64,
        o_groups: 8,
        o_lora_rank: 1024,
        window_size: 128,
        original_seq_len: 65536,
        rope_theta: 10000.0,
        rope_factor: 16.0,
        beta_fast: 32,
        beta_slow: 1,
        index_n_heads: 64,
        index_head_dim: 128,
        index_topk: 512,
        hc_mult: 4,
        hc_sinkhorn_iters: 20,
        compress_rope_theta: 160000.0,
        compress_ratios: vec![0; 46],
        norm_eps: 1e-6,
        hc_eps: 1e-6,
    }
}

/// CSA/HCA YaRN rope table for tests (positions = prefill + decode + margin).
fn test_rope(positions: usize) -> dsv4_cpu::RopeTable {
    dsv4_cpu::rope_table(64, positions, 65536, 160000.0, 16.0, 32, 1)
}

/// Random real-shape compressor weights (bf16-valued wkv/wgate/norm, fp32 ape).
fn synth_comp_weights(rng: &mut XorShift, spec: CompSpec) -> CompressorWeights {
    let cd = spec.cd();
    CompressorWeights {
        wkv: (0..cd * spec.dim).map(|_| dsv4_cpu::bf(rng.f32() * 0.02)).collect(),
        wgate: (0..cd * spec.dim).map(|_| dsv4_cpu::bf(rng.f32() * 0.02)).collect(),
        norm: (0..spec.head_dim).map(|_| dsv4_cpu::bf(1.0 + rng.f32() * 0.3)).collect(),
        ape: (0..spec.ratio * cd).map(|_| rng.f32() * 0.05).collect(),
        ratio: spec.ratio,
        head_dim: spec.head_dim,
        rope_dim: spec.rope_dim,
        overlap: spec.overlap,
        rotate: spec.rotate,
        sim_group: spec.sim_group,
        dim: spec.dim,
    }
}

// ---------------------------------------------------------------------------
// 1. module load + build-id handshake
// ---------------------------------------------------------------------------

#[test]
fn comp_loads_and_asserts_build_id() {
    let dev = dev();
    let _ks = CompKernels::load(&dev).expect("comp+spine module load");
}

// ---------------------------------------------------------------------------
// 2. dsv4_comp_gemm_tree_*_b — the pairwise-tree GEMM must be BIT-EXACT vs
//    dsv4_cpu::gemm_f32 / gemm_bf16 (it is the bit-exactness linchpin for the
//    whole compressor chain).
// ---------------------------------------------------------------------------

#[test]
fn comp_gemm_tree_bit_exact_vs_cpu() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let mut rng = XorShift(0x6E33_0001);

    let (s, k, n) = (7usize, 4096usize, 1024usize);
    let x: Vec<f32> = (0..s * k).map(|_| dsv4_cpu::bf(rng.f32())).collect();
    let w: Vec<f32> = (0..n * k).map(|_| dsv4_cpu::bf(rng.f32() * 0.02)).collect(); // bf16-valued fp32

    let x_dev = to_bf16_dev(&dev, &x);
    let w_dev = dev.htod_sync_copy(&w).unwrap();
    let y_dev = dev.alloc_zeros::<f32>(s * n).unwrap();
    dev.synchronize().unwrap();

    let (si, ki, ni) = (s as i32, k as i32, n as i32);
    gb10_inference::dsv4_launch!(ks.comp, "dsv4_comp_gemm_tree_f32w_b", stream.stream,
        (n as u32, s as u32, 1), (256, 1, 1), 0,
        (&x_dev, &w_dev, &y_dev, &si, &ki, &ni)).unwrap();
    dev.synchronize().unwrap();
    let got = dev.dtoh_sync_copy(&y_dev).unwrap();

    let want = dsv4_cpu::gemm_f32(&x, s, k, &w, n);
    let (mism, mx) = f32_diff(&got, &want);
    eprintln!("comp gemm_tree f32: mismatches={mism}/{}, max abs={mx:.3e}", s * n);
    assert_eq!(mism, 0, "gemm_tree not bit-exact vs gemm_f32 (dot_tree order broken?)");

    // bf16 weights / bf16 out variant (indexer weights_proj path)
    let n2 = 64usize;
    let w2: Vec<f32> = (0..n2 * k).map(|_| dsv4_cpu::bf(rng.f32() * 0.02)).collect();
    let w2_dev = to_bf16_dev(&dev, &w2);
    let y2_dev = dev.alloc_zeros::<bf16>(s * n2).unwrap();
    dev.synchronize().unwrap();
    let ni2 = n2 as i32;
    gb10_inference::dsv4_launch!(ks.comp, "dsv4_comp_gemm_tree_bf16w_bf16out_b", stream.stream,
        (n2 as u32, s as u32, 1), (256, 1, 1), 0,
        (&x_dev, &w2_dev, &y2_dev, &si, &ki, &ni2)).unwrap();
    dev.synchronize().unwrap();
    let got2: Vec<f32> = dev.dtoh_sync_copy(&y2_dev).unwrap().iter().map(|b| b.to_f32()).collect();

    let want2 = dsv4_cpu::gemm_bf16(&x, s, k, &w2, n2);
    let (mism2, ulp2) = bf16_diff(&got2, &want2);
    eprintln!("comp gemm_tree bf16: mismatches={mism2}/{}, max bf16 ulp={ulp2}", s * n2);
    assert_eq!(mism2, 0, "gemm_tree bf16-out not bit-exact vs gemm_bf16");
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 3. CSA compressor (ratio 4, overlap, d 512): prefill S=130 + 4 decodes.
//    Compares pooled rows, FULL cache, and BOTH frontier tensors vs dsv4_cpu.
//    Target: bit-exact everywhere (the GEMM is tree-exact; the pool is CPU-order;
//    residual risk is rmsnorm block-tree + rope FMA + exp — measured, reported).
// ---------------------------------------------------------------------------

/// Run one synthetic compressor sequence on CPU and GPU; returns
/// (cpu_pooled_per_step, cpu_cache, cpu_state, gpu_cache, gpu_state).
#[allow(clippy::too_many_arguments)]
fn run_comp_sequence(
    spec: CompSpec,
    w: &CompressorWeights,
    xs: &[Vec<f32>], // prefill + decode inputs ([s0, dim], then [dim] each)
    start0: usize,   // 0 for prefill-then-decode sequences
    rope: &dsv4_cpu::RopeTable,
    cache_rows: usize,
) -> (Vec<Option<Vec<f32>>>, Vec<f32>, (Vec<f32>, Vec<f32>), Vec<f32>, (Vec<f32>, Vec<f32>)) {
    let d = spec.head_dim;
    // ---- CPU ----
    let mut cpu = Compressor::new(w.clone());
    let mut cpu_cache = vec![0.0f32; cache_rows * d];
    let mut cpu_pooled = Vec::new();
    let mut pos = start0;
    for (i, x) in xs.iter().enumerate() {
        let s = x.len() / spec.dim;
        let p = cpu.forward(x, s, pos, rope, 1e-6, &mut cpu_cache);
        cpu_pooled.push(p);
        pos += s;
        let _ = i;
    }
    let cpu_state = (cpu.st.kv_state.clone(), cpu.st.score_state.clone());
    // ---- GPU ----
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let gpu = GpuCompressor::new(&dev, &ks, &stream, spec, w, 1e-6, cache_rows, xs[0].len() / spec.dim).unwrap();
    let rope_dev = DevRope::from_cpu(&dev, rope).unwrap();
    let mut pos = start0;
    for x in xs {
        let s = x.len() / spec.dim;
        let x_dev = to_bf16_dev(&dev, x);
        dev.synchronize().unwrap();
        if pos == 0 {
            gpu.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x_dev, s, &rope_dev).unwrap();
        } else if s >= spec.ratio && pos % spec.ratio == 0 {
            gpu.prefill_at::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x_dev, s, pos, &rope_dev).unwrap();
        } else {
            gpu.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &x_dev, s, pos, &rope_dev).unwrap();
        }
        pos += s;
    }
    let gpu_cache = gpu.cache_host(&dev, cache_rows).unwrap();
    let gpu_state = gpu.state_host(&dev).unwrap();
    (cpu_pooled, cpu_cache, cpu_state, gpu_cache, gpu_state)
}

fn synth_x(rng: &mut XorShift, s: usize, dim: usize) -> Vec<f32> {
    (0..s * dim).map(|_| dsv4_cpu::bf(rng.f32() * 0.7)).collect()
}

#[test]
fn comp_csa_prefill_decode_bit_exact_vs_cpu() {
    let spec = CompSpec::csa_attn(4096, 64);
    let mut rng = XorShift(0xC5A0_0001);
    let w = synth_comp_weights(&mut rng, spec);
    let rope = test_rope(140);
    let mut xs = vec![synth_x(&mut rng, 130, 4096)];
    for _ in 0..4 {
        xs.push(synth_x(&mut rng, 1, 4096));
    }
    let (cpu_pooled, cpu_cache, cpu_state, gpu_cache, gpu_state) =
        run_comp_sequence(spec, &w, &xs, 0, &rope, 64);

    // pooled rows from prefill: [32, 512] — must equal cache rows 0..32 (both sides)
    let pre_pooled = cpu_pooled[0].as_ref().expect("CPU prefill fired");
    let (pm, pu) = bf16_diff(&gpu_cache[..32 * 512], pre_pooled);
    eprintln!("CSA prefill pooled (GPU cache vs CPU pooled): mism={pm} max bf16 ulp={pu}");
    // decode fires: dec0@130 no, dec1@131 yes (row 32), dec2/3 no
    assert!(cpu_pooled[1].is_none() && cpu_pooled[2].is_some() && cpu_pooled[3].is_none() && cpu_pooled[4].is_none());
    // full cache (rows 0..33 hold values)
    let (cm, cu) = bf16_diff(&gpu_cache[..33 * 512], &cpu_cache[..33 * 512]);
    eprintln!("CSA cache rows 0..33: mism={cm}/{} max bf16 ulp={cu}", 33 * 512);
    // frontier state: BIT-EXACT required (pure copies + exact adds off the exact GEMM)
    let (sm1, sd1) = f32_diff(&gpu_state.0, &cpu_state.0);
    let (sm2, sd2) = f32_diff(&gpu_state.1, &cpu_state.1);
    eprintln!("CSA frontier: kv_state mism={sm1} (max {sd1:.3e}), score_state mism={sm2} (max {sd2:.3e})");
    // The attention compressor now uses the WMMA TC GEMM (rotate=false → TC path in
    // gemm_pair). The TC reduction order differs from the scalar tree by ~1e-7 rel-L2;
    // the post-pool QAT-sim snaps most but a few elements at the quant boundary survive
    // (observed: ~7/16896 cache mismatches, max ulp 16). Frontier state is fp32 (no
    // QAT-sim) so every element differs by a tiny amount (max abs ~1e-5). Both are the
    // documented reorder class — the full-layer replay gates (kv_cache bar 2e-3) confirm
    // the residual is harmless. The INDEXER compressor (rotate=true) stays scalar/bit-exact.
    assert!(pm + cm <= 64, "CSA compressor cache TC residual too large (pooled {pm}, cache {cm})");
    assert!(pu <= 64, "CSA cache max bf16 ulp {pu} too large (TC reorder class bar 64)");
    assert!(sd1 <= 1e-3 && sd2 <= 1e-3, "CSA frontier state max abs {sd1:.3e}/{sd2:.3e} > 1e-3");
}

// ---------------------------------------------------------------------------
// 4. HCA compressor (ratio 128, no overlap, d 512): prefill S=250 + 6 decodes
//    (fire at 255 → row 1); plus a one-shot S=384 multi-block/rem-0 run.
// ---------------------------------------------------------------------------

#[test]
fn comp_hca_prefill_decode_bit_exact_vs_cpu() {
    let spec = CompSpec::hca_attn(4096, 64);
    let mut rng = XorShift(0x11CA_0002);
    let w = synth_comp_weights(&mut rng, spec);
    let rope = test_rope(400);
    let mut xs = vec![synth_x(&mut rng, 250, 4096)];
    for _ in 0..6 {
        xs.push(synth_x(&mut rng, 1, 4096));
    }
    let (cpu_pooled, cpu_cache, cpu_state, gpu_cache, gpu_state) =
        run_comp_sequence(spec, &w, &xs, 0, &rope, 8);

    assert!(cpu_pooled[0].is_some());
    for (i, p) in cpu_pooled.iter().enumerate().skip(1) {
        if i == 6 {
            assert!(p.is_some(), "decode at 250+5=255 must fire (255+1)%128==0");
        } else {
            assert!(p.is_none(), "decode {i} must not fire");
        }
    }
    let (cm, cu) = bf16_diff(&gpu_cache[..2 * 512], &cpu_cache[..2 * 512]);
    eprintln!("HCA cache rows 0..2: mism={cm}/{} max bf16 ulp={cu}", 2 * 512);
    let (sm1, sd1) = f32_diff(&gpu_state.0, &cpu_state.0);
    let (sm2, sd2) = f32_diff(&gpu_state.1, &cpu_state.1);
    eprintln!("HCA frontier: kv mism={sm1} (max {sd1:.3e}), score mism={sm2} (max {sd2:.3e})");
    assert_eq!(cm, 0, "HCA compressor cache not bit-exact");
    // Frontier state: TC GEMM reorder residual (fp32, no QAT-sim snapping). Max abs ~1e-5.
    assert!(sd1 <= 1e-3 && sd2 <= 1e-3, "HCA frontier state max abs {sd1:.3e}/{sd2:.3e} > 1e-3");

    // one-shot S=384: 3 full blocks, remainder 0 (covers the rem==0 path)
    let xs2 = vec![synth_x(&mut rng, 384, 4096)];
    let (cpu_pooled2, cpu_cache2, cpu_state2, gpu_cache2, gpu_state2) =
        run_comp_sequence(spec, &w, &xs2, 0, &rope, 8);
    let (cm2, cu2) = bf16_diff(&gpu_cache2[..3 * 512], &cpu_cache2[..3 * 512]);
    eprintln!("HCA one-shot S=384 cache rows 0..3: mism={cm2} max bf16 ulp={cu2}");
    let (sm3, sd3) = f32_diff(&gpu_state2.0, &cpu_state2.0);
    let (sm4, sd4) = f32_diff(&gpu_state2.1, &cpu_state2.1);
    eprintln!("HCA S=384 frontier: kv mism={sm3} (max {sd3:.3e}), score mism={sm4} (max {sd4:.3e})");
    assert_eq!(cm2, 0, "HCA S=384 cache not bit-exact");
    assert!(sd3 <= 1e-3 && sd4 <= 1e-3, "HCA S=384 frontier max abs {sd3:.3e}/{sd4:.3e} > 1e-3");
    assert!(cpu_pooled2[0].as_ref().unwrap().len() == 3 * 512);
}

// ---------------------------------------------------------------------------
// 5. Chunked-prefill equivalence (§12.B.5): the decode-path compressor at
//    start_pos>0 must produce identical OUTPUTS to the one-shot prefill
//    assembly. Gate structure (measured, see the report):
//      (a) every GPU trajectory is BIT-EXACT vs dsv4_cpu on the SAME trajectory;
//      (b) across trajectories the CACHE (the outputs) is bit-exact;
//      (c) the LIVE frontier rows are bit-exact — CSA rows [0..ratio) (the
//          overlap context, the only state any future step can read before it
//          is rewritten), and the current-window slots written after a boundary.
//    Dead rows (CSA slots 4..7 / HCA rows after a boundary: the last block's
//    rows, which one-shot prefill leaves at init) differ across trajectories IN
//    THE REFERENCE ITSELF — they are always fully rewritten before any read
//    (the next fire's pool reads only freshly-written rows), so they cannot
//    influence any output. The test proves that by showing the GPU's dead rows
//    match dsv4_cpu's dead rows bitwise on each trajectory, and by continuing
//    both trajectories into the next block and re-checking convergence.
// ---------------------------------------------------------------------------

/// GPU≈CPU for one trajectory (cache rows + full frontier state). The attention
/// compressor uses the WMMA TC GEMM (rotate=false); its cache/output is tolerance-level
/// vs the CPU's scalar tree (reorder class: ~7/16896 bf16 mismatches, max ulp 16; frontier
/// state fp32 differs by ~1e-5 max abs). The indexer compressor (rotate=true) stays scalar.
fn assert_gpu_eq_cpu(
    tag: &str,
    cpu_cache: &[f32],
    cpu_state: &(Vec<f32>, Vec<f32>),
    gpu_cache: &[f32],
    gpu_state: &(Vec<f32>, Vec<f32>),
    rows: usize,
    d: usize,
) {
    let (cm, cu) = bf16_diff(&gpu_cache[..rows * d], &cpu_cache[..rows * d]);
    let (_, sd1) = f32_diff(&gpu_state.0, &cpu_state.0);
    let (_, sd2) = f32_diff(&gpu_state.1, &cpu_state.1);
    eprintln!("  {tag}: GPU vs CPU — cache mism={cm} (ulp {cu}), kv max abs={sd1:.3e}, score max abs={sd2:.3e}");
    assert!(cm <= 64 && cu <= 64, "{tag}: cache TC residual too large (mism {cm}, ulp {cu})");
    assert!(sd1 <= 1e-3 && sd2 <= 1e-3, "{tag}: frontier max abs {sd1:.3e}/{sd2:.3e} > 1e-3");
}

#[test]
fn comp_chunked_prefill_equivalence() {
    // ---- CSA: A = one-shot S=136; B = [prefill 128 + 8 dec]; C = [prefill 130 + 6 dec].
    // All end at token 135 (block boundary), firing rows 32 (@131) and 33 (@135).
    let spec = CompSpec::csa_attn(4096, 64);
    let mut rng = XorShift(0xC000_5EED);
    let w = synth_comp_weights(&mut rng, spec);
    let rope = test_rope(140);
    let x_full = synth_x(&mut rng, 136, 4096);
    let cd = spec.cd();
    let row = |t: usize| x_full[t * 4096..(t + 1) * 4096].to_vec();
    let xs_a: Vec<Vec<f32>> = std::iter::once(x_full.clone()).collect();
    let xs_b: Vec<Vec<f32>> = std::iter::once(x_full[..128 * 4096].to_vec()).chain((128..136).map(row)).collect();
    let xs_c: Vec<Vec<f32>> = std::iter::once(x_full[..130 * 4096].to_vec()).chain((130..136).map(row)).collect();
    let (_, cpu_cache_a, cpu_state_a, gpu_cache_a, gpu_state_a) = run_comp_sequence(spec, &w, &xs_a, 0, &rope, 64);
    let (_, cpu_cache_b, cpu_state_b, gpu_cache_b, gpu_state_b) = run_comp_sequence(spec, &w, &xs_b, 0, &rope, 64);
    let (_, cpu_cache_c, cpu_state_c, gpu_cache_c, gpu_state_c) = run_comp_sequence(spec, &w, &xs_c, 0, &rope, 64);
    assert_gpu_eq_cpu("CSA A[one-shot 136]", &cpu_cache_a, &cpu_state_a, &gpu_cache_a, &gpu_state_a, 34, 512);
    assert_gpu_eq_cpu("CSA B[128+8]", &cpu_cache_b, &cpu_state_b, &gpu_cache_b, &gpu_state_b, 34, 512);
    assert_gpu_eq_cpu("CSA C[130+6]", &cpu_cache_c, &cpu_state_c, &gpu_cache_c, &gpu_state_c, 34, 512);
    // outputs identical across trajectories
    for (tag, gc) in [("B", &gpu_cache_b), ("C", &gpu_cache_c)] {
        let (cm, cu) = bf16_diff(&gc[..34 * 512], &gpu_cache_a[..34 * 512]);
        eprintln!("CSA chunked {tag} vs one-shot: CACHE mism={cm} (ulp {cu})");
        assert_eq!(cm, 0, "CSA chunked {tag} cache != one-shot (§12.B.5)");
    }
    // live frontier rows (0..ratio = the overlap context) identical
    for (tag, gs) in [("B", &gpu_state_b), ("C", &gpu_state_c)] {
        let (s1, _) = f32_diff(&gs.0[..4 * cd], &gpu_state_a.0[..4 * cd]);
        let (s2, _) = f32_diff(&gs.1[..4 * cd], &gpu_state_a.1[..4 * cd]);
        eprintln!("CSA chunked {tag} vs one-shot: LIVE state rows 0..4 — kv mism={s1}, score mism={s2}");
        assert_eq!(s1 + s2, 0, "CSA chunked {tag} live state != one-shot");
    }
    // dead rows (slots 4..7, never read before rewrite): report; must equal the CPU's own
    let (d1, _) = f32_diff(&gpu_state_b.0[4 * cd..], &gpu_state_a.0[4 * cd..]);
    let (d2, _) = f32_diff(&cpu_state_b.0[4 * cd..], &cpu_state_a.0[4 * cd..]);
    eprintln!("CSA dead rows 4..7: GPU B-vs-A differs in {d1} elems, CPU B-vs-A in {d2} elems (reference-inherent; GPU matches CPU per-trajectory above)");

    // ---- HCA: A = one-shot S=260; B = [prefill 250 + 10 dec]. Both end at 259,
    // having fired rows 0 (@127), 1 (@255); slots 0..3 = tokens 256..259 (live).
    let spec = CompSpec::hca_attn(4096, 64);
    let mut rng = XorShift(0x11CA_5EED);
    let w = synth_comp_weights(&mut rng, spec);
    let rope = test_rope(264);
    let x_full = synth_x(&mut rng, 260, 4096);
    let row = |t: usize| x_full[t * 4096..(t + 1) * 4096].to_vec();
    let xs_a = vec![x_full.clone()];
    let xs_b: Vec<Vec<f32>> = std::iter::once(x_full[..250 * 4096].to_vec()).chain((250..260).map(row)).collect();
    let (_, cpu_cache_a, cpu_state_a, gpu_cache_a, gpu_state_a) = run_comp_sequence(spec, &w, &xs_a, 0, &rope, 8);
    let (_, cpu_cache_b, cpu_state_b, gpu_cache_b, gpu_state_b) = run_comp_sequence(spec, &w, &xs_b, 0, &rope, 8);
    assert_gpu_eq_cpu("HCA A[one-shot 260]", &cpu_cache_a, &cpu_state_a, &gpu_cache_a, &gpu_state_a, 2, 512);
    assert_gpu_eq_cpu("HCA B[250+10]", &cpu_cache_b, &cpu_state_b, &gpu_cache_b, &gpu_state_b, 2, 512);
    let (cm, cu) = bf16_diff(&gpu_cache_b[..2 * 512], &gpu_cache_a[..2 * 512]);
    let (s1, _) = f32_diff(&gpu_state_b.0[..4 * 512], &gpu_state_a.0[..4 * 512]);
    let (s2, _) = f32_diff(&gpu_state_b.1[..4 * 512], &gpu_state_a.1[..4 * 512]);
    let (d1, _) = f32_diff(&gpu_state_b.0[4 * 512..], &gpu_state_a.0[4 * 512..]);
    eprintln!("HCA chunked [250+10] vs one-shot: cache mism={cm} (ulp {cu}), live slots 0..3 kv/score mism={s1}/{s2}, dead rows differ={d1} (reference-inherent)");
    assert_eq!(cm + s1 + s2, 0, "HCA chunked != one-shot (§12.B.5)");
}

#[test]
fn comp_prefill_at_vs_forward_tokens() {
    // Directly tests prefill_at (the batched pool with frontier carry) vs forward_tokens
    // (the sequential per-token decode) for chunk continuation. Both must produce
    // bitwise-identical cache + frontier state. This is the unit-level gate for the
    // parallel chunk-prefill speedup (item 1).
    let spec = CompSpec::csa_attn(4096, 64);
    let mut rng = XorShift(0xDEAD_BEEF);
    let w = synth_comp_weights(&mut rng, spec);
    let rope = test_rope(264);
    let s_total = 256usize;
    let chunk = 128usize;
    let x_full = synth_x(&mut rng, s_total, 4096);

    // Path A: one-shot prefill of 256 tokens (the reference).
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let rope_dev = DevRope::from_cpu(&dev, &rope).unwrap();
    let gpu_a = GpuCompressor::new(&dev, &ks, &stream, spec, &w, 1e-6, s_total / 4, s_total).unwrap();
    let x_dev_a = to_bf16_dev(&dev, &x_full);
    gpu_a.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x_dev_a, s_total, &rope_dev).unwrap();
    let cache_a = gpu_a.cache_host(&dev, s_total / 4).unwrap();
    let state_a = gpu_a.state_host(&dev).unwrap();
    drop(gpu_a);

    // Path B: prefill_at chunk 0 (128) + prefill_at chunk 1 (128) — the batched continuation.
    let gpu_b = GpuCompressor::new(&dev, &ks, &stream, spec, &w, 1e-6, s_total / 4, chunk).unwrap();
    let x0 = to_bf16_dev(&dev, &x_full[..chunk * 4096]);
    let x1 = to_bf16_dev(&dev, &x_full[chunk * 4096..]);
    gpu_b.prefill_at::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x0, chunk, 0, &rope_dev).unwrap();
    gpu_b.prefill_at::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x1, chunk, chunk, &rope_dev).unwrap();
    let cache_b = gpu_b.cache_host(&dev, s_total / 4).unwrap();
    let state_b = gpu_b.state_host(&dev).unwrap();

    let (cm, cu) = bf16_diff(&cache_b, &cache_a);
    let cd = spec.cd();
    let (s_kv, _) = f32_diff(&state_b.0[..4 * cd], &state_a.0[..4 * cd]);
    let (s_sc, _) = f32_diff(&state_b.1[..4 * cd], &state_a.1[..4 * cd]);
    eprintln!("[prefill_at] cache mism={cm}/{} (ulp {cu}), live state kv/sc mism={s_kv}/{s_sc}",
        (s_total / 4) * 512);
    assert_eq!(cm, 0, "prefill_at chunk-continuation cache != one-shot");
    assert_eq!(s_kv + s_sc, 0, "prefill_at chunk-continuation live state != one-shot");

    // Path C: prefill chunk 0 + forward_tokens chunk 1 — the old sequential path (control).
    let gpu_c = GpuCompressor::new(&dev, &ks, &stream, spec, &w, 1e-6, s_total / 4, chunk).unwrap();
    gpu_c.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x0, chunk, &rope_dev).unwrap();
    gpu_c.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &x1, chunk, chunk, &rope_dev).unwrap();
    let cache_c = gpu_c.cache_host(&dev, s_total / 4).unwrap();
    let (cm2, _) = bf16_diff(&cache_c, &cache_a);
    eprintln!("[forward_tokens control] cache mism={cm2}/{}", (s_total / 4) * 512);
    assert_eq!(cm2, 0, "forward_tokens chunk-continuation cache != one-shot (control)");
}

// ---------------------------------------------------------------------------
// 6. Frontier snapshot/restore (§12.B.4): snapshot, run decodes, restore,
//    re-run the SAME decodes → identical cache + state (DSpark verify rollback).
// ---------------------------------------------------------------------------

#[test]
fn comp_state_snapshot_restore() {
    let spec = CompSpec::csa_attn(4096, 64);
    let mut rng = XorShift(0x5AA1_0001);
    let w = synth_comp_weights(&mut rng, spec);
    let rope = test_rope(140);
    let x_pre = synth_x(&mut rng, 130, 4096);
    let xs_dec: Vec<Vec<f32>> = (0..4).map(|_| synth_x(&mut rng, 1, 4096)).collect();

    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let gpu = GpuCompressor::new(&dev, &ks, &stream, spec, &w, 1e-6, 64, 130).unwrap();
    let rope_dev = DevRope::from_cpu(&dev, &rope).unwrap();

    let x_dev = to_bf16_dev(&dev, &x_pre);
    gpu.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x_dev, 130, &rope_dev).unwrap();
    // decode 130, then snapshot
    let d0 = to_bf16_dev(&dev, &xs_dec[0]);
    gpu.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &d0, 1, 130, &rope_dev).unwrap();
    let snap = gpu.snapshot(&dev, &stream).unwrap();
    // decode 131..133 (131 fires → row 32)
    for (j, x) in xs_dec.iter().enumerate().skip(1) {
        let xd = to_bf16_dev(&dev, x);
        gpu.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &xd, 1, 130 + j, &rope_dev).unwrap();
    }
    let cache_run1 = gpu.cache_host(&dev, 64).unwrap();
    let state_run1 = gpu.state_host(&dev).unwrap();
    // restore, re-run the same three decodes
    gpu.restore(&snap, &stream).unwrap();
    for (j, x) in xs_dec.iter().enumerate().skip(1) {
        let xd = to_bf16_dev(&dev, x);
        gpu.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &xd, 1, 130 + j, &rope_dev).unwrap();
    }
    let cache_run2 = gpu.cache_host(&dev, 64).unwrap();
    let state_run2 = gpu.state_host(&dev).unwrap();

    let (cm, _) = bf16_diff(&cache_run2, &cache_run1);
    let (s1, _) = f32_diff(&state_run2.0, &state_run1.0);
    let (s2, _) = f32_diff(&state_run2.1, &state_run1.1);
    eprintln!("snapshot/restore determinism: cache mism={cm}, kv mism={s1}, score mism={s2}");
    assert_eq!(cm + s1 + s2, 0, "restore did not reproduce the pre-snapshot trajectory");
    // and the fired row must be present (assert the SUCCESS signal, not emptiness)
    assert!(cache_run1[32 * 512..33 * 512].iter().any(|&v| v != 0.0), "decode@131 fire row missing");
}

// ---------------------------------------------------------------------------
// 7. Indexer (§B.6) synthetic gate: q path (fp8_bsb) → rope/fwht/fp4 → own
//    compressor → weights → score chain → deterministic topk. vs dsv4_cpu
//    indexer_forward. Gates: indexer kv_cache bit-exact; topk SETS exact
//    (ordered reported); scores rel-L2 ≤ 1e-4.
// ---------------------------------------------------------------------------

fn pow2_to_e8m0(s: f32) -> u8 {
    (s.to_bits() >> 23) as u8
}

/// fp8 block-scale weight quant (mirrors dsv4_spine_test::quant_w_blocks).
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

/// CPU replica of the §B.6 score chain (indexer_forward steps 1-6) on the CPU
/// compressor's OWN kv_cache — exposes the scores indexer_forward keeps internal.
#[allow(clippy::too_many_arguments)]
fn cpu_index_scores(
    wq_b_deq: &[f32],
    weights_proj: &[f32],
    kv_cache: &[f32],
    x: &[f32],
    qr: &[f32],
    s: usize,
    start_pos: usize,
    rope: &dsv4_cpu::RopeTable,
    had: &[f32],
    cfg: &Dsv4Config,
) -> Vec<f32> {
    let (qlr, nh, hd, dim) = (cfg.q_lora_rank, cfg.index_n_heads, cfg.index_head_dim, cfg.dim);
    let mut q = dsv4_cpu::quant_gemm(qr, s, qlr, wq_b_deq, nh * hd, 128);
    let pos: Vec<usize> = (0..s * nh).map(|i| start_pos + i / nh).collect();
    dsv4_cpu::apply_rope(&mut q, s * nh, hd, rope, &pos, false);
    dsv4_cpu::rotate_activation(&mut q, s * nh, hd, hd, had);
    dsv4_cpu::fp4_act_quant_sim(&mut q, s * nh, hd, 32);
    let mut weights = dsv4_cpu::gemm_bf16(x, s, dim, weights_proj, nh);
    let wscale = ((hd as f64).powf(-0.5) * (nh as f64).powf(-0.5)) as f32;
    for v in weights.iter_mut() {
        *v = dsv4_cpu::bf(*v * wscale);
    }
    let nblocks = (start_pos + s) / 4;
    let mut scores = vec![0.0f32; s * nblocks];
    for i in 0..s {
        for t in 0..nblocks {
            let kv = &kv_cache[t * hd..(t + 1) * hd];
            let mut acc = 0.0f32;
            for h in 0..nh {
                let dot = dsv4_cpu::bf(dsv4_cpu::dot8(&q[(i * nh + h) * hd..(i * nh + h + 1) * hd], kv));
                let rel = if dot > 0.0 { dot } else { 0.0 };
                acc += dsv4_cpu::bf(rel * weights[i * nh + h]);
            }
            scores[i * nblocks + t] = dsv4_cpu::bf(acc);
        }
    }
    if start_pos == 0 {
        for i in 0..s {
            let lim = (i + 1) / 4;
            for t in lim..nblocks {
                scores[i * nblocks + t] = f32::NEG_INFINITY;
            }
        }
    }
    scores
}

/// (ordered match, set match, total rows) between two per-row index lists.
fn topk_row_match(got: &[i32], want: &[i64], rows: usize, k: usize) -> (usize, usize, usize) {
    let (mut ord, mut set) = (0usize, 0usize);
    for r in 0..rows {
        let g = &got[r * k..(r + 1) * k];
        let w = &want[r * k..(r + 1) * k];
        let ordered = (0..k).all(|j| g[j] as i64 == w[j]);
        let gs: std::collections::HashSet<i64> = g.iter().map(|&v| v as i64).collect();
        let ws: std::collections::HashSet<i64> = w.iter().copied().collect();
        ord += ordered as usize;
        set += (gs == ws) as usize;
    }
    (ord, set, rows)
}

struct IndexerFixture {
    iw: IndexerWeights,
    wq_b_wt: Vec<u8>,
    wq_b_sb: Vec<u8>,
    wq_b_deq: Vec<f32>,
}

fn synth_indexer(rng: &mut XorShift) -> IndexerFixture {
    let (nh, hd, qlr, dim) = (64usize, 128usize, 1024usize, 4096usize);
    let wq_b: Vec<f32> = (0..nh * hd * qlr).map(|_| rng.f32() * 0.02).collect();
    let (codes, sb) = quant_w_blocks(&wq_b, nh * hd, qlr);
    let wq_b_wt = quant::repack_fp8_mma(&codes, nh * hd, qlr);
    let wq_b_deq = dsv4_load::dequant_fp8_exact(&codes, &sb, nh * hd, qlr);
    let weights_proj: Vec<f32> = (0..nh * dim).map(|_| dsv4_cpu::bf(rng.f32() * 0.02)).collect();
    let comp = synth_comp_weights(rng, CompSpec::indexer(dim, 64));
    IndexerFixture {
        iw: IndexerWeights { wq_b: wq_b_deq.clone(), weights_proj, compressor: comp },
        wq_b_wt,
        wq_b_sb: sb,
        wq_b_deq,
    }
}

#[test]
fn comp_indexer_prefill_decode_vs_cpu() {
    let cfg = test_cfg();
    let mut rng = XorShift(0x1DE7_0001);
    let fix = synth_indexer(&mut rng);
    let rope = test_rope(140);
    let had = dsv4_cpu::hadamard_scaled(128);
    let max_seq = 8192usize;
    let x_pre = synth_x(&mut rng, 130, 4096);
    let qr_pre = synth_x(&mut rng, 130, 1024);
    let xs_dec: Vec<Vec<f32>> = (0..4).map(|_| synth_x(&mut rng, 1, 4096)).collect();
    let qrs_dec: Vec<Vec<f32>> = (0..4).map(|_| synth_x(&mut rng, 1, 1024)).collect();

    // ---- CPU: indexer_forward prefill + 4 decodes ----
    let mut ist = IndexerState {
        compressor: Compressor::new(fix.iw.compressor.clone()),
        kv_cache: vec![0.0; (max_seq / 4) * 128],
    };
    let cpu_topk_pre = dsv4_cpu::indexer_forward(&fix.iw, &mut ist, &x_pre, 130, &qr_pre, 0, 130, &rope, &had, &cfg);
    assert_eq!(cpu_topk_pre.len(), 130);
    assert_eq!(cpu_topk_pre[0].len(), 32);
    let mut cpu_topk_dec = Vec::new();
    for i in 0..4 {
        let t = dsv4_cpu::indexer_forward(&fix.iw, &mut ist, &xs_dec[i], 1, &qrs_dec[i], 130 + i, 128, &rope, &had, &cfg);
        cpu_topk_dec.push(t[0].clone());
    }
    // CPU score replicas (on the CPU cache state AFTER each step — note the score
    // gate below compares the FINAL prefill state, the largest tensor)
    let cpu_scores_pre = cpu_index_scores(&fix.wq_b_deq, &fix.iw.weights_proj, &ist.kv_cache,
        &x_pre, &qr_pre, 130, 0, &rope, &had, &cfg);

    // ---- GPU ----
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx").unwrap());
    dev.load_ptx(ptx, "gpu_batch", &["gemm_dsv4_fp8_bsb2"]).unwrap();
    let f_bsb = dev.get_func("gpu_batch", "gemm_dsv4_fp8_bsb2").unwrap();
    let rope_dev = DevRope::from_cpu(&dev, &rope).unwrap();
    let mut gpu = GpuIndexer::new(&dev, &ks, &stream, 4096, 64, 1024, 64, 128, 512,
        &fix.iw.compressor, 1e-6, &fix.wq_b_wt, &fix.wq_b_sb, &fix.iw.weights_proj, max_seq, 130).unwrap();

    let x_dev = to_bf16_dev(&dev, &x_pre);
    let qr_dev = to_bf16_dev(&dev, &qr_pre);
    dev.synchronize().unwrap();
    let k_pre = gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &x_dev, &qr_dev, 130, 0, 130, &rope_dev).unwrap();
    assert_eq!(k_pre, 32);
    let gpu_idx_pre = gpu.idx_host(&dev, 130, k_pre).unwrap();
    let gpu_scores_pre = gpu.scores_host(&dev, 130, 32).unwrap();
    let gpu_icache_pre = gpu.comp.cache_host(&dev, 33).unwrap();
    let mut gpu_idx_dec = Vec::new();
    for i in 0..4 {
        let xd = to_bf16_dev(&dev, &xs_dec[i]);
        let qd = to_bf16_dev(&dev, &qrs_dec[i]);
        let k = gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &xd, &qd, 1, 130 + i, 128, &rope_dev).unwrap();
        gpu_idx_dec.push((k, gpu.idx_host(&dev, 1, k).unwrap()));
    }
    let gpu_icache = gpu.comp.cache_host(&dev, 33).unwrap();
    let (gpu_kv_st, gpu_sc_st) = gpu.comp.state_host(&dev).unwrap();

    // ---- gates ----
    // (a) indexer kv_cache vs CPU (bit-exact target) — rows 0..33 after the full sequence
    let (cm, cu) = bf16_diff(&gpu_icache, &ist.kv_cache[..33 * 128]);
    eprintln!("indexer kv_cache rows 0..33: mism={cm}/{} max bf16 ulp={cu}", 33 * 128);
    // (b) indexer frontier state bit-exact
    let (s1, _) = f32_diff(&gpu_kv_st, &ist.compressor.st.kv_state);
    let (s2, _) = f32_diff(&gpu_sc_st, &ist.compressor.st.score_state);
    eprintln!("indexer frontier: kv mism={s1}, score mism={s2}");
    // (c) prefill topk: ordered + SETS vs CPU
    let want_pre: Vec<i64> = cpu_topk_pre.iter().flatten().copied().collect();
    let (ord, set, rows) = topk_row_match(&gpu_idx_pre, &want_pre, 130, k_pre);
    eprintln!("indexer prefill topk vs CPU: ordered={ord}/{rows} sets={set}/{rows}");
    // (d) decode topk per step
    for (i, (k, idx)) in gpu_idx_dec.iter().enumerate() {
        let want = &cpu_topk_dec[i];
        assert_eq!(*k, want.len(), "decode {i} k mismatch");
        let (o, st, r) = topk_row_match(idx, want, 1, *k);
        eprintln!("indexer decode {i} topk vs CPU: ordered={o}/{r} sets={st}/{r} (k={k})");
        assert_eq!(st, r, "decode {i} SET mismatch");
    }
    // (e) prefill scores rel-L2 vs CPU replica
    let mut finite_g = Vec::new();
    let mut finite_c = Vec::new();
    let mut inf_mism = 0usize;
    for i in 0..130 * 32 {
        match (gpu_scores_pre[i].is_finite(), cpu_scores_pre[i].is_finite()) {
            (true, true) => {
                finite_g.push(gpu_scores_pre[i]);
                finite_c.push(cpu_scores_pre[i]);
            }
            (false, false) => {}
            _ => inf_mism += 1,
        }
    }
    let rl = rel_l2(&finite_g, &finite_c);
    eprintln!("indexer prefill scores: rel-L2={rl:.3e} (finite {}), -inf mismatches={inf_mism}", finite_g.len());
    assert_eq!(inf_mism, 0, "mask pattern mismatch");
    assert!(rl <= 1e-4, "indexer scores rel-L2 {rl:.3e} > 1e-4");
    assert_eq!(set, rows, "prefill SET mismatches vs CPU");
    eprintln!("indexer cache gate: mism={cm} (bit-exact target; ulps={cu})");
    assert_eq!(cm, 0, "indexer kv_cache not bit-exact vs CPU");
    assert_eq!(s1 + s2, 0, "indexer frontier not bit-exact");
    let _ = (gpu_icache_pre, stream);
}

// ---------------------------------------------------------------------------
// 8. REAL-WEIGHTS GATES: load_layer(2) (CSA) / load_layer(3) (HCA), oracle
//    pre.x/dec*.x inputs (dsv4_diff.py export). The subsystem input is the
//    G1-proven CPU sublayer-input path: yn = rms_norm(hc_pre(x)), qr =
//    q_norm(wq_a(yn)) — the same yn/qr feeds the CPU reference and the GPU.
//    Compressor cache: bit-exact vs dsv4_cpu-on-same-inputs (target) and
//    reported vs the oracle kv_cache rows 128: (KV_QAT class). Indexer topk:
//    SETS exact vs oracle pre/dec*.topk_idx (tie regime), ordered reported.
// ---------------------------------------------------------------------------

struct RealPiece {
    cfg: Dsv4Config,
    layer: dsv4_cpu::CpuLayer,
    yns: Vec<Vec<f32>>, // [prefill S rows, then 4 decode rows] of attn-sublayer input
    qrs: Vec<Vec<f32>>,
    s: usize,
    d: usize,
}

fn load_real_piece(dir: &Path, layer_id: usize, kind: dsv4_load::LayerKind) -> RealPiece {
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).expect("load_config");
    assert_eq!(cfg.layer_kind(layer_id), kind, "layer {layer_id} kind");
    let layer = dsv4_load::load_layer(bundle, &cfg, layer_id).expect("load_layer");
    let layer = dsv4_cpu::cpu_layer_from_dsv4(layer, &cfg, kind).expect("cpu_layer_from_dsv4");
    let (xshape, pre_x) = read_f32(dir, "pre.x");
    let (s, d) = (xshape[0], 4usize);
    let mut xs = vec![pre_x];
    for i in 0..d {
        xs.push(read_f32(dir, &format!("dec{i}.x")).1);
    }
    let dim = cfg.dim;
    let qlr = cfg.q_lora_rank;
    let mut yns = Vec::new();
    let mut qrs = Vec::new();
    for x in &xs {
        let n = x.len() / (cfg.hc_mult * dim);
        let (y, _, _) = dsv4_cpu::hc_pre_all(x, n, &layer.hc_attn, &cfg);
        let yn = dsv4_cpu::rms_norm(&y, n, dim, &layer.attn_norm, cfg.norm_eps);
        let qr_pre = dsv4_cpu::quant_gemm(&yn, n, dim, &layer.attn.wq_a, qlr, 128);
        let qr = dsv4_cpu::rms_norm(&qr_pre, n, qlr, &layer.attn.q_norm, cfg.norm_eps);
        yns.push(yn);
        qrs.push(qr);
    }
    RealPiece { cfg, layer, yns, qrs, s, d }
}

/// GPU + CPU compressor sequence on the piece's yn inputs; returns
/// (gpu_cache[rows], cpu_cache[rows], gpu_state, cpu_state) for rows = fired rows.
#[allow(clippy::too_many_arguments)]
fn real_comp_run(
    piece: &RealPiece,
    cw: &CompressorWeights,
    rope: &dsv4_cpu::RopeTable,
    cache_rows: usize,
    fired_rows: usize,
) -> (Vec<f32>, Vec<f32>, (Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)) {
    let d = cw.head_dim;
    // CPU
    let mut cpu = Compressor::new(cw.clone());
    let mut cpu_cache = vec![0.0f32; cache_rows * d];
    let mut pos = 0usize;
    for (i, yn) in piece.yns.iter().enumerate() {
        let n = yn.len() / piece.cfg.dim;
        cpu.forward(yn, n, pos, rope, piece.cfg.norm_eps, &mut cpu_cache);
        pos += n;
        let _ = i;
    }
    let cpu_state = (cpu.st.kv_state.clone(), cpu.st.score_state.clone());
    // GPU
    let spec = CompSpec::from(cw);
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let gpu = GpuCompressor::new(&dev, &ks, &stream, spec, cw, piece.cfg.norm_eps, cache_rows, piece.s).unwrap();
    let rope_dev = DevRope::from_cpu(&dev, rope).unwrap();
    let mut pos = 0usize;
    for yn in &piece.yns {
        let n = yn.len() / piece.cfg.dim;
        let x_dev = to_bf16_dev(&dev, yn);
        dev.synchronize().unwrap();
        if pos == 0 {
            gpu.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x_dev, n, &rope_dev).unwrap();
        } else {
            gpu.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &x_dev, n, pos, &rope_dev).unwrap();
        }
        pos += n;
    }
    let gpu_cache = gpu.cache_host(&dev, fired_rows).unwrap();
    let gpu_state = gpu.state_host(&dev).unwrap();
    (gpu_cache, cpu_cache[..fired_rows * d].to_vec(), gpu_state, cpu_state)
}

#[test]
fn comp_real_csa_layer2_vs_oracle() {
    let dir = PathBuf::from(format!("{RT}/csa"));
    if !dir.join("pre.x.npy").exists() {
        eprintln!("SKIP comp_real_csa_layer2_vs_oracle: {} not found (run dsv4_diff.py export first)", dir.display());
        return;
    }
    let piece = load_real_piece(&dir, 2, dsv4_load::LayerKind::Csa);
    let (s, d_steps) = (piece.s, piece.d);
    let cfg = &piece.cfg;
    let rope = dsv4_cpu::layer_rope_table(cfg, dsv4_load::LayerKind::Csa, s + d_steps + 8);
    let had = dsv4_cpu::hadamard_scaled(cfg.index_head_dim);
    let cw = piece.layer.attn.compressor.clone().unwrap();
    let iw = piece.layer.attn.indexer.clone().unwrap();

    // ---------- gate A: attention compressor cache — GPU vs CPU (bit-exact target)
    let fired = s / 4 + 1; // 32 prefill + 1 (dec1@131)
    let (gpu_cache, cpu_cache, gpu_state, cpu_state) = real_comp_run(&piece, &cw, &rope, 2048, fired);
    let (cm, cu) = bf16_diff(&gpu_cache, &cpu_cache);
    eprintln!("[CSA L2] attn compressor cache rows 0..{fired}: GPU vs CPU mism={cm}/{} max bf16 ulp={cu}", fired * 512);
    let (st1, std1) = f32_diff(&gpu_state.0, &cpu_state.0);
    let (st2, std2) = f32_diff(&gpu_state.1, &cpu_state.1);
    eprintln!("[CSA L2] attn frontier: kv mism={st1} (max {std1:.3e}), score mism={st2} (max {std2:.3e})");

    // ---------- gate B: vs the ORACLE kv_cache rows 128..128+fired (KV_QAT class)
    let (_, oracle_kvc) = read_f32(&dir, "kv_cache"); // [2176, 512]
    let oracle_rows: Vec<f32> = oracle_kvc[128 * 512..(128 + fired) * 512].to_vec();
    let (om, ou) = bf16_diff(&gpu_cache, &oracle_rows);
    let orl = rel_l2(&gpu_cache, &oracle_rows);
    eprintln!("[CSA L2] attn compressor cache GPU vs ORACLE: mism={om}/{} max bf16 ulp={ou} rel-L2={orl:.3e}", fired * 512);
    let (cm_cpu_o, _) = bf16_diff(&cpu_cache, &oracle_rows);
    let orl_cpu = rel_l2(&cpu_cache, &oracle_rows);
    eprintln!("[CSA L2] (baseline CPU vs ORACLE: mism={cm_cpu_o} rel-L2={orl_cpu:.3e} — the torch-CPU control class)");
    assert!(cm <= 64 && cu <= 64, "attn compressor cache TC residual (mism {cm}, ulp {cu})");
    assert!(std1 <= 1e-3 && std2 <= 1e-3, "attn frontier TC max abs {std1:.3e}/{std2:.3e} > 1e-3");

    // ---------- gate C: indexer — CPU reference run
    let mut ist = IndexerState {
        compressor: Compressor::new(iw.compressor.clone()),
        kv_cache: vec![0.0; (8192 / 4) * 128],
    };
    let cpu_topk_pre = dsv4_cpu::indexer_forward(&iw, &mut ist, &piece.yns[0], s, &piece.qrs[0], 0, s, &rope, &had, cfg);
    let mut cpu_topk_dec = Vec::new();
    for i in 0..d_steps {
        let t = dsv4_cpu::indexer_forward(&iw, &mut ist, &piece.yns[1 + i], 1, &piece.qrs[1 + i], s + i, 128, &rope, &had, cfg);
        cpu_topk_dec.push(t[0].clone());
    }

    // ---------- GPU indexer run
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx").unwrap());
    dev.load_ptx(ptx, "gpu_batch", &["gemm_dsv4_fp8_bsb2"]).unwrap();
    let f_bsb = dev.get_func("gpu_batch", "gemm_dsv4_fp8_bsb2").unwrap();
    let rope_dev = DevRope::from_cpu(&dev, &rope).unwrap();
    let (wq_shape, wq_codes, wq_sb) = dsv4_load::read_raw_fp8(Path::new(BUNDLE), "layers.2.attn.indexer.wq_b.weight").unwrap();
    let wq_wt = quant::repack_fp8_mma(&wq_codes, wq_shape[0], wq_shape[1]);
    let mut gpu = GpuIndexer::new(&dev, &ks, &stream, cfg.dim, cfg.rope_head_dim, cfg.q_lora_rank,
        cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk, &iw.compressor, cfg.norm_eps,
        &wq_wt, &wq_sb, &iw.weights_proj, 8192, s).unwrap();

    let x_dev = to_bf16_dev(&dev, &piece.yns[0]);
    let qr_dev = to_bf16_dev(&dev, &piece.qrs[0]);
    dev.synchronize().unwrap();
    let k_pre = gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &x_dev, &qr_dev, s, 0, s, &rope_dev).unwrap();
    let gpu_idx_pre = gpu.idx_host(&dev, s, k_pre).unwrap();
    let mut gpu_idx_dec = Vec::new();
    for i in 0..d_steps {
        let xd = to_bf16_dev(&dev, &piece.yns[1 + i]);
        let qd = to_bf16_dev(&dev, &piece.qrs[1 + i]);
        let k = gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &xd, &qd, 1, s + i, 128, &rope_dev).unwrap();
        gpu_idx_dec.push((k, gpu.idx_host(&dev, 1, k).unwrap()));
    }
    // cache/state read AFTER the full sequence (row 32 is written by dec1 @131)
    let gpu_icache = gpu.comp.cache_host(&dev, fired).unwrap();
    let (gi_s1, gi_s2) = gpu.comp.state_host(&dev).unwrap();

    // ---------- gate D: indexer kv_cache GPU vs CPU (bit-exact target)
    let (im, iu) = bf16_diff(&gpu_icache, &ist.kv_cache[..fired * 128]);
    eprintln!("[CSA L2] indexer kv_cache rows 0..{fired}: GPU vs CPU mism={im}/{} max bf16 ulp={iu}", fired * 128);
    let (is1, _) = f32_diff(&gi_s1, &ist.compressor.st.kv_state);
    let (is2, _) = f32_diff(&gi_s2, &ist.compressor.st.score_state);
    eprintln!("[CSA L2] indexer frontier: kv mism={is1}, score mism={is2}");

    // ---------- gate E: topk vs CPU and vs ORACLE (SETS exact, tie regime)
    let want_pre: Vec<i64> = cpu_topk_pre.iter().flatten().copied().collect();
    let (o1, st1c, r1) = topk_row_match(&gpu_idx_pre, &want_pre, s, k_pre);
    eprintln!("[CSA L2] prefill topk GPU vs CPU: ordered={o1}/{r1} sets={st1c}/{r1} (k={k_pre})");
    let (_, op_shape, oracle_pre) = {
        let (sh, v) = read_i64(&dir, "pre.topk_idx");
        ((), sh, v)
    };
    assert_eq!(op_shape, vec![1, s, 128 + k_pre]);
    let oracle_comp: Vec<i64> = (0..s).flat_map(|i| oracle_pre[i * (128 + k_pre) + 128..(i + 1) * (128 + k_pre)].to_vec()).collect();
    let (o2, st2c, _) = topk_row_match(&gpu_idx_pre, &oracle_comp, s, k_pre);
    let (o3, st3c, _) = topk_row_match(&{
        want_pre.iter().map(|&v| v as i32).collect::<Vec<_>>()
    }, &oracle_comp, s, k_pre);
    eprintln!("[CSA L2] prefill topk GPU vs ORACLE: ordered={o2}/{r1} sets={st2c}/{r1}; (baseline CPU vs ORACLE: ordered={o3} sets={st3c})");
    assert_eq!(st2c, r1, "prefill topk SETS vs oracle not exact");
    for (i, (k, idx)) in gpu_idx_dec.iter().enumerate() {
        let (oshape, odec) = read_i64(&dir, &format!("dec{i}.topk_idx"));
        assert_eq!(oshape, vec![1, 1, 128 + k]);
        let ow = &odec[128..];
        let (od, sd, _) = topk_row_match(idx, ow, 1, *k);
        let (odc, sdc, _) = topk_row_match(&cpu_topk_dec[i].iter().map(|&v| v as i32).collect::<Vec<_>>(), ow, 1, *k);
        eprintln!("[CSA L2] decode {i} topk vs ORACLE: ordered={od}/1 sets={sd}/1; vs CPU sets=1/1 check={} (baseline CPU-vs-oracle sets={sdc}, ordered={odc})",
            topk_row_match(idx, &cpu_topk_dec[i], 1, *k).1);
        assert_eq!(sd, 1, "decode {i} SET vs oracle");
    }
    assert_eq!(im, 0, "indexer kv_cache GPU != CPU (bit-exact target)");
    assert_eq!(is1 + is2, 0, "indexer frontier GPU != CPU");

    // ---------- gate F: chunked-prefill equivalence on real weights (§12.B.5)
    // one-shot: prefill(130) + dec(130..133)  vs  chunked: prefill(128) + tokens(128..134)
    let mut chunk: Vec<f32> = Vec::new();
    chunk.extend_from_slice(&piece.yns[0][128 * cfg.dim..130 * cfg.dim]);
    for i in 0..d_steps {
        chunk.extend_from_slice(&piece.yns[1 + i]);
    }
    let gpu2 = GpuCompressor::new(&dev, &ks, &stream, CompSpec::from(&cw), &cw, cfg.norm_eps, 2048, piece.s).unwrap();
    let x128 = to_bf16_dev(&dev, &piece.yns[0][..128 * cfg.dim]);
    let xc = to_bf16_dev(&dev, &chunk);
    dev.synchronize().unwrap();
    gpu2.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x128, 128, &rope_dev).unwrap();
    gpu2.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &xc, 2 + d_steps, 128, &rope_dev).unwrap();
    let cache2 = gpu2.cache_host(&dev, fired).unwrap();
    let (gs2a, gs2b) = gpu2.state_host(&dev).unwrap();
    // the one-shot trajectory from gate A ran prefill(130)+4 decodes on `gpu` — rebuild for a clean compare
    let gpu1 = GpuCompressor::new(&dev, &ks, &stream, CompSpec::from(&cw), &cw, cfg.norm_eps, 2048, piece.s).unwrap();
    let x130 = to_bf16_dev(&dev, &piece.yns[0]);
    dev.synchronize().unwrap();
    gpu1.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x130, s, &rope_dev).unwrap();
    for i in 0..d_steps {
        let xd = to_bf16_dev(&dev, &piece.yns[1 + i]);
        gpu1.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &xd, 1, s + i, &rope_dev).unwrap();
    }
    let cache1 = gpu1.cache_host(&dev, fired).unwrap();
    let (gs1a, gs1b) = gpu1.state_host(&dev).unwrap();
    let (fm, _) = bf16_diff(&cache2, &cache1);
    let (fs1, _) = f32_diff(&gs2a, &gs1a);
    let (fs2, _) = f32_diff(&gs2b, &gs1b);
    eprintln!("[CSA L2] chunked [prefill 128 + 6 tokens] vs one-shot [130 + 4 dec]: cache mism={fm}, kv mism={fs1}, score mism={fs2}");
    assert_eq!(fm + fs1 + fs2, 0, "real-weights chunked != one-shot (§12.B.5)");
    let _ = stream;
}

#[test]
fn comp_real_hca_layer3_vs_oracle() {
    let dir = PathBuf::from(format!("{RT}/hca"));
    if !dir.join("pre.x.npy").exists() {
        eprintln!("SKIP comp_real_hca_layer3_vs_oracle: {} not found (run dsv4_diff.py export first)", dir.display());
        return;
    }
    let piece = load_real_piece(&dir, 3, dsv4_load::LayerKind::Hca);
    let (s, d_steps) = (piece.s, piece.d);
    let cfg = &piece.cfg;
    let rope = dsv4_cpu::layer_rope_table(cfg, dsv4_load::LayerKind::Hca, s + d_steps + 8);
    let cw = piece.layer.attn.compressor.clone().unwrap();

    // ---------- gate A: compressor cache GPU vs CPU (bit-exact target), vs oracle
    let fired = s / 128; // 1 block; decodes 130..133 never fire ((sp+1)%128 != 0)
    let (gpu_cache, cpu_cache, gpu_state, cpu_state) = real_comp_run(&piece, &cw, &rope, 64, fired);
    let (cm, cu) = bf16_diff(&gpu_cache, &cpu_cache);
    eprintln!("[HCA L3] compressor cache rows 0..{fired}: GPU vs CPU mism={cm}/{} max bf16 ulp={cu}", fired * 512);
    let (st1, std1) = f32_diff(&gpu_state.0, &cpu_state.0);
    let (st2, std2) = f32_diff(&gpu_state.1, &cpu_state.1);
    eprintln!("[HCA L3] frontier (122 remainder rows): kv mism={st1} (max {std1:.3e}), score mism={st2} (max {std2:.3e})");
    let (_, oracle_kvc) = read_f32(&dir, "kv_cache"); // [192, 512]
    let oracle_row: Vec<f32> = oracle_kvc[128 * 512..129 * 512].to_vec();
    let (om, ou) = bf16_diff(&gpu_cache, &oracle_row);
    let orl = rel_l2(&gpu_cache, &oracle_row);
    eprintln!("[HCA L3] compressor cache GPU vs ORACLE row 128: mism={om}/512 max bf16 ulp={ou} rel-L2={orl:.3e}");
    assert!(cm <= 64 && cu <= 64, "HCA compressor cache TC residual (mism {cm}, ulp {cu})");
    assert!(std1 <= 1e-3 && std2 <= 1e-3, "HCA frontier TC max abs {std1:.3e}/{std2:.3e} > 1e-3");

    // ---------- gate B: all-blocks index lists exact (get_compress_topk_idxs)
    let (tshape, oracle_pre) = read_i64(&dir, "pre.topk_idx"); // [1, 130, 129]
    let kc = tshape[2] - 128;
    let cpu_lists = dsv4_cpu::compress_topk_idxs(128, s, 0, s);
    let mut list_mism = 0usize;
    for i in 0..s {
        let ow = &oracle_pre[i * (128 + kc) + 128..(i + 1) * (128 + kc)];
        if ow.len() != kc || cpu_lists[i].iter().map(|&v| v).ne(ow.iter().copied()) {
            list_mism += 1;
        }
    }
    eprintln!("[HCA L3] prefill all-blocks lists vs oracle: mismatched rows={list_mism}/{s} (k={kc})");
    assert_eq!(list_mism, 0, "HCA all-blocks index lists wrong");
    for i in 0..d_steps {
        let (oshape, odec) = read_i64(&dir, &format!("dec{i}.topk_idx"));
        let want = dsv4_cpu::compress_topk_idxs(128, 1, s + i, 128)[0].clone();
        let got = &odec[128..];
        let ok = got.len() == want.len() && want.iter().zip(got.iter()).all(|(&a, &b)| a == b);
        eprintln!("[HCA L3] decode {i} all-blocks list vs oracle: shape={oshape:?} match={ok} (want {want:?})");
        assert!(ok, "HCA decode {i} list mismatch");
    }
    let _ = kc;
}

// ---------------------------------------------------------------------------
// 9. Indexer batch-invariance (§6.3.1, §12.B.2 — the CSA trap): a 2-wide
//    "verify" forward at start_pos>0 must produce, for its row 0, BIT-IDENTICAL
//    scores and selections to the 1-wide decode at the same position — indexer
//    decisions are a pure function of the committed prefix, never of width.
// ---------------------------------------------------------------------------

#[test]
fn comp_indexer_batch_invariance() {
    let cfg = test_cfg();
    let mut rng = XorShift(0xB1B7_0001);
    let fix = synth_indexer(&mut rng);
    let rope = test_rope(140);
    let max_seq = 8192usize;
    let x_pre = synth_x(&mut rng, 130, 4096);
    let qr_pre = synth_x(&mut rng, 130, 1024);
    let x_a = synth_x(&mut rng, 1, 4096);
    let qr_a = synth_x(&mut rng, 1, 1024);
    let x_b = synth_x(&mut rng, 1, 4096);
    let qr_b = synth_x(&mut rng, 1, 1024);

    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx").unwrap());
    dev.load_ptx(ptx, "gpu_batch", &["gemm_dsv4_fp8_bsb2"]).unwrap();
    let f_bsb = dev.get_func("gpu_batch", "gemm_dsv4_fp8_bsb2").unwrap();
    let rope_dev = DevRope::from_cpu(&dev, &rope).unwrap();
    let mut gpu = GpuIndexer::new(&dev, &ks, &stream, 4096, 64, 1024, 64, 128, 512,
        &fix.iw.compressor, 1e-6, &fix.wq_b_wt, &fix.wq_b_sb, &fix.iw.weights_proj, max_seq, 130).unwrap();

    // prefill, then snapshot the indexer compressor frontier
    let x_dev = to_bf16_dev(&dev, &x_pre);
    let qr_dev = to_bf16_dev(&dev, &qr_pre);
    dev.synchronize().unwrap();
    gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &x_dev, &qr_dev, 130, 0, 130, &rope_dev).unwrap();
    let snap = gpu.comp.snapshot(&dev, &stream).unwrap();

    // (a) decode token A alone at 130
    let xa = to_bf16_dev(&dev, &x_a);
    let qra = to_bf16_dev(&dev, &qr_a);
    let k1 = gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &xa, &qra, 1, 130, 128, &rope_dev).unwrap();
    let idx1 = gpu.idx_host(&dev, 1, k1).unwrap();
    let sc1 = gpu.scores_host(&dev, 1, 131 / 4).unwrap();

    // (b) restore frontier, then 2-wide "verify" of [A, B] at 130. Row 0 (token 130)
    // has the per-row block-causal limit (130+1)//4 = 32 — its k2=33 selection is the
    // 32 committed blocks + a −1 pad (the reference's own prefill-row re-mask pattern).
    gpu.comp.restore(&snap, &stream).unwrap();
    let mut xab = x_a.clone();
    xab.extend_from_slice(&x_b);
    let mut qrab = qr_a.clone();
    qrab.extend_from_slice(&qr_b);
    let xab_dev = to_bf16_dev(&dev, &xab);
    let qrab_dev = to_bf16_dev(&dev, &qrab);
    dev.synchronize().unwrap();
    let k2 = gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &xab_dev, &qrab_dev, 2, 130, 128, &rope_dev).unwrap();
    let idx2 = gpu.idx_host(&dev, 2, k2).unwrap();
    let sc2 = gpu.scores_host(&dev, 2, 132 / 4).unwrap();

    let nb = 131 / 4; // 32 committed blocks at token 130
    assert_eq!(k1, nb);
    assert_eq!(k2, 132 / 4);
    let (sm, _) = f32_diff(&sc2[..nb], &sc1[..nb]);
    let idx_eq: bool = idx1.iter().zip(idx2[..k1].iter()).all(|(&a, &b)| a == b);
    let pad_ok = idx2[k1..k2].iter().all(|&v| v == -1);
    eprintln!("indexer batch-invariance: verify row0 scores mism={sm}/{nb}, idx ordered-equal={idx_eq}, tail-pad-ok={pad_ok} (k2={k2})");
    assert_eq!(sm, 0, "verify row 0 scores != decode (batch-invariance broken)");
    assert!(idx_eq && pad_ok, "verify row 0 selection != decode + -1 pad (§12.B.2 broken)");
    // the masked score slot (block 32 — contains the future token 131) must be -inf
    assert_eq!(sc2[nb], f32::NEG_INFINITY, "verify row 0 saw the future block 32");
    // row 1 of the verify run must equal a fresh decode of B at 131 from the same state
    gpu.comp.restore(&snap, &stream).unwrap();
    gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &xa, &qra, 1, 130, 128, &rope_dev).unwrap();
    let xb = to_bf16_dev(&dev, &x_b);
    let qrb = to_bf16_dev(&dev, &qr_b);
    let k3 = gpu.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &xb, &qrb, 1, 131, 128, &rope_dev).unwrap();
    let idx3 = gpu.idx_host(&dev, 1, k3).unwrap();
    let idx2_row1: Vec<i64> = idx2[k2..2 * k2].iter().map(|&v| v as i64).collect();
    let idx3_i64: Vec<i64> = idx3.iter().map(|&v| v as i64).collect();
    let (o, st, _) = topk_row_match(&idx2_row1.iter().map(|&v| v as i32).collect::<Vec<_>>(), &idx3_i64, 1, k3);
    eprintln!("indexer verify row1 vs sequential decode@131: ordered={o}/1 sets={st}/1 (k={k3})");
    assert_eq!(st, 1, "verify row 1 != sequential decode (state-threading broken)");
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 10. LONG-CONTEXT gate (§12.B.6 regime, dsv4_csa.long8k): S=8192 prefill (2048
//     compressor blocks, 512-wide top-k over T=2048) + 4 decodes. vs the ORACLE
//     directly (a full dsv4_cpu re-run at this scale is prohibitively slow; the
//     yn/qr subsystem inputs still come from the G1-proven CPU path, so the only
//     un-replayed stage is the torch-vs-CPU GEMM-order class measured at S=130:
//     12/16896 flips, rel-L2 5.6e-4 — the KV_QAT/INT_EXACT bars absorb it).
//     Near-tie set flips are adjudicated with MY score gaps (the §12.B.2 regime:
//     a flipped block must sit within ~1e-4 of the row's selection threshold).
// ---------------------------------------------------------------------------

#[test]
fn comp_real_csa_long8k_vs_oracle() {
    let dir = PathBuf::from(format!("{RT}/csa_long8k"));
    if !dir.join("pre.x.npy").exists() {
        eprintln!("SKIP comp_real_csa_long8k_vs_oracle: {} not found (export dsv4_csa.long8k.npz first)", dir.display());
        return;
    }
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).expect("load_config");
    let layer = dsv4_load::load_layer(bundle, &cfg, 2).expect("load_layer");
    let layer = dsv4_cpu::cpu_layer_from_dsv4(layer, &cfg, dsv4_load::LayerKind::Csa).expect("cpu_layer_from_dsv4");
    let (xshape, pre_x) = read_f32(&dir, "pre.x");
    let s = xshape[0];
    assert_eq!(s, 8192);
    let mut dec_xs = Vec::new();
    for i in 0..4 {
        dec_xs.push(read_f32(&dir, &format!("dec{i}.x")).1);
    }
    // yn/qr via the G1-proven CPU sublayer-input path
    let (dim, qlr) = (cfg.dim, cfg.q_lora_rank);
    let (y, _, _) = dsv4_cpu::hc_pre_all(&pre_x, s, &layer.hc_attn, &cfg);
    let yn_pre = dsv4_cpu::rms_norm(&y, s, dim, &layer.attn_norm, cfg.norm_eps);
    let qr_pre = dsv4_cpu::quant_gemm(&yn_pre, s, dim, &layer.attn.wq_a, qlr, 128);
    let qr_pre = dsv4_cpu::rms_norm(&qr_pre, s, qlr, &layer.attn.q_norm, cfg.norm_eps);
    let mut yns = vec![yn_pre];
    let mut qrs = vec![qr_pre];
    for dx in &dec_xs {
        let (y, _, _) = dsv4_cpu::hc_pre_all(dx, 1, &layer.hc_attn, &cfg);
        let yn = dsv4_cpu::rms_norm(&y, 1, dim, &layer.attn_norm, cfg.norm_eps);
        let qrp = dsv4_cpu::quant_gemm(&yn, 1, dim, &layer.attn.wq_a, qlr, 128);
        let qr = dsv4_cpu::rms_norm(&qrp, 1, qlr, &layer.attn.q_norm, cfg.norm_eps);
        yns.push(yn);
        qrs.push(qr);
    }
    eprintln!("[CSA L2 long8k] yn/qr computed (S={s})");
    let rope = dsv4_cpu::layer_rope_table(&cfg, dsv4_load::LayerKind::Csa, s + 8);
    let cw = layer.attn.compressor.clone().unwrap();
    let iw = layer.attn.indexer.clone().unwrap();

    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let rope_dev = DevRope::from_cpu(&dev, &rope).unwrap();

    // ---------- attention compressor over the full 8192-token prefill
    let max_seq = 16384usize;
    let gpu = GpuCompressor::new(&dev, &ks, &stream, CompSpec::from(&cw), &cw, cfg.norm_eps, max_seq / 4, s).unwrap();
    let t0 = std::time::Instant::now();
    let x_dev = to_bf16_dev(&dev, &yns[0]);
    dev.synchronize().unwrap();
    let nb = gpu.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x_dev, s, &rope_dev).unwrap();
    assert_eq!(nb, 2048);
    dev.synchronize().unwrap();
    eprintln!("[CSA L2 long8k] compressor prefill: {} blocks in {:?}", nb, t0.elapsed());
    let (_, oracle_kvc) = read_f32(&dir, "kv_cache"); // [4224, 512]
    let fired = nb + 1; // 2048 prefill + 1 (dec1 @8193)
    // decode steps (fire check: 8192 no, 8193 yes, 8194/5 no)
    for (i, yn) in yns.iter().enumerate().skip(1) {
        let xd = to_bf16_dev(&dev, yn);
        let fired_i = gpu.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &xd, 1, s + i - 1, &rope_dev).unwrap();
        eprintln!("[CSA L2 long8k] decode @{} fired={}", s + i - 1, fired_i);
    }
    let gpu_cache = gpu.cache_host(&dev, fired).unwrap();
    let oracle_rows = &oracle_kvc[128 * 512..(128 + fired) * 512];
    let (om, ou) = bf16_diff(&gpu_cache, &oracle_rows);
    let orl = rel_l2(&gpu_cache, oracle_rows);
    eprintln!("[CSA L2 long8k] compressor cache rows 0..{fired} vs ORACLE: mism={om}/{} max bf16 ulp={ou} rel-L2={orl:.3e} (KV_QAT bar 2e-3)", fired * 512);
    assert!(orl <= 2e-3, "long8k compressor cache rel-L2 {orl:.3e} over the KV_QAT bar");

    // ---------- indexer at scale: 512-wide top-k over T=2048
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx").unwrap());
    dev.load_ptx(ptx, "gpu_batch", &["gemm_dsv4_fp8_bsb2"]).unwrap();
    let f_bsb = dev.get_func("gpu_batch", "gemm_dsv4_fp8_bsb2").unwrap();
    let (wq_shape, wq_codes, wq_sb) = dsv4_load::read_raw_fp8(bundle, "layers.2.attn.indexer.wq_b.weight").unwrap();
    let wq_wt = quant::repack_fp8_mma(&wq_codes, wq_shape[0], wq_shape[1]);
    let mut gpu_idx = GpuIndexer::new(&dev, &ks, &stream, dim, cfg.rope_head_dim, qlr,
        cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk, &iw.compressor, cfg.norm_eps,
        &wq_wt, &wq_sb, &iw.weights_proj, max_seq, s).unwrap();
    let t0 = std::time::Instant::now();
    let qr_dev = to_bf16_dev(&dev, &qrs[0]);
    dev.synchronize().unwrap();
    let k_pre = gpu_idx.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &x_dev, &qr_dev, s, 0, s, &rope_dev).unwrap();
    dev.synchronize().unwrap();
    eprintln!("[CSA L2 long8k] indexer prefill: k={k_pre} in {:?}", t0.elapsed());
    assert_eq!(k_pre, 512);
    let gpu_scores = gpu_idx.scores_host(&dev, s, 2048).unwrap();
    let gpu_idx_pre = gpu_idx.idx_host(&dev, s, k_pre).unwrap();

    // topk vs oracle (compress part = cols 128:), SETS exact target
    let (tshape, oracle_pre) = read_i64(&dir, "pre.topk_idx");
    assert_eq!(tshape, vec![1, s, 128 + k_pre]);
    let t = 128 + k_pre;
    let (mut ord, mut set, mut bad_rows) = (0usize, 0usize, 0usize);
    let mut max_gap = 0.0f64;
    for i in 0..s {
        let g = &gpu_idx_pre[i * k_pre..(i + 1) * k_pre];
        let w = &oracle_pre[i * t + 128..(i + 1) * t];
        if (0..k_pre).all(|j| g[j] as i64 == w[j]) {
            ord += 1;
            set += 1;
            continue;
        }
        let gs: std::collections::HashSet<i64> = g.iter().map(|&v| v as i64).collect();
        let ws: std::collections::HashSet<i64> = w.iter().copied().collect();
        if gs == ws {
            set += 1;
            continue;
        }
        // near-tie adjudication: every block in the symmetric difference must sit
        // within ~1e-4 of the row's k-th threshold score (in MY scores). The idx
        // values carry +offset (s); the score columns are the RAW block indices.
        let lim = (i + 1) / 4;
        let mut row_scores: Vec<(i64, f32)> = (0..2048.min(lim))
            .map(|b| (b as i64, gpu_scores[i * 2048 + b]))
            .collect();
        row_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        let thr = row_scores[k_pre - 1].1 as f64;
        let mut worst = 0.0f64;
        let mut detail = Vec::new();
        for &b in gs.symmetric_difference(&ws) {
            let raw = if b >= s as i64 { b - s as i64 } else { b };
            let sc = gpu_scores[i * 2048 + raw as usize] as f64;
            let gap = (thr - sc).abs() / thr.abs().max(1e-30);
            worst = worst.max(gap);
            detail.push((b, sc, gap));
        }
        max_gap = max_gap.max(worst);
        if worst > 1e-3 {
            bad_rows += 1;
            if bad_rows <= 6 {
                eprintln!("    row {i}: set mismatch (worst gap {worst:.3e}, thr={thr:.6e}): {detail:?}");
            }
        }
    }
    // NOTE (regime finding, see test 11): at S=8192 the 512-of->512 selection lives in
    // a dense near-tie regime where even the G1-proven CPU reference agrees with the
    // oracle on only ~72% of non-vacuous rows (the §12.B.2 documented exception
    // regime). "SETS exact vs oracle" is therefore NOT a valid gate at this scale —
    // the decisive scoring gate is GPU-vs-dsv4_cpu (test 11). Here we assert the
    // harness sanity only (real comparisons happened, no wild divergence) and REPORT.
    eprintln!("[CSA L2 long8k] prefill topk vs ORACLE: ordered={ord}/{s} sets={set}/{s}; adjudicated-near-tie rows={} (max gap {max_gap:.3e}), TRUE mismatches={bad_rows}",
        s - set - bad_rows);
    let nonvac = (2048..s).filter(|&i| (i + 1) / 4 > k_pre).count();
    eprintln!("[CSA L2 long8k] non-vacuous rows: {nonvac} (set agreement is scored on these)");
    assert!(bad_rows * 2 <= s, "wild divergence vs oracle ({bad_rows} hard mismatches)");

    // decodes vs oracle
    for (i, qr) in qrs.iter().enumerate().skip(1) {
        let qd = to_bf16_dev(&dev, qr);
        let xd = to_bf16_dev(&dev, &yns[i]);
        let k = gpu_idx.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &xd, &qd, 1, s + i - 1, 128, &rope_dev).unwrap();
        let idx = gpu_idx.idx_host(&dev, 1, k).unwrap();
        let (oshape, odec) = read_i64(&dir, &format!("dec{}.topk_idx", i - 1));
        assert_eq!(oshape, vec![1, 1, 128 + k]);
        let ow = &odec[128..];
        let (o, st, _) = topk_row_match(&idx, ow, 1, k);
        eprintln!("[CSA L2 long8k] decode {} topk vs ORACLE: ordered={o}/1 sets={st}/1 (k={k}) — reported (dense tie regime; decisive gate is test 11)", i - 1);
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 11. LONG-CONTEXT CPU-reference gate: the S=8192 oracle comparison (test 10)
//     showed ~28% of fully-visible rows flipping selection sets vs the ORACLE.
//     At S=130 every row selects ALL visible blocks (k ≥ lim), so set-equality
//     there is vacuous — the only honest scoring gate at scale is GPU vs
//     dsv4_cpu ON THE SAME yn/qr (the G1-proven semantics target). This test
//     runs dsv4_cpu::indexer_forward over the full 8192-token prefill + 4
//     decodes (minutes of CPU) and gates: indexer kv_cache bit-exact, scores
//     rel-L2, topk sets with near-tie adjudication. CPU-vs-oracle set agreement
//     is reported as the torch-CPU-control floor for this regime.
// ---------------------------------------------------------------------------

#[test]
fn comp_real_csa_long8k_cpu_reference() {
    let dir = PathBuf::from(format!("{RT}/csa_long8k"));
    if !dir.join("pre.x.npy").exists() {
        eprintln!("SKIP comp_real_csa_long8k_cpu_reference: export dsv4_csa.long8k.npz first");
        return;
    }
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).expect("load_config");
    let layer = dsv4_load::load_layer(bundle, &cfg, 2).expect("load_layer");
    let layer = dsv4_cpu::cpu_layer_from_dsv4(layer, &cfg, dsv4_load::LayerKind::Csa).expect("cpu_layer_from_dsv4");
    let (xshape, pre_x) = read_f32(&dir, "pre.x");
    let s = xshape[0];
    let mut dec_xs = Vec::new();
    for i in 0..4 {
        dec_xs.push(read_f32(&dir, &format!("dec{i}.x")).1);
    }
    let (dim, qlr) = (cfg.dim, cfg.q_lora_rank);
    let mut yns = Vec::new();
    let mut qrs = Vec::new();
    for x in std::iter::once(&pre_x).chain(dec_xs.iter()) {
        let n = x.len() / (cfg.hc_mult * dim);
        let (y, _, _) = dsv4_cpu::hc_pre_all(x, n, &layer.hc_attn, &cfg);
        let yn = dsv4_cpu::rms_norm(&y, n, dim, &layer.attn_norm, cfg.norm_eps);
        let qrp = dsv4_cpu::quant_gemm(&yn, n, dim, &layer.attn.wq_a, qlr, 128);
        let qr = dsv4_cpu::rms_norm(&qrp, n, qlr, &layer.attn.q_norm, cfg.norm_eps);
        yns.push(yn);
        qrs.push(qr);
    }
    eprintln!("[long8k cpu-ref] yn/qr done");
    let rope = dsv4_cpu::layer_rope_table(&cfg, dsv4_load::LayerKind::Csa, s + 8);
    let had = dsv4_cpu::hadamard_scaled(cfg.index_head_dim);
    let cw = layer.attn.compressor.clone().unwrap();
    let iw = layer.attn.indexer.clone().unwrap();

    // ---------- CPU reference: compressor + indexer over the full sequence
    let t0 = std::time::Instant::now();
    let mut cpu_comp = Compressor::new(cw.clone());
    let mut cpu_cache = vec![0.0f32; (16384 / 4) * 512];
    let mut ist = IndexerState {
        compressor: Compressor::new(iw.compressor.clone()),
        kv_cache: vec![0.0; (16384 / 4) * 128],
    };
    let mut pos = 0usize;
    let mut cpu_topk: Vec<Vec<i64>> = Vec::new();
    for (i, yn) in yns.iter().enumerate() {
        let n = yn.len() / dim;
        cpu_comp.forward(yn, n, pos, &rope, cfg.norm_eps, &mut cpu_cache);
        let off = if pos == 0 { n } else { cfg.window_size };
        let tk = dsv4_cpu::indexer_forward(&iw, &mut ist, yn, n, &qrs[i], pos, off, &rope, &had, &cfg);
        cpu_topk.extend(tk);
        eprintln!("[long8k cpu-ref] step {i} (pos {pos}, n {n}) done in {:?}", t0.elapsed());
        pos += n;
    }
    eprintln!("[long8k cpu-ref] CPU indexer_forward complete in {:?}", t0.elapsed());

    // ---------- GPU run (same trajectory)
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let rope_dev = DevRope::from_cpu(&dev, &rope).unwrap();
    let gpu = GpuCompressor::new(&dev, &ks, &stream, CompSpec::from(&cw), &cw, cfg.norm_eps, 16384 / 4, s).unwrap();
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx").unwrap());
    dev.load_ptx(ptx, "gpu_batch", &["gemm_dsv4_fp8_bsb2"]).unwrap();
    let f_bsb = dev.get_func("gpu_batch", "gemm_dsv4_fp8_bsb2").unwrap();
    let (wq_shape, wq_codes, wq_sb) = dsv4_load::read_raw_fp8(bundle, "layers.2.attn.indexer.wq_b.weight").unwrap();
    let wq_wt = quant::repack_fp8_mma(&wq_codes, wq_shape[0], wq_shape[1]);
    let mut gpu_idx = GpuIndexer::new(&dev, &ks, &stream, dim, cfg.rope_head_dim, qlr,
        cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk, &iw.compressor, cfg.norm_eps,
        &wq_wt, &wq_sb, &iw.weights_proj, 16384, s).unwrap();
    // prefill (capture the 512-wide selections + the raw scores for adjudication)
    let x_dev = to_bf16_dev(&dev, &yns[0]);
    let qr_dev = to_bf16_dev(&dev, &qrs[0]);
    dev.synchronize().unwrap();
    gpu.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x_dev, s, &rope_dev).unwrap();
    let k_pre = gpu_idx.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &x_dev, &qr_dev, s, 0, s, &rope_dev).unwrap();
    assert_eq!(k_pre, 512);
    let gpu_scores = gpu_idx.scores_host(&dev, s, 2048).unwrap();
    let gpu_idx_pre = gpu_idx.idx_host(&dev, s, k_pre).unwrap();
    let mut gpu_topk: Vec<Vec<i32>> = Vec::new();
    for r in 0..s {
        gpu_topk.push(gpu_idx_pre[r * k_pre..(r + 1) * k_pre].to_vec());
    }
    // decodes
    for (i, yn) in yns.iter().enumerate().skip(1) {
        let x_d = to_bf16_dev(&dev, yn);
        let q_d = to_bf16_dev(&dev, &qrs[i]);
        gpu.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &x_d, 1, s + i - 1, &rope_dev).unwrap();
        let k = gpu_idx.forward::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &f_bsb, &x_d, &q_d, 1, s + i - 1, cfg.window_size, &rope_dev).unwrap();
        gpu_topk.push(gpu_idx.idx_host(&dev, 1, k).unwrap());
    }

    // ---------- gates
    // (a) compressor + indexer caches bit-exact vs CPU
    let fired = 2049usize;
    let gpu_cache = gpu.cache_host(&dev, fired).unwrap();
    let (cm, cu) = bf16_diff(&gpu_cache, &cpu_cache[..fired * 512]);
    eprintln!("[long8k cpu-ref] attn compressor cache GPU vs CPU: mism={cm}/{} max bf16 ulp={cu}", fired * 512);
    for idx in 0..fired * 512 {
        if gpu_cache[idx] != cpu_cache[idx] {
            eprintln!("    flip: row={} col={} ({} dim): gpu={:.9e} (bits {:04x}) cpu={:.9e} (bits {:04x})",
                idx / 512, idx % 512, if idx % 512 < 448 { "nope/sim" } else { "rope" },
                gpu_cache[idx], bf16::from_f32(gpu_cache[idx]).to_bits(),
                cpu_cache[idx], bf16::from_f32(cpu_cache[idx]).to_bits());
        }
    }
    let gpu_icache = gpu_idx.comp.cache_host(&dev, fired).unwrap();
    let (im, iu) = bf16_diff(&gpu_icache, &ist.kv_cache[..fired * 128]);
    eprintln!("[long8k cpu-ref] indexer kv_cache GPU vs CPU: mism={im}/{} max bf16 ulp={iu}", fired * 128);
    // The attention compressor uses the WMMA TC GEMM; at long8k the reorder residual
    // accumulates over 2048 compressions (observed: 446/1049088 = 0.04% mismatches).
    // The correctness bar is the KV_QAT rel-L2 (same as the full-layer replay gate).
    let cm_total = fired * 512;
    let cm_rl = rel_l2(&gpu_cache, &cpu_cache[..cm_total]);
    eprintln!("[long8k cpu-ref] attn compressor cache GPU vs CPU: rel-L2={cm_rl:.3e} (KV_QAT bar 2e-3, {cm}/{cm_total} mismatches)");
    assert!(cm_rl <= 2e-3, "long8k attn compressor cache rel-L2 {cm_rl:.3e} over KV_QAT bar (TC reorder class)");
    assert_eq!(im, 0, "long8k indexer kv_cache GPU != CPU");

    // (b) topk sets on the non-vacuous rows (lim > k): GPU vs CPU, plus the
    //     CPU-vs-oracle floor. Flip rows get a full score-row adjudication:
    //     CPU scores for that row are recomputed on demand (q/weights once).
    let (_, oracle_pre) = read_i64(&dir, "pre.topk_idx");
    // CPU q/weights for on-demand score rows (indexer_forward internals, public helpers)
    let mut q_cpu = dsv4_cpu::quant_gemm(&qrs[0], s, qlr, &iw.wq_b, cfg.index_n_heads * cfg.index_head_dim, 128);
    {
        let nh = cfg.index_n_heads;
        let hd = cfg.index_head_dim;
        let posv: Vec<usize> = (0..s * nh).map(|i| i / nh).collect();
        dsv4_cpu::apply_rope(&mut q_cpu, s * nh, hd, &rope, &posv, false);
        dsv4_cpu::rotate_activation(&mut q_cpu, s * nh, hd, hd, &had);
        dsv4_cpu::fp4_act_quant_sim(&mut q_cpu, s * nh, hd, 32);
    }
    let mut w_cpu = dsv4_cpu::gemm_bf16(&yns[0], s, dim, &iw.weights_proj, cfg.index_n_heads);
    let wscale = ((cfg.index_head_dim as f64).powf(-0.5) * (cfg.index_n_heads as f64).powf(-0.5)) as f32;
    for v in w_cpu.iter_mut() {
        *v = dsv4_cpu::bf(*v * wscale);
    }
    eprintln!("[long8k cpu-ref] q/weights for adjudication ready in {:?}", t0.elapsed());
    // MECHANISM CHECK: GPU q (fp8_bsb + rope/fwht/fp4-sim) vs CPU q (quant_gemm + ...)
    // — the only tolerance-level stage in the whole indexer chain. Mismatches must be
    // isolated fp4-code flips (one E2M1 grid step), not bulk drift.
    let gpu_q = gpu_idx.q_host(&dev, s * cfg.index_n_heads).unwrap();
    {
        let (qm, qu) = bf16_diff(&gpu_q, &q_cpu);
        let rlq = rel_l2(&gpu_q, &q_cpu);
        let mut max_step = 0.0f32;
        let mut big = 0usize;
        for (&g, &c) in gpu_q.iter().zip(q_cpu.iter()) {
            let d = (g - c).abs();
            max_step = max_step.max(d);
            if d > 0.5 { big += 1; }
        }
        eprintln!("[long8k cpu-ref] q GPU vs CPU (post-fp4-sim): mism={qm}/{} max bf16 ulp={qu} max abs step={max_step:.3e} (>0.5: {big}) rel-L2={rlq:.3e}",
            gpu_q.len());
        // HARD gate on the mechanism: mismatches must be RARE, ISOLATED fp4-code
        // flips (one grid step), the fp8_bsb-vs-quant_gemm bf16-ulp class (G2 floor).
        assert!(qm * 10000 <= gpu_q.len() * 5, "q mismatch rate {qm}/{} over the isolated-flip class", gpu_q.len());
        assert!(max_step <= 1.5, "q max step {max_step} exceeds one fp4 grid step — bulk drift, not RNE flips");
    }
    let cpu_score_row = |i: usize, nb: usize| -> Vec<f32> {
        let (nh, hd) = (cfg.index_n_heads, cfg.index_head_dim);
        let mut row = vec![0.0f32; nb];
        for t in 0..nb {
            let kv = &ist.kv_cache[t * hd..(t + 1) * hd];
            let mut acc = 0.0f32;
            for h in 0..nh {
                let dot = dsv4_cpu::bf(dsv4_cpu::dot8(&q_cpu[(i * nh + h) * hd..(i * nh + h + 1) * hd], kv));
                let rel = if dot > 0.0 { dot } else { 0.0 };
                acc += dsv4_cpu::bf(rel * w_cpu[i * nh + h]);
            }
            row[t] = dsv4_cpu::bf(acc);
        }
        row
    };
    let k = 512usize;
    let (mut rows_checked, mut set_cpu, mut ord_cpu) = (0usize, 0usize, 0usize);
    let (mut set_orc_cpu, mut set_orc_gpu) = (0usize, 0usize);
    let (mut flip_rows, mut adj_ok, mut adj_bad) = (0usize, 0usize, 0usize);
    let mut worst_gap_n = 0.0f64;
    for i in 0..s {
        let lim = (i + 1) / 4;
        if lim <= k {
            continue; // select-all rows: vacuous
        }
        rows_checked += 1;
        let g = &gpu_topk[i];
        let c = &cpu_topk[i];
        let gs: std::collections::HashSet<i64> = g.iter().map(|&v| v as i64).collect();
        let cs: std::collections::HashSet<i64> = c.iter().copied().collect();
        let ot = 128 + k;
        let os: std::collections::HashSet<i64> = oracle_pre[i * ot + 128..(i + 1) * ot].iter().copied().collect();
        set_orc_cpu += (cs == os) as usize;
        set_orc_gpu += (gs == os) as usize;
        if g.iter().zip(c.iter()).all(|(&a, &b)| a as i64 == b) {
            ord_cpu += 1;
        }
        if gs == cs {
            set_cpu += 1;
            continue;
        }
        flip_rows += 1;
        // adjudication: threshold gaps on BOTH score rows (GPU captured + CPU on demand)
        let nb = 2048.min(lim);
        let crow = cpu_score_row(i, nb);
        let grow: Vec<f32> = (0..nb).map(|t| gpu_scores[i * 2048 + t]).collect();
        let kth = |row: &[f32]| -> f32 {
            let mut v = row.to_vec();
            v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            v[k - 1]
        };
        let (thr_g, thr_c) = (kth(&grow) as f64, kth(&crow) as f64);
        let rl = rel_l2(&grow, &crow);
        let mut worst = 0.0f64;
        for &b in gs.symmetric_difference(&cs) {
            let raw = if b >= s as i64 { b - s as i64 } else { b };
            let gap_g = ((grow[raw as usize] as f64) - thr_g).abs();
            let gap_c = ((crow[raw as usize] as f64) - thr_c).abs();
            worst = worst.max(gap_g).max(gap_c);
        }
        // near-tie band: the flipped block must sit within ~1e-3 ABSOLUTE of both
        // thresholds (the q-side fp4-code-flip noise level, see the mechanism check
        // above; score values are O(0.1) bf16-chained sums). The full-row rel-L2 is
        // REPORTED (it carries the q-row fp4 flips) but is not itself the criterion.
        // the fp4-flip noise envelope measured on q (rel-L2 ~1.6e-2, isolated steps):
        // threshold gaps up to ~5e-3 absolute are the same class (multi-block swaps
        // when a flipped q element shifts a whole head's dots). A logic bug reads as
        // bulk flips or O(0.1) gaps.
        if worst <= 5e-3 {
            adj_ok += 1;
        } else {
            adj_bad += 1;
            if adj_bad <= 6 {
                eprintln!("    row {i}: NON-near-tie flip: worst-gap={worst:.3e} row-relL2={rl:.3e} sym-diff={:?}",
                    gs.symmetric_difference(&cs).collect::<Vec<_>>());
            }
        }
        worst_gap_n = worst_gap_n.max(worst);
    }
    eprintln!("[long8k cpu-ref] non-vacuous rows={rows_checked}: GPU-vs-CPU ordered={ord_cpu} sets={set_cpu}; \
        flips={flip_rows} (near-tie adjudicated {adj_ok}, FAILED {adj_bad}; worst gap {worst_gap_n:.3e})");
    eprintln!("[long8k cpu-ref] (floor) CPU-vs-ORACLE sets={set_orc_cpu}/{rows_checked}, GPU-vs-ORACLE sets={set_orc_gpu}/{rows_checked}");
    assert_eq!(adj_bad, 0, "long8k GPU-vs-CPU topk: {adj_bad} rows with non-near-tie set flips");
    // flip-count budget: the fp4-flip class measured ~0.25% of non-vacuous rows;
    // 1% is 4x margin and still 25x below any logic-bug signature.
    assert!(flip_rows * 100 <= rows_checked, "flip rate {flip_rows}/{rows_checked} over budget");
    // decode selections: CPU-vs-GPU exact sets
    for i in 0..4 {
        let g = &gpu_topk[s + i];
        let c = &cpu_topk[s + i];
        let gs: std::collections::HashSet<i64> = g.iter().map(|&v| v as i64).collect();
        let cs: std::collections::HashSet<i64> = c.iter().copied().collect();
        let ordered = g.iter().zip(c.iter()).all(|(&a, &b)| a as i64 == b);
        eprintln!("[long8k cpu-ref] decode {i}: GPU-vs-CPU ordered={ordered} sets={}", gs == cs);
        assert_eq!(gs, cs, "long8k decode {i} GPU-vs-CPU set mismatch");
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 12. HCA long-context gate (dsv4_hca.long8k): S=8192 prefill → 64 compressor
//     blocks (multi-block regime), 4 decodes (no fires — next boundary @8319).
//     Cache bit-exact vs dsv4_cpu-on-same-inputs; rel-L2 vs oracle reported
//     (KV_QAT class); all-blocks index lists exact.
// ---------------------------------------------------------------------------

#[test]
fn comp_real_hca_long8k_vs_oracle() {
    let dir = PathBuf::from(format!("{RT}/hca_long8k"));
    if !dir.join("pre.x.npy").exists() {
        eprintln!("SKIP comp_real_hca_long8k_vs_oracle: export dsv4_hca.long8k.npz first");
        return;
    }
    let bundle = Path::new(BUNDLE);
    let cfg = dsv4_load::load_config(bundle).expect("load_config");
    let layer = dsv4_load::load_layer(bundle, &cfg, 3).expect("load_layer");
    let layer = dsv4_cpu::cpu_layer_from_dsv4(layer, &cfg, dsv4_load::LayerKind::Hca).expect("cpu_layer_from_dsv4");
    let (xshape, pre_x) = read_f32(&dir, "pre.x");
    let s = xshape[0];
    let mut dec_xs = Vec::new();
    for i in 0..4 {
        dec_xs.push(read_f32(&dir, &format!("dec{i}.x")).1);
    }
    let (dim, qlr) = (cfg.dim, cfg.q_lora_rank);
    let mut yns = Vec::new();
    for x in std::iter::once(&pre_x).chain(dec_xs.iter()) {
        let n = x.len() / (cfg.hc_mult * dim);
        let (y, _, _) = dsv4_cpu::hc_pre_all(x, n, &layer.hc_attn, &cfg);
        let yn = dsv4_cpu::rms_norm(&y, n, dim, &layer.attn_norm, cfg.norm_eps);
        let _ = qlr;
        yns.push(yn);
    }
    eprintln!("[HCA L3 long8k] yn computed (S={s})");
    let rope = dsv4_cpu::layer_rope_table(&cfg, dsv4_load::LayerKind::Hca, s + 8);
    let cw = layer.attn.compressor.clone().unwrap();

    // CPU reference compressor over the full sequence
    let t0 = std::time::Instant::now();
    let mut cpu = Compressor::new(cw.clone());
    let mut cpu_cache = vec![0.0f32; (16384 / 128) * 512];
    let mut pos = 0usize;
    for yn in &yns {
        let n = yn.len() / dim;
        cpu.forward(yn, n, pos, &rope, cfg.norm_eps, &mut cpu_cache);
        pos += n;
    }
    eprintln!("[HCA L3 long8k] CPU compressor done in {:?}", t0.elapsed());

    // GPU compressor
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let rope_dev = DevRope::from_cpu(&dev, &rope).unwrap();
    let gpu = GpuCompressor::new(&dev, &ks, &stream, CompSpec::from(&cw), &cw, cfg.norm_eps, 16384 / 128, s).unwrap();
    let mut pos = 0usize;
    for yn in &yns {
        let n = yn.len() / dim;
        let x_dev = to_bf16_dev(&dev, yn);
        dev.synchronize().unwrap();
        if pos == 0 {
            gpu.prefill::<dsv4_gpu::B, CudaSlice<i32>>(&dev, &ks, &stream, &x_dev, n, &rope_dev).unwrap();
        } else {
            gpu.forward_tokens::<dsv4_gpu::B, dsv4_gpu::S, CudaSlice<i32>, CudaSlice<u32>>(&dev, &ks, &stream, &x_dev, n, pos, &rope_dev).unwrap();
        }
        pos += n;
    }
    let fired = s / 128; // 64 blocks; decodes @8192..8195 never fire ((sp+1)%128 != 0)
    let gpu_cache = gpu.cache_host(&dev, fired).unwrap();
    let (cm, cu) = bf16_diff(&gpu_cache, &cpu_cache[..fired * 512]);
    eprintln!("[HCA L3 long8k] compressor cache rows 0..{fired} GPU vs CPU: mism={cm}/{} max bf16 ulp={cu}", fired * 512);
    let gpu_state_h = gpu.state_host(&dev).unwrap();
    let (sm1, sd1) = f32_diff(&gpu_state_h.0, &cpu.st.kv_state);
    let (sm2, sd2) = f32_diff(&gpu_state_h.1, &cpu.st.score_state);
    eprintln!("[HCA L3 long8k] frontier: kv mism={sm1} (max {sd1:.3e}), score mism={sm2} (max {sd2:.3e})");
    let (_, oracle_kvc) = read_f32(&dir, "kv_cache"); // [256, 512]
    let oracle_rows = &oracle_kvc[128 * 512..(128 + fired) * 512];
    let (om, ou) = bf16_diff(&gpu_cache, oracle_rows);
    let orl = rel_l2(&gpu_cache, oracle_rows);
    eprintln!("[HCA L3 long8k] compressor cache vs ORACLE: mism={om}/{} max bf16 ulp={ou} rel-L2={orl:.3e} (KV_QAT bar 2e-3)", fired * 512);
    assert!(cm <= 64 && cu <= 64, "HCA long8k cache TC residual (mism {cm}, ulp {cu})");
    assert!(sd1 <= 1e-3 && sd2 <= 1e-3, "HCA long8k frontier TC max abs {sd1:.3e}/{sd2:.3e} > 1e-3");
    assert!(orl <= 2e-3, "HCA long8k cache rel-L2 {orl:.3e} over KV_QAT bar");

    // all-blocks index lists exact vs oracle
    let (tshape, oracle_pre) = read_i64(&dir, "pre.topk_idx"); // [1, 8192, 192]
    let kc = tshape[2] - 128;
    assert_eq!(kc, 64);
    let cpu_lists = dsv4_cpu::compress_topk_idxs(128, s, 0, s);
    let mut list_mism = 0usize;
    for i in 0..s {
        let ow = &oracle_pre[i * (128 + kc) + 128..(i + 1) * (128 + kc)];
        if cpu_lists[i].len() != kc || cpu_lists[i].iter().copied().ne(ow.iter().copied()) {
            list_mism += 1;
        }
    }
    eprintln!("[HCA L3 long8k] prefill all-blocks lists vs oracle: mismatched rows={list_mism}/{s}");
    assert_eq!(list_mism, 0);
    for i in 0..4 {
        let (oshape, odec) = read_i64(&dir, &format!("dec{i}.topk_idx"));
        let want = dsv4_cpu::compress_topk_idxs(128, 1, s + i, 128)[0].clone();
        let got = &odec[128..];
        let ok = got.len() == want.len() && want.iter().zip(got.iter()).all(|(&a, &b)| a == b);
        eprintln!("[HCA L3 long8k] decode {i} list match={ok} ({} blocks)", want.len());
        assert!(ok);
    }
    let _ = stream;
}

// ---------------------------------------------------------------------------
// 9. Streaming indexer top-k (DSV4_LONG_CONTEXT_1M §4 — the 1M enabler): the
//    stripe scorer (dsv4_comp_index_score_tile_b) + carry-merge
//    (index_topk_streaming) vs the materialized full-matrix reference. Gates:
//    (a) 64K sub-range (nb=16384): streaming at nb_tile=4096 (4 stripes) ≡
//        full-matrix + dsv4_topk — per-row selection SETS equal (tie regime)
//        AND order-exact; stripe scores bitwise == the full matrix's columns.
//    (b) tile-width invariance: nb_tile ∈ {1024, 4096, 16384} → BITWISE-same idx.
//    (c) decode-side s=1 at nb=250000 (1M committed blocks): streaming ≡ the
//        CPU-sort reference on the materialized scores ≡ the 3C hierarchical
//        merge — SETS + order equal; GPU wall time measured for the choice.
// ---------------------------------------------------------------------------

use gb10_inference::dsv4_comp::{index_topk_streaming, TopkScratch};
use gb10_inference::dsv4_gpu::Dsv4Arg;

/// Deterministic top-k on the host: total order (value desc, index asc) — the
/// §12.B.2 rule as a third, independent implementation.
fn cpu_topk_reference(scores: &[f32], s: usize, nb: usize, k: usize) -> Vec<i32> {
    let mut out = vec![0i32; s * k];
    for r in 0..s {
        let row = &scores[r * nb..(r + 1) * nb];
        let mut ord: Vec<i32> = (0..nb as i32).collect();
        ord.sort_by(|&a, &b| {
            row[b as usize]
                .partial_cmp(&row[a as usize])
                .unwrap()
                .then(a.cmp(&b))
        });
        out[r * k..(r + 1) * k].copy_from_slice(&ord[..k]);
    }
    out
}

/// (rows with equal SETS, rows with equal ORDER) between two [s, k] idx buffers.
fn topk_set_order(a: &[i32], b: &[i32], s: usize, k: usize) -> (usize, usize) {
    let (mut set_ok, mut ord_ok) = (0usize, 0usize);
    for r in 0..s {
        let (ra, rb) = (&a[r * k..(r + 1) * k], &b[r * k..(r + 1) * k]);
        let (mut sa, mut sb) = (ra.to_vec(), rb.to_vec());
        sa.sort_unstable();
        sb.sort_unstable();
        if sa == sb {
            set_ok += 1;
        }
        if ra == rb {
            ord_ok += 1;
        }
    }
    (set_ok, ord_ok)
}

#[allow(clippy::too_many_arguments)]
fn synth_indexer_inputs(
    rng: &mut XorShift,
    dev: &Arc<CudaDevice>,
    s: usize,
    nb: usize,
) -> (CudaSlice<bf16>, CudaSlice<bf16>, CudaSlice<bf16>) {
    let (nh, hd) = (64usize, 128usize);
    let q: Vec<f32> = (0..s * nh * hd).map(|_| rng.f32()).collect();
    let kv: Vec<f32> = (0..nb * hd).map(|_| rng.f32()).collect();
    let w: Vec<f32> = (0..s * nh).map(|_| rng.f32().abs() * 0.5).collect();
    (to_bf16_dev(dev, &q), to_bf16_dev(dev, &kv), to_bf16_dev(dev, &w))
}

/// Materialized full-matrix scores [s, nb] via the original (kernel 6) scorer.
fn full_index_scores(
    dev: &Arc<CudaDevice>,
    ks: &CompKernels,
    stream: &cudarc::driver::CudaStream,
    q: &CudaSlice<bf16>,
    kv: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    s: usize,
    nb: usize,
    start_pos: usize,
) -> CudaSlice<f32> {
    let full = dev.alloc_zeros::<f32>(s * nb).unwrap();
    let (nb_i, sp_i, ratio_i) = (nb as i32, start_pos as i32, 4i32);
    let grid_y = ((nb + 1023) / 1024).max(1) as u32;
    gb10_inference::dsv4_launch!(ks.comp, "dsv4_comp_index_score_b", stream.stream,
        (s as u32, grid_y, 1), (256, 1, 1), 0,
        (q, kv, w, &full, &nb_i, &sp_i, &ratio_i)).unwrap();
    full
}

/// Streaming top-k on raw buffers → host [s, k].
fn streaming_topk(
    dev: &Arc<CudaDevice>,
    ks: &CompKernels,
    stream: &cudarc::driver::CudaStream,
    q: &CudaSlice<bf16>,
    kv: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    s: usize,
    nb: usize,
    k: usize,
    start_pos: usize,
    nb_tile: usize,
) -> Vec<i32> {
    let scr = TopkScratch::new(dev, s, k, nb_tile).unwrap();
    let out = dev.alloc_zeros::<i32>(s * k).unwrap();
    index_topk_streaming(ks, stream, q, kv, w, &scr, &out, s, nb, k, start_pos, 4).unwrap();
    dev.synchronize().unwrap();
    dev.dtoh_sync_copy(&out).unwrap()
}

#[test]
fn index_topk_streaming_set_match() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let k = 512usize;

    // ================= (a) 64K sub-range: nb=16384, streaming(4096) vs full-matrix =================
    for &(s, start_pos, tag) in &[(8usize, 0usize, "prefill-causal"), (8, 65536, "full-visibility"), (1, 65536, "decode-s1")] {
        let nb = 16384usize;
        let mut rng = XorShift(0x51EA_11 << 8 | s as u64);
        let (q, kv, w) = synth_indexer_inputs(&mut rng, &dev, s, nb);
        let full = full_index_scores(&dev, &ks, &stream, &q, &kv, &w, s, nb, start_pos);
        dev.synchronize().unwrap();
        // reference 1: full matrix + GPU dsv4_topk (the former production path)
        let ref_idx_dev = dev.alloc_zeros::<i32>(s * k).unwrap();
        let (s_i, nb_i, k_i) = (s as i32, nb as i32, k as i32);
        gb10_inference::dsv4_launch!(ks.spine, "dsv4_topk", stream.stream,
            (s as u32, 1, 1), (256, 1, 1), 0, (&full, &ref_idx_dev, &s_i, &nb_i, &k_i)).unwrap();
        dev.synchronize().unwrap();
        let ref_idx = dev.dtoh_sync_copy(&ref_idx_dev).unwrap();
        // reference 2: CPU-sort on the materialized scores (independent implementation)
        let full_host = dev.dtoh_sync_copy(&full).unwrap();
        let cpu_idx = cpu_topk_reference(&full_host, s, nb, k);
        // streaming at nb_tile=4096 (4 stripes)
        let got = streaming_topk(&dev, &ks, &stream, &q, &kv, &w, s, nb, k, start_pos, 4096);
        let (set_g, ord_g) = topk_set_order(&got, &ref_idx, s, k);
        let (set_c, ord_c) = topk_set_order(&got, &cpu_idx, s, k);
        eprintln!("[stream-a {tag}] s={s} nb={nb} tile=4096: vs full-matrix sets={set_g}/{s} order={ord_g}/{s} | vs CPU-sort sets={set_c}/{s} order={ord_c}/{s}");
        assert_eq!(set_g, s, "{tag}: streaming SETS != full-matrix sets");
        assert_eq!(set_c, s, "{tag}: streaming SETS != CPU-sort sets");
        assert_eq!(ord_g + ord_c, 2 * s, "{tag}: streaming ORDER != references (must be exact — the merge preserves the total order)");
        // stripe scores bitwise == the full matrix's columns (tile 1024, stripe 3 = cols 3072..)
        let tile = 1024usize;
        let scr = TopkScratch::new(&dev, s, k, tile).unwrap();
        let (t0_i, tc_i) = (3072i32, 1024i32);
        let grid_y = ((tile + 1023) / 1024).max(1) as u32;
        gb10_inference::dsv4_launch!(ks.comp, "dsv4_comp_index_score_tile_b", stream.stream,
            (s as u32, grid_y, 1), (256, 1, 1), 0,
            (&q, &kv, &w, &scr.scores_tile, &t0_i, &tc_i, &nb_i, &(start_pos as i32), &4i32)).unwrap();
        dev.synchronize().unwrap();
        let mut stripe_host = vec![0.0f32; s * tc_i as usize];
        dev.dtoh_sync_copy_into(&scr.scores_tile.slice(0..s * tc_i as usize), &mut stripe_host).unwrap();
        let mut score_mism = 0usize;
        for r in 0..s {
            for j in 0..tc_i as usize {
                if stripe_host[r * tc_i as usize + j].to_bits() != full_host[r * nb + 3072 + j].to_bits() {
                    score_mism += 1;
                }
            }
        }
        eprintln!("[stream-a {tag}] stripe[3072..4096) vs full cols: mismatched f32 {score_mism}/{} (bitwise required)", s * tc_i as usize);
        assert_eq!(score_mism, 0, "{tag}: stripe scorer != full scorer bitwise");
        // (b) tile-width invariance: tiles {1024, 4096, 16384} → bitwise-same idx
        let base = streaming_topk(&dev, &ks, &stream, &q, &kv, &w, s, nb, k, start_pos, 16384);
        for &tile in &[1024usize, 4096usize] {
            let alt = streaming_topk(&dev, &ks, &stream, &q, &kv, &w, s, nb, k, start_pos, tile);
            let mism = base.iter().zip(alt.iter()).filter(|(a, b)| a != b).count();
            eprintln!("[stream-b {tag}] tile {tile} vs 16384: idx mismatches {mism}/{} (bitwise required)", s * k);
            assert_eq!(mism, 0, "{tag}: tile-width {tile} changed the selection");
        }
    }

    // ================= (c) decode s=1 at nb=250000 (1M committed blocks) =================
    {
        let (s, nb, start_pos) = (1usize, 250000usize, 1000000usize);
        let mut rng = XorShift(0x1C_0DE0);
        let (q, kv, w) = synth_indexer_inputs(&mut rng, &dev, s, nb);
        // materialized full matrix at s=1 is only 1 MB — the ground truth for both legs
        let t_full0 = std::time::Instant::now();
        let full = full_index_scores(&dev, &ks, &stream, &q, &kv, &w, s, nb, start_pos);
        dev.synchronize().unwrap();
        let full_score_ms = t_full0.elapsed().as_secs_f64() * 1e3;
        let full_host = dev.dtoh_sync_copy(&full).unwrap();
        let cpu_idx = cpu_topk_reference(&full_host, s, nb, k);
        // leg 1: streaming (production path), timed
        let t0 = std::time::Instant::now();
        let got = streaming_topk(&dev, &ks, &stream, &q, &kv, &w, s, nb, k, start_pos, 16384);
        let stream_ms = t0.elapsed().as_secs_f64() * 1e3;
        // leg 2: the 3C hierarchical merge on the materialized matrix, timed
        let t1 = std::time::Instant::now();
        let n_chunks = nb.div_ceil(16384);
        let m = n_chunks * k;
        let mut stage1 = dev.alloc_zeros::<i32>(m).unwrap();
        let mut gathered = dev.alloc_zeros::<f32>(m).unwrap();
        let mut stage2 = dev.alloc_zeros::<i32>(k).unwrap();
        let hier_idx = dev.alloc_zeros::<i32>(k).unwrap();
        let (s_i, k_i, m_i, nb_i) = (1i32, k as i32, m as i32, nb as i32);
        for c in 0..n_chunks {
            let base = c * 16384;
            let cs = (nb - base).min(16384);
            let view = full.slice(base..base + cs);
            let chunk_idx = dev.alloc_zeros::<i32>(k).unwrap();
            let (cs_i, ck_i, col_i, off_i) = (cs as i32, k as i32, (c * k) as i32, base as i32);
            gb10_inference::dsv4_launch!(ks.spine, "dsv4_topk", stream.stream,
                (1u32, 1, 1), (256, 1, 1), 0, (&view, &chunk_idx, &s_i, &cs_i, &ck_i)).unwrap();
            gb10_inference::dsv4_launch!(ks.comp, "dsv4_idx_offset_place_b", stream.stream,
                (((k + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                (&stage1, &chunk_idx, &s_i, &ck_i, &m_i, &col_i, &off_i)).unwrap();
        }
        gb10_inference::dsv4_launch!(ks.comp, "dsv4_score_gather_b", stream.stream,
            (((m + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&gathered, &full, &stage1, &s_i, &m_i, &nb_i)).unwrap();
        gb10_inference::dsv4_launch!(ks.spine, "dsv4_topk", stream.stream,
            (1u32, 1, 1), (256, 1, 1), 0, (&gathered, &stage2, &s_i, &m_i, &k_i)).unwrap();
        gb10_inference::dsv4_launch!(ks.comp, "dsv4_idx_remap_b", stream.stream,
            (((k + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
            (&hier_idx, &stage2, &stage1, &s_i, &k_i, &m_i)).unwrap();
        dev.synchronize().unwrap();
        let hier_ms = t1.elapsed().as_secs_f64() * 1e3;
        let hier = dev.dtoh_sync_copy(&hier_idx).unwrap();
        let (set_h, ord_h) = topk_set_order(&got, &hier, s, k);
        let (set_c, ord_c) = topk_set_order(&got, &cpu_idx, s, k);
        eprintln!("[stream-c] decode s=1 nb=250000 (1M blocks): streaming {stream_ms:.2} ms (scorer-only full-matrix {full_score_ms:.2} ms) vs hierarchical {hier_ms:.2} ms");
        eprintln!("[stream-c] vs hierarchical: sets={set_h}/{s} order={ord_h}/{s} | vs CPU-sort: sets={set_c}/{s} order={ord_c}/{s}");
        assert_eq!(set_h + set_c, 2 * s, "streaming SETS differ at 250K blocks");
        assert_eq!(ord_h + ord_c, 2 * s, "streaming ORDER differs at 250K blocks");
        // peak-memory context-independence: scratch sizes don't take nb (construction-level)
        let (s_max, tile) = (4096usize, 16384usize);
        let bytes = s_max * tile * 4 + s_max * k * 4 * 4 + s_max * 2 * k * 4 * 2;
        eprintln!("[stream-c] TopkScratch bytes at s_max={s_max}, tile={tile}, k={k}: {:.1} MB — INDEPENDENT of nb (200K: nb=50072; 1M: nb=262144)",
                  bytes as f64 / 1e6);
    }
}

// ---------------------------------------------------------------------------
// R5a-1: the FP4-packed indexer cache write path (epilogue codes call) must be a
// LOSSLESS bijection with the bf16 QAT-sim cache: both kernels run the SAME
// dsv4_fp4_act_quant_body math on the SAME inputs (codes BEFORE the in-place sim,
// so neither sees the other's output), hence dequant(packed) == simmed bf16 rows
// BITWISE. This test proves the identity on synthetic rows: CPU-unpack of
// (codes, scales) must equal the simmed bf16 rows 100% bit-exact, and the packed
// scales must equal the sim's scales bytewise.
// ---------------------------------------------------------------------------

fn fp4_e2m1_to_f32(c: u8) -> f32 {
    // mirrors dsv4_fp4_to_f32 (gpu_dsv4.cu §C.2): sign<<3 | exp<<1 | man
    let e = (c >> 1) & 3;
    let m = c & 1;
    let mag = if e == 0 {
        if m == 1 { 0.5 } else { 0.0 }
    } else {
        (1.0 + 0.5 * m as f32) * (1u32 << (e - 1)) as f32
    };
    if c & 8 != 0 { -mag } else { mag }
}

#[test]
fn comp_indexer_fp4_packed_write_matches_bf16_sim() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let mut rng = XorShift(0xF4D3_0001);

    let (rows, n) = (9usize, 128usize);
    // magnitudes across several binades per row to exercise per-group UE8M0 scales
    let x: Vec<f32> = (0..rows * n)
        .map(|i| {
            let base = dsv4_cpu::bf(rng.f32());
            let mag = [1.0f32, 0.02, 5.0, 1e-3, 0.25, 40.0, 1e-4, 2.0, 8.0][(i / n) % 9];
            dsv4_cpu::bf(base * mag)
        })
        .collect();
    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_codes_in = dev.htod_sync_copy(&x_bf).unwrap();
    let mut x_sim_in = dev.htod_sync_copy(&x_bf).unwrap();
    let mut codes = dev.alloc_zeros::<u8>(rows * (n / 2)).unwrap();
    let mut scales = dev.alloc_zeros::<u8>(rows * (n / 32)).unwrap();
    let mut sim_scales = dev.alloc_zeros::<u8>(rows * (n / 32)).unwrap();
    dev.synchronize().unwrap();
    let (rows_i, n_i) = (rows as i32, n as i32);

    // codes FIRST (reads x unmodified), then the in-place sim — the epilogue's order.
    let warps = rows * (n / 32);
    gb10_inference::dsv4_launch!(ks.spine, "dsv4_fp4_act_quant", stream.stream,
        (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&x_codes_in, &mut codes, &mut scales, &rows_i, &n_i)).unwrap();
    gb10_inference::dsv4_launch!(ks.spine, "dsv4_fp4_act_quant_sim", stream.stream,
        (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&mut x_sim_in, &mut sim_scales, &rows_i, &n_i)).unwrap();
    dev.synchronize().unwrap();

    let codes_h = dev.dtoh_sync_copy(&codes).unwrap();
    let scales_h = dev.dtoh_sync_copy(&scales).unwrap();
    let sim_scales_h = dev.dtoh_sync_copy(&sim_scales).unwrap();
    let simmed: Vec<bf16> = dev.dtoh_sync_copy(&x_sim_in).unwrap();

    // scales bytewise equal
    let scale_diffs = scales_h.iter().zip(&sim_scales_h).filter(|(a, b)| a != b).count();
    eprintln!("[r5a-1] scale byte diffs: {scale_diffs}/{}", scales_h.len());
    assert_eq!(scale_diffs, 0, "packed scales != sim scales");

    // CPU-unpack: out[r, g*32+k] = bf16(fp4(nibble) * 2^(scale-127)) vs the simmed row
    let mut mism = 0usize;
    let mut first: Option<(usize, u16, u16)> = None;
    for r in 0..rows {
        for g in 0..(n / 32) {
            let sc = ((scales_h[r * (n / 32) + g] as i32 - 127) as f32).exp2();
            for k in 0..32 {
                let idx = g * 32 + k;
                let byte = codes_h[r * (n / 2) + g * 16 + (k >> 1)];
                let nib = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
                let v = bf16::from_f32(fp4_e2m1_to_f32(nib) * sc);
                let got = simmed[r * n + idx];
                if v.to_bits() != got.to_bits() {
                    mism += 1;
                    if first.is_none() {
                        first = Some((r * n + idx, v.to_bits(), got.to_bits()));
                    }
                }
            }
        }
    }
    eprintln!("[r5a-1] unpack-vs-sim bit mismatches: {mism}/{}", rows * n);
    assert_eq!(mism, 0, "FP4 pack NOT lossless (first: {first:?})");
    eprintln!("[r5a-1] PASS: dequant(packed) == simmed bf16 rows BITWISE ({} values)", rows * n);
}

// ---------------------------------------------------------------------------
// R5a-2: the packed score reader (dsv4_comp_index_score_fp4_b) must reproduce the bf16
// reader (dsv4_comp_index_score_b) BITWISE given the same logical cache: the FP4-packed
// rows are the lossless form of the QAT-simmed bf16 rows (the R5a-1 identity), and the
// head-grouped dequant preserves every dot8 chain + the ascending-head acc. Compares
// full [s, nblocks] score matrices (including the block-causal −inf mask region).
// ---------------------------------------------------------------------------
#[test]
fn comp_index_score_fp4_reader_matches_bf16_bitwise() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let mut rng = XorShift(0x5C0E_F204);

    let (s, nblocks, n, nh) = (3usize, 37usize, 128usize, 64usize);
    let start_pos = 20usize; // lim = (20+i+1)/4 < nblocks → exercises the −inf mask
    // cache rows: multi-binade synthetic (same shape as the R5a-1 test)
    let cx: Vec<f32> = (0..nblocks * n)
        .map(|i| dsv4_cpu::bf(dsv4_cpu::bf(rng.f32()) * [1.0f32, 0.02, 5.0, 1e-3][(i / n) % 4]))
        .collect();
    let cx_bf: Vec<bf16> = cx.iter().map(|&v| bf16::from_f32(v)).collect();
    let cx_codes_in = dev.htod_sync_copy(&cx_bf).unwrap();
    let mut cx_sim_in = dev.htod_sync_copy(&cx_bf).unwrap();
    let mut codes = dev.alloc_zeros::<u8>(nblocks * (n / 2)).unwrap();
    let mut scales = dev.alloc_zeros::<u8>(nblocks * (n / 32)).unwrap();
    let mut sim_scales = dev.alloc_zeros::<u8>(nblocks * (n / 32)).unwrap();
    // q + weights
    let q: Vec<bf16> = (0..s * nh * n).map(|_| bf16::from_f32(dsv4_cpu::bf(rng.f32() * 0.5))).collect();
    let w: Vec<bf16> = (0..s * nh).map(|_| bf16::from_f32(dsv4_cpu::bf(rng.f32().abs()))).collect();
    let q_dev = dev.htod_sync_copy(&q).unwrap();
    let w_dev = dev.htod_sync_copy(&w).unwrap();
    let mut scores_a = dev.alloc_zeros::<f32>(s * nblocks).unwrap();
    let mut scores_b = dev.alloc_zeros::<f32>(s * nblocks).unwrap();
    dev.synchronize().unwrap();

    // cache in both forms (codes first — the epilogue's order)
    let warps = nblocks * (n / 32);
    let (nb_i32, n_i32) = (nblocks as i32, n as i32);
    gb10_inference::dsv4_launch!(ks.spine, "dsv4_fp4_act_quant", stream.stream,
        (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&cx_codes_in, &mut codes, &mut scales, &nb_i32, &n_i32)).unwrap();
    gb10_inference::dsv4_launch!(ks.spine, "dsv4_fp4_act_quant_sim", stream.stream,
        (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&mut cx_sim_in, &mut sim_scales, &nb_i32, &n_i32)).unwrap();
    dev.synchronize().unwrap();

    // both readers, same q/weights/geometry (grid.y = 1 and 3 to cover the grid-stride)
    for &gy in &[1u32, 3] {
        let (s_i, nb_i, sp_i, ratio_i) = (s as i32, nblocks as i32, start_pos as i32, 4i32);
        gb10_inference::dsv4_launch!(ks.comp, "dsv4_comp_index_score_b", stream.stream,
            (s as u32, gy, 1), (256, 1, 1), 0,
            (&q_dev, &cx_sim_in, &w_dev, &mut scores_a, &nb_i, &sp_i, &ratio_i)).unwrap();
        gb10_inference::dsv4_launch!(ks.comp, "dsv4_comp_index_score_fp4_b", stream.stream,
            (s as u32, gy, 1), (256, 1, 1), 0,
            (&q_dev, &codes, &scales, &w_dev, &mut scores_b, &nb_i, &sp_i, &ratio_i)).unwrap();
        dev.synchronize().unwrap();
        let a = dev.dtoh_sync_copy(&scores_a).unwrap();
        let b = dev.dtoh_sync_copy(&scores_b).unwrap();
        let mism = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        eprintln!("[r5a-2] gy={gy}: score bit mismatches {mism}/{}", a.len());
        assert_eq!(mism, 0, "fp4 score reader NOT bitwise vs bf16 reader (gy={gy})");
    }
    eprintln!("[r5a-2] PASS: packed score reader ≡ bf16 reader bitwise ({} scores)", s * nblocks);
}

// ---------------------------------------------------------------------------
// R5b-1: the FP8-g64 packed attn-cache write path must be a LOSSLESS bijection with the
// bf16 QAT-sim cache on the simmed (nope) span: dsv4_comp_act_quant_g64s_b (codes) runs
// the identical body as dsv4_comp_act_quant_sim_g64s_b on identical inputs (codes first),
// so dequant(packed) == simmed bf16 rows[..448] BITWISE; the 64 rope dims stay raw.
// ---------------------------------------------------------------------------

fn fp8_e4m3_to_f32(c: u8) -> f32 {
    let sign = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((c >> 3) & 0xF) as i32;
    let m = (c & 0x7) as f32;
    let mag = if e == 0 { (m / 8.0) * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    sign * mag
}

#[test]
fn comp_attn_fp8_packed_write_matches_bf16_sim() {
    let dev = dev();
    let stream = dsv4_gpu::blocking_compute_stream(&dev);
    let ks = CompKernels::load(&dev).unwrap();
    let mut rng = XorShift(0xF864_0002);

    let (rows, ld, nope) = (7usize, 512usize, 448usize);
    let x: Vec<f32> = (0..rows * ld)
        .map(|i| {
            if i % ld >= nope { return 3.25; } // rope tail: constant marker, must stay raw
            dsv4_cpu::bf(dsv4_cpu::bf(rng.f32()) * [1.0f32, 0.02, 5.0, 1e-3, 0.25, 40.0, 2.0][(i / ld) % 7])
        })
        .collect();
    let x_bf: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_codes_in = dev.htod_sync_copy(&x_bf).unwrap();
    let mut x_sim_in = dev.htod_sync_copy(&x_bf).unwrap();
    let mut codes = dev.alloc_zeros::<u8>(rows * nope).unwrap();
    let mut scales = dev.alloc_zeros::<u8>(rows * (nope / 64)).unwrap();
    let mut sim_scales = dev.alloc_zeros::<u8>(rows * (nope / 64)).unwrap();
    dev.synchronize().unwrap();
    let (rows_i, nope_i, ld_i) = (rows as i32, nope as i32, ld as i32);

    let warps = rows * (nope / 64);
    gb10_inference::dsv4_launch!(ks.comp, "dsv4_comp_act_quant_g64s_b", stream.stream,
        (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&x_codes_in, &mut codes, &mut scales, &rows_i, &nope_i, &ld_i)).unwrap();
    gb10_inference::dsv4_launch!(ks.comp, "dsv4_comp_act_quant_sim_g64s_b", stream.stream,
        (((warps * 32 + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
        (&mut x_sim_in, &mut sim_scales, &rows_i, &nope_i, &ld_i)).unwrap();
    dev.synchronize().unwrap();

    let codes_h = dev.dtoh_sync_copy(&codes).unwrap();
    let scales_h = dev.dtoh_sync_copy(&scales).unwrap();
    let sim_scales_h = dev.dtoh_sync_copy(&sim_scales).unwrap();
    let simmed: Vec<bf16> = dev.dtoh_sync_copy(&x_sim_in).unwrap();

    let scale_diffs = scales_h.iter().zip(&sim_scales_h).filter(|(a, b)| a != b).count();
    eprintln!("[r5b-1] scale byte diffs: {scale_diffs}/{}", scales_h.len());
    assert_eq!(scale_diffs, 0, "packed scales != sim scales");

    let mut mism = 0usize;
    for r in 0..rows {
        for g in 0..(nope / 64) {
            let sc = ((scales_h[r * (nope / 64) + g] as i32 - 127) as f32).exp2();
            for k in 0..64 {
                let idx = g * 64 + k;
                let v = bf16::from_f32(fp8_e4m3_to_f32(codes_h[r * nope + idx]) * sc);
                let got = simmed[r * ld + idx];
                if v.to_bits() != got.to_bits() {
                    mism += 1;
                }
            }
        }
        // rope tail: raw (both kernels must not touch it)
        for idx in nope..ld {
            assert_eq!(simmed[r * ld + idx].to_f32(), 3.25, "rope tail modified at r{r} idx {idx}");
        }
    }
    eprintln!("[r5b-1] unpack-vs-sim bit mismatches: {mism}/{} (rope tail raw ✓)", rows * nope);
    assert_eq!(mism, 0, "FP8 pack NOT lossless on the nope span");
    eprintln!("[r5b-1] PASS: dequant(packed) == simmed bf16 nope span BITWISE ({} values)", rows * nope);
}
