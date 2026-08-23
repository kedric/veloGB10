//! Whole-layer A/B: dsv4_cpu vs the bundle's own model.py (torch CPU with §C
//! emulations) on a synthetic tiny model. Gate for §B end-to-end composition
//! BEFORE the Lane A loader merge.
//!
//! Prerequisite: `python3 scripts/dsv4_tiny_ab.py --out /tmp/dsv4_tiny_ab`
//! (dumps tiny-model weights/inputs/expects). If the dump is absent the tests
//! SKIP (so the suite stays green on machines without torch).

use gb10_inference::dsv4_cpu::*;
use gb10_inference::dsv4_load::{Dsv4Config, LayerKind};
use gb10_inference::quant::{e2m1_to_f32, e4m3_to_f32};
use half::bf16;
use std::path::{Path, PathBuf};

const ROOT: &str = "/tmp/dsv4_tiny_ab";

// ---------------------------------------------------------------------------
// mini npy reader (v1.0, LE, C-order; <f4 / <i8 / |u1) — test-local, Lane A's
// dsv4_load::read_npy is the production path (todo!() until the merge)
// ---------------------------------------------------------------------------

enum Npy {
    F32(Vec<f32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

// npy writers (test-local)
fn npy_hdr(shape: &[usize], descr: &str) -> Vec<u8> {
    let shape_s = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }}");
    let mut h = b"\x93NUMPY\x01\x00".to_vec();
    let total = 10 + dict.len() + 1;
    let pad = (64 - (total % 64)) % 64;
    let dict = format!("{}{}\n", dict, " ".repeat(pad));
    h.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    h.extend_from_slice(dict.as_bytes());
    h
}

fn write_npy_f32_t(path: &PathBuf, shape: &[usize], data: &[f32]) {
    let mut b = npy_hdr(shape, "<f4");
    for v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

fn write_npy_i64_t(path: &PathBuf, shape: &[usize], data: &[i64]) {
    let mut b = npy_hdr(shape, "<i8");
    for v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

fn read_npy(path: &Path) -> (Vec<usize>, Npy) {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(raw.starts_with(b"\x93NUMPY"), "{}: bad magic", path.display());
    let (major, hlen) = (raw[6], u16::from_le_bytes([raw[8], raw[9]]) as usize);
    assert_eq!(major, 1, "only npy v1.0");
    let hdr = std::str::from_utf8(&raw[10..10 + hlen]).unwrap().to_string();
    let descr = hdr.split("'descr':").nth(1).unwrap().split(',').next().unwrap().trim().trim_matches('\'').trim_matches('"');
    let sh = hdr.split("'shape':").nth(1).unwrap().split(')').next().unwrap().split('(').nth(1).unwrap();
    let shape: Vec<usize> = sh
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let numel: usize = shape.iter().product();
    let data = &raw[10 + hlen..];
    match descr {
        "<f4" => {
            assert_eq!(data.len(), numel * 4, "{}: f32 size", path.display());
            let v = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            (shape, Npy::F32(v))
        }
        "<i8" => {
            assert_eq!(data.len(), numel * 8, "{}: i64 size", path.display());
            let v = data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            (shape, Npy::I64(v))
        }
        "|u1" => {
            assert_eq!(data.len(), numel, "{}: u8 size", path.display());
            (shape, Npy::U8(data.to_vec()))
        }
        other => panic!("{}: unsupported descr {other}", path.display()),
    }
}

fn f32w(dir: &Path, key: &str) -> Vec<f32> {
    match read_npy(&dir.join(format!("{key}.npy"))) {
        (_, Npy::F32(v)) => v,
        _ => panic!("{key}: expected f32"),
    }
}

fn i64w(dir: &Path, key: &str) -> Vec<i64> {
    match read_npy(&dir.join(format!("{key}.npy"))) {
        (_, Npy::I64(v)) => v,
        _ => panic!("{key}: expected i64"),
    }
}

fn u8w(dir: &Path, key: &str) -> Vec<u8> {
    match read_npy(&dir.join(format!("{key}.npy"))) {
        (_, Npy::U8(v)) => v,
        _ => panic!("{key}: expected u8"),
    }
}

/// fp8 codes + ue8m0 128×128 block scales → exact f32 (Lane A's
/// dequant_fp8_exact semantics: e4m3·2^(b-127), each product exact).
fn fp8_dequant(dir: &Path, key: &str, out: usize, k: usize) -> Vec<f32> {
    let codes = u8w(dir, key);
    let scales = u8w(dir, &format!("{}.scale", key.replace(".weight", "")));
    assert_eq!(codes.len(), out * k);
    let nb_o = out.div_ceil(128);
    let nb_k = k.div_ceil(128);
    assert_eq!(scales.len(), nb_o * nb_k);
    let mut w = vec![0.0f32; out * k];
    for o in 0..out {
        for kk in 0..k {
            let sb = scales[(o / 128) * nb_k + kk / 128];
            let s = 2f32.powi(sb as i32 - 127);
            w[o * k + kk] = e4m3_to_f32(codes[o * k + kk]) * s;
        }
    }
    w
}

/// on-disk fp4 (packed nibbles, low = even K) + ue8m0 per-32 → exact f32
/// (the reference-side dequant; §F.3 proves Lane A's NVFP4 repack equals it).
fn fp4_dequant(dir: &Path, key: &str, out: usize, k: usize) -> Vec<f32> {
    let packed = u8w(dir, key);
    let scales = u8w(dir, &format!("{}.scale", key.replace(".weight", "")));
    assert_eq!(packed.len(), out * k / 2);
    assert_eq!(scales.len(), out * (k / 32));
    let mut w = vec![0.0f32; out * k];
    for o in 0..out {
        for kk in 0..k {
            let byte = packed[o * (k / 2) + kk / 2];
            let nib = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let sb = scales[o * (k / 32) + kk / 32];
            let s = 2f32.powi(sb as i32 - 127);
            w[o * k + kk] = e2m1_to_f32(nib) * s;
        }
    }
    w
}

// ---------------------------------------------------------------------------
// tiny config + CpuLayer builder (mirrors the harness's tiny model)
// ---------------------------------------------------------------------------

fn tiny_cfg() -> Dsv4Config {
    Dsv4Config {
        vocab_size: 32,
        dim: 128,
        moe_inter_dim: 128,
        n_layers: 4,
        n_hash_layers: 3,
        n_mtp_layers: 3,
        dspark_block_size: 5,
        dspark_noise_token_id: 31,
        dspark_target_layer_ids: vec![1, 2, 3],
        dspark_markov_rank: 16,
        n_heads: 2,
        n_routed_experts: 8,
        n_shared_experts: 1,
        n_activated_experts: 2,
        route_scale: 1.5,
        swiglu_limit: 10.0,
        q_lora_rank: 128,
        head_dim: 512,
        rope_head_dim: 64,
        o_groups: 2,
        o_lora_rank: 64,
        window_size: 4,
        original_seq_len: 65536,
        rope_theta: 10000.0,
        rope_factor: 16.0,
        beta_fast: 32,
        beta_slow: 1,
        index_n_heads: 2,
        index_head_dim: 128,
        index_topk: 6,
        hc_mult: 4,
        hc_sinkhorn_iters: 20,
        compress_rope_theta: 160000.0,
        compress_ratios: vec![0, 0, 4, 128, 0, 0, 0],
        norm_eps: 1e-6,
        hc_eps: 1e-6,
    }
}

fn load_tiny_layer(dir: &Path, cfg: &Dsv4Config, kind: LayerKind, hash_layer: bool) -> CpuLayer {
    let (dim, qlr, hd, nh) = (cfg.dim, cfg.q_lora_rank, cfg.head_dim, cfg.n_heads);
    let (g, r) = (cfg.o_groups, cfg.o_lora_rank);
    let compressor = match kind {
        LayerKind::Swa => None,
        LayerKind::Csa => Some(CompressorWeights {
            wkv: f32w(dir, "attn.compressor.wkv.weight"),
            wgate: f32w(dir, "attn.compressor.wgate.weight"),
            norm: f32w(dir, "attn.compressor.norm.weight"),
            ape: f32w(dir, "attn.compressor.ape"),
            ratio: 4,
            head_dim: hd,
            rope_dim: cfg.rope_head_dim,
            overlap: true,
            rotate: false,
            sim_group: 64,
            dim,
        }),
        LayerKind::Hca => Some(CompressorWeights {
            wkv: f32w(dir, "attn.compressor.wkv.weight"),
            wgate: f32w(dir, "attn.compressor.wgate.weight"),
            norm: f32w(dir, "attn.compressor.norm.weight"),
            ape: f32w(dir, "attn.compressor.ape"),
            ratio: 128,
            head_dim: hd,
            rope_dim: cfg.rope_head_dim,
            overlap: false,
            rotate: false,
            sim_group: 64,
            dim,
        }),
    };
    let indexer = if matches!(kind, LayerKind::Csa) {
        let ihd = cfg.index_head_dim;
        Some(IndexerWeights {
            wq_b: fp8_dequant(dir, "attn.indexer.wq_b.weight", cfg.index_n_heads * ihd, qlr),
            weights_proj: f32w(dir, "attn.indexer.weights_proj.weight"),
            compressor: CompressorWeights {
                wkv: f32w(dir, "attn.indexer.compressor.wkv.weight"),
                wgate: f32w(dir, "attn.indexer.compressor.wgate.weight"),
                norm: f32w(dir, "attn.indexer.compressor.norm.weight"),
                ape: f32w(dir, "attn.indexer.compressor.ape"),
                ratio: 4,
                head_dim: ihd,
                rope_dim: cfg.rope_head_dim,
                overlap: true,
                rotate: true,
                sim_group: 32,
                dim,
            },
        })
    } else {
        None
    };
    let ne = cfg.n_routed_experts;
    let experts: Vec<ExpertF32> = (0..ne)
        .map(|e| ExpertF32 {
            w1: fp4_dequant(dir, &format!("ffn.experts.{e}.w1.weight"), dim, dim),
            w2: fp4_dequant(dir, &format!("ffn.experts.{e}.w2.weight"), dim, dim),
            w3: fp4_dequant(dir, &format!("ffn.experts.{e}.w3.weight"), dim, dim),
        })
        .collect();
    let hc = |prefix: &str| HcParams {
        hc_fn: f32w(dir, &format!("{prefix}_fn")),
        hc_base: f32w(dir, &format!("{prefix}_base")),
        hc_scale: {
            let v = f32w(dir, &format!("{prefix}_scale"));
            [v[0], v[1], v[2]]
        },
    };
    CpuLayer {
        kind,
        attn: AttnWeights {
            wq_a: fp8_dequant(dir, "attn.wq_a.weight", qlr, dim),
            q_norm: f32w(dir, "attn.q_norm.weight"),
            wq_b: fp8_dequant(dir, "attn.wq_b.weight", nh * hd, qlr),
            wkv: fp8_dequant(dir, "attn.wkv.weight", hd, dim),
            kv_norm: f32w(dir, "attn.kv_norm.weight"),
            sink: f32w(dir, "attn.attn_sink"),
            wo_a: f32w(dir, "attn.wo_a.weight"),
            wo_b: fp8_dequant(dir, "attn.wo_b.weight", dim, g * r),
            compressor,
            indexer,
            kind,
        },
        attn_norm: f32w(dir, "attn_norm.weight"),
        ffn_norm: f32w(dir, "ffn_norm.weight"),
        hc_attn: hc("hc_attn"),
        hc_ffn: hc("hc_ffn"),
        moe: MoeWeights {
            gate_w: f32w(dir, "ffn.gate.weight"),
            gate_bias: if hash_layer { None } else { Some(f32w(dir, "ffn.gate.bias")) },
            tid2eid: if hash_layer {
                Some(i64w(dir, "ffn.gate.tid2eid").iter().map(|&v| v as i32).collect())
            } else {
                None
            },
            shared: ExpertF32 {
                w1: fp8_dequant(dir, "ffn.shared_experts.w1.weight", dim, dim),
                w2: fp8_dequant(dir, "ffn.shared_experts.w2.weight", dim, dim),
                w3: fp8_dequant(dir, "ffn.shared_experts.w3.weight", dim, dim),
            },
            experts: ExpertBank::from_f32(experts),
        },
        main_proj: None,
        main_norm: None,
        norm: None,
        hc_head: None,
        markov_w1: None,
        markov_w2: None,
        confidence: None,
    }
}

// ---------------------------------------------------------------------------
// comparison helpers
// ---------------------------------------------------------------------------

struct Diff {
    rel_l2: f64,
    max_abs: f32,
    n: usize,
}

fn diff(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        let d = (g - w) as f64;
        num += d * d;
        den += (*w as f64) * (*w as f64);
        max_abs = max_abs.max((g - w).abs());
    }
    Diff {
        rel_l2: (num / den.max(1e-30)).sqrt(),
        max_abs,
        n: got.len(),
    }
}

fn check_f32(name: &str, got: &[f32], want: &[f32], rel_tol: f64, abs_tol: f32) {
    let d = diff(got, want);
    eprintln!("    {name}: rel_l2 {:.3e} max_abs {:.3e} (n={})", d.rel_l2, d.max_abs, d.n);
    assert!(
        d.rel_l2 < rel_tol || d.max_abs < abs_tol,
        "{name}: rel_l2 {:.3e} (tol {rel_tol}), max_abs {:.3e} (tol {abs_tol})",
        d.rel_l2,
        d.max_abs
    );
}

fn check_i64(name: &str, got: &[i64], want: &[i64]) {
    if got != want {
        let nbad = got.iter().zip(want).filter(|(a, b)| a != b).count();
        panic!("{name}: {nbad}/{} mismatched indices", got.len());
    }
    eprintln!("    {name}: exact ({} indices)", got.len());
}

/// topk lists: per row, the SET of non-(-1) entries must match (order among
/// tied bf16 scores is implementation-defined in torch.topk; §12.B.2 pins ours).
/// `scores_for_near_tie`: optional [rows, nblocks] score array (the harness's
/// dbg.ix.scores) — rows whose sets differ only by blocks within a 2% score
/// gap of a swapped partner count as legitimate near-tie flips (summation-order
/// noise, not a selection bug).
fn check_topk(name: &str, got: &[i64], want: &[i64], rows: usize, t: usize, win: usize, offset: i64, scores_for_near_tie: Option<&[f32]>) {
    assert_eq!(got.len(), rows * t);
    assert_eq!(want.len(), rows * t);
    let mut order_diffs = 0usize;
    let mut flips = 0usize;
    for r in 0..rows {
        // window columns are pure functions — must be EXACT
        let gw = &got[r * t..r * t + win];
        let ww = &want[r * t..r * t + win];
        assert_eq!(gw, ww, "{name} row {r}: window columns differ: {gw:?} vs {ww:?}");
        let mut a: Vec<i64> = got[r * t + win..(r + 1) * t].iter().copied().filter(|&v| v != -1).collect();
        let mut b: Vec<i64> = want[r * t + win..(r + 1) * t].iter().copied().filter(|&v| v != -1).collect();
        if a != b {
            order_diffs += 1;
            let au = a.clone();
            let bu = b.clone();
            a.sort_unstable();
            b.sort_unstable();
            if a != b {
                flips += 1;
                let scores = scores_for_near_tie.unwrap_or_else(|| {
                    panic!("{name} row {r}: selected SETS differ (no scores to arbitrate): {a:?} vs {b:?}")
                });
                let erow = &scores[r * (scores.len() / rows)..(r + 1) * (scores.len() / rows)];
                let only_a: Vec<i64> = a.iter().copied().filter(|v| !b.contains(v)).collect();
                let only_b: Vec<i64> = b.iter().copied().filter(|v| !a.contains(v)).collect();
                for v in only_a.iter().chain(only_b.iter()) {
                    let blk = (v - offset) as usize;
                    let mut close = false;
                    for v2 in only_a.iter().chain(only_b.iter()) {
                        let blk2 = (v2 - offset) as usize;
                        if (erow[blk] - erow[blk2]).abs() <= 0.02 * erow[blk].abs().max(1e-6) {
                            close = true;
                        }
                    }
                    assert!(close, "{name} row {r}: HARD selection diff on block {v} (sets {au:?} vs {bu:?})");
                }
            }
        }
    }
    eprintln!("    {name}: window exact, {order_diffs} order-only rows, {flips} near-tie flip rows (sets arbitrated)");
}

fn dump_dir(piece: &str) -> Option<PathBuf> {
    let p = Path::new(ROOT).join(piece);
    if p.join("expect").is_dir() {
        Some(p)
    } else {
        eprintln!("SKIP {piece}: {} not found (run scripts/dsv4_tiny_ab.py first)", p.display());
        None
    }
}

// ---------------------------------------------------------------------------
// layer pieces
// ---------------------------------------------------------------------------

fn ab_layer(piece: &str, kind: LayerKind, hash_layer: bool) {
    let Some(root) = dump_dir(piece) else { return };
    let cfg = tiny_cfg();
    let wdir = root.join("weights");
    let idir = root.join("inputs");
    let edir = root.join("expect");
    let layer = load_tiny_layer(&wdir, &cfg, kind, hash_layer);
    let ids = i64w(&idir, "ids");
    let pre_x = f32w(&idir, "pre.x");
    let mut dec_xs = Vec::new();
    for i in 0..2 {
        dec_xs.push(f32w(&idir, &format!("dec{i}.x")));
    }
    let outs = run_layer_piece(&cfg, &layer, &ids, &pre_x, &dec_xs, 256);
    // write got/ for offline analysis
    let gdir = root.join("got");
    std::fs::create_dir_all(&gdir).unwrap();
    for (key, shape, data) in &outs.f32_arrays {
        write_npy_f32_t(&gdir.join(format!("{key}.npy")), shape, data);
    }
    for (key, shape, data) in &outs.i64_arrays {
        write_npy_i64_t(&gdir.join(format!("{key}.npy")), shape, data);
    }
    for (key, shape, data) in &outs.f32_arrays {
        let (eshape, edata) = match read_npy(&edir.join(format!("{key}.npy"))) {
            (s, Npy::F32(v)) => (s, v),
            _ => panic!("{key}: expected f32"),
        };
        assert_eq!(shape, &eshape, "{key}: shape");
        // Tolerances are GEMM-order-noise bounds, not correctness targets: a
        // composition bug shows as rel_l2 ≥ 1e-1 (e.g. the sim-writeback bug
        // gave 3e-1 on o_raw). Observed post-fix: ≤9.4e-3 rel on bf16 tensors,
        // ≤8.7e-5 on fp32 router_w, bit-exact router_idx / output_ids.
        let tol = if key.contains("router_w") { 1e-3 } else { 2e-2 };
        let atol = if key.contains("router_w") { 1e-3 } else { 5e-2 };
        check_f32(&format!("{piece}/{key}"), data, &edata, tol, atol);
    }
    for (key, shape, data) in &outs.i64_arrays {
        let (eshape, edata) = i64w_shape(&edir, key);
        assert_eq!(shape, &eshape, "{key}: shape");
        if key.contains("router_idx") {
            check_i64(&format!("{piece}/{key}"), data, &edata);
        } else if key.contains("topk_idx") {
            let rows = shape[1];
            let t = shape[2];
            // CSA prefill: offset = S; decode: offset = window. Scores for
            // near-tie arbitration come from the harness's dbg.ix.scores dump.
            let scores = if matches!(kind, LayerKind::Csa) && key == "pre.topk_idx" {
                Some(f32w(&edir, "dbg.ix.scores"))
            } else {
                None
            };
            let offset = if key.starts_with("pre") { 130 } else { cfg.window_size as i64 };
            check_topk(&format!("{piece}/{key}"), data, &edata, rows, t, cfg.window_size, offset, scores.as_deref());
        }
    }
}

fn i64w_shape(dir: &Path, key: &str) -> (Vec<usize>, Vec<i64>) {
    match read_npy(&dir.join(format!("{key}.npy"))) {
        (s, Npy::I64(v)) => (s, v),
        _ => panic!("{key}: expected i64"),
    }
}

#[test]
fn ab_swa_layer() {
    ab_layer("swa", LayerKind::Swa, true);
}

#[test]
fn ab_csa_layer() {
    ab_layer("csa", LayerKind::Csa, true); // layer 2 is hash-routed (layers 0–2)
}

#[test]
fn ab_hca_layer() {
    ab_layer("hca", LayerKind::Hca, false);
}

/// Bisection for the CSA indexer: q post-fp4, indexer cache, head weights,
/// block scores — each compared against the harness's dbg.ix.* dumps.
#[test]
fn ab_csa_indexer_stages() {
    let Some(root) = dump_dir("csa") else { return };
    let cfg = tiny_cfg();
    let layer = load_tiny_layer(&root.join("weights"), &cfg, LayerKind::Csa, true);
    let idir = root.join("inputs");
    let edir = root.join("expect");
    let pre_x = f32w(&idir, "pre.x");
    let s = 130usize;
    let dim = cfg.dim;
    let rope = layer_rope_table(&cfg, LayerKind::Csa, s + 16);
    let (y, _p, _c) = hc_pre_all(&pre_x, s, &layer.hc_attn, &cfg);
    let xn = rms_norm(&y, s, dim, &layer.attn_norm, cfg.norm_eps);
    // qr (attention latent shared with the indexer)
    let qr_pre = quant_gemm(&xn, s, dim, &layer.attn.wq_a, cfg.q_lora_rank, 128);
    let qr = rms_norm(&qr_pre, s, cfg.q_lora_rank, &layer.attn.q_norm, cfg.norm_eps);
    let iw = layer.attn.indexer.as_ref().unwrap();
    let ihd = cfg.index_head_dim;
    let inh = cfg.index_n_heads;
    let had = hadamard_scaled(ihd);
    // q = wq_b(qr); rope; hadamard; fp4 sim
    let mut q = quant_gemm(&qr, s, cfg.q_lora_rank, &iw.wq_b, inh * ihd, 128);
    {
        let rows = s * inh;
        let pos: Vec<usize> = (0..rows).map(|i| i / inh).collect();
        apply_rope(&mut q, rows, ihd, &rope, &pos, false);
        rotate_activation(&mut q, rows, ihd, ihd, &had);
        fp4_act_quant_sim(&mut q, rows, ihd, 32);
    }
    let eq = f32w(&edir, "dbg.ix.q");
    let myq: Vec<f32> = (0..s).flat_map(|i| q[(i * inh) * ihd..(i * inh + 1) * ihd].to_vec()).collect();
    // pre-hadamard/pre-fp4 comparison: recompute without them
    let mut q0 = quant_gemm(&qr, s, cfg.q_lora_rank, &iw.wq_b, inh * ihd, 128);
    {
        let rows = s * inh;
        let pos: Vec<usize> = (0..rows).map(|i| i / inh).collect();
        apply_rope(&mut q0, rows, ihd, &rope, &pos, false);
    }
    let eq0 = f32w(&edir, "dbg.ix.q_nofp4");
    // harness's q_nofp4 is post-hadamard (pre-fp4); reproduce mine that way
    let mut q0h = q0.clone();
    rotate_activation(&mut q0h, s * inh, ihd, ihd, &had);
    let myq0h: Vec<f32> = (0..s).flat_map(|i| q0h[(i * inh) * ihd..(i * inh + 1) * ihd].to_vec()).collect();
    check_f32("dbg.ix.q_nofp4", &myq0h, &eq0, 2e-3, 2e-2);
    check_f32("dbg.ix.q", &myq, &eq, 2e-3, 2e-2);
    // indexer compressor (its own cache) — driven through prefill + 2 decodes
    let mut comp = Compressor::new(iw.compressor.clone());
    let mut icache = vec![0.0f32; 64 * ihd];
    comp.forward(&xn, s, 0, &rope, cfg.norm_eps, &mut icache);
    for (i, sp) in [(130usize), (131)].iter().enumerate() {
        let dx = f32w(&idir, &format!("dec{i}.x"));
        let (dy, _p, _c) = hc_pre_all(&dx, 1, &layer.hc_attn, &cfg);
        let dxn = rms_norm(&dy, 1, dim, &layer.attn_norm, cfg.norm_eps);
        comp.forward(&dxn, 1, *sp, &rope, cfg.norm_eps, &mut icache);
    }
    let ecache = f32w(&edir, "dbg.ix.cache");
    // per-row report
    let mut worst = (0usize, 0.0f32);
    for r in 0..33usize {
        let a = &icache[r * ihd..(r + 1) * ihd];
        let b = &ecache[r * ihd..(r + 1) * ihd];
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        let den: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        let rel = if den > 0.0 { num / den } else { num };
        if rel > worst.1 {
            worst = (r, rel);
        }
        if r < 4 || rel > 1e-2 {
            eprintln!("  ix.cache row {r}: rel {rel:.3e} mine[:6] {:?} exp[:6] {:?}", &a[..6].iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>(), &b[..6].iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>());
        }
    }
    eprintln!("  worst ix.cache row: {} rel {:.3e}", worst.0, worst.1);
    write_npy_f32_t(&root.join("got").join("dbg.ix.cache.npy"), &[64, ihd], &icache);
    check_f32("dbg.ix.cache", &icache, &ecache[..64 * ihd], 1e-2, 3e-1);
    // weights
    let mut weights = gemm_bf16(&xn, s, dim, &iw.weights_proj, inh);
    let wscale = ((ihd as f64).powf(-0.5) * (inh as f64).powf(-0.5)) as f32;
    for v in weights.iter_mut() {
        *v = bf16::from_f32(*v * wscale).to_f32();
    }
    check_f32("dbg.ix.weights", &weights, &f32w(&edir, "dbg.ix.weights"), 2e-3, 2e-2);
    // scores
    let nblocks = s / 4;
    let mut score = vec![0.0f32; s * nblocks];
    for i in 0..s {
        for t in 0..nblocks {
            let kvrow = &icache[t * ihd..(t + 1) * ihd];
            let mut acc = 0.0f32;
            for h in 0..inh {
                let qrow = &q[(i * inh + h) * ihd..(i * inh + h + 1) * ihd];
                let dot = bf16::from_f32(dot8(qrow, kvrow)).to_f32();
                let rel = if dot > 0.0 { dot } else { 0.0 };
                acc += bf16::from_f32(rel * weights[i * inh + h]).to_f32();
            }
            score[i * nblocks + t] = bf16::from_f32(acc).to_f32();
        }
        let lim = (i + 1) / 4;
        for t in 0..nblocks {
            if t >= lim {
                score[i * nblocks + t] = f32::NEG_INFINITY;
            }
        }
    }
    let escore = f32w(&edir, "dbg.ix.scores");
    // masked entries are -inf on both sides (pure-function mask) — zero them for the diff
    let score_c: Vec<f32> = score.iter().map(|&v| if v.is_finite() { v } else { 0.0 }).collect();
    let escore_c: Vec<f32> = escore.iter().map(|&v| if v.is_finite() { v } else { 0.0 }).collect();
    check_f32("dbg.ix.scores", &score_c, &escore_c, 2e-2, 5e-2);
    // top-k sets, near-tie aware: a swap is legitimate only when the swapped
    // blocks' scores are close (§12.B.2: near-ties across summation orders)
    let want = i64w(&edir, "pre.topk_idx");
    let t = want.len() / s;
    let mut flips = 0usize;
    let mut hard = 0usize;
    for i in 0..s {
        let k = cfg.index_topk.min(nblocks);
        let lim = (i + 1) / 4;
        let got_sel = topk_deterministic(&score[i * nblocks..(i + 1) * nblocks], k);
        let mut gs: Vec<i64> = got_sel.iter().map(|&v| if v >= lim as i64 { -1 } else { v }).filter(|&v| v != -1).collect();
        let mut ws: Vec<i64> = want[i * t + cfg.window_size..i * t + t].iter().copied().filter(|&v| v != -1).map(|v| v - 130).collect();
        gs.sort_unstable();
        ws.sort_unstable();
        if gs != ws {
            flips += 1;
            // every block in the symmetric difference must be a near-tie of a
            // swapped partner (score gap ≤ 2% of the row's k-th score scale)
            let srow = &score[i * nblocks..(i + 1) * nblocks];
            let erow = &escore[i * nblocks..(i + 1) * nblocks];
            let only_g: Vec<i64> = gs.iter().copied().filter(|v| !ws.contains(v)).collect();
            let only_w: Vec<i64> = ws.iter().copied().filter(|v| !gs.contains(v)).collect();
            for b in only_g.iter().chain(only_w.iter()) {
                let b = *b as usize;
                let gap_mine = srow[b];
                let mut close = false;
                for b2 in only_g.iter().chain(only_w.iter()) {
                    let b2 = *b2 as usize;
                    if (srow[b] - srow[b2]).abs() <= 0.02 * srow[b].abs().max(1e-6)
                        || (erow[b] - erow[b2]).abs() <= 0.02 * erow[b].abs().max(1e-6)
                    {
                        close = true;
                    }
                }
                let _ = gap_mine;
                if !close {
                    hard += 1;
                    eprintln!("  HARD selection diff row {i}: block {b} mine {} want-sets {gs:?}/{ws:?}", srow[b]);
                }
            }
        }
    }
    eprintln!("  topk: {flips} near-tie flip rows, {hard} hard diffs");
    assert_eq!(hard, 0, "selection diffs must all be near-ties");
}

/// Bisection: recompute the swa attention sub-path stage by stage and report
/// per-stage rel_l2 against the harness's dbg.* dumps.
#[test]
fn ab_swa_stages() {
    let Some(root) = dump_dir("swa") else { return };
    let cfg = tiny_cfg();
    let layer = load_tiny_layer(&root.join("weights"), &cfg, LayerKind::Swa, true);
    let idir = root.join("inputs");
    let edir = root.join("expect");
    let pre_x = f32w(&idir, "pre.x");
    let s = 130usize;
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let rope = layer_rope_table(&cfg, LayerKind::Swa, s + 16);
    // collapsed
    // collapsed/normed/qr were observed BIT-EXACT (the fp8 GEMM's 128-wide K
    // happens to match torch's order on this config); accept exact or ≤1e-6.
    let (y, _posts, _combs) = hc_pre_all(&pre_x, s, &layer.hc_attn, &cfg);
    let ey = f32w(&edir, "dbg.collapsed");
    eprintln!("    dbg.collapsed: {}", if y == ey { "BIT-EXACT".into() } else { format!("rel_l2 {:.3e}", diff(&y, &ey).rel_l2) });
    assert!(diff(&y, &ey).rel_l2 < 1e-6, "collapsed");
    let xn = rms_norm(&y, s, dim, &layer.attn_norm, cfg.norm_eps);
    let exn = f32w(&edir, "dbg.normed");
    eprintln!("    dbg.normed: {}", if xn == exn { "BIT-EXACT".into() } else { format!("rel_l2 {:.3e}", diff(&xn, &exn).rel_l2) });
    assert!(diff(&xn, &exn).rel_l2 < 1e-6, "normed");
    let (qr, q, kv) = attn_qkv(&layer.attn, &xn, s, 0, &rope, &cfg);
    let eqr = f32w(&edir, "dbg.qr");
    eprintln!("    dbg.qr: {}", if qr == eqr { "BIT-EXACT".into() } else { format!("rel_l2 {:.3e}", diff(&qr, &eqr).rel_l2) });
    assert!(diff(&qr, &eqr).rel_l2 < 1e-6, "qr");
    // no-rope q (wq_b + per-head rescale, pre-RoPE)
    let qlr = cfg.q_lora_rank;
    let nh = cfg.n_heads;
    let hd = cfg.head_dim;
    let mut qnr = quant_gemm(&qr, s, qlr, &layer.attn.wq_b, nh * hd, 128);
    for i in 0..s * nh {
        let row = &mut qnr[i * hd..(i + 1) * hd];
        let mut ss = 0.0f32;
        for &v in row.iter() {
            ss += bf16::from_f32(v * v).to_f32();
        }
        let mean = bf16::from_f32(ss / hd as f32).to_f32();
        let arg = bf16::from_f32(mean + cfg.norm_eps).to_f32();
        let r = bf16::from_f32(arg.sqrt().recip()).to_f32();
        for v in row.iter_mut() {
            *v = bf16::from_f32(*v * r).to_f32();
        }
    }
    let myqnr: Vec<f32> = (0..s).flat_map(|i| qnr[(i * nh) * hd..(i * nh + 1) * hd].to_vec()).collect();
    check_f32("dbg.q_norope", &myqnr, &f32w(&edir, "dbg.q_norope"), 1e-2, 1e-1);
    // no-sim kv (kv_norm + RoPE, pre-QAT-sim)
    let kv_pre = quant_gemm(&xn, s, dim, &layer.attn.wkv, hd, 128);
    let mut kvn = rms_norm(&kv_pre, s, hd, &layer.attn.kv_norm, cfg.norm_eps);
    let posk: Vec<usize> = (0..s).collect();
    apply_rope(&mut kvn, s, hd, &rope, &posk, false);
    check_f32("dbg.kv_norope", &kvn, &f32w(&edir, "dbg.kv_norope"), 1e-3, 1e-2);
    let eq = f32w(&edir, "dbg.q");
    let myq: Vec<f32> = (0..s).flat_map(|i| q[(i * cfg.n_heads) * cfg.head_dim..(i * cfg.n_heads + 1) * cfg.head_dim].to_vec()).collect();
    check_f32("dbg.q", &myq, &eq, 1e-2, 1e-1);
    check_f32("dbg.kv", &kv, &f32w(&edir, "dbg.kv"), 1e-3, 1e-2);
    // raw sparse attention (window only)
    let idxs = window_topk_idxs(cfg.window_size, s, 0);
    let flat: Vec<i64> = idxs.into_iter().flatten().collect();
    let scale = (cfg.head_dim as f64).powf(-0.5) as f32;
    let o = sparse_attn(&q, s, cfg.n_heads, cfg.head_dim, &kv, s, &layer.attn.sink, &flat, 4, scale);
    let eo = f32w(&edir, "dbg.o_raw");
    let myo: Vec<f32> = (0..s).flat_map(|i| o[(i * cfg.n_heads) * cfg.head_dim..(i * cfg.n_heads + 1) * cfg.head_dim].to_vec()).collect();
    check_f32("dbg.o_raw", &myo, &eo, 1e-2, 5e-2);
    // de-rotation
    let mut od = o.clone();
    let rows = s * cfg.n_heads;
    let pos: Vec<usize> = (0..rows).map(|i| i / cfg.n_heads).collect();
    apply_rope(&mut od, rows, cfg.head_dim, &rope, &pos, true);
    let eod = f32w(&edir, "dbg.o_derot");
    let myod: Vec<f32> = (0..s).flat_map(|i| od[(i * cfg.n_heads) * cfg.head_dim..(i * cfg.n_heads + 1) * cfg.head_dim].to_vec()).collect();
    check_f32("dbg.o_derot", &myod, &eod, 1e-2, 5e-2);
}

// ---------------------------------------------------------------------------
// head piece
// ---------------------------------------------------------------------------

#[test]
fn ab_head_piece() {
    let Some(root) = dump_dir("head") else { return };
    let cfg = tiny_cfg();
    let wdir = root.join("weights");
    let hc_head = HcHeadParams {
        hc_fn: f32w(&wdir, "hc_head_fn"),
        hc_base: f32w(&wdir, "hc_head_base"),
        hc_scale: f32w(&wdir, "hc_head_scale")[0],
    };
    let norm = f32w(&wdir, "norm.weight");
    let head = f32w(&wdir, "head.weight");
    let x = f32w(&root.join("inputs"), "x");
    let outs = run_head_piece(&cfg, &hc_head, &norm, &head, &x);
    for (key, shape, data) in &outs.f32_arrays {
        let (eshape, edata) = match read_npy(&root.join("expect").join(format!("{key}.npy"))) {
            (s, Npy::F32(v)) => (s, v),
            _ => panic!("{key}"),
        };
        assert_eq!(shape, &eshape, "{key}: shape");
        let tol = if key == "logits" { 1e-4 } else { 2e-3 };
        check_f32(&format!("head/{key}"), data, &edata, tol, 5e-2);
    }
}

// ---------------------------------------------------------------------------
// dspark piece
// ---------------------------------------------------------------------------

#[test]
fn ab_dspark_piece() {
    let Some(root) = dump_dir("dspark") else { return };
    let cfg = tiny_cfg();
    let stages: Vec<CpuLayer> = (0..3)
        .map(|s| {
            let wdir = root.join("weights").join(format!("stage{s}"));
            let mut layer = load_tiny_layer(&wdir, &cfg, LayerKind::Swa, false);
            let dim = cfg.dim;
            if s == 0 {
                layer.main_proj = Some(fp8_dequant(&wdir, "main_proj.weight", dim, 3 * dim));
                layer.main_norm = Some(f32w(&wdir, "main_norm.weight"));
            }
            if s == 2 {
                layer.norm = Some(f32w(&wdir, "norm.weight"));
                layer.hc_head = Some(HcHeadParams {
                    hc_fn: f32w(&wdir, "hc_head_fn"),
                    hc_base: f32w(&wdir, "hc_head_base"),
                    hc_scale: f32w(&wdir, "hc_head_scale")[0],
                });
                layer.markov_w1 = Some(f32w(&wdir, "markov_head.markov_w1.weight"));
                layer.markov_w2 = Some(f32w(&wdir, "markov_head.markov_w2.weight"));
                layer.confidence = Some(f32w(&wdir, "confidence_head.proj.weight"));
            }
            layer
        })
        .collect();
    let stages: [CpuLayer; 3] = match stages.try_into() {
        Ok(s) => s,
        Err(_) => panic!("expected 3 stages"),
    };
    let embed = f32w(&root.join("weights"), "embed.weight");
    let head = f32w(&root.join("weights"), "head.weight");
    let warm = f32w(&root.join("inputs"), "warm.main_hidden");
    let draft = f32w(&root.join("inputs"), "draft.main_hidden");
    let real = i64w(&root.join("inputs"), "draft.real_token")[0];
    // stage-by-stage traced forward: compare each stage's attn/ffn/router too
    let sw = warm.len() / (3 * cfg.dim);
    let rope = layer_rope_table(&cfg, LayerKind::Swa, sw + cfg.dspark_block_size + 8);
    let mut rings: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0f32; cfg.window_size * cfg.head_dim]).collect();
    let st0 = &stages[0];
    let mxw = quant_gemm(&warm, sw, 3 * cfg.dim, st0.main_proj.as_ref().unwrap(), cfg.dim, 128);
    let main_x_w = rms_norm(&mxw, sw, cfg.dim, st0.main_norm.as_ref().unwrap(), cfg.norm_eps);
    let block = cfg.dspark_block_size;
    let draft_ids: Vec<i64> = std::iter::once(real)
        .chain(std::iter::repeat(cfg.dspark_noise_token_id as i64).take(block - 1))
        .collect();
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let mut h = vec![0.0f32; block * hc * dim];
    for (i, &id) in draft_ids.iter().enumerate() {
        for s in 0..hc {
            h[(i * hc + s) * dim..(i * hc + s + 1) * dim].copy_from_slice(&embed[id as usize * dim..(id as usize + 1) * dim]);
        }
    }
    for (i, st) in stages.iter().enumerate() {
        let _ = dspark_block_forward(st, &mut rings[i], &h, block, 0, &main_x_w, &rope, &cfg);
    }
    let mxd = quant_gemm(&draft, 1, 3 * cfg.dim, st0.main_proj.as_ref().unwrap(), cfg.dim, 128);
    let main_x_d = rms_norm(&mxd, 1, cfg.dim, st0.main_norm.as_ref().unwrap(), cfg.norm_eps);
    for (i, st) in stages.iter().enumerate() {
        let (h2, trace) = dspark_block_forward_traced(st, &mut rings[i], &h, block, sw, &main_x_d, &rope, &cfg);
        let edir = root.join("expect");
        check_f32(&format!("dspark/draft.attn_out{i}"), &trace.attn_out, &f32w(&edir, &format!("draft.attn_out{i}")), 2e-2, 1e-2);
        check_f32(&format!("dspark/draft.ffn_out{i}"), &trace.ffn_out, &f32w(&edir, &format!("draft.ffn_out{i}")), 2e-2, 1e-2);
        check_f32(&format!("dspark/draft.router_w{i}"), &trace.router_w, &f32w(&edir, &format!("draft.router_w{i}")), 1e-3, 1e-3);
        h = h2;
    }
    let outs = run_dspark_piece(&cfg, &stages, &embed, &head, &warm, &draft, real);
    let gdir = root.join("got");
    std::fs::create_dir_all(&gdir).unwrap();
    for (key, shape, data) in &outs.f32_arrays {
        write_npy_f32_t(&gdir.join(format!("{key}.npy")), shape, data);
    }
    for (key, shape, data) in &outs.i64_arrays {
        write_npy_i64_t(&gdir.join(format!("{key}.npy")), shape, data);
    }
    for (key, shape, data) in &outs.f32_arrays {
        let (eshape, edata) = match read_npy(&root.join("expect").join(format!("{key}.npy"))) {
            (s, Npy::F32(v)) => (s, v),
            _ => panic!("{key}"),
        };
        assert_eq!(shape, &eshape, "{key}: shape");
        // observed post-fix: h* ≤ 9.9e-3 rel / 3.9e-3 abs; logits/conf ≤ 8.6e-3
        let tol = 5e-2;
        let atol = if key.contains("logits") || key.contains("confidence") { 2e-2 } else { 1e-2 };
        check_f32(&format!("dspark/{key}"), data, &edata, tol, atol);
    }
    for (key, shape, data) in &outs.i64_arrays {
        let (eshape, edata) = i64w_shape(&root.join("expect"), key);
        assert_eq!(shape, &eshape, "{key}: shape");
        check_i64(&format!("dspark/{key}"), data, &edata);
    }
}
