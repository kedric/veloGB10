//! Unit tests for the DSV4 CPU reference model (Lane B owns this file).
//!
//! All tensors are synthetic — no bundle or oracle files are touched. References
//! are written from first principles (f64 or independent f32 replicas of
//! DEEPSEEK_V4_PORT.md §B/§C and model.py/kernel.py), so a green suite means the
//! implementation matches the *documented* math, not itself.

use gb10_inference::dsv4_cpu::*;
use gb10_inference::dsv4_load::{Dsv4Config, LayerKind};
use gb10_inference::quant;
use half::bf16;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn test_cfg() -> Dsv4Config {
    let mut compress_ratios = vec![0i32; 46];
    compress_ratios[0] = 0;
    compress_ratios[1] = 0;
    for i in 2..43 {
        compress_ratios[i] = if i % 2 == 0 { 4 } else { 128 };
    }
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
        compress_ratios,
        norm_eps: 1e-6,
        hc_eps: 1e-6,
    }
}

/// Deterministic pseudo-random f32 in [-scale, scale] (xorshift; no deps).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // 24-bit mantissa uniform in [0,1)
        let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

// ---------------------------------------------------------------------------
// 1. RoPE: table values, rotation vs first-principles complex multiply,
//    YaRN correction boundary dims 15..25 (§B.1.3)
// ---------------------------------------------------------------------------

#[test]
fn rope_plain_theta_table_and_rotation() {
    let rd = 64;
    let table = rope_table(rd, 4, 0, 10000.0, 16.0, 32, 1);
    // freq[i] = 10000^(-2i/64); position t angle = t * freq[i]
    for i in 0..32 {
        let f64_freq = 10000f64.powf(-2.0 * i as f64 / 64.0);
        let t = 3usize;
        let ang = t as f64 * f64_freq;
        let c = table.cos[t * 32 + i] as f64;
        let s = table.sin[t * 32 + i] as f64;
        assert!((c - ang.cos()).abs() < 1e-6, "cos[{t},{i}]: {} vs {}", c, ang.cos());
        assert!((s - ang.sin()).abs() < 1e-6, "sin[{t},{i}]: {} vs {}", s, ang.sin());
    }
    // rotation of a row's last-64 dims == adjacent-pair complex multiply
    let dim = 96; // 32 non-rope + 64 rope
    let mut rng = Rng(0x1234);
    let mut x = rng.vec(dim, 1.0);
    x.iter_mut().for_each(|v| *v = bf(*v)); // bf16-valued like the real path
    let orig = x.clone();
    apply_rope(&mut x, 1, dim, &table, &[3], false);
    // first 32 dims untouched
    assert_eq!(&x[..32], &orig[..32]);
    for j in 0..32 {
        let (re, im) = (orig[32 + 2 * j] as f64, orig[32 + 2 * j + 1] as f64);
        let ang = 3.0f64 * 10000f64.powf(-2.0 * j as f64 / 64.0);
        let (c, s) = (ang.cos(), ang.sin());
        let er = bf((re * c - im * s) as f32) as f64;
        let ei = bf((re * s + im * c) as f32) as f64;
        assert!((x[32 + 2 * j] as f64 - er).abs() < 8e-3, "rope re[{j}]: {} vs {er}", x[32 + 2 * j]);
        assert!((x[32 + 2 * j + 1] as f64 - ei).abs() < 8e-3, "rope im[{j}]: {} vs {ei}", x[32 + 2 * j + 1]);
    }
    // inverse == conjugate: rotate then inverse-rotate ≈ identity (bf16 round-trip)
    let mut y = orig.clone();
    apply_rope(&mut y, 1, dim, &table, &[3], false);
    apply_rope(&mut y, 1, dim, &table, &[3], true);
    for d in 32..dim {
        assert!((y[d] - orig[d]).abs() < 2e-2, "roundtrip dim {d}: {} vs {}", y[d], orig[d]);
    }
}

#[test]
fn rope_yarn_correction_boundary_dims() {
    let cfg = test_cfg();
    // compress-layer table: θ=160000, original_seq_len=65536, factor 16, β 32/1
    let yarn = rope_table(64, 2, cfg.original_seq_len, cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow);
    let plain = rope_table(64, 2, 0, cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow);
    // First-principles check of the correction range (model.py:211-217, f64):
    // low = floor(64·ln(65536/(32·2π)) / (2·ln 160000)) = 15
    // high = ceil(64·ln(65536/(1·2π)) / (2·ln 160000)) = 25
    let find = |num_rot: f64| 64.0 * (65536f64 / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * 160000f64.ln());
    assert_eq!(find(32.0).floor() as i32, 15);
    assert_eq!(find(1.0).ceil() as i32, 25);
    // Position-0 angle is 0 everywhere; compare the underlying freqs via a
    // nonzero position: angle[t=1, i] = freq[i]. Extract from cos/sin.
    let freq_of = |t: &RopeTable, i: usize| t.sin[32 + i].asin(); // t=1 → angle = freq (small, asin safe for i ≥ ~4)
    // i = 15: smooth = 1 → freq unchanged (plain θ table)
    {
        let f_y = yarn.sin[32 + 15].atan2(yarn.cos[32 + 15]);
        let f_p = plain.sin[32 + 15].atan2(plain.cos[32 + 15]);
        assert!((f_y - f_p).abs() < 1e-7, "dim 15 must be UNCORRECTED: {f_y} vs {f_p}");
    }
    // i = 25: smooth = 0 → freq = plain / 16
    {
        let f_y = yarn.sin[32 + 25].atan2(yarn.cos[32 + 25]);
        let f_p = plain.sin[32 + 25].atan2(plain.cos[32 + 25]);
        assert!((f_y - f_p / 16.0).abs() < 1e-7, "dim 25 must be fully interpolated: {f_y} vs {}", f_p / 16.0);
    }
    // i = 20: smooth = 0.5 → freq = plain/16·0.5 + plain·0.5
    {
        let f_y = freq_of(&yarn, 20);
        let f_p = freq_of(&plain, 20);
        let expect = f_p / 16.0 * 0.5 + f_p * 0.5;
        assert!((f_y - expect).abs() < 1e-6, "dim 20 halfway ramp: {f_y} vs {expect}");
    }
    // monotonic decrease across the band 15..=25
    for i in 15..25 {
        let a = freq_of(&yarn, i);
        let b = freq_of(&yarn, i + 1);
        assert!(a > b, "freqs must decrease: dim {i} {a} !> dim {} {b}", i + 1);
    }
}

// ---------------------------------------------------------------------------
// 2. mHC Sinkhorn vs a from-first-principles replica of the torch reference
//    (dsv4_ref.py selftest / kernel.py:371-438) (§B.8)
// ---------------------------------------------------------------------------

#[test]
fn sinkhorn_matches_torch_replica() {
    let mut rng = Rng(0x777);
    let eps = 1e-6f32;
    for _trial in 0..8 {
        let mixes: Vec<f32> = (0..24).map(|_| rng.next() * 3.0).collect();
        let scale = [rng.next().abs() + 0.5, rng.next().abs() + 0.5, rng.next().abs() + 0.5];
        let base: Vec<f32> = (0..24).map(|_| rng.next() * 2.0).collect();
        let (pre, post, comb) = hc_split_sinkhorn(&mixes, &scale, &base, 4, 20, eps);
        // independent replica (f32, same formulas, written naively)
        let sig = |x: f32| 1.0f32 / (1.0f32 + (-x).exp());
        for j in 0..4 {
            assert!((pre[j] - (sig(mixes[j] * scale[0] + base[j]) + eps)).abs() < 1e-7);
            assert!((post[j] - 2.0 * sig(mixes[4 + j] * scale[1] + base[4 + j])).abs() < 1e-7);
        }
        let mut c = [[0.0f32; 4]; 4];
        for j in 0..4 {
            for k in 0..4 {
                c[j][k] = mixes[8 + 4 * j + k] * scale[2] + base[8 + 4 * j + k];
            }
        }
        // row softmax + eps
        for j in 0..4 {
            let mx = c[j].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0f32;
            for k in 0..4 {
                c[j][k] = (c[j][k] - mx).exp();
                z += c[j][k];
            }
            for k in 0..4 {
                c[j][k] = c[j][k] / z + eps;
            }
        }
        // col norm, then 19×(row, col)
        let col_norm = |c: &mut [[f32; 4]; 4]| {
            for k in 0..4 {
                let s: f32 = (0..4).map(|j| c[j][k]).sum();
                for j in 0..4 {
                    c[j][k] /= s + eps;
                }
            }
        };
        col_norm(&mut c);
        for _ in 0..19 {
            for j in 0..4 {
                let s: f32 = c[j].iter().sum();
                for k in 0..4 {
                    c[j][k] /= s + eps;
                }
            }
            col_norm(&mut c);
        }
        for j in 0..4 {
            for k in 0..4 {
                assert!((comb[j * 4 + k] - c[j][k]).abs() < 2e-6, "comb[{j},{k}]: {} vs {}", comb[j * 4 + k], c[j][k]);
            }
        }
        // doubly-stochastic-ish: column sums ≈ 1 (final op is a column norm)
        for k in 0..4 {
            let s: f32 = (0..4).map(|j| comb[j * 4 + k]).sum();
            assert!((s - 1.0).abs() < 1e-2, "col {k} sum {s}");
        }
    }
}

#[test]
fn mhc_collapse_values() {
    // hc_pre_token / hc_head_token vs hand-computed mixing + collapse
    let mut rng = Rng(0xBEEF);
    let (hc, dim) = (4usize, 8usize);
    let hcd = hc * dim;
    let hc_fn = rng.vec(24 * hcd, 0.05);
    let hc_base = rng.vec(24, 0.5);
    let hc_scale = [1.3f32, 0.7, 2.1];
    let p = HcParams { hc_fn: hc_fn.clone(), hc_base: hc_base.clone(), hc_scale };
    let xf: Vec<f32> = rng.vec(hcd, 2.0).iter().map(|&v| bf(v)).collect();
    let (y, post, comb) = hc_pre_token(&xf, hc, dim, &p, 1e-6, 20, 1e-6);
    // independent: rsqrt, mixes, sinkhorn, collapse
    let ss = sumsq_tree(&xf);
    let rsqrt = (ss / hcd as f32 + 1e-6).sqrt().recip();
    let mut mixes = vec![0.0f32; 24];
    for m in 0..24 {
        mixes[m] = dot_tree(&hc_fn[m * hcd..(m + 1) * hcd], &xf) * rsqrt; // same order as the impl → bitwise mixes
    }
    let (pre2, post2, comb2) = hc_split_sinkhorn(&mixes, &hc_scale, &hc_base, 4, 20, 1e-6);
    assert_eq!(post, post2);
    assert_eq!(comb, comb2);
    let mut y2 = vec![0.0f32; dim];
    for h in 0..hc {
        for d in 0..dim {
            y2[d] += pre2[h] * xf[h * dim + d];
        }
    }
    y2.iter_mut().for_each(|v| *v = bf(*v));
    assert_eq!(y, y2, "hc_pre collapse must match the mixing pipeline bitwise");
    // hc_head: sigmoid-only collapse
    let hfn = rng.vec(hc * hcd, 0.05);
    let hbase = rng.vec(hc, 0.5);
    let hscale = 0.9f32;
    let yh = hc_head_token(&xf, hc, dim, &hfn, &hbase, hscale, 1e-6, 1e-6);
    let mut yh2 = vec![0.0f32; dim];
    for h in 0..hc {
        let mut acc = 0.0f32;
        for d in 0..hcd {
            acc += hfn[h * hcd + d] * xf[d];
        }
        let mixes = acc * rsqrt;
        let pre = (1.0f32 / (1.0f32 + (-(mixes * hscale + hbase[h])).exp())) + 1e-6;
        for d in 0..dim {
            yh2[d] += pre * xf[h * dim + d];
        }
    }
    yh2.iter_mut().for_each(|v| *v = bf(*v));
    assert!(max_abs_diff(&yh, &yh2) < 1e-6, "hc_head collapse");
}

// ---------------------------------------------------------------------------
// 3. QAT-sim bit-exactness (§C.1–2): floors, pow2 round-up, RNE ties,
//    sign-of-zero, nibble packing, RNE equivalence vs quant.rs
// ---------------------------------------------------------------------------

#[test]
fn ieee_bit_tricks() {
    // fast_log2_ceil on crafted normals/subnormals
    assert_eq!(fast_log2_ceil(1.0), 0);
    assert_eq!(fast_log2_ceil(1.5), 1);
    assert_eq!(fast_log2_ceil(8.0), 3);
    assert_eq!(fast_log2_ceil(f32::from_bits(0x41000001)), 4); // just above 8
    assert_eq!(fast_log2_ceil(2f32.powi(-126)), -126);
    // subnormal inputs follow the kernel's bit math (exp field 0): -126
    assert_eq!(fast_log2_ceil(f32::from_bits(0x007FFFFF)), -126);
    assert_eq!(fast_log2_ceil(f32::from_bits(0x00000001)), -126);
    assert_eq!(fast_pow2(0), 1.0);
    assert_eq!(fast_pow2(-22), 2f32.powi(-22));
    assert_eq!(fast_pow2(127), 2f32.powi(127));
}

#[test]
fn rne_e4m3_encoder() {
    // ties-to-even: midpoints between adjacent codes
    assert_eq!(f32_to_e4m3_rne(1.0625), 0x38); // 1.0 (even) vs 1.125
    assert_eq!(f32_to_e4m3_rne(1.1875), 0x3A); // 1.125 vs 1.25 (even)
    assert_eq!(f32_to_e4m3_rne(-1.1875), 0x80 | 0x3A);
    // sign of zero preserved (cvt.rn behavior)
    assert_eq!(f32_to_e4m3_rne(0.0), 0x00);
    assert_eq!(f32_to_e4m3_rne(-0.0), 0x80);
    // saturation
    assert_eq!(f32_to_e4m3_rne(448.0), 0x7E);
    assert_eq!(f32_to_e4m3_rne(500.0), 0x7E);
    assert_eq!(f32_to_e4m3_rne(-500.0), 0xFE);
    // RNE equivalence with quant::f32_to_e4m3 on all non-tie values: sweep every
    // bf16 bit pattern (they are the sim's actual inputs) and assert the two
    // encoders differ ONLY at exact ties (and there mine picks the even code).
    let table: Vec<f32> = (0u8..127).map(quant::e4m3_to_f32).collect();
    let mut n_ties = 0u32;
    for bits in 0u16..=0x7F80 {
        for sign in [0u16, 0x8000] {
            let b = bf16::from_bits(bits | sign);
            let x = b.to_f32();
            if !x.is_finite() || x.abs() > 448.0 {
                continue;
            }
            let a = f32_to_e4m3_rne(x);
            let q = quant::f32_to_e4m3(x);
            if a != q {
                // must be an exact tie between adjacent codes; rne chose the even one
                let av = quant::e4m3_to_f32(a & 0x7F).abs();
                let qv = quant::e4m3_to_f32(q & 0x7F).abs();
                let da = (av - x.abs()).abs();
                let dq = (qv - x.abs()).abs();
                assert_eq!(da, dq, "non-tie disagreement at x={x}: rne {a:#x} vs quant {q:#x}");
                assert_eq!(a & 1, 0, "tie must go to even code at x={x}");
                // -0.0 case: quant.rs flattens the sign
                if x == 0.0 {
                    assert_eq!(a, 0x80);
                }
                n_ties += 1;
            }
        }
    }
    assert!(n_ties > 0, "sweep must actually hit ties (got {n_ties})");
    let _ = table;
}

#[test]
fn rne_e2m1_encoder() {
    // all seven midpoints of the E2M1 grid, ties to even code
    let cases = [
        (0.25f32, 0u8), // 0 (even) vs 0.5
        (0.75, 2),      // 0.5 vs 1.0 (even)
        (1.25, 2),      // 1.0 (even) vs 1.5
        (1.75, 4),      // 1.5 vs 2.0 (even)
        (2.5, 4),       // 2.0 (even) vs 3.0
        (3.5, 6),       // 3.0 vs 4.0 (even)
        (5.0, 6),       // 4.0 (even) vs 6.0
    ];
    for (x, code) in cases {
        assert_eq!(f32_to_e2m1_rne(x), code, "tie at {x}");
        assert_eq!(f32_to_e2m1_rne(-x), 0x8 | code, "tie at {x} (neg)");
    }
    assert_eq!(f32_to_e2m1_rne(-0.0), 0x8);
    assert_eq!(f32_to_e2m1_rne(0.0), 0x0);
    assert_eq!(f32_to_e2m1_rne(6.0), 7);
    assert_eq!(f32_to_e2m1_rne(7.0), 7);
    // equivalence with quant::f32_to_e2m1 on non-zero values
    let mut rng = Rng(0x42);
    for _ in 0..10000 {
        let x = rng.next() * 6.0;
        assert_eq!(f32_to_e2m1_rne(x), quant::f32_to_e2m1(x), "mismatch at {x}");
    }
    // nibble packing: low nibble = even K index (§A.2/§C.2)
    assert_eq!(pack_e2m1_pair(0xA, 0x5), 0x5A);
    assert_eq!(pack_e2m1_pair(0x0, 0xF), 0xF0);
}

/// Independent slow RNE search over the positive E4M3 grid (linear scan,
/// ties-to-even) — the reference the fast encoders are checked against.
/// Takes a signed value, returns the signed code (sign bit 0x80).
fn slow_e4m3_rne_signed(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    sign | slow_e4m3_rne(x.abs())
}

fn slow_e4m3_rne(a: f32) -> u8 {
    let t: Vec<f32> = (0u8..127).map(quant::e4m3_to_f32).collect();
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (i, &v) in t.iter().enumerate() {
        let d = (v - a).abs();
        if d < bd {
            bd = d;
            best = i;
        } else if d == bd && i % 2 == 0 {
            best = i;
        }
    }
    best as u8
}

#[test]
fn act_quant_sim_bit_exact_crafted() {
    // floor: an all-zero group takes amax = 1e-4 → s = 2^ceil(log2(1e-4/448)) = 2^-22
    let mut x = vec![0.0f32; 64];
    x[3] = -0.0; // sign-of-zero must survive the round-trip
    act_quant_sim(&mut x, 1, 64, 64);
    assert_eq!(x[0], 0.0);
    assert!(x[3] == 0.0 && x[3].is_sign_negative(), "-0.0 must stay -0.0");

    // pow2 round-UP: amax slightly above 448·2^-22 → s = 2^-21 (not 2^-22)
    let amax = 448.0 * 2f32.powi(-22) * (1.0 + 2f32.powi(-23));
    let mut x = vec![0.0f32; 64];
    x[0] = amax;
    x[1] = -amax;
    act_quant_sim(&mut x, 1, 64, 64);
    let s = 2f32.powi(-21);
    let q = (amax / s).clamp(-448.0, 448.0);
    let expect = bf(quant::e4m3_to_f32(slow_e4m3_rne(q)) * s);
    assert_eq!(x[0], expect, "scale must round UP to the next pow2");
    assert_eq!(x[1], -expect);

    // table-driven check on a random group: independent per-element reference
    let mut rng = Rng(0x99);
    let mut x: Vec<f32> = rng.vec(128, 3.0).iter().map(|&v| bf(v)).collect();
    let orig = x.clone();
    act_quant_sim(&mut x, 2, 64, 64);
    for row in 0..2 {
        let blk = &orig[row * 64..(row + 1) * 64];
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-4);
        // replicate the bit trick on the same f32 product (independent of act_quant_sim internals)
        let prod = amax * (1.0f32 / 448.0f32);
        let bits = prod.to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32;
        let man = bits & 0x7FFFFF;
        let s = f32::from_bits((((exp - 127 + if man != 0 { 1 } else { 0 }) + 127) as u32) << 23);
        for j in 0..64 {
            let q = slow_e4m3_rne_signed((blk[j] / s).clamp(-448.0, 448.0));
            let expect = bf(quant::e4m3_to_f32(q) * s);
            assert_eq!(x[row * 64 + j], expect, "row {row} j {j}");
        }
    }
}

#[test]
fn fp4_act_quant_sim_bit_exact_crafted() {
    // floor: amax = 6·2^-126 exactly (the floor value) → s = 2^-125 (the
    // (1/6)_f32 rounding pushes the product just above 2^-126 and the scale
    // always rounds UP). x/s = 3.0 exactly → dequant 3·2^-125.
    let floor = 6.0f32 * 2f32.powi(-126);
    let mut x = vec![0.0f32; 32];
    x[5] = floor;
    x[6] = -floor;
    fp4_act_quant_sim(&mut x, 1, 32, 32);
    let s = 2f32.powi(-125);
    assert_eq!(x[5], bf(3.0 * s), "floored scale + exact quotient");
    assert_eq!(x[6], bf(-3.0 * s));
    // all-zero group: floored scale, every output is signed zero (no inf/NaN)
    let mut z = vec![0.0f32; 32];
    z[1] = -0.0;
    fp4_act_quant_sim(&mut z, 1, 32, 32);
    assert_eq!(z[0], 0.0);
    assert!(z[1].is_sign_negative());

    // table-driven random-group check (independent reference, same bit trick)
    let mut rng = Rng(0xABC);
    let mut x: Vec<f32> = rng.vec(64, 2.0).iter().map(|&v| bf(v)).collect();
    let orig = x.clone();
    fp4_act_quant_sim(&mut x, 2, 32, 32);
    let floor = 6.0f32 * 2f32.powi(-126);
    for row in 0..2 {
        let blk = &orig[row * 32..(row + 1) * 32];
        let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(floor);
        let prod = amax * (1.0f32 / 6.0f32);
        let bits = prod.to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32;
        let man = bits & 0x7FFFFF;
        let s = f32::from_bits((((exp - 127 + if man != 0 { 1 } else { 0 }) + 127) as u32) << 23);
        for j in 0..32 {
            let q = slow_e2m1_rne_signed((blk[j] / s).clamp(-6.0, 6.0));
            let expect = bf(quant::e2m1_to_f32(q) * s);
            assert_eq!(x[row * 32 + j], expect, "row {row} j {j}");
        }
    }
}

fn slow_e2m1_rne_signed(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x8u8 } else { 0 };
    sign | slow_e2m1_rne(x.abs())
}

fn slow_e2m1_rne(a: f32) -> u8 {
    const T: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (i, &v) in T.iter().enumerate() {
        let d = (v - a).abs();
        if d < bd {
            bd = d;
            best = i;
        } else if d == bd && i % 2 == 0 {
            best = i;
        }
    }
    best as u8
}

// ---------------------------------------------------------------------------
// 4. Deterministic top-k (§12.B.2): value desc, index asc on ties, -inf last
// ---------------------------------------------------------------------------

#[test]
fn topk_deterministic_tie_order() {
    let scores = [1.0f32, 2.0, 2.0, f32::NEG_INFINITY, 0.5];
    assert_eq!(topk_deterministic(&scores, 3), vec![1, 2, 0]);
    assert_eq!(topk_deterministic(&scores, 5), vec![1, 2, 0, 4, 3]);
    // k > n clamps to n
    assert_eq!(topk_deterministic(&scores, 99).len(), 5);
    // all -inf (the prefill fully-masked row): ascending indices
    let masked = [f32::NEG_INFINITY; 4];
    assert_eq!(topk_deterministic(&masked, 4), vec![0, 1, 2, 3]);
}

// ---------------------------------------------------------------------------
// 5. Index-list helpers vs the ORACLE's ground-truth rows (bit-exact)
// ---------------------------------------------------------------------------

#[test]
fn index_lists_match_oracle_rows() {
    // SWA prefill S=130 (oracle dsv4_swa.npz pre.topk_idx)
    let w = window_topk_idxs(128, 130, 0);
    assert_eq!(w.len(), 130);
    assert_eq!(w[0][..4].to_vec(), vec![0, -1, -1, -1]);
    assert!(w[0].iter().skip(1).all(|&v| v == -1));
    assert_eq!(w[127], (0..=127).collect::<Vec<i64>>());
    assert_eq!(w[128], (1..=128).collect::<Vec<i64>>());
    assert_eq!(w[129], (2..=129).collect::<Vec<i64>>());
    // SWA decode at start_pos=130: all 128 physical slots, oldest→newest
    let d = window_topk_idxs(128, 1, 130);
    let expect: Vec<i64> = (3..128).chain(0..=2).map(|v| v as i64).collect();
    assert_eq!(d[0], expect);
    // early decode (start_pos < 127): [0..sp] + -1 pad
    let e = window_topk_idxs(128, 1, 5);
    assert_eq!(e[0][..6].to_vec(), vec![0, 1, 2, 3, 4, 5]);
    assert!(e[0].iter().skip(6).all(|&v| v == -1));
    // HCA prefill S=130 (offset 130): row 0 all -1, rows 127..129 → [130]
    let h = compress_topk_idxs(128, 130, 0, 130);
    assert_eq!(h[0], vec![-1]);
    assert_eq!(h[127], vec![130]);
    assert_eq!(h[129], vec![130]);
    // HCA decode at 130: [128] (block 0 + decode offset)
    assert_eq!(compress_topk_idxs(128, 1, 130, 128)[0], vec![128]);
    // DSpark 133-entry non-causal block list at start_pos=130
    let ds = dspark_topk_idxs(128, 5, 130);
    assert_eq!(ds.len(), 133);
    assert_eq!(ds[..4].to_vec(), vec![0, 1, 2, 3]);
    assert_eq!(ds[128..].to_vec(), vec![128, 129, 130, 131, 132]);
}

// ---------------------------------------------------------------------------
// 6. sparse_attn (§B.7) vs a naive per-row loop
// ---------------------------------------------------------------------------

#[test]
fn sparse_attn_vs_naive_row_loop() {
    let (m, h, d, n, t) = (3usize, 4usize, 16usize, 20usize, 7usize);
    let mut rng = Rng(0x5150);
    let q: Vec<f32> = rng.vec(m * h * d, 1.0).iter().map(|&v| bf(v)).collect();
    let kv: Vec<f32> = rng.vec(n * d, 1.0).iter().map(|&v| bf(v)).collect();
    let sink = rng.vec(h, 1.0);
    let mut idxs = vec![0i64; m * t];
    for i in 0..m * t {
        let r = ((rng.next() + 1.0) * 0.5 * (n + 2) as f32) as i64 - 2; // some -1 entries
        idxs[i] = if r < 0 { -1 } else { r.min(n as i64 - 1) };
    }
    let scale = (d as f64).powf(-0.5) as f32;
    let got = sparse_attn(&q, m, h, d, &kv, n, &sink, &idxs, t, scale);
    // naive per-row reference (sequential dots — the same math, one row at a time)
    let mut want = vec![0.0f32; m * h * d];
    for mi in 0..m {
        for hh in 0..h {
            let valid: Vec<usize> = idxs[mi * t..(mi + 1) * t]
                .iter()
                .enumerate()
                .filter(|&(_, &ix)| ix >= 0)
                .map(|(tt, _)| tt)
                .collect();
            assert!(!valid.is_empty(), "test must keep at least one valid index per row");
            let qrow = &q[(mi * h + hh) * d..(mi * h + hh + 1) * d];
            let mut scores = vec![0.0f32; valid.len()];
            for (vi, &tt) in valid.iter().enumerate() {
                let kvrow = &kv[idxs[mi * t + tt] as usize * d..][..d];
                let mut acc = 0.0f32;
                for dd in 0..d {
                    acc += qrow[dd] * kvrow[dd];
                }
                scores[vi] = acc * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
            let denom: f32 = p.iter().sum::<f32>() + (sink[hh] - mx).exp();
            let orow = &mut want[(mi * h + hh) * d..(mi * h + hh + 1) * d];
            for (vi, &tt) in valid.iter().enumerate() {
                let pbf = bf(p[vi]); // bf16-rounded probability in the numerator only
                let kvrow = &kv[idxs[mi * t + tt] as usize * d..][..d];
                for dd in 0..d {
                    orow[dd] += pbf * kvrow[dd];
                }
            }
            for dd in 0..d {
                orow[dd] = bf(orow[dd] / denom);
            }
        }
    }
    // dot-order differences (dot8 vs sequential) are the only slack
    let err = max_abs_diff(&got, &want);
    assert!(err < 2e-5, "sparse_attn vs naive: max abs diff {err}");
}

// ---------------------------------------------------------------------------
// 7. Compressor state machine (§B.5): tiny ratio-4 overlap vs hand-computed
// ---------------------------------------------------------------------------

/// Tiny overlap compressor fixture: dim 4 → d 4, rd 2, ratio 4, coff 2 (cd 8).
/// wkv rows are a_r·[1,0,0,0] (a = 1..8) so kv row of token i = (i+1)·a.
/// wgate = 0 → scores come from ape alone. Compares state, cache and pooled
/// rows against an independent f32 replica of §B.5 written in the test.
struct TinyCompressor {
    w: CompressorWeights,
    a: [f32; 8],
    ape: Vec<f32>, // [4, 8]
}

impl TinyCompressor {
    fn new() -> Self {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut wkv = vec![0.0f32; 8 * 4];
        for r in 0..8 {
            wkv[r * 4] = a[r];
        }
        let wgate = vec![0.0f32; 8 * 4];
        // distinctive-but-tiny ape values (keeps softmax math well-conditioned)
        let ape: Vec<f32> = (0..32).map(|i| 0.01 * (i as f32 - 16.0)).collect();
        TinyCompressor {
            w: CompressorWeights {
                wkv,
                wgate,
                norm: vec![1.0; 4],
                ape: ape.clone(),
                ratio: 4,
                head_dim: 4,
                rope_dim: 2,
                overlap: true,
                rotate: false,
                sim_group: 2, // nope = d - rd = 2 → one group of 2
                dim: 4,
            },
            a,
            ape,
        }
    }

    fn kv_of(&self, token0: usize) -> [f32; 8] {
        let mut r = [0.0f32; 8];
        for c in 0..8 {
            r[c] = (token0 + 1) as f32 * self.a[c];
        }
        r
    }

    /// Independent replica of the overlap pool for one block (§B.5):
    /// rows 0..3 = PREV block dims :4 (+ape), rows 4..7 = current dims 4: (+ape);
    /// block 0 prev = zeros/−inf. Returns [4] pre-norm pooled row.
    fn pool_block(&self, b: usize, nblocks: usize) -> Vec<f32> {
        assert!(b < nblocks);
        let mut out = vec![0.0f32; 4];
        for dd in 0..4 {
            let mut kvs = [0.0f32; 8];
            let mut scs = [f32::NEG_INFINITY; 8];
            for j in 0..8 {
                if j < 4 {
                    if b > 0 {
                        let tok = (b - 1) * 4 + j;
                        kvs[j] = (tok + 1) as f32 * self.a[dd];
                        scs[j] = self.ape[j * 8 + dd];
                    }
                } else {
                    let tok = b * 4 + (j - 4);
                    kvs[j] = (tok + 1) as f32 * self.a[4 + dd];
                    scs[j] = self.ape[(j - 4) * 8 + 4 + dd];
                }
            }
            let mx = scs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f32 = scs.iter().map(|&s| (s - mx).exp()).sum();
            let mut acc = 0.0f32;
            for j in 0..8 {
                let p = (scs[j] - mx).exp() / z;
                acc += kvs[j] * p;
            }
            out[dd] = acc;
        }
        out
    }

    /// Post-pool finishing (bf16 → RMSNorm(ones) → RoPE → FP8-sim on :2, group 2),
    /// replicated independently.
    fn finish(&self, pooled: &[f32], pos: usize, rope: &RopeTable) -> Vec<f32> {
        let mut row: Vec<f32> = pooled.iter().map(|&v| bf(v)).collect();
        // rms_norm_row with weight = ones
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let inv = (ss / 4.0 + 1e-6).sqrt().recip();
        for v in row.iter_mut() {
            *v = bf(*v * inv);
        }
        apply_rope(&mut row, 1, 4, rope, &[pos], false);
        act_quant_sim(&mut row[..2], 1, 2, 2);
        row
    }
}

#[test]
fn compressor_tiny_overlap_state_machine() {
    let fix = TinyCompressor::new();
    let rope = rope_table(2, 32, 0, 10000.0, 16.0, 32, 1);
    let mut comp = Compressor::new(fix.w.clone());
    let mut cache = vec![0.0f32; 8 * 4]; // 8 rows of d=4
    // prefill S = 10: two full blocks (tokens 0..7) + remainder (tokens 8,9)
    let x: Vec<f32> = (0..10)
        .flat_map(|i| vec![(i + 1) as f32, 0.0, 0.0, 0.0])
        .collect();
    let pooled = comp.forward(&x, 10, 0, &rope, 1e-6, &mut cache).expect("prefill must compress");
    assert_eq!(pooled.len(), 2 * 4);
    // independent expected pooled rows
    let e0 = fix.pool_block(0, 2);
    let e1 = fix.pool_block(1, 2);
    assert_eq!(&pooled[..4], &fix.finish(&e0, 0, &rope)[..], "block 0 cache row");
    assert_eq!(&pooled[4..], &fix.finish(&e1, 4, &rope)[..], "block 1 cache row");
    assert_eq!(&cache[..4], &pooled[..4], "cache row 0");
    assert_eq!(&cache[4..8], &pooled[4..], "cache row 1");
    // stash: kv_state[:4] = last FULL block (tokens 4..7) raw rows; score += ape
    for j in 0..4 {
        let kv = fix.kv_of(4 + j);
        assert_eq!(&comp.st.kv_state[j * 8..(j + 1) * 8], &kv[..], "stashed kv row {j}");
        for c in 0..8 {
            assert_eq!(comp.st.score_state[j * 8 + c], fix.ape[j * 8 + c], "stashed score");
        }
    }
    // remainder: kv_state[4..6] = tokens 8,9; score += ape[:2]; score_state[6..] = -inf
    for j in 0..2 {
        let kv = fix.kv_of(8 + j);
        assert_eq!(&comp.st.kv_state[(4 + j) * 8..(4 + j + 1) * 8], &kv[..], "remainder kv row {j}");
        for c in 0..8 {
            assert_eq!(comp.st.score_state[(4 + j) * 8 + c], fix.ape[j * 8 + c], "remainder score");
        }
    }
    for c in 0..8 {
        assert_eq!(comp.st.score_state[6 * 8 + c], f32::NEG_INFINITY);
    }

    // decode at start_pos=10: (10+1)%4 != 0 → no compression, state slot 6 updated
    let x10 = vec![11.0f32, 0.0, 0.0, 0.0];
    assert!(comp.forward(&x10, 1, 10, &rope, 1e-6, &mut cache).is_none());
    let kv10 = fix.kv_of(10);
    assert_eq!(&comp.st.kv_state[6 * 8..7 * 8], &kv10[..]);
    for c in 0..8 {
        assert_eq!(comp.st.score_state[6 * 8 + c], fix.ape[2 * 8 + c], "ape row 10%4=2 added");
    }
    assert_eq!(&cache[8..12], &[0.0; 4], "no cache write on non-compressing decode");

    // decode at start_pos=11: (11+1)%4 == 0 → compress: rows 0..3 = stashed prev
    // block dims :4, rows 4..7 = current window (tokens 8..11) dims 4:; shift after.
    let x11 = vec![12.0f32, 0.0, 0.0, 0.0];
    let row = comp.forward(&x11, 1, 11, &rope, 1e-6, &mut cache).expect("decode must compress");
    // independent expected pooled row
    let mut epool = vec![0.0f32; 4];
    for dd in 0..4 {
        let mut kvs = [0.0f32; 8];
        let mut scs = [0.0f32; 8];
        for j in 0..4 {
            let tok = 4 + j; // stashed previous block, dims :4
            kvs[j] = (tok + 1) as f32 * fix.a[dd];
            scs[j] = fix.ape[j * 8 + dd];
        }
        for j in 4..8 {
            let tok = 4 + j; // current window tokens 8..11, dims 4:
            kvs[j] = (tok + 1) as f32 * fix.a[4 + dd];
            scs[j] = fix.ape[(j - 4) * 8 + 4 + dd];
        }
        let mx = scs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = scs.iter().map(|&s| (s - mx).exp()).sum();
        for j in 0..8 {
            let p = (scs[j] - mx).exp() / z;
            epool[dd] += kvs[j] * p;
        }
    }
    let efin = fix.finish(&epool, 11 + 1 - 4, &rope);
    assert_eq!(row, efin, "decode compressed row");
    assert_eq!(&cache[8..12], &row[..], "cache row 11//4 = 2");
    // shift: kv_state[:4] must now be tokens 8..11 raw rows
    for j in 0..4 {
        let kv = fix.kv_of(8 + j);
        assert_eq!(&comp.st.kv_state[j * 8..(j + 1) * 8], &kv[..], "shifted kv row {j}");
    }
}

// ---------------------------------------------------------------------------
// 8. Router (§B.9): hash-layer gather semantics + bias-for-selection-only
// ---------------------------------------------------------------------------

#[test]
fn router_hash_layer_gather_semantics() {
    // 8 experts, dim 4, topk 3; two tokens. gate_w = e_i rows → scores = x[i].
    let (ne, dim, topk) = (8usize, 4usize, 3usize);
    let mut gate_w = vec![0.0f32; ne * dim];
    for e in 0..4 {
        gate_w[e * dim + e] = 1.0;
    } // experts 4..7 get zero rows → scores 0
    let x = vec![1.0f32, -2.0, 0.5, 4.0, 2.0, 1.0, -1.0, 3.0]; // [2, 4]
    // tid2eid table: token 5 → [6, 1, 3]; token 9 → [2, 7, 0]
    let mut table = vec![0i32; 16 * topk];
    table[5 * topk..5 * topk + 3].copy_from_slice(&[6, 1, 3]);
    table[9 * topk..9 * topk + 3].copy_from_slice(&[2, 7, 0]);
    let ids = vec![5i64, 9];
    let (w, idx) = gate_forward(&x, 2, dim, &gate_w, None, Some((&table, &ids)), ne, topk, 1.5);
    assert_eq!(idx, vec![6, 1, 3, 2, 7, 0], "indices come from the TABLE, not scores");
    // weights = computed (un-biased, sqrt-softplus) scores at the table indices, renorm ×1.5
    let sp = |s: f32| if s > 20.0 { s } else { s.exp().ln_1p() };
    let score = |v: f32| sp(v).sqrt();
    for (tok, row) in [(0usize, [6usize, 1, 3]), (1usize, [2, 7, 0])] {
        let xrow = &x[tok * dim..(tok + 1) * dim];
        let orig: Vec<f32> = (0..ne)
            .map(|e| score(if e < dim { xrow[e] } else { 0.0 }))
            .collect();
        let sum: f32 = row.iter().map(|&e| orig[e]).sum();
        for (j, &e) in row.iter().enumerate() {
            let expect = orig[e] / sum * 1.5;
            assert!((w[tok * topk + j] - expect).abs() < 1e-7, "tok {tok} slot {j}");
        }
    }
    // softplus threshold-20 boundary
    assert_eq!(softplus_torch(21.0), 21.0);
    assert!((softplus_torch(19.0) - 19.0f32.exp().ln_1p()).abs() < 1e-7);
}

#[test]
fn router_bias_for_selection_only() {
    // experts scores: e0 = 1.0, e1 = 0.9, e2 = 0.8 (dim 1 proxy); bias pushes e2 on top
    let (ne, dim, topk) = (3usize, 1usize, 2usize);
    let gate_w = vec![1.0f32, 0.9, 0.8]; // [3, 1]
    let x = vec![1.0f32];
    let bias = vec![0.0f32, 0.0, 5.0]; // e2 wins selection with bias
    let (w, idx) = gate_forward(&x, 1, dim, &gate_w, Some(&bias), None, ne, topk, 1.5);
    assert_eq!(idx, vec![2, 0], "biased selection order");
    // weights from UN-biased original scores
    let sp = |s: f32| if s > 20.0 { s } else { s.exp().ln_1p() };
    let orig: Vec<f32> = [1.0f32, 0.9, 0.8].iter().map(|&v| sp(v).sqrt()).collect();
    let sum = orig[2] + orig[0];
    assert!((w[0] - orig[2] / sum * 1.5).abs() < 1e-7);
    assert!((w[1] - orig[0] / sum * 1.5).abs() < 1e-7);
}

// ---------------------------------------------------------------------------
// 9. quant GEMM (§C.3/§C.4) vs an independent per-block reference
// ---------------------------------------------------------------------------

#[test]
fn quant_gemm_blocked_structure() {
    let (t, k, n) = (2usize, 256usize, 3usize);
    let mut rng = Rng(0xFEED);
    let x: Vec<f32> = rng.vec(t * k, 1.5).iter().map(|&v| bf(v)).collect();
    // weights that are exactly e4m3·pow2 (like a dequantized fp8 checkpoint)
    let w: Vec<f32> = rng
        .vec(n * k, 0.02)
        .iter()
        .map(|&v| quant::e4m3_to_f32(quant::f32_to_e4m3(v)) * 2f32.powi(-3))
        .collect();
    for inner in [128usize, 32usize] {
        let got = quant_gemm(&x, t, k, &w, n, inner);
        // independent: quantize activation per 128-group (slow reference), then
        // per inner-block raw dot × sa accumulation
        for m in 0..t {
            let row = &x[m * k..(m + 1) * k];
            let mut codes = vec![0.0f32; k];
            let mut sa = vec![0.0f32; k / 128];
            for g in 0..k / 128 {
                let blk = &row[g * 128..(g + 1) * 128];
                let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-4);
                let prod = amax * (1.0f32 / 448.0);
                let bits = prod.to_bits();
                let exp = ((bits >> 23) & 0xFF) as i32;
                let s = f32::from_bits((((exp - 127 + if bits & 0x7FFFFF != 0 { 1 } else { 0 }) + 127) as u32) << 23);
                sa[g] = s;
                for j in 0..128 {
                    codes[g * 128 + j] = quant::e4m3_to_f32(slow_e4m3_rne_signed((blk[j] / s).clamp(-448.0, 448.0)));
                }
            }
            for n0 in 0..n {
                let wrow = &w[n0 * k..(n0 + 1) * k];
                let mut acc = 0.0f32;
                for kb in 0..k / inner {
                    let mut raw = 0.0f32;
                    for j in 0..inner {
                        raw += codes[kb * inner + j] * wrow[kb * inner + j];
                    }
                    acc += raw * sa[kb * inner / 128];
                }
                let expect = bf(acc);
                let diff = (got[m * n + n0] - expect).abs();
                let tol = 2e-5 * expect.abs().max(1.0); // dot-order slack only
                assert!(diff <= tol, "inner {inner} m {m} n {n0}: {} vs {expect}", got[m * n + n0]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 10. expert SwiGLU clamps (§B.9) — asymmetric up ±10 / gate ≤10
// ---------------------------------------------------------------------------

#[test]
fn expert_asymmetric_clamps() {
    // inter 128, dim 128 (activation quant groups are 128-wide): w1/w3 rows =
    // scaled identity so gate/up values are known.
    // Expected values replicate the documented formula (§B.9) with the same
    // blocked quant structure (act_quant_codes is bit-tested separately above).
    let (dim, inter) = (128usize, 128usize);
    let mut w1 = vec![0.0f32; inter * dim];
    let mut w3 = vec![0.0f32; inter * dim];
    let mut w2 = vec![0.0f32; dim * inter];
    for i in 0..inter {
        w1[i * dim + i] = 30.0; // gate = 30·x[i]
        w3[i * dim + i] = -40.0; // up = -40·x[i]  (hits the -10 lower clamp when x[i] ≥ 0.25)
        w2[i * dim + i] = 1.0;
    }
    // reference pipeline replica: quant-GEMM -> clamp -> silu·up -> weight -> bf16 -> quant-GEMM
    let replica = |x: &[f32], wgt: Option<f32>| -> Vec<f32> {
        let q1 = quant_gemm(x, 1, dim, &w1, inter, 32);
        let q3 = quant_gemm(x, 1, dim, &w3, inter, 32);
        let mut h = vec![0.0f32; inter];
        for i in 0..inter {
            let g = q1[i].min(10.0); // gate: clamp max only
            let u = q3[i].clamp(-10.0, 10.0); // up: clamp ±10
            let silu = g * (1.0f32 / (1.0f32 + (-g).exp()));
            let mut v = silu * u;
            if let Some(w) = wgt {
                v *= w;
            }
            h[i] = bf(v);
        }
        quant_gemm(&h, 1, inter, &w2, dim, 32)
    };
    let x: Vec<f32> = (0..dim).map(|i| 0.1 * (i + 1) as f32).collect();
    let got = expert_forward_token(&x, &w1, &w2, &w3, dim, inter, 32, 10.0, Some(0.5));
    let want = replica(&x, Some(0.5));
    assert!(max_abs_diff(&got, &want) < 1e-6, "weighted expert forward");
    // gate values that engage the clamps asymmetrically: gate = -60 (unbounded
    // below), up = +80 → clamped to +10
    let xn: Vec<f32> = vec![-2.0f32; dim];
    let got = expert_forward_token(&xn, &w1, &w2, &w3, dim, inter, 32, 10.0, None);
    let want = replica(&xn, None);
    assert!(max_abs_diff(&got, &want) < 1e-6, "gate must NOT be clamped below");
    // sanity that the clamps actually engaged in the replica path: q1 = -60
    let q1 = quant_gemm(&xn, 1, dim, &w1, inter, 32);
    let q3 = quant_gemm(&xn, 1, dim, &w3, inter, 32);
    assert!(q1[0] < -50.0 && q1[0].min(10.0) == q1[0], "gate below -10 must survive");
    assert_eq!(q3[0].clamp(-10.0, 10.0), 10.0, "up above +10 must clamp");
}

// ---------------------------------------------------------------------------
// 11. argmax sampler + bf16 RNE sanity (half crate contract)
// ---------------------------------------------------------------------------

#[test]
fn argmax_and_bf16_rounding() {
    assert_eq!(argmax_first(&[1.0, 3.0, 3.0, 2.0]), 1); // first max on ties
    assert_eq!(argmax_first(&[-5.0, -1.0]), 1);
    // bf16 RNE ties-to-even (the rounding the whole model relies on)
    assert_eq!(bf16::from_f32(1.0 + 2f32.powi(-9)).to_bits(), 0x3F80); // → 1.0 (even)
    assert_eq!(bf16::from_f32(1.0 + 3.0 * 2f32.powi(-9)).to_bits(), 0x3F81);
    assert_eq!(bf16::from_f32(-0.0f32).to_bits(), 0x8000);
}
