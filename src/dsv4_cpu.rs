//! DSV4 CPU reference model (Lane B owns this file).
//!
//! Plain-Rust reimplementation of the bundle reference forward
//! (`/mnt/models/DeepSeek-V4-Flash-DSpark/inference/model.py`) following
//! DEEPSEEK_V4_PORT.md §B exactly, feeding on `crate::dsv4_load` (Lane A's loader).
//! Replays the oracle npz inputs and emits per-piece outputs for
//! `scripts/dsv4_diff.py` (Lane C).
//!
//! # Numerics policy (what is bit-faithful vs tolerance-level)
//!
//! Bit-faithful by construction (integer/bit-trick math, §C.1–2): the QAT-sim
//! round-trips (`act_quant_sim` / `fp4_act_quant_sim`), the UE8M0 pow2 scale
//! computation, the E4M3/E2M1 RNE casts (incl. sign-of-zero and ties-to-even),
//! the fp8/fp4 code extraction for GEMM activations, all index-list helpers
//! (pure functions), the deterministic top-k (§12.B.2), and the Sinkhorn
//! iteration *order* (§B.8).
//!
//! Tolerance-level (vs the torch/tilelang oracle): anything involving GEMM or
//! torch reduction accumulation *order* (cuBLAS/tilelang reduction trees vs our
//! sequential f32 dots) and transcendental ulps (exp/sigmoid/pow/cos/sin between
//! CUDA libm and the host libm). The Hadamard rotation is the oracle's fp32
//! matmul emulation (single rounding) — same structure, order-tolerance only.
//!
//! Activations are stored as `Vec<f32>` holding **bf16-rounded values** at every
//! point the reference materializes a bf16 tensor (`bf16::from_f32` RNE
//! round-trip, `bf()`). This is value-identical to the reference's bf16 storage
//! and keeps the code in one dtype; the two places the reference does *not*
//! float first (the weight-free per-head q rescale) are reproduced with
//! per-op bf16 roundings exactly.

use anyhow::{anyhow, bail, Result};
use half::bf16;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::dsv4_load::{Dsv4Config, Dsv4Layer, HostTensor, LayerKind};
use crate::quant::{e2m1_to_f32, e4m3_to_f32, Nvfp4Tensor};

// ---------------------------------------------------------------------------
// bf16 rounding helpers
// ---------------------------------------------------------------------------

/// Round f32 -> bf16 (RNE) -> f32. Every point the reference materializes a
/// bf16 tensor goes through this.
#[inline]
pub fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

#[inline]
pub fn round_bf16(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = bf(*x);
    }
}

// ---------------------------------------------------------------------------
// RNE cast encoders (§C.1–2: `cvt.rn` semantics — ties-to-even, sign-of-zero)
// ---------------------------------------------------------------------------

/// f32 -> FP8-E4M3 byte, round-to-nearest-even on the target grid, satfinite.
/// Input is expected pre-clamped to ±448 by the caller (the quant kernels clamp
/// before casting); we still saturate for safety. `quant::f32_to_e4m3` agrees
/// on every value except exact ties (it rounds ties down) and -0.0 (it returns
/// +0); this version is the exact `cvt.rn.satfinite.e4m3x2.f32` behavior the
/// tilelang kernels get from hardware. Proven equivalent in tests.
pub fn f32_to_e4m3_rne(x: f32) -> u8 {
    const MAX: f32 = 448.0;
    if x.is_nan() {
        return 0x7F; // e4m3 NaN
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0u8 };
    let a = x.abs().min(MAX);
    if a == 0.0 {
        return sign; // preserve -0.0 like cvt.rn
    }
    // The 127 finite non-negative codes are monotonic; locate the bracketing pair.
    let t = e4m3_pos_table();
    let hi = t.partition_point(|&v| v < a);
    let code = if hi == 0 {
        0usize
    } else if hi >= 127 {
        126usize
    } else {
        let d_hi = t[hi] - a;
        let d_lo = a - t[hi - 1];
        if d_hi < d_lo {
            hi
        } else if d_hi > d_lo {
            hi - 1
        } else if hi % 2 == 0 {
            hi // exact tie -> even code (RNE)
        } else {
            hi - 1
        }
    };
    sign | code as u8
}

fn e4m3_pos_table() -> &'static [f32; 127] {
    static T: OnceLock<[f32; 127]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0.0f32; 127];
        for (c, slot) in t.iter_mut().enumerate() {
            *slot = e4m3_to_f32(c as u8);
        }
        t
    })
}

/// f32 -> FP4-E2M1 nibble (0..=15), round-to-nearest-even, sign-of-zero kept.
/// Caller clamps to ±6 (we saturate regardless). quant::f32_to_e2m1 is also
/// ties-to-even but flattens -0.0; proven equivalent on non-zero values in tests.
pub fn f32_to_e2m1_rne(x: f32) -> u8 {
    const T: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    if x.is_nan() {
        return 0x7; // e2m1fn NaN (S.111)
    }
    let sign = if x.is_sign_negative() { 0x8u8 } else { 0u8 };
    let a = x.abs().min(6.0);
    if a == 0.0 {
        return sign;
    }
    let hi = T.partition_point(|&v| v < a);
    let code = if hi == 0 {
        0usize
    } else if hi >= 8 {
        7usize
    } else {
        let d_hi = T[hi] - a;
        let d_lo = a - T[hi - 1];
        if d_hi < d_lo {
            hi
        } else if d_hi > d_lo {
            hi - 1
        } else if hi % 2 == 0 {
            hi
        } else {
            hi - 1
        }
    };
    sign | code as u8
}

/// Pack two E2M1 nibbles into one byte: low nibble = even K index (§A.2, §C.2).
/// The QAT-sim path never packs (it round-trips values), but the convention is
/// load-bearing for Lane A's repack and pinned down by a unit test here.
#[inline]
pub fn pack_e2m1_pair(even_k: u8, odd_k: u8) -> u8 {
    (even_k & 0xF) | ((odd_k & 0xF) << 4)
}

// ---------------------------------------------------------------------------
// UE8M0 scale bit tricks (kernel.py:22-37, replicated operation-for-operation)
// ---------------------------------------------------------------------------

/// `fast_log2_ceil`: ceil(log2(x)) for x > 0 via IEEE 754 bits — including the
/// kernel's exact subnormal behavior (exp field 0 -> -127, +1 if mantissa != 0).
#[inline]
pub fn fast_log2_ceil(x: f32) -> i32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    exp - 127 + if man != 0 { 1 } else { 0 }
}

/// `fast_pow2`: 2^x for integer x via IEEE bits (valid for x in [-126, 127];
/// the quant callers' amax floors keep the argument in range).
#[inline]
pub fn fast_pow2(x: i32) -> f32 {
    f32::from_bits(((x + 127) as u32) << 23)
}

/// `fast_round_scale(amax, 1/max)`: 2^ceil(log2(amax/max)) — always rounds UP.
#[inline]
pub fn fast_round_scale(amax: f32, max_inv: f32) -> f32 {
    fast_pow2(fast_log2_ceil(amax * max_inv))
}

// ---------------------------------------------------------------------------
// QAT-sim round-trips (§C.1 act_quant inplace, §C.2 fp4_act_quant inplace)
// ---------------------------------------------------------------------------

/// FP8-E4M3 QAT-sim, in place, over bf16-valued f32 data: per row per group,
/// `amax = max|x|` floored at 1e-4; `s = 2^ceil(log2(amax/448))` (UE8M0, always
/// up); `x <- bf16(f32(fp8_RNE(clamp(x/s, ±448))) * s)`. The product fp8·s is
/// exact in f32 (≤4 significand bits × pow2), so the bf16 store is exact — the
/// round-trip value is exactly `code·s`.
pub fn act_quant_sim(x: &mut [f32], rows: usize, n: usize, group: usize) {
    assert_eq!(n % group, 0, "act_quant_sim: n={} not divisible by group={}", n, group);
    const FP8_MAX_INV: f32 = 1.0 / 448.0;
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        for g in 0..n / group {
            let blk = &mut row[g * group..(g + 1) * group];
            let mut amax = 0.0f32;
            for &v in blk.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(1e-4);
            let s = fast_round_scale(amax, FP8_MAX_INV);
            for v in blk.iter_mut() {
                let q = f32_to_e4m3_rne((*v / s).clamp(-448.0, 448.0));
                *v = bf(e4m3_to_f32(q) * s);
            }
        }
    }
}

/// FP4-E2M1 QAT-sim, in place: group 32; `amax` floored at `6·2^-126`;
/// `s = 2^ceil(log2(amax/6))`; `x <- bf16(f32(fp4_RNE(clamp(x/s, ±6))) * s)`.
pub fn fp4_act_quant_sim(x: &mut [f32], rows: usize, n: usize, group: usize) {
    assert_eq!(n % group, 0, "fp4_act_quant_sim: n={} not divisible by group={}", n, group);
    const FP4_MAX_INV: f32 = 1.0 / 6.0;
    let floor = 6.0f32 * 2f32.powi(-126);
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        for g in 0..n / group {
            let blk = &mut row[g * group..(g + 1) * group];
            let mut amax = 0.0f32;
            for &v in blk.iter() {
                amax = amax.max(v.abs());
            }
            amax = amax.max(floor);
            let s = fast_round_scale(amax, FP4_MAX_INV);
            for v in blk.iter_mut() {
                let q = f32_to_e2m1_rne((*v / s).clamp(-6.0, 6.0));
                *v = bf(e2m1_to_f32(q) * s);
            }
        }
    }
}

/// Non-inplace FP8 activation quant for the quant GEMMs (§C.1 non-inplace path,
/// group always 128): returns (code values as f32, ue8m0 scales as f32), one
/// scale per 128-wide group per row.
pub fn act_quant_codes(x: &[f32], rows: usize, n: usize, group: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(n % group, 0);
    const FP8_MAX_INV: f32 = 1.0 / 448.0;
    let ng = n / group;
    let mut codes = vec![0.0f32; rows * n];
    let mut scales = vec![0.0f32; rows * ng];
    for r in 0..rows {
        for g in 0..ng {
            let base = r * n + g * group;
            let mut amax = 0.0f32;
            for j in 0..group {
                amax = amax.max(x[base + j].abs());
            }
            amax = amax.max(1e-4);
            let s = fast_round_scale(amax, FP8_MAX_INV);
            scales[r * ng + g] = s;
            for j in 0..group {
                let q = f32_to_e4m3_rne((x[base + j] / s).clamp(-448.0, 448.0));
                codes[base + j] = e4m3_to_f32(q);
            }
        }
    }
    (codes, scales)
}

// ---------------------------------------------------------------------------
// Parallel row helper (deterministic: rows are independent)
// ---------------------------------------------------------------------------

fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Apply `f` to each disjoint `chunk`-sized row of `out` in parallel.
/// `f(row_index, &mut [T])` receives row `i`'s mutable slice. Static assignment
/// — deterministic (rows are independent).
pub fn par_chunks<T: Send>(out: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let n = out.len() / chunk;
    let nt = n_threads().min(n.max(1));
    if nt <= 1 {
        for (i, c) in out.chunks_mut(chunk).enumerate() {
            f(i, c);
        }
        return;
    }
    std::thread::scope(|sc| {
        let per = n.div_ceil(nt);
        let mut rest: &mut [T] = out;
        let mut start = 0usize;
        for _ in 0..nt {
            let take_chunks = per.min(rest.len() / chunk);
            if take_chunks == 0 {
                break;
            }
            let (head, tail) = rest.split_at_mut(take_chunks * chunk);
            rest = tail;
            let f = &f;
            let s0 = start;
            sc.spawn(move || {
                for (j, c) in head.chunks_mut(chunk).enumerate() {
                    f(s0 + j, c);
                }
            });
            start += take_chunks;
        }
    });
}

/// Apply `f` to each disjoint row-chunk of `out` in parallel.
/// `f(row_index, &mut T)`; T: Send. Chunk assignment is static — deterministic.
pub fn par_rows<T: Send>(out: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
    let n = out.len();
    let nt = n_threads().min(n.max(1));
    if nt <= 1 {
        for (i, slot) in out.iter_mut().enumerate() {
            f(i, slot);
        }
        return;
    }
    let chunk = n.div_ceil(nt);
    std::thread::scope(|sc| {
        let mut rest: &mut [T] = out;
        for t in 0..nt {
            let take = chunk.min(rest.len());
            let (head, tail) = rest.split_at_mut(take);
            rest = tail;
            if head.is_empty() {
                break;
            }
            let f = &f;
            let start = t * chunk;
            sc.spawn(move || {
                for (i, slot) in head.iter_mut().enumerate() {
                    f(start + i, slot);
                }
            });
        }
    });
}

/// 8-accumulator f32 dot product (lets LLVM vectorize; summation order is our
/// own — GEMM-order is tolerance-level vs the oracle by design, see module docs).
/// Used where the products are EXACT in f32 (quant-GEMM code·pow2-dequant and
/// sparse-attn bf16·bf16 dots) — there mul+add equals FMA and this order was
/// measured bit-identical to torch's block matmul on the real shapes.
#[inline]
pub fn dot8(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let i = c * 8;
        for l in 0..8 {
            acc[l] += a[i + l] * b[i + l];
        }
    }
    for l in chunks * 8..n {
        acc[l % 8] += a[l] * b[l];
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// Pairwise-tree reduction (in place). Measured against torch's vectorized
/// reduce on the real layer-0 tensors: ~1 ulp off torch's mean-of-squares,
/// where an 8-accumulator chain is ~18 ulps off. Used for the long-K f32
/// reductions (RMSNorm/hc_pre mean-of-squares, f32 GEMV dots) — the paths whose
/// noise snaps act_quant codes at midpoints (the G1 kv_cache diagnostic).
pub fn pairwise_sum(v: &mut [f32]) -> f32 {
    let mut n = v.len();
    if n == 0 {
        return 0.0;
    }
    while n > 1 {
        let mut w = 0;
        let mut r = 0;
        while r + 1 < n {
            v[w] = v[r] + v[r + 1];
            w += 1;
            r += 2;
        }
        if r < n {
            v[w] = v[r];
            w += 1;
        }
        n = w;
    }
    v[0]
}

/// Per-element square (f32, rounded like torch's materialized `.square()`)
/// then pairwise-tree sum.
pub fn sumsq_tree(x: &[f32]) -> f32 {
    let mut buf: Vec<f32> = x.iter().map(|v| v * v).collect();
    pairwise_sum(&mut buf)
}

/// f32 dot for long-K f32 GEMV/GEMM paths (hc_fn, router, compressor, LM head):
/// per-element product (f32-rounded) then pairwise-tree sum.
pub fn dot_tree(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut buf: Vec<f32> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    pairwise_sum(&mut buf)
}

// ---------------------------------------------------------------------------
// GEMMs
// ---------------------------------------------------------------------------

/// §C.3/§C.4 blocked quant GEMM, output bf16-valued f32 [t, n].
///
/// `x` [t,k] bf16-valued f32 activations; `w` [n,k] f32 **exactly dequantized**
/// weights (fp8: e4m3·ue8m0; fp4: e2m1·e8m0 — both exact in f32, so the weight
/// block scale sb is a power of two already folded into `w`). The activation is
/// quantized per 128-group to FP8 codes + ue8m0 scales (§C.1 non-inplace).
/// Structure reproduces the dual-accumulator kernels: per inner K-block
/// (`inner_block` = 128 for fp8 weights / 32 for fp4), a raw f32 dot of
/// code·w_deq, then `acc += raw · sa[m, kb·B/128]`.
///
/// Exactness note: code·w_deq products scale the reference's code·code products
/// by the pow2 sb, and pow2 scaling commutes exactly with f32 addition, so each
/// block dot equals sb·(reference raw block dot) up to summation order (which
/// is tolerance-level anyway). `acc` and the final bf16 cast mirror the kernels.
pub fn quant_gemm(x: &[f32], t: usize, k: usize, w: &[f32], n: usize, inner_block: usize) -> Vec<f32> {
    assert!(k % 128 == 0 && (inner_block == 128 || inner_block == 32));
    assert_eq!(x.len(), t * k);
    assert_eq!(w.len(), n * k);
    let nkb = k / inner_block;
    let mut out = vec![0.0f32; t * n];
    par_chunks(&mut out, n, |m, row_out| {
        let (codes, sa) = act_quant_codes(&x[m * k..(m + 1) * k], 1, k, 128);
        for n0 in 0..n {
            let wrow = &w[n0 * k..(n0 + 1) * k];
            let mut acc = 0.0f32;
            for kb in 0..nkb {
                let base = kb * inner_block;
                let raw = dot8(&codes[base..base + inner_block], &wrow[base..base + inner_block]);
                acc += raw * sa[kb * inner_block / 128]; // act scale: per-128 on K (§C.4)
            }
            row_out[n0] = acc;
        }
    });
    round_bf16(&mut out);
    out
}

/// Plain f32 GEMM, out [t,n] f32 (router / mHC GEMV / compressor / LM head).
pub fn gemm_f32(x: &[f32], t: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), t * k);
    assert_eq!(w.len(), n * k);
    let mut out = vec![0.0f32; t * n];
    par_chunks(&mut out, n, |i, row_out| {
        for n0 in 0..n {
            row_out[n0] = dot_tree(&x[i * k..(i + 1) * k], &w[n0 * k..(n0 + 1) * k]);
        }
    });
    out
}

/// bf16-in/out GEMM with f32 accumulate (weights_proj, wo_a einsum, markov_w2
/// is f32 — use gemm_f32 there). x and w hold bf16-valued f32; out bf16-rounded.
pub fn gemm_bf16(x: &[f32], t: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut out = gemm_f32(x, t, k, w, n);
    round_bf16(&mut out);
    out
}

// ---------------------------------------------------------------------------
// RoPE (model.py:205-250; §B.1.3)
// ---------------------------------------------------------------------------

/// Complex rotation table, fp32, [positions, rd/2] — adjacent-pair convention.
pub struct RopeTable {
    pub cos: Vec<f32>, // [positions, rd/2]
    pub sin: Vec<f32>,
    pub rd: usize,
}

/// `precompute_freqs_cis` (model.py:205-235) exactly: plain θ table, or the
/// YaRN correction when `original_seq_len > 0` (compress layers only; SWA layers
/// force-disable YaRN, §B.1.3). Correction-range math is python f64; the ramp
/// and the frequency blend are torch f32 op order.
pub fn rope_table(
    rd: usize,
    positions: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: u32,
    beta_slow: u32,
) -> RopeTable {
    let half = rd / 2;
    let dim = rd as f64;
    let mut freqs = vec![0.0f32; half];
    for i in 0..half {
        let e = (2 * i) as f32 / rd as f32;
        freqs[i] = 1.0f32 / base.powf(e);
    }
    if original_seq_len > 0 {
        // find_correction_dim in python f64 (math.log), then floor/ceil.
        let find = |num_rot: f64| -> f64 {
            dim * ((original_seq_len as f64) / (num_rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = find(beta_fast as f64).floor().max(0.0) as i32;
        let high = find(beta_slow as f64).ceil().min(dim - 1.0) as i32;
        // linear_ramp_factor(low, high, half): clamp((i-low)/(high-low), 0, 1), f32
        let (minv, maxv) = if low == high { (low as f32, low as f32 + 0.001) } else { (low as f32, high as f32) };
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - minv) / (maxv - minv)).clamp(0.0, 1.0);
            let smooth = 1.0f32 - ramp;
            *f = (*f / factor) * (1.0f32 - smooth) + (*f) * smooth;
        }
    }
    let mut cos = vec![0.0f32; positions * half];
    let mut sin = vec![0.0f32; positions * half];
    for t in 0..positions {
        for (i, &f) in freqs.iter().enumerate() {
            let ang = (t as f32) * f; // torch.polar(1, ang)
            cos[t * half + i] = ang.cos();
            sin[t * half + i] = ang.sin();
        }
    }
    RopeTable { cos, sin, rd }
}

/// `apply_rotary_emb` on the last `rd` dims of each row, in place (bf16 round
/// on write-back). `row_pos[i]` is the absolute position of row i. Complex mul
/// in f32: (re+im·i)(c+s·i) = (re·c − im·s) + (re·s + im·c)i.
pub fn apply_rope(x: &mut [f32], rows: usize, dim: usize, table: &RopeTable, row_pos: &[usize], inverse: bool) {
    let rd = table.rd;
    let half = rd / 2;
    assert_eq!(x.len(), rows * dim);
    for (i, &p) in row_pos.iter().enumerate().take(rows) {
        let off = dim - rd;
        let row = &mut x[i * dim..(i + 1) * dim];
        for j in 0..half {
            let re = row[off + 2 * j];
            let im = row[off + 2 * j + 1];
            let c = table.cos[p * half + j];
            let s = if inverse { -table.sin[p * half + j] } else { table.sin[p * half + j] };
            row[off + 2 * j] = bf(re * c - im * s);
            row[off + 2 * j + 1] = bf(re * s + im * c);
        }
    }
}

// ---------------------------------------------------------------------------
// Hadamard rotation (oracle emu: exact fp32 Walsh-Hadamard matmul, scale d^-0.5)
// ---------------------------------------------------------------------------

/// Sylvester ±1 Hadamard matrix × d^-0.5, f32, row-major [d,d]. d must be a
/// power of two. Mirrors dsv4_ref.py's `hadamard_matrix(d) * (d ** -0.5)`.
pub fn hadamard_scaled(d: usize) -> Vec<f32> {
    assert!(d.is_power_of_two());
    let scale = (d as f64).powf(-0.5) as f32;
    let mut h = vec![1.0f32];
    let mut n = 1;
    while n < d {
        let mut nh = vec![0.0f32; 4 * n * n];
        for r in 0..n {
            for c in 0..n {
                let v = h[r * n + c];
                nh[r * 2 * n + c] = v;
                nh[r * 2 * n + c + n] = v;
                nh[(r + n) * 2 * n + c] = v;
                nh[(r + n) * 2 * n + c + n] = -v;
            }
        }
        h = nh;
        n *= 2;
    }
    for v in h.iter_mut() {
        *v *= scale;
    }
    h
}

/// `rotate_activation`: y = bf16(x_f32 @ H·d^-0.5) on the last d dims of each row.
pub fn rotate_activation(x: &mut [f32], rows: usize, dim: usize, d: usize, h_scaled: &[f32]) {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(h_scaled.len(), d * d);
    let off = dim - d;
    for r in 0..rows {
        let row = &mut x[r * dim..(r + 1) * dim];
        let src: Vec<f32> = row[off..].to_vec();
        for c in 0..d {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += src[j] * h_scaled[j * d + c];
            }
            row[off + c] = bf(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// sparse_attn (§B.7 — the oracle's pure-torch emulation of kernel.py:276-368)
// ---------------------------------------------------------------------------

/// fp32 scores with a single global max per (row, head); probabilities rounded
/// to bf16 for the P·V numerator only; fp32 denominator; sink added to the
/// DENOMINATOR only; −1 index = masked (zero KV row, −inf score).
/// q [m,h,d] / kv [n,d] bf16-valued f32; idxs [m,t] (i64, −1 masked).
/// Returns o [m,h,d] bf16-rounded.
pub fn sparse_attn(
    q: &[f32],
    m: usize,
    h: usize,
    d: usize,
    kv: &[f32],
    n: usize,
    sink: &[f32],
    idxs: &[i64],
    t: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), m * h * d);
    assert_eq!(kv.len(), n * d);
    assert_eq!(sink.len(), h);
    assert_eq!(idxs.len(), m * t);
    let mut out = vec![0.0f32; m * h * d];
    par_chunks(&mut out, h * d, |mi, orow| {
        let ids = &idxs[mi * t..(mi + 1) * t];
        for hh in 0..h {
            let qrow = &q[(mi * h + hh) * d..(mi * h + hh + 1) * d];
            // scores (masked = -inf), global row max
            let mut scores = vec![f32::NEG_INFINITY; t];
            let mut row_max = f32::NEG_INFINITY;
            for (tt, &ix) in ids.iter().enumerate() {
                if ix < 0 {
                    continue;
                }
                let kvrow = &kv[ix as usize * d..(ix as usize + 1) * d];
                let s = dot8(qrow, kvrow) * scale;
                scores[tt] = s;
                if s > row_max {
                    row_max = s;
                }
            }
            // p = exp(s - max); denominator fp32 incl. denominator-only sink
            let mut denom = 0.0f32;
            let mut p = vec![0.0f32; t];
            for tt in 0..t {
                if scores[tt].is_finite() {
                    let e = (scores[tt] - row_max).exp();
                    p[tt] = e;
                    denom += e;
                }
            }
            denom += (sink[hh] - row_max).exp();
            // numerator uses bf16-rounded probabilities (acc_s_cast), fp32 accum
            let orow = &mut orow[hh * d..(hh + 1) * d];
            for (tt, &ix) in ids.iter().enumerate() {
                if ix < 0 {
                    continue;
                }
                let pbf = bf(p[tt]);
                let kvrow = &kv[ix as usize * d..(ix as usize + 1) * d];
                for dd in 0..d {
                    orow[dd] += pbf * kvrow[dd];
                }
            }
            for dd in 0..d {
                orow[dd] = bf(orow[dd] / denom);
            }
        }
    });
    out
}

// ---------------------------------------------------------------------------
// RMSNorm (model.py:189-202): fp32 math, bf16 result, eps = norm_eps (1e-6)
// ---------------------------------------------------------------------------

pub fn rms_norm(x: &[f32], rows: usize, dim: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(weight.len(), dim);
    let mut out = vec![0.0f32; rows * dim];
    par_chunks(&mut out, dim, |r, orow| {
        let row = &x[r * dim..(r + 1) * dim];
        let ss = sumsq_tree(row);
        let var = ss / dim as f32;
        let inv = (var + eps).sqrt().recip();
        for d in 0..dim {
            orow[d] = bf(weight[d] * (row[d] * inv));
        }
    });
    out
}

/// In-place RMSNorm variant over one row slice (used where rows share scratch).
pub fn rms_norm_row(x: &mut [f32], weight: &[f32], eps: f32) {
    let ss = sumsq_tree(x);
    let inv = (ss / x.len() as f32 + eps).sqrt().recip();
    for d in 0..x.len() {
        x[d] = bf(weight[d] * (x[d] * inv));
    }
}

// ---------------------------------------------------------------------------
// mHC (§B.8; kernel.py:371-438 exact iteration order)
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

/// `hc_split_sinkhorn` for one token's 24 mixes (hc=4). Returns (pre[4], post[4], comb[16]).
/// Exact sequence, all fp32, eps = hc_eps (1e-6) in every denominator:
/// pre = sigmoid+eps; post = 2·sigmoid; comb: row-softmax+eps → col-norm →
/// 19×(row-norm, col-norm).
pub fn hc_split_sinkhorn(mixes: &[f32], scale: &[f32; 3], base: &[f32], hc: usize, iters: usize, eps: f32) -> ([f32; 4], [f32; 4], [f32; 16]) {
    assert_eq!(mixes.len(), (2 + hc) * hc);
    assert_eq!(base.len(), (2 + hc) * hc);
    let mut pre = [0.0f32; 4];
    let mut post = [0.0f32; 4];
    let mut comb = [0.0f32; 16];
    for j in 0..hc {
        pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
    }
    for j in 0..hc {
        post[j] = 2.0 * sigmoid(mixes[hc + j] * scale[1] + base[hc + j]);
    }
    for j in 0..hc {
        for k in 0..hc {
            comb[j * hc + k] = mixes[hc * 2 + j * hc + k] * scale[2] + base[hc * 2 + j * hc + k];
        }
    }
    // comb = softmax_rows(comb) + eps  (max-subtracted exp / row sum, then + eps)
    for j in 0..hc {
        let mut mx = f32::NEG_INFINITY;
        for k in 0..hc {
            mx = mx.max(comb[j * hc + k]);
        }
        let mut rs = 0.0f32;
        for k in 0..hc {
            let e = (comb[j * hc + k] - mx).exp();
            comb[j * hc + k] = e;
            rs += e;
        }
        for k in 0..hc {
            comb[j * hc + k] = comb[j * hc + k] / rs + eps;
        }
    }
    // comb /= (col_sum + eps)
    let mut col_norm = |comb: &mut [f32; 16]| {
        for k in 0..hc {
            let mut cs = 0.0f32;
            for j in 0..hc {
                cs += comb[j * hc + k];
            }
            for j in 0..hc {
                comb[j * hc + k] /= cs + eps;
            }
        }
    };
    col_norm(&mut comb);
    for _ in 0..iters - 1 {
        // row-norm
        for j in 0..hc {
            let mut rs = 0.0f32;
            for k in 0..hc {
                rs += comb[j * hc + k];
            }
            for k in 0..hc {
                comb[j * hc + k] /= rs + eps;
            }
        }
        col_norm(&mut comb);
    }
    (pre, post, comb)
}

/// mHC mixing parameters for one sublayer (all fp32).
#[derive(Debug, Clone)]
pub struct HcParams {
    pub hc_fn: Vec<f32>,   // [24, hc*dim]
    pub hc_base: Vec<f32>, // [24]
    pub hc_scale: [f32; 3],
}

/// `hc_pre` for one token: xf [hc*dim] (bf16-valued f32 streams flattened) →
/// (y [dim] bf16-rounded, post [4], comb [16]).
pub fn hc_pre_token(xf: &[f32], hc: usize, dim: usize, p: &HcParams, norm_eps: f32, iters: usize, hc_eps: f32) -> (Vec<f32>, [f32; 4], [f32; 16]) {
    let hcd = hc * dim;
    assert_eq!(xf.len(), hcd);
    let ss = sumsq_tree(xf);
    let rsqrt = (ss / hcd as f32 + norm_eps).sqrt().recip();
    let nmix = (2 + hc) * hc;
    let mut mixes = vec![0.0f32; nmix];
    for m in 0..nmix {
        mixes[m] = dot_tree(&p.hc_fn[m * hcd..(m + 1) * hcd], xf) * rsqrt;
    }
    let (pre, post, comb) = hc_split_sinkhorn(&mixes, &p.hc_scale, &p.hc_base, hc, iters, hc_eps);
    // y = sum_h pre[h] * x[h] (fp32), then bf16
    let mut y = vec![0.0f32; dim];
    for h in 0..hc {
        for d in 0..dim {
            y[d] += pre[h] * xf[h * dim + d];
        }
    }
    round_bf16(&mut y);
    (y, post, comb)
}

/// `hc_post` for one token: out[k,d] = post[k]·x[d] + Σ_j comb[j,k]·res[j,d],
/// result bf16-rounded (streams update).
pub fn hc_post_token(x: &[f32], residual: &[f32], post: &[f32; 4], comb: &[f32; 16], hc: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), dim);
    debug_assert_eq!(residual.len(), hc * dim);
    debug_assert_eq!(out.len(), hc * dim);
    for k in 0..hc {
        for d in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..hc {
                acc += comb[j * hc + k] * residual[j * dim + d];
            }
            out[k * dim + d] = bf(post[k] * x[d] + acc);
        }
    }
}

/// `hc_head` (final collapse, sigmoid-only — no post/comb/Sinkhorn; model.py:709-716).
/// xf [hc*dim] flattened streams → y [dim] bf16-rounded.
pub fn hc_head_token(xf: &[f32], hc: usize, dim: usize, hc_fn: &[f32], hc_base: &[f32], hc_scale: f32, norm_eps: f32, hc_eps: f32) -> Vec<f32> {
    let hcd = hc * dim;
    assert_eq!(xf.len(), hcd);
    assert_eq!(hc_fn.len(), hc * hcd);
    let ss = sumsq_tree(xf);
    let rsqrt = (ss / hcd as f32 + norm_eps).sqrt().recip();
    let mut pre = [0.0f32; 4];
    for h in 0..hc {
        let mixes = dot_tree(&hc_fn[h * hcd..(h + 1) * hcd], xf) * rsqrt;
        pre[h] = sigmoid(mixes * hc_scale + hc_base[h]) + hc_eps;
    }
    let mut y = vec![0.0f32; dim];
    for h in 0..hc {
        for d in 0..dim {
            y[d] += pre[h] * xf[h * dim + d];
        }
    }
    round_bf16(&mut y);
    y
}

// ---------------------------------------------------------------------------
// Deterministic top-k (§12.B.2: value desc, index asc; -inf entries sort last)
// ---------------------------------------------------------------------------

/// Indices of the k largest values, sorted value-desc with index-asc tie-break.
/// This is the CSA batch-invariance contract — selection is a pure function of
/// the scores, independent of batch width and platform (torch.topk tie order is
/// implementation-defined; near-ties in identical inputs still agree exactly).
pub fn topk_deterministic(scores: &[f32], k: usize) -> Vec<i64> {
    let k = k.min(scores.len());
    let mut idx: Vec<i64> = (0..scores.len() as i64).collect();
    // Full sort is O(n log n) with n ≤ 262144 for the indexer — fine on CPU.
    idx.sort_by(|&a, &b| {
        let (sa, sb) = (scores[a as usize], scores[b as usize]);
        // desc by value; NaN/-inf naturally last via partial_cmp reversal; ties by index asc
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

// ---------------------------------------------------------------------------
// Router + experts (§B.9)
// ---------------------------------------------------------------------------

/// PyTorch softplus (threshold 20): log1p(exp(s)) for s ≤ 20 else s.
#[inline]
pub fn softplus_torch(s: f32) -> f32 {
    if s > 20.0 {
        s
    } else {
        s.exp().ln_1p()
    }
}

/// `Gate.forward` for a batch of tokens. x [t, dim] (bf16-valued f32, floated
/// exactly like `linear(x.float(), weight.float())`). Returns (weights [t,6]
/// fp32, indices [t,6]).
#[allow(clippy::too_many_arguments)]
pub fn gate_forward(
    x: &[f32],
    t: usize,
    dim: usize,
    gate_w: &[f32],          // [n_exp, dim] fp32
    bias: Option<&[f32]>,    // [n_exp] fp32, selection-only (layers ≥ 3)
    tid2eid: Option<(&[i32], &[i64])>, // (table [vocab,6] int32, token ids [t]) — hash layers 0–2
    n_exp: usize,
    topk: usize,
    route_scale: f32,
) -> (Vec<f32>, Vec<i64>) {
    assert_eq!(x.len(), t * dim);
    assert_eq!(gate_w.len(), n_exp * dim);
    let scores = gemm_f32(x, t, dim, gate_w, n_exp); // fp32 GEMM always
    let mut slots: Vec<(Vec<f32>, Vec<i64>)> = (0..t).map(|_| (vec![0.0; topk], vec![0; topk])).collect();
    par_rows(&mut slots, |i, slot| {
        let (wrow, irow) = slot;
        let mut s: Vec<f32> = scores[i * n_exp..(i + 1) * n_exp].to_vec();
        for v in s.iter_mut() {
            *v = softplus_torch(*v).sqrt();
        }
        let original = s.clone(); // weights are gathered from UN-biased scores
        let sel = match bias {
            Some(b) => {
                let mut bsel = s;
                for (e, v) in bsel.iter_mut().enumerate() {
                    *v += b[e];
                }
                bsel
            }
            None => s,
        };
        let idx: Vec<i64> = match tid2eid {
            Some((table, ids)) => {
                let id = ids[i] as usize;
                (0..topk).map(|j| table[id * topk + j] as i64).collect()
            }
            None => topk_deterministic(&sel, topk),
        };
        let mut wsum = 0.0f32;
        for (j, &e) in idx.iter().enumerate() {
            let w = original[e as usize];
            wrow[j] = w;
            wsum += w;
        }
        for j in 0..topk {
            wrow[j] = wrow[j] / wsum * route_scale;
        }
        irow.copy_from_slice(&idx);
    });
    let mut weights = vec![0.0f32; t * topk];
    let mut indices = vec![0i64; t * topk];
    for (i, (w, ix)) in slots.into_iter().enumerate() {
        weights[i * topk..(i + 1) * topk].copy_from_slice(&w);
        indices[i * topk..(i + 1) * topk].copy_from_slice(&ix);
    }
    (weights, indices)
}

/// One expert's forward on one token (§B.9 Expert): fp32 inner math,
/// asymmetric clamps (up ±limit, gate ≤ +limit), silu(gate)·up, optional
/// routing weight, w2 on the bf16-rounded intermediate. Weight matrices are the
/// exact f32 dequant; `inner_block` is 32 for FP4 routed experts / 128 for FP8
/// shared expert. Returns [dim] bf16-rounded.
pub fn expert_forward_token(
    x: &[f32], // [dim] bf16-valued f32
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    dim: usize,
    inter: usize,
    inner_block: usize,
    swiglu_limit: f32,
    routing_weight: Option<f32>,
) -> Vec<f32> {
    let mut gate = quant_gemm(x, 1, dim, w1, inter, inner_block);
    let mut up = quant_gemm(x, 1, dim, w3, inter, inner_block);
    if swiglu_limit > 0.0 {
        for v in up.iter_mut() {
            *v = v.clamp(-swiglu_limit, swiglu_limit);
        }
        for v in gate.iter_mut() {
            *v = v.min(swiglu_limit);
        }
    }
    let mut h = vec![0.0f32; inter];
    for i in 0..inter {
        let g = gate[i];
        let mut val = (g * sigmoid(g)) * up[i]; // silu(gate) * up, fp32
        if let Some(w) = routing_weight {
            val *= w;
        }
        h[i] = bf(val); // x.to(dtype) before w2
    }
    quant_gemm(&h, 1, inter, w2, dim, inner_block)
}

// ---------------------------------------------------------------------------
// Weight extraction from Lane A's Dsv4Layer (replay path; unit tests build the
// Cpu* structs directly with synthetic tensors)
// ---------------------------------------------------------------------------

pub(crate) fn take_f32(map: &mut HashMap<String, HostTensor>, key: &str, numel: usize) -> Result<Vec<f32>> {
    match map.remove(key) {
        Some(HostTensor::F32 { data, .. }) => {
            if data.len() != numel {
                bail!("{key}: expected {} elements, got {}", numel, data.len());
            }
            Ok(data)
        }
        Some(other) => Err(anyhow!("{key}: expected F32, got {:?}", other.shape())),
        None => Err(anyhow!("{key}: missing")),
    }
}

pub(crate) fn take_bf16_as_f32(map: &mut HashMap<String, HostTensor>, key: &str, numel: usize) -> Result<Vec<f32>> {
    match map.remove(key) {
        Some(HostTensor::BF16 { data, .. }) => {
            if data.len() != numel {
                bail!("{key}: expected {} elements, got {}", numel, data.len());
            }
            Ok(data.iter().map(|v| v.to_f32()).collect())
        }
        Some(HostTensor::F32 { data, .. }) => {
            // tolerate already-upcast (cast rules upcast several bf16 tensors)
            if data.len() != numel {
                bail!("{key}: expected {} elements, got {}", numel, data.len());
            }
            Ok(data)
        }
        Some(other) => Err(anyhow!("{key}: expected BF16, got {:?}", other.shape())),
        None => Err(anyhow!("{key}: missing")),
    }
}

pub(crate) fn take_i32(map: &mut HashMap<String, HostTensor>, key: &str, numel: usize) -> Result<Vec<i32>> {
    match map.remove(key) {
        Some(HostTensor::I32 { data, .. }) => {
            if data.len() != numel {
                bail!("{key}: expected {} elements, got {}", numel, data.len());
            }
            Ok(data)
        }
        Some(other) => Err(anyhow!("{key}: expected I32, got {:?}", other.shape())),
        None => Err(anyhow!("{key}: missing")),
    }
}

pub(crate) fn opt_f32(map: &mut HashMap<String, HostTensor>, key: &str, numel: usize) -> Result<Option<Vec<f32>>> {
    if !map.contains_key(key) {
        return Ok(None);
    }
    take_f32(map, key, numel).map(Some)
}

// ---------------------------------------------------------------------------
// Index-list helpers (model.py:260-282, 743-747 — pure functions, bit-exact)
// ---------------------------------------------------------------------------

/// `get_window_topk_idxs`: SWA ring index lists.
/// prefill (start_pos == 0): row i = [max(0,i−127)..i], −1-padded to min(s,128).
/// decode start_pos ≥ 127: all 128 physical slots, oldest→newest
/// (cat([sp+1..128), [0..sp])), sp = start_pos % 128. Early decode: [0..sp], −1 pad.
pub fn window_topk_idxs(window: usize, s: usize, start_pos: usize) -> Vec<Vec<i64>> {
    if start_pos >= window - 1 {
        let sp = start_pos % window;
        let mut row = Vec::with_capacity(window);
        row.extend((sp + 1..window).map(|v| v as i64));
        row.extend((0..=sp).map(|v| v as i64));
        vec![row; s]
    } else if start_pos > 0 {
        let mut row: Vec<i64> = (0..=start_pos as i64).collect();
        row.resize(window, -1);
        vec![row; s]
    } else {
        let tw = s.min(window);
        (0..s)
            .map(|i| {
                let base = i.saturating_sub(window - 1);
                (0..tw)
                    .map(|j| {
                        let v = base + j;
                        if v > i {
                            -1
                        } else {
                            v as i64
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// `get_compress_topk_idxs` (HCA): every completed 128-token block, no top-k.
/// prefill row i: blocks 0..(i+1)//ratio − 1 (−1-masked), + offset.
/// decode: arange(0, (start_pos+1)//ratio) + offset.
pub fn compress_topk_idxs(ratio: usize, s: usize, start_pos: usize, offset: usize) -> Vec<Vec<i64>> {
    if start_pos > 0 {
        vec![(0..(start_pos + 1) / ratio).map(|v| (v + offset) as i64).collect(); s]
    } else {
        let nb = s / ratio;
        (0..s)
            .map(|i| {
                let lim = (i + 1) / ratio;
                (0..nb)
                    .map(|j| if j >= lim { -1 } else { (j + offset) as i64 })
                    .collect()
            })
            .collect()
    }
}

/// `get_dspark_topk_idxs`: 133-entry non-causal block list —
/// cat([arange(min(128, start_pos+1)), 128+arange(block)]), identical for all draft rows.
pub fn dspark_topk_idxs(window: usize, block: usize, start_pos: usize) -> Vec<i64> {
    let mut v: Vec<i64> = (0..window.min(start_pos + 1)).map(|x| x as i64).collect();
    v.extend((0..block).map(|i| (window + i) as i64));
    v
}

// ---------------------------------------------------------------------------
// Compressor (§B.5, model.py:285-383 — exact prefill + decode state machine)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompressorWeights {
    pub wkv: Vec<f32>,   // [coff*d, dim] fp32
    pub wgate: Vec<f32>, // [coff*d, dim] fp32
    pub norm: Vec<f32>,  // [d] fp32
    pub ape: Vec<f32>,   // [ratio, coff*d] fp32 — added to SCORES only
    pub ratio: usize,
    pub head_dim: usize,  // d
    pub rope_dim: usize,  // rd
    pub overlap: bool,    // ratio == 4 (CSA); false for ratio 128 (HCA)
    pub rotate: bool,     // indexer compressor: Hadamard + FP4 sim on full d
    pub sim_group: usize, // 64 (attention fp8-sim) / 32 (indexer fp4-sim)
    pub dim: usize,       // input dim
}

impl CompressorWeights {
    pub fn coff(&self) -> usize {
        1 + self.overlap as usize
    }
}

/// Incremental decode state (`kv_state`/`score_state`, model.py:309-310).
/// With overlap: state rows [0..ratio) = previous block context, [ratio..2·ratio) = current
/// window. Without: [0..ratio) accumulation rows. score_state inits to −inf.
#[derive(Debug, Clone)]
pub struct CompressorState {
    pub kv_state: Vec<f32>,    // [coff*ratio, coff*d] fp32
    pub score_state: Vec<f32>, // [coff*ratio, coff*d] fp32
}

pub struct Compressor {
    pub w: CompressorWeights,
    pub st: CompressorState,
    pub hadamard: Option<Vec<f32>>, // [d,d] scaled — rotate only
}

impl Compressor {
    pub fn new(w: CompressorWeights) -> Self {
        let coff = w.coff();
        let rows = coff * w.ratio;
        let cols = coff * w.head_dim;
        let hadamard = if w.rotate { Some(hadamard_scaled(w.head_dim)) } else { None };
        Compressor {
            st: CompressorState {
                kv_state: vec![0.0; rows * cols],
                score_state: vec![f32::NEG_INFINITY; rows * cols],
            },
            w,
            hadamard,
        }
    }

    /// `Compressor.forward`. x [s, dim] bf16-valued f32. `rope` is the layer's
    /// shared freqs table. `cache` is this compressor's own cache slice
    /// (attention compressor: aliased rows 128.. of the layer kv_cache; indexer:
    /// its own [max_seq//4, 128]). Returns the pooled rows ([nblocks, d] prefill,
    /// [1, d] decode) when a compression fired, else None.
    pub fn forward(
        &mut self,
        x: &[f32],
        s: usize,
        start_pos: usize,
        rope: &RopeTable,
        norm_eps: f32,
        cache: &mut [f32],
    ) -> Option<Vec<f32>> {
        let w = &self.w;
        let (ratio, d, rd, overlap, rotate) = (w.ratio, w.head_dim, w.rope_dim, w.overlap, w.rotate);
        let coff = w.coff();
        let cd = coff * d;
        assert_eq!(x.len(), s * w.dim);
        // compression in fp32 (x values already f32); wkv/wgate are fp32 Linears
        let kv_full = gemm_f32(x, s, w.dim, &w.wkv, cd);
        let score_full = gemm_f32(x, s, w.dim, &w.wgate, cd);
        let mut pooled: Vec<f32>; // [nblocks, d]
        let mut block_positions: Vec<usize>; // first-token position per pooled row
        let mut cache_row0: usize;
        if start_pos == 0 {
            let should_compress = s >= ratio;
            let remainder = s % ratio;
            let cutoff = s - remainder;
            let offset = if overlap { ratio } else { 0 };
            if overlap && cutoff >= ratio {
                // stash the LAST FULL block as overlap context (raw rows, all coff*d dims)
                for j in 0..ratio {
                    let dst = j * cd;
                    let src = (cutoff - ratio + j) * cd;
                    self.st.kv_state[dst..dst + cd].copy_from_slice(&kv_full[src..src + cd]);
                    for c in 0..cd {
                        self.st.score_state[dst + c] = score_full[src + c] + w.ape[j * cd + c];
                    }
                }
            }
            if remainder > 0 {
                for j in 0..remainder {
                    let dst = (offset + j) * cd;
                    let src = (cutoff + j) * cd;
                    self.st.kv_state[dst..dst + cd].copy_from_slice(&kv_full[src..src + cd]);
                    for c in 0..cd {
                        self.st.score_state[dst + c] = score_full[src + c] + w.ape[j * cd + c];
                    }
                }
            }
            let nb = cutoff / ratio;
            pooled = vec![0.0f32; nb * d];
            for b in 0..nb {
                // assemble the pool rows: overlap → 2·ratio rows (prev block dims :d
                // then current block dims d:), else ratio rows.
                let nrow = if overlap { 2 * ratio } else { ratio };
                // per-column softmax over rows
                let mut kvs = vec![0.0f32; nrow * d];
                let mut scs = vec![f32::NEG_INFINITY; nrow * d];
                for j in 0..nrow {
                    for dd in 0..d {
                        let (kv_v, sc_v);
                        if overlap {
                            if j < ratio {
                                if b == 0 {
                                    kv_v = 0.0;
                                    sc_v = f32::NEG_INFINITY;
                                } else {
                                    let src = ((b - 1) * ratio + j) * cd + dd;
                                    kv_v = kv_full[src];
                                    sc_v = score_full[src] + w.ape[j * cd + dd];
                                }
                            } else {
                                let src = (b * ratio + (j - ratio)) * cd + d + dd;
                                kv_v = kv_full[src];
                                sc_v = score_full[src] + w.ape[(j - ratio) * cd + d + dd];
                            }
                        } else {
                            let src = (b * ratio + j) * cd + dd;
                            kv_v = kv_full[src];
                            sc_v = score_full[src] + w.ape[j * cd + dd];
                        }
                        kvs[j * d + dd] = kv_v;
                        scs[j * d + dd] = sc_v;
                    }
                }
                // pooled = (kvs · softmax(scs, dim=rows)).sum(rows), fp32.
                // torch softmax NORMALIZES first (p_j = e_j/z), then Σ p_j·kv_j
                // — that rounding order is load-bearing (do not divide at the end).
                for dd in 0..d {
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..nrow {
                        mx = mx.max(scs[j * d + dd]);
                    }
                    let mut z = 0.0f32;
                    for j in 0..nrow {
                        z += (scs[j * d + dd] - mx).exp();
                    }
                    let mut acc = 0.0f32;
                    for j in 0..nrow {
                        let p = (scs[j * d + dd] - mx).exp() / z;
                        acc += kvs[j * d + dd] * p;
                    }
                    pooled[b * d + dd] = acc;
                }
            }
            block_positions = (0..nb).map(|b| b * ratio).collect();
            cache_row0 = 0;
            if !should_compress {
                return None;
            }
        } else {
            let should_compress = (start_pos + 1) % ratio == 0;
            // score += ape[start_pos % ratio]  (score row for this token only)
            let mut score0 = score_full[..cd].to_vec();
            for c in 0..cd {
                score0[c] += w.ape[(start_pos % ratio) * cd + c];
            }
            pooled = Vec::new();
            if overlap {
                let slot = ratio + start_pos % ratio;
                self.st.kv_state[slot * cd..(slot + 1) * cd].copy_from_slice(&kv_full[..cd]);
                self.st.score_state[slot * cd..(slot + 1) * cd].copy_from_slice(&score0);
                if should_compress {
                    let nrow = 2 * ratio;
                    let mut kvs = vec![0.0f32; nrow * d];
                    let mut scs = vec![0.0f32; nrow * d];
                    for j in 0..nrow {
                        for dd in 0..d {
                            if j < ratio {
                                kvs[j * d + dd] = self.st.kv_state[j * cd + dd];
                                scs[j * d + dd] = self.st.score_state[j * cd + dd];
                            } else {
                                kvs[j * d + dd] = self.st.kv_state[j * cd + d + dd];
                                scs[j * d + dd] = self.st.score_state[j * cd + d + dd];
                            }
                        }
                    }
                    pooled = vec![0.0f32; d];
                    for dd in 0..d {
                        let mut mx = f32::NEG_INFINITY;
                        for j in 0..nrow {
                            mx = mx.max(scs[j * d + dd]);
                        }
                        let mut z = 0.0f32;
                        for j in 0..nrow {
                            z += (scs[j * d + dd] - mx).exp();
                        }
                        let mut acc = 0.0f32;
                        for j in 0..nrow {
                            let p = (scs[j * d + dd] - mx).exp() / z;
                            acc += kvs[j * d + dd] * p;
                        }
                        pooled[dd] = acc;
                    }
                    // shift: state rows [0..ratio) <- [ratio..2ratio)
                    for j in 0..ratio {
                        let dst = j * cd;
                        let src = (ratio + j) * cd;
                        let (a, b) = self.st.kv_state.split_at_mut(src);
                        a[dst..dst + cd].copy_from_slice(&b[..cd]);
                        let (a, b) = self.st.score_state.split_at_mut(src);
                        a[dst..dst + cd].copy_from_slice(&b[..cd]);
                    }
                }
            } else {
                let slot = start_pos % ratio;
                self.st.kv_state[slot * cd..(slot + 1) * cd].copy_from_slice(&kv_full[..cd]);
                self.st.score_state[slot * cd..(slot + 1) * cd].copy_from_slice(&score0);
                if should_compress {
                    pooled = vec![0.0f32; d];
                    for dd in 0..d {
                        let mut mx = f32::NEG_INFINITY;
                        for j in 0..ratio {
                            mx = mx.max(self.st.score_state[j * cd + dd]);
                        }
                        let mut z = 0.0f32;
                        for j in 0..ratio {
                            z += (self.st.score_state[j * cd + dd] - mx).exp();
                        }
                        let mut acc = 0.0f32;
                        for j in 0..ratio {
                            let p = (self.st.score_state[j * cd + dd] - mx).exp() / z;
                            acc += self.st.kv_state[j * cd + dd] * p;
                        }
                        pooled[dd] = acc;
                    }
                }
            }
            if !should_compress {
                return None;
            }
            block_positions = vec![start_pos + 1 - ratio];
            cache_row0 = start_pos / ratio;
        }
        // post-pooling: bf16 -> RMSNorm -> RoPE at first-token position -> QAT-sim
        let nrows = pooled.len() / d;
        for (b, pos) in block_positions.iter().enumerate().take(nrows) {
            let row = &mut pooled[b * d..(b + 1) * d];
            round_bf16(row);
            rms_norm_row(row, &w.norm, norm_eps);
            apply_rope(row, 1, d, rope, &[*pos], false);
            if rotate {
                let h = self.hadamard.as_ref().unwrap();
                rotate_activation(row, 1, d, d, h);
                fp4_act_quant_sim(row, 1, d, w.sim_group);
            } else {
                act_quant_sim(&mut row[..d - rd], 1, d - rd, w.sim_group);
            }
            cache[(cache_row0 + b) * d..(cache_row0 + b + 1) * d].copy_from_slice(row);
        }
        Some(pooled)
    }
}

// ---------------------------------------------------------------------------
// Indexer (§B.6, model.py:386-439 — CSA only)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct IndexerWeights {
    pub wq_b: Vec<f32>,         // [64*128, q_lora_rank] fp8-dequant
    pub weights_proj: Vec<f32>, // [64, dim] bf16-valued
    pub compressor: CompressorWeights, // ratio 4, d 128, rotate = true
}

pub struct IndexerState {
    pub compressor: Compressor,
    pub kv_cache: Vec<f32>, // [max_seq//4, 128] bf16-valued
}

/// `Indexer.forward` → per-token compress-block index lists (with `offset`
/// already added; −1 = masked). `qr` is the attention's q_lora latent [s, 1024]
/// (bf16-valued). `rope` is the layer table. `hadamard` is the 128×128 scaled
/// rotation. Deterministic top-k per §12.B.2.
#[allow(clippy::too_many_arguments)]
pub fn indexer_forward(
    w: &IndexerWeights,
    st: &mut IndexerState,
    x: &[f32],
    s: usize,
    qr: &[f32],
    start_pos: usize,
    offset: usize,
    rope: &RopeTable,
    hadamard: &[f32],
    cfg: &Dsv4Config,
) -> Vec<Vec<i64>> {
    let dim = cfg.dim;
    let nh = cfg.index_n_heads;
    let hd = cfg.index_head_dim;
    let rd = cfg.rope_head_dim;
    let ratio = 4usize;
    let end_pos = start_pos + s;
    // q = wq_b(qr) -> [s, 64, 128]; RoPE last 64; Hadamard; FP4 sim
    let mut q = quant_gemm(qr, s, cfg.q_lora_rank, &w.wq_b, nh * hd, 128);
    {
        let rows = s * nh;
        let pos: Vec<usize> = (0..rows).map(|i| start_pos + i / nh).collect();
        apply_rope(&mut q, rows, hd, rope, &pos, false);
        rotate_activation(&mut q, rows, hd, hd, hadamard);
        fp4_act_quant_sim(&mut q, rows, hd, 32);
    }
    // own compressor updates the indexer kv cache
    st.compressor.forward(x, s, start_pos, rope, cfg.norm_eps, &mut st.kv_cache);
    // head weights: weights_proj(x) · (128^-0.5 · 64^-0.5), bf16
    let mut weights = gemm_bf16(x, s, dim, &w.weights_proj, nh);
    let wscale = ((hd as f64).powf(-0.5) * (nh as f64).powf(-0.5)) as f32;
    for v in weights.iter_mut() {
        *v = bf(*v * wscale);
    }
    let nblocks = end_pos / ratio;
    let k = cfg.index_topk.min(nblocks);
    let mut out: Vec<Vec<i64>> = Vec::with_capacity(s);
    for i in 0..s {
        // index_score = einsum(q, kv_cache[:nblocks]); relu; ×weights; sum over heads
        // (bf16 tensor semantics: einsum out bf16, relu exact, mul bf16, sum f32-acc→bf16)
        let mut score = vec![0.0f32; nblocks];
        for t in 0..nblocks {
            let kvrow = &st.kv_cache[t * hd..(t + 1) * hd];
            let mut acc = 0.0f32;
            for h in 0..nh {
                let qrow = &q[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                let dot = bf(dot8(qrow, kvrow)); // einsum output is bf16
                let rel = if dot > 0.0 { dot } else { 0.0 }; // relu_ (bf16, exact)
                acc += bf(rel * weights[i * nh + h]); // bf16 mul
            }
            score[t] = bf(acc);
        }
        if start_pos == 0 {
            // block-causal: token i sees block t ⟺ t < (i+1)//ratio
            let lim = (i + 1) / ratio;
            for (t, v) in score.iter_mut().enumerate() {
                if t >= lim {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        let mut idx = topk_deterministic(&score, k);
        if start_pos == 0 {
            let lim = (i + 1) / ratio;
            for v in idx.iter_mut() {
                if *v >= lim as i64 {
                    *v = -1;
                } else {
                    *v += offset as i64;
                }
            }
        } else {
            for v in idx.iter_mut() {
                *v += offset as i64;
            }
        }
        out.push(idx);
    }
    out
}

// ---------------------------------------------------------------------------
// Attention (§B.1–B.4, model.py:442-548)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AttnWeights {
    pub wq_a: Vec<f32>,     // [q_lora_rank, dim] fp8-dequant
    pub q_norm: Vec<f32>,   // [q_lora_rank]
    pub wq_b: Vec<f32>,     // [64*512, q_lora_rank] fp8-dequant
    pub wkv: Vec<f32>,      // [512, dim] fp8-dequant
    pub kv_norm: Vec<f32>,  // [512]
    pub sink: Vec<f32>,     // [64] fp32 — denominator only
    pub wo_a: Vec<f32>,     // [8*1024, 4096] bf16-valued (§F.2 dequant)
    pub wo_b: Vec<f32>,     // [4096, 8192] fp8-dequant
    pub compressor: Option<CompressorWeights>,
    pub indexer: Option<IndexerWeights>,
    pub kind: LayerKind,
}

pub struct AttnState {
    pub kv_cache: Vec<f32>, // [cache_rows, 512] bf16-valued
    pub cache_rows: usize,
    pub compressor: Option<Compressor>,
    pub indexer: Option<IndexerState>,
}

impl AttnState {
    pub fn new(cfg: &Dsv4Config, w: &AttnWeights, max_seq_len: usize) -> Self {
        let win = cfg.window_size;
        let cache_rows = match w.kind {
            LayerKind::Swa => win,
            LayerKind::Csa => win + max_seq_len / 4,
            LayerKind::Hca => win + max_seq_len / 128,
        };
        let compressor = w.compressor.as_ref().map(|cw| Compressor::new(cw.clone()));
        let indexer = w.indexer.as_ref().map(|iw| IndexerState {
            compressor: Compressor::new(iw.compressor.clone()),
            kv_cache: vec![0.0; (max_seq_len / 4) * cfg.index_head_dim],
        });
        AttnState {
            kv_cache: vec![0.0; cache_rows * cfg.head_dim],
            cache_rows,
            compressor,
            indexer,
        }
    }
}

/// Q/KV common sub-path (§B.1.1–2). Returns (qr [s,1024], q [s,64,512], kv [s,512]).
/// All bf16-valued f32. `x` [s, dim] is the attn_norm output.
pub fn attn_qkv(
    w: &AttnWeights,
    x: &[f32],
    s: usize,
    start_pos: usize,
    rope: &RopeTable,
    cfg: &Dsv4Config,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = cfg.dim;
    let qlr = cfg.q_lora_rank;
    let nh = cfg.n_heads;
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    // qr = q_norm(wq_a(x))   [s, 1024]
    let qr_pre = quant_gemm(x, s, dim, &w.wq_a, qlr, 128);
    let qr = rms_norm(&qr_pre, s, qlr, &w.q_norm, cfg.norm_eps);
    // q = wq_b(qr) -> [s, 64, 512]; weight-free per-head RMS rescale (bf16 per-op)
    let mut q = quant_gemm(&qr, s, qlr, &w.wq_b, nh * hd, 128);
    for i in 0..s * nh {
        let row = &mut q[i * hd..(i + 1) * hd];
        // q.square().mean(-1) + eps -> rsqrt -> q *= — torch bf16 op-by-op:
        // square is a bf16 tensor (per-op rounding), mean reduces f32 (tree)
        let mut sq: Vec<f32> = row.iter().map(|&v| bf(v * v)).collect();
        let ss = pairwise_sum(&mut sq);
        let mean = bf(ss / hd as f32);
        let arg = bf(mean + cfg.norm_eps);
        let r = bf(arg.sqrt().recip());
        for v in row.iter_mut() {
            *v = bf(*v * r);
        }
    }
    {
        let rows = s * nh;
        let pos: Vec<usize> = (0..rows).map(|i| start_pos + i / nh).collect();
        apply_rope(&mut q, rows, hd, rope, &pos, false);
    }
    // kv = kv_norm(wkv(x)); RoPE last 64; QAT-sim nope dims (group 64)
    let kv_pre = quant_gemm(x, s, dim, &w.wkv, hd, 128);
    let mut kv = rms_norm(&kv_pre, s, hd, &w.kv_norm, cfg.norm_eps);
    {
        let pos: Vec<usize> = (0..s).map(|i| start_pos + i).collect();
        apply_rope(&mut kv, s, hd, rope, &pos, false);
    }
    {
        let nope = hd - rd;
        // act_quant on the [.., :448] view: treat as rows of length 448
        let mut tmp = vec![0.0f32; s * nope];
        for i in 0..s {
            tmp[i * nope..(i + 1) * nope].copy_from_slice(&kv[i * hd..i * hd + nope]);
        }
        act_quant_sim(&mut tmp, s, nope, 64);
        for i in 0..s {
            kv[i * hd..i * hd + nope].copy_from_slice(&tmp[i * nope..(i + 1) * nope]);
        }
    }
    (qr, q, kv)
}

/// Grouped-LoRA O (§B.1.4): de-rotation (caller does it), then
/// o.view(s,8,4096) einsum with wo_a.view(8,1024,4096) → wo_b.
pub fn attn_out_proj(w: &AttnWeights, o: &[f32], s: usize, cfg: &Dsv4Config) -> Vec<f32> {
    let g = cfg.o_groups;
    let r = cfg.o_lora_rank;
    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let gd = nh * hd / g; // 4096 per group
    // einsum "bsgd,grd->bsgr": per group, [s, gd] @ wo_a[g]ᵀ -> [s, r] (bf16 GEMM)
    let mut oflat = vec![0.0f32; s * g * r];
    for grp in 0..g {
        let og = &o[grp * gd..]; // strided view per token — gather
        let mut xg = vec![0.0f32; s * gd];
        for i in 0..s {
            xg[i * gd..(i + 1) * gd].copy_from_slice(&og[i * nh * hd..i * nh * hd + gd]);
        }
        let wag = &w.wo_a[grp * r * gd..(grp + 1) * r * gd];
        let yg = gemm_bf16(&xg, s, gd, wag, r);
        for i in 0..s {
            oflat[i * g * r + grp * r..i * g * r + (grp + 1) * r].copy_from_slice(&yg[i * r..(i + 1) * r]);
        }
    }
    quant_gemm(&oflat, s, g * r, &w.wo_b, cfg.dim, 128)
}

/// Trunk attention forward (model.py:490-548). x [s, dim] = attn_norm output.
/// Returns (out [s, dim] bf16, topk_idxs flattened [s, T] with −1 masking).
#[allow(clippy::too_many_arguments)]
pub fn attn_forward(
    w: &AttnWeights,
    st: &mut AttnState,
    x: &[f32],
    s: usize,
    start_pos: usize,
    rope: &RopeTable,
    hadamard128: Option<&[f32]>,
    cfg: &Dsv4Config,
) -> (Vec<f32>, Vec<i64>, usize) {
    let win = cfg.window_size;
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let (qr, q, kv) = attn_qkv(w, x, s, start_pos, rope, cfg);
    // index lists
    let win_idxs = window_topk_idxs(win, s, start_pos);
    let topk: Vec<Vec<i64>> = match w.kind {
        LayerKind::Swa => win_idxs,
        LayerKind::Csa => {
            let offset = if start_pos == 0 { s } else { win };
            let iw = w.indexer.as_ref().unwrap();
            let ist = st.indexer.as_mut().unwrap();
            let comp = indexer_forward(iw, ist, x, s, &qr, start_pos, offset, rope, hadamard128.unwrap(), cfg);
            win_idxs
                .into_iter()
                .zip(comp)
                .map(|(mut a, b)| {
                    a.extend(b);
                    a
                })
                .collect()
        }
        LayerKind::Hca => {
            let offset = if start_pos == 0 { s } else { win };
            let comp = compress_topk_idxs(128, s, start_pos, offset);
            win_idxs
                .into_iter()
                .zip(comp)
                .map(|(mut a, b)| {
                    a.extend(b);
                    a
                })
                .collect()
        }
    };
    let t = topk[0].len();
    let scale = (hd as f64).powf(-0.5) as f32; // 512^-0.5
    let o: Vec<f32>;
    if start_pos == 0 {
        // ring write (rotated at s > win), then attention over current kv (+ compressor rows)
        if s <= win {
            st.kv_cache[..s * hd].copy_from_slice(&kv);
        } else {
            let cutoff = s % win;
            // cache[cutoff:win], cache[:cutoff] = kv[-win:].split([win-cutoff, cutoff])
            st.kv_cache[cutoff * hd..win * hd].copy_from_slice(&kv[(s - win) * hd..(s - cutoff) * hd]);
            st.kv_cache[..cutoff * hd].copy_from_slice(&kv[(s - cutoff) * hd..s * hd]);
        }
        let mut kv_attn = kv.clone();
        if let Some(comp) = st.compressor.as_mut() {
            let (head, tail) = st.kv_cache.split_at_mut(win * hd);
            let _ = head;
            if let Some(kvc) = comp.forward(x, s, 0, rope, cfg.norm_eps, tail) {
                kv_attn.extend_from_slice(&kvc);
            }
        }
        let flat: Vec<i64> = topk.iter().flatten().copied().collect();
        o = sparse_attn(&q, s, cfg.n_heads, hd, &kv_attn, kv_attn.len() / hd, &w.sink, &flat, t, scale);
    } else {
        // write-before-attention: current token attends to itself
        st.kv_cache[(start_pos % win) * hd..(start_pos % win + 1) * hd].copy_from_slice(&kv[..hd]);
        if let Some(comp) = st.compressor.as_mut() {
            let (head, tail) = st.kv_cache.split_at_mut(win * hd);
            let _ = head;
            comp.forward(x, s, start_pos, rope, cfg.norm_eps, tail);
        }
        let flat: Vec<i64> = topk.iter().flatten().copied().collect();
        o = sparse_attn(&q, s, cfg.n_heads, hd, &st.kv_cache, st.cache_rows, &w.sink, &flat, t, scale);
    }
    // de-rotation (inverse RoPE on the attention output, compensating K≡V leak)
    let mut o = o;
    {
        let rows = s * cfg.n_heads;
        let pos: Vec<usize> = (0..rows).map(|i| start_pos + i / cfg.n_heads).collect();
        apply_rope(&mut o, rows, hd, rope, &pos, true);
    }
    let _ = rd;
    let out = attn_out_proj(w, &o, s, cfg);
    let flat: Vec<i64> = topk.into_iter().flatten().collect();
    (out, flat, t)
}

// ---------------------------------------------------------------------------
// MoE (§B.9)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ExpertF32 {
    pub w1: Vec<f32>, // [inter, dim]
    pub w2: Vec<f32>, // [dim, inter]
    pub w3: Vec<f32>, // [inter, dim]
}

/// Routed-expert bank: NVFP4 tensors (Lane A repack) with lazy exact-f32 dequant
/// cache (256 slots, one per expert; get_or_init is thread-safe).
pub struct ExpertBank {
    pub w1: Vec<Nvfp4Tensor>,
    pub w2: Vec<Nvfp4Tensor>,
    pub w3: Vec<Nvfp4Tensor>,
    cache: Vec<OnceLock<ExpertF32>>,
}

impl ExpertBank {
    pub fn from_nvfp4(w1: Vec<Nvfp4Tensor>, w2: Vec<Nvfp4Tensor>, w3: Vec<Nvfp4Tensor>) -> Self {
        let n = w1.len();
        ExpertBank {
            w1,
            w2,
            w3,
            cache: (0..n).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Test path: pre-dequantized experts (no NVFP4 backing needed).
    pub fn from_f32(experts: Vec<ExpertF32>) -> Self {
        let n = experts.len();
        let cache: Vec<OnceLock<ExpertF32>> = (0..n).map(|_| OnceLock::new()).collect();
        for (slot, e) in cache.iter().zip(experts) {
            let _ = slot.set(e);
        }
        ExpertBank { w1: Vec::new(), w2: Vec::new(), w3: Vec::new(), cache }
    }

    pub fn get(&self, e: usize) -> &ExpertF32 {
        self.cache[e].get_or_init(|| {
            ExpertF32 {
                w1: crate::dsv4_load::dequant_nvfp4_f32(&self.w1[e]),
                w2: crate::dsv4_load::dequant_nvfp4_f32(&self.w2[e]),
                w3: crate::dsv4_load::dequant_nvfp4_f32(&self.w3[e]),
            }
        })
    }
}

pub struct MoeWeights {
    pub gate_w: Vec<f32>,            // [n_exp, dim] fp32 (bf16 upcast)
    pub gate_bias: Option<Vec<f32>>, // [n_exp] fp32 — selection only (layers ≥ 3)
    pub tid2eid: Option<Vec<i32>>,   // [vocab, 6] — hash layers 0–2
    pub shared: ExpertF32,           // fp8 dequant — inner block 128
    pub experts: ExpertBank,         // fp4 — inner block 32
}

/// `MoE.forward`: fp32 accumulator, per-expert row gather (expert-ascending
/// merge order, matching the reference's loop), shared expert added last,
/// output bf16. Returns (out [s,dim], router_w [s,6] fp32, router_idx [s,6]).
pub fn moe_forward(
    w: &MoeWeights,
    x: &[f32],
    s: usize,
    ids: &[i64],
    cfg: &Dsv4Config,
) -> (Vec<f32>, Vec<f32>, Vec<i64>) {
    let dim = cfg.dim;
    let inter = cfg.moe_inter_dim;
    let ne = cfg.n_routed_experts;
    let topk = cfg.n_activated_experts;
    let tid = w.tid2eid.as_deref().map(|t| (t, ids));
    let (weights, indices) = gate_forward(x, s, dim, &w.gate_w, w.gate_bias.as_deref(), tid, ne, topk, cfg.route_scale);
    // expert-major partials (parallel over experts; each expert's rows are disjoint)
    let mut partials: Vec<Vec<(usize, Vec<f32>)>> = (0..ne).map(|_| Vec::new()).collect();
    par_rows(&mut partials, |e, slot| {
        let mut rows: Vec<(usize, f32)> = Vec::new(); // (token, which slot)
        for i in 0..s {
            for j in 0..topk {
                if indices[i * topk + j] as usize == e {
                    rows.push((i, weights[i * topk + j]));
                }
            }
        }
        if rows.is_empty() {
            return;
        }
        let exp = w.experts.get(e);
        for (i, wgt) in rows {
            let xi = &x[i * dim..(i + 1) * dim];
            let y = expert_forward_token(xi, &exp.w1, &exp.w2, &exp.w3, dim, inter, 32, cfg.swiglu_limit, Some(wgt));
            slot.push((i, y));
        }
    });
    // fp32 accumulator; expert-ascending merge order (reference loop order)
    let mut y = vec![0.0f32; s * dim];
    for part in partials.iter() {
        for (i, row) in part {
            for d in 0..dim {
                y[i * dim + d] += row[d];
            }
        }
    }
    // shared expert on the original bf16 x, no routing weight
    for i in 0..s {
        let xi = &x[i * dim..(i + 1) * dim];
        let ys = expert_forward_token(xi, &w.shared.w1, &w.shared.w2, &w.shared.w3, dim, inter, 128, cfg.swiglu_limit, None);
        for d in 0..dim {
            y[i * dim + d] += ys[d];
        }
    }
    round_bf16(&mut y);
    (y, weights, indices)
}

// ---------------------------------------------------------------------------
// Block (§B.8 ordering: hc_pre -> norm -> sublayer -> hc_post, twice)
// ---------------------------------------------------------------------------

/// mHC head-collapse parameters (trunk `hc_head_*` and mtp.2's own).
#[derive(Debug, Clone)]
pub struct HcHeadParams {
    pub hc_fn: Vec<f32>,   // [4, hc*dim]
    pub hc_base: Vec<f32>, // [4]
    pub hc_scale: f32,     // [1]
}

pub struct CpuLayer {
    pub kind: LayerKind,
    pub attn: AttnWeights,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc_attn: HcParams,
    pub hc_ffn: HcParams,
    pub moe: MoeWeights,
    // DSpark stage extras (None on trunk layers)
    pub main_proj: Option<Vec<f32>>, // [4096, 12288] fp8-dequant (stage 0)
    pub main_norm: Option<Vec<f32>>, // [4096] (stage 0)
    pub norm: Option<Vec<f32>>,      // [4096] (stage 2)
    pub hc_head: Option<HcHeadParams>, // (stage 2)
    pub markov_w1: Option<Vec<f32>>, // [vocab, 256] bf16-valued (stage 2)
    pub markov_w2: Option<Vec<f32>>, // [vocab, 256] fp32 (stage 2)
    pub confidence: Option<Vec<f32>>, // [4352] fp32 (stage 2)
}

#[derive(Debug, Default)]
pub struct BlockTrace {
    pub attn_out: Vec<f32>,   // [s, dim]
    pub ffn_out: Vec<f32>,    // [s, dim]
    pub router_w: Vec<f32>,   // [s, 6]
    pub router_idx: Vec<i64>, // [s, 6]
    pub topk_idx: Vec<i64>,   // [s, T] flattened
    pub topk_t: usize,
}

/// hc_pre over all tokens (parallel). x [s, hc*dim] flattened streams.
/// Returns (y [s, dim] bf16, posts [s], combs [s]).
pub fn hc_pre_all(x: &[f32], s: usize, p: &HcParams, cfg: &Dsv4Config) -> (Vec<f32>, Vec<[f32; 4]>, Vec<[f32; 16]>) {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let mut slots: Vec<(Vec<f32>, [f32; 4], [f32; 16])> = (0..s).map(|_| (Vec::new(), [0.0; 4], [0.0; 16])).collect();
    par_rows(&mut slots, |i, slot| {
        let xf = &x[i * hc * dim..(i + 1) * hc * dim];
        *slot = hc_pre_token(xf, hc, dim, p, cfg.norm_eps, cfg.hc_sinkhorn_iters, cfg.hc_eps);
    });
    let mut y = vec![0.0f32; s * dim];
    let mut posts = Vec::with_capacity(s);
    let mut combs = Vec::with_capacity(s);
    for (i, (yi, pi, ci)) in slots.into_iter().enumerate() {
        y[i * dim..(i + 1) * dim].copy_from_slice(&yi);
        posts.push(pi);
        combs.push(ci);
    }
    (y, posts, combs)
}

/// hc_post over all tokens: out [s, hc*dim] streams.
fn hc_post_all(x_out: &[f32], residual: &[f32], posts: &[[f32; 4]], combs: &[[f32; 16]], cfg: &Dsv4Config) -> Vec<f32> {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let s = posts.len();
    let mut out = vec![0.0f32; s * hc * dim];
    par_chunks(&mut out, hc * dim, |i, orow| {
        hc_post_token(
            &x_out[i * dim..(i + 1) * dim],
            &residual[i * hc * dim..(i + 1) * hc * dim],
            &posts[i],
            &combs[i],
            hc,
            dim,
            orow,
        );
    });
    out
}

/// Trunk `Block.forward`. x [s, hc*dim] streams (bf16-valued), ids [s].
/// Returns (new streams [s, hc*dim], trace).
#[allow(clippy::too_many_arguments)]
pub fn block_forward(
    layer: &CpuLayer,
    st: &mut AttnState,
    x: &[f32],
    s: usize,
    start_pos: usize,
    ids: &[i64],
    rope: &RopeTable,
    hadamard128: Option<&[f32]>,
    cfg: &Dsv4Config,
) -> (Vec<f32>, BlockTrace) {
    let dim = cfg.dim;
    let mut trace = BlockTrace::default();
    // --- attention sublayer ---
    let (y, posts, combs) = hc_pre_all(x, s, &layer.hc_attn, cfg);
    let yn = rms_norm(&y, s, dim, &layer.attn_norm, cfg.norm_eps);
    let (attn_out, topk, t) = attn_forward(&layer.attn, st, &yn, s, start_pos, rope, hadamard128, cfg);
    trace.attn_out = attn_out.clone();
    trace.topk_idx = topk;
    trace.topk_t = t;
    let x2 = hc_post_all(&attn_out, x, &posts, &combs, cfg);
    // --- ffn sublayer ---
    let (y2, posts2, combs2) = hc_pre_all(&x2, s, &layer.hc_ffn, cfg);
    let y2n = rms_norm(&y2, s, dim, &layer.ffn_norm, cfg.norm_eps);
    let (ffn_out, rw, ri) = moe_forward(&layer.moe, &y2n, s, ids, cfg);
    trace.ffn_out = ffn_out.clone();
    trace.router_w = rw;
    trace.router_idx = ri;
    let x3 = hc_post_all(&ffn_out, &x2, &posts2, &combs2, cfg);
    (x3, trace)
}

// ---------------------------------------------------------------------------
// DSpark (§B.10)
// ---------------------------------------------------------------------------

/// `DSparkAttention` prefill/warm branch: compute main_kv for ALL trunk
/// positions, RoPE 0..S−1, FP8-sim, ring-write. Returns nothing (h passes
/// through unchanged — no attention, no FFN in the warm pass).
pub fn dspark_attn_warm(
    w: &AttnWeights,
    kv_cache: &mut [f32], // [128, 512]
    main_x: &[f32],
    s: usize,
    rope: &RopeTable,
    cfg: &Dsv4Config,
) {
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let win = cfg.window_size;
    let mk = quant_gemm(main_x, s, cfg.dim, &w.wkv, hd, 128);
    let mut main_kv = rms_norm(&mk, s, hd, &w.kv_norm, cfg.norm_eps);
    let pos: Vec<usize> = (0..s).collect();
    apply_rope(&mut main_kv, s, hd, rope, &pos, false);
    let nope = hd - rd;
    let mut tmp = vec![0.0f32; s * nope];
    for i in 0..s {
        tmp[i * nope..(i + 1) * nope].copy_from_slice(&main_kv[i * hd..i * hd + nope]);
    }
    act_quant_sim(&mut tmp, s, nope, 64);
    for i in 0..s {
        main_kv[i * hd..i * hd + nope].copy_from_slice(&tmp[i * nope..(i + 1) * nope]);
    }
    if s <= win {
        kv_cache[..s * hd].copy_from_slice(&main_kv);
    } else {
        let cutoff = s % win;
        kv_cache[cutoff * hd..win * hd].copy_from_slice(&main_kv[(s - win) * hd..(s - cutoff) * hd]);
        kv_cache[..cutoff * hd].copy_from_slice(&main_kv[(s - cutoff) * hd..s * hd]);
    }
}

/// `DSparkAttention` decode branch (model.py:771-792): main_kv at start_pos,
/// draft q/kv at positions start_pos+seqlen .. +block, 133-entry non-causal
/// block index list, sparse_attn, draft-freq de-rotation, grouped-O.
/// x [block, dim] is the collapsed+normed draft input. main_x [1, dim].
#[allow(clippy::too_many_arguments)]
pub fn dspark_attn_forward(
    w: &AttnWeights,
    kv_cache: &mut [f32], // [128, 512]
    x: &[f32],
    block: usize,
    start_pos: usize,
    main_x: &[f32],
    rope: &RopeTable,
    cfg: &Dsv4Config,
) -> Vec<f32> {
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let win = cfg.window_size;
    let sm = main_x.len() / cfg.dim;
    // main kv at start_pos (sm = 1 for the draft pass)
    dspark_attn_warm_deque(w, kv_cache, main_x, sm, start_pos, rope, cfg);
    // draft q/kv at positions start_pos+sm .. start_pos+sm+block-1
    let pos0 = start_pos + sm;
    let (_qr, q, kv) = attn_qkv(w, x, block, pos0, rope, cfg);
    // index list: cat([arange(min(128, start_pos+1)), 128+arange(block)]) × block rows
    let idx_row = dspark_topk_idxs(win, block, start_pos);
    let t = idx_row.len();
    let mut flat = Vec::with_capacity(block * t);
    for _ in 0..block {
        flat.extend_from_slice(&idx_row);
    }
    let mut kv_cat = kv_cache.to_vec();
    kv_cat.extend_from_slice(&kv);
    let scale = (hd as f64).powf(-0.5) as f32;
    let mut o = sparse_attn(&q, block, cfg.n_heads, hd, &kv_cat, win + block, &w.sink, &flat, t, scale);
    // de-rotation with the DRAFT freqs
    let rows = block * cfg.n_heads;
    let pos: Vec<usize> = (0..rows).map(|i| pos0 + i / cfg.n_heads).collect();
    apply_rope(&mut o, rows, hd, rope, &pos, true);
    let _ = rd;
    attn_out_proj(w, &o, block, cfg)
}

/// main_kv single-position write used by the decode branch (RoPE at start_pos,
/// FP8-sim, ring slot start_pos % 128 — written BEFORE the draft attention).
fn dspark_attn_warm_deque(
    w: &AttnWeights,
    kv_cache: &mut [f32],
    main_x: &[f32],
    sm: usize,
    start_pos: usize,
    rope: &RopeTable,
    cfg: &Dsv4Config,
) {
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let mk = quant_gemm(main_x, sm, cfg.dim, &w.wkv, hd, 128);
    let mut main_kv = rms_norm(&mk, sm, hd, &w.kv_norm, cfg.norm_eps);
    let pos: Vec<usize> = (0..sm).map(|i| start_pos + i).collect();
    apply_rope(&mut main_kv, sm, hd, rope, &pos, false);
    let nope = hd - rd;
    let mut tmp = vec![0.0f32; sm * nope];
    for i in 0..sm {
        tmp[i * nope..(i + 1) * nope].copy_from_slice(&main_kv[i * hd..i * hd + nope]);
    }
    act_quant_sim(&mut tmp, sm, nope, 64);
    for i in 0..sm {
        main_kv[i * hd..i * hd + nope].copy_from_slice(&tmp[i * nope..(i + 1) * nope]);
    }
    kv_cache[(start_pos % cfg.window_size) * hd..(start_pos % cfg.window_size + 1) * hd]
        .copy_from_slice(&main_kv[..hd]);
}

/// `DSparkBlock.forward`: warm (start_pos == 0) writes the ring and returns h
/// unchanged; decode runs the full Block (hc_pre → norm → DSparkAttention →
/// hc_post → hc_pre → norm → MoE → hc_post).
#[allow(clippy::too_many_arguments)]
pub fn dspark_block_forward(
    layer: &CpuLayer,
    kv_cache: &mut [f32],
    h: &[f32], // [block, hc*dim] streams
    block: usize,
    start_pos: usize,
    main_x: &[f32],
    rope: &RopeTable,
    cfg: &Dsv4Config,
) -> Vec<f32> {
    dspark_block_forward_traced(layer, kv_cache, h, block, start_pos, main_x, rope, cfg).0
}

/// Traced variant (A/B debugging): also returns the sublayer outputs.
#[allow(clippy::too_many_arguments)]
pub fn dspark_block_forward_traced(
    layer: &CpuLayer,
    kv_cache: &mut [f32],
    h: &[f32], // [block, hc*dim] streams
    block: usize,
    start_pos: usize,
    main_x: &[f32],
    rope: &RopeTable,
    cfg: &Dsv4Config,
) -> (Vec<f32>, BlockTrace) {
    let dim = cfg.dim;
    let mut trace = BlockTrace::default();
    if start_pos == 0 {
        let s = main_x.len() / dim;
        dspark_attn_warm(&layer.attn, kv_cache, main_x, s, rope, cfg);
        return (h.to_vec(), trace);
    }
    let (y, posts, combs) = hc_pre_all(h, block, &layer.hc_attn, cfg);
    let yn = rms_norm(&y, block, dim, &layer.attn_norm, cfg.norm_eps);
    let attn_out = dspark_attn_forward(&layer.attn, kv_cache, &yn, block, start_pos, main_x, rope, cfg);
    trace.attn_out = attn_out.clone();
    let h2 = hc_post_all(&attn_out, h, &posts, &combs, cfg);
    let (y2, posts2, combs2) = hc_pre_all(&h2, block, &layer.hc_ffn, cfg);
    let y2n = rms_norm(&y2, block, dim, &layer.ffn_norm, cfg.norm_eps);
    // stages are never hash-routed; ids unused (bias path)
    let ids = vec![0i64; block];
    let (ffn_out, rw, ri) = moe_forward(&layer.moe, &y2n, block, &ids, cfg);
    trace.ffn_out = ffn_out.clone();
    trace.router_w = rw;
    trace.router_idx = ri;
    (hc_post_all(&ffn_out, &h2, &posts2, &combs2, cfg), trace)
}

/// argmax, first-max-index (matches torch.argmax tie behavior claim; f32 logits
/// ties are measure-zero). Used for the temperature=0 oracle sampler.
pub fn argmax_first(x: &[f32]) -> i64 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as i64
}

// ---------------------------------------------------------------------------
// Piece outputs (the replay driver writes these as .npy under the same keys)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct PieceOutputs {
    pub f32_arrays: Vec<(String, Vec<usize>, Vec<f32>)>,
    pub i64_arrays: Vec<(String, Vec<usize>, Vec<i64>)>,
}

impl PieceOutputs {
    pub fn push_f32(&mut self, key: &str, shape: &[usize], data: Vec<f32>) {
        self.f32_arrays.push((key.to_string(), shape.to_vec(), data));
    }
    pub fn push_i64(&mut self, key: &str, shape: &[usize], data: Vec<i64>) {
        self.i64_arrays.push((key.to_string(), shape.to_vec(), data));
    }
}

/// Layer piece (swa/csa/hca): prefill over pre.x [S, hc*dim] + D decode steps
/// over dec_xs[i] [1, hc*dim]; ids [S+D]. Emits keys exactly like the oracle
/// npz (pre.*, dec{i}.*, kv_cache). Decode inputs are independent synthetic
/// per step (the replay contract).
pub fn run_layer_piece(
    cfg: &Dsv4Config,
    layer: &CpuLayer,
    ids: &[i64],
    pre_x: &[f32],
    dec_xs: &[Vec<f32>],
    max_seq_len: usize,
) -> PieceOutputs {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let s = pre_x.len() / (hc * dim);
    let d = dec_xs.len();
    assert_eq!(ids.len(), s + d);
    // RoPE table for this layer kind; positions cover prefill + decode + 1
    let positions = s + d + 8;
    let rope = layer_rope_table(cfg, layer.kind, positions);
    let had = if matches!(layer.kind, LayerKind::Csa) {
        Some(hadamard_scaled(cfg.index_head_dim))
    } else {
        None
    };
    let mut st = AttnState::new(cfg, &layer.attn, max_seq_len);
    let mut out = PieceOutputs::default();
    // prefill
    let (y, tr) = block_forward(layer, &mut st, pre_x, s, 0, &ids[..s], &rope, had.as_deref(), cfg);
    out.push_f32("pre.y", &[s, hc, dim], y);
    out.push_f32("pre.attn_out", &[s, dim], tr.attn_out);
    out.push_f32("pre.ffn_out", &[s, dim], tr.ffn_out);
    out.push_f32("pre.router_w", &[s, cfg.n_activated_experts], tr.router_w);
    out.push_i64("pre.router_idx", &[s, cfg.n_activated_experts], tr.router_idx);
    out.push_i64("pre.topk_idx", &[1, s, tr.topk_t], tr.topk_idx);
    // decode steps
    for (i, dx) in dec_xs.iter().enumerate() {
        assert_eq!(dx.len(), hc * dim);
        let sp = s + i;
        let (y, tr) = block_forward(layer, &mut st, dx, 1, sp, &ids[sp..sp + 1], &rope, had.as_deref(), cfg);
        out.push_f32(&format!("dec{i}.y"), &[1, hc, dim], y);
        out.push_f32(&format!("dec{i}.attn_out"), &[1, dim], tr.attn_out);
        // oracle saves dec router_w squeezed to [6], router_idx as [1,6]
        out.push_f32(&format!("dec{i}.router_w"), &[cfg.n_activated_experts], tr.router_w);
        out.push_i64(&format!("dec{i}.router_idx"), &[1, cfg.n_activated_experts], tr.router_idx);
        out.push_i64(&format!("dec{i}.topk_idx"), &[1, 1, tr.topk_t], tr.topk_idx);
    }
    out.push_f32("kv_cache", &[st.cache_rows, cfg.head_dim], st.kv_cache);
    out
}

/// RoPE table per layer kind (§B.1.3): SWA plain θ=10000 (YaRN force-disabled);
/// CSA/HCA YaRN θ=160000, original_seq_len=65536, factor 16, β 32/1.
pub fn layer_rope_table(cfg: &Dsv4Config, kind: LayerKind, positions: usize) -> RopeTable {
    match kind {
        LayerKind::Swa => rope_table(cfg.rope_head_dim, positions, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow),
        _ => rope_table(
            cfg.rope_head_dim,
            positions,
            cfg.original_seq_len,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        ),
    }
}

/// DSpark piece: warm all 3 stages' rings from warm_main_hidden [Sw, 3*dim],
/// then the 5-position draft at start_pos = Sw. Emits draft.h_in, draft.h0..2,
/// draft.output_ids, draft.logits, draft.confidence.
pub fn run_dspark_piece(
    cfg: &Dsv4Config,
    stages: &[CpuLayer; 3],
    embed: &[f32], // [vocab, dim] bf16-valued
    head: &[f32],  // [vocab, dim] fp32
    warm_main_hidden: &[f32],
    draft_main_hidden: &[f32],
    real_token: i64,
) -> PieceOutputs {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let block = cfg.dspark_block_size;
    let sw = warm_main_hidden.len() / (3 * dim);
    let positions = sw + block + 8;
    let rope = layer_rope_table(cfg, LayerKind::Swa, positions);
    let mut rings: Vec<Vec<f32>> = (0..3).map(|_| vec![0.0f32; cfg.window_size * cfg.head_dim]).collect();
    // warm: main_x for ALL trunk positions; all 3 stages ring-write
    let st0 = &stages[0];
    let main_proj = st0.main_proj.as_ref().unwrap();
    let main_norm = st0.main_norm.as_ref().unwrap();
    let mxw = quant_gemm(warm_main_hidden, sw, 3 * dim, main_proj, dim, 128);
    let main_x_w = rms_norm(&mxw, sw, dim, main_norm, cfg.norm_eps);
    let mut out = PieceOutputs::default();
    {
        // h is computed by forward_embed but unused by the warm pass
        let draft_ids: Vec<i64> = std::iter::once(real_token)
            .chain(std::iter::repeat(cfg.dspark_noise_token_id as i64).take(block - 1))
            .collect();
        let h = dspark_embed(embed, &draft_ids, hc, dim);
        for (i, st) in stages.iter().enumerate() {
            let _ = dspark_block_forward(st, &mut rings[i], &h, block, 0, &main_x_w, &rope, cfg);
        }
    }
    // draft pass at start_pos = sw
    let mxd = quant_gemm(draft_main_hidden, 1, 3 * dim, main_proj, dim, 128);
    let main_x_d = rms_norm(&mxd, 1, dim, main_norm, cfg.norm_eps);
    let draft_ids: Vec<i64> = std::iter::once(real_token)
        .chain(std::iter::repeat(cfg.dspark_noise_token_id as i64).take(block - 1))
        .collect();
    let mut h = dspark_embed(embed, &draft_ids, hc, dim);
    out.push_f32("draft.h_in", &[block, hc, dim], h.clone());
    for (i, st) in stages.iter().enumerate() {
        h = dspark_block_forward(st, &mut rings[i], &h, block, sw, &main_x_d, &rope, cfg);
        out.push_f32(&format!("draft.h{i}"), &[block, hc, dim], h.clone());
    }
    // forward_head (mtp.2): hc_head -> norm -> LM head (fp32) -> Markov chain -> confidence
    let st2 = &stages[2];
    let hch = st2.hc_head.as_ref().unwrap();
    let mut collapse = vec![0.0f32; block * dim];
    for i in 0..block {
        let xf = &h[i * hc * dim..(i + 1) * hc * dim];
        let y = hc_head_token(xf, hc, dim, &hch.hc_fn, &hch.hc_base, hch.hc_scale, cfg.norm_eps, cfg.hc_eps);
        collapse[i * dim..(i + 1) * dim].copy_from_slice(&y);
    }
    let yn = rms_norm(&collapse, block, dim, st2.norm.as_ref().unwrap(), cfg.norm_eps);
    let vocab = cfg.vocab_size;
    let mut logits = gemm_f32(&yn, block, dim, head, vocab); // fp32 [block, vocab]
    // sequential Markov bigram chain over output_ids [6]
    let mw1 = st2.markov_w1.as_ref().unwrap();
    let mw2 = st2.markov_w2.as_ref().unwrap();
    let rank = cfg.dspark_markov_rank;
    let mut output_ids = vec![0i64; block + 1];
    output_ids[0] = real_token;
    let mut markov_embeds: Vec<Vec<f32>> = Vec::with_capacity(block);
    for i in 0..block {
        let id = output_ids[i] as usize;
        let e: Vec<f32> = mw1[id * rank..(id + 1) * rank].to_vec(); // bf16-valued embedding
        let bias = gemm_f32(&e, 1, rank, mw2, vocab); // fp32 bigram bias
        for v in 0..vocab {
            logits[i * vocab + v] += bias[v];
        }
        markov_embeds.push(e);
        output_ids[i + 1] = argmax_first(&logits[i * vocab..(i + 1) * vocab]);
    }
    // confidence head: cat([collapse, markov_embeds], -1) -> fp32 Linear, raw score
    let conf_w = st2.confidence.as_ref().unwrap();
    let mut confidence = vec![0.0f32; block];
    for i in 0..block {
        let mut catv = Vec::with_capacity(dim + rank);
        catv.extend_from_slice(&collapse[i * dim..(i + 1) * dim]);
        catv.extend_from_slice(&markov_embeds[i]);
        confidence[i] = dot_tree(&catv, conf_w);
    }
    out.push_i64("draft.output_ids", &[block + 1], output_ids);
    out.push_f32("draft.logits", &[block, vocab], logits);
    out.push_f32("draft.confidence", &[block], confidence);
    out
}

/// Embed draft ids and replicate to hc streams (forward_embed).
fn dspark_embed(embed: &[f32], ids: &[i64], hc: usize, dim: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; ids.len() * hc * dim];
    for (i, &id) in ids.iter().enumerate() {
        let row = &embed[id as usize * dim..(id as usize + 1) * dim];
        for s in 0..hc {
            h[(i * hc + s) * dim..(i * hc + s + 1) * dim].copy_from_slice(row);
        }
    }
    round_bf16(&mut h); // embedding output is bf16
    h
}

/// Head piece (trunk collapse): hc_head (sigmoid-only) -> RMSNorm -> fp32 LM
/// head on the LAST row. x [S, hc*dim]. Emits collapsed [S, dim], logits [vocab].
pub fn run_head_piece(
    cfg: &Dsv4Config,
    hc_head: &HcHeadParams,
    norm: &[f32],
    head: &[f32],
    x: &[f32],
) -> PieceOutputs {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let s = x.len() / (hc * dim);
    let mut collapse = vec![0.0f32; s * dim];
    for i in 0..s {
        let xf = &x[i * hc * dim..(i + 1) * hc * dim];
        let y = hc_head_token(xf, hc, dim, &hc_head.hc_fn, &hc_head.hc_base, hc_head.hc_scale, cfg.norm_eps, cfg.hc_eps);
        collapse[i * dim..(i + 1) * dim].copy_from_slice(&y);
    }
    let yn = rms_norm(&collapse, s, dim, norm, cfg.norm_eps);
    let logits = gemm_f32(&yn[(s - 1) * dim..s * dim], 1, dim, head, cfg.vocab_size);
    let mut out = PieceOutputs::default();
    out.push_f32("collapsed", &[s, dim], collapse);
    out.push_f32("logits", &[cfg.vocab_size], logits);
    out
}

// ---------------------------------------------------------------------------
// Weight extraction (Lane A Dsv4Layer -> CpuLayer; replay path only — unit
// tests construct CpuLayer directly)
// ---------------------------------------------------------------------------

/// Build a trunk CpuLayer from Lane A's strict-loaded layer.
pub fn cpu_layer_from_dsv4(layer: Dsv4Layer, cfg: &Dsv4Config, kind: LayerKind) -> Result<CpuLayer> {
    let Dsv4Layer { mut tensors, experts_w1, experts_w2, experts_w3 } = layer;
    cpu_layer_core(&mut tensors, (experts_w1, experts_w2, experts_w3), cfg, kind)
}

/// Shared extraction core: pulls every tensor a Block needs out of `map`
/// (stripped keys, §A), leaving any extras (DSpark stage-specifics) behind.
fn cpu_layer_core(
    mut map: &mut HashMap<String, HostTensor>,
    experts: (Vec<Nvfp4Tensor>, Vec<Nvfp4Tensor>, Vec<Nvfp4Tensor>),
    cfg: &Dsv4Config,
    kind: LayerKind,
) -> Result<CpuLayer> {
    let dim = cfg.dim;
    let qlr = cfg.q_lora_rank;
    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let g = cfg.o_groups;
    let r = cfg.o_lora_rank;
    let ne = cfg.n_routed_experts;
    let inter = cfg.moe_inter_dim;
    let compressor = match kind {
        LayerKind::Swa => None,
        LayerKind::Csa => Some(CompressorWeights {
            wkv: take_f32(&mut map, "attn.compressor.wkv.weight", 2 * hd * dim)?,
            wgate: take_f32(&mut map, "attn.compressor.wgate.weight", 2 * hd * dim)?,
            norm: take_f32(&mut map, "attn.compressor.norm.weight", hd)?,
            ape: take_f32(&mut map, "attn.compressor.ape", 4 * 2 * hd)?,
            ratio: 4,
            head_dim: hd,
            rope_dim: cfg.rope_head_dim,
            overlap: true,
            rotate: false,
            sim_group: 64,
            dim,
        }),
        LayerKind::Hca => Some(CompressorWeights {
            wkv: take_f32(&mut map, "attn.compressor.wkv.weight", hd * dim)?,
            wgate: take_f32(&mut map, "attn.compressor.wgate.weight", hd * dim)?,
            norm: take_f32(&mut map, "attn.compressor.norm.weight", hd)?,
            ape: take_f32(&mut map, "attn.compressor.ape", 128 * hd)?,
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
            wq_b: take_f32(&mut map, "attn.indexer.wq_b.weight", cfg.index_n_heads * ihd * qlr)?,
            weights_proj: take_bf16_as_f32(&mut map, "attn.indexer.weights_proj.weight", cfg.index_n_heads * dim)?,
            compressor: CompressorWeights {
                wkv: take_f32(&mut map, "attn.indexer.compressor.wkv.weight", 2 * ihd * dim)?,
                wgate: take_f32(&mut map, "attn.indexer.compressor.wgate.weight", 2 * ihd * dim)?,
                norm: take_f32(&mut map, "attn.indexer.compressor.norm.weight", ihd)?,
                ape: take_f32(&mut map, "attn.indexer.compressor.ape", 4 * 2 * ihd)?,
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
    let attn = AttnWeights {
        wq_a: take_f32(&mut map, "attn.wq_a.weight", qlr * dim)?,
        q_norm: take_f32(&mut map, "attn.q_norm.weight", qlr)?,
        wq_b: take_f32(&mut map, "attn.wq_b.weight", nh * hd * qlr)?,
        wkv: take_f32(&mut map, "attn.wkv.weight", hd * dim)?,
        kv_norm: take_f32(&mut map, "attn.kv_norm.weight", hd)?,
        sink: take_f32(&mut map, "attn.attn_sink", nh)?,
        wo_a: take_bf16_as_f32(&mut map, "attn.wo_a.weight", g * r * nh * hd / g)?,
        wo_b: take_f32(&mut map, "attn.wo_b.weight", dim * g * r)?,
        compressor,
        indexer,
        kind,
    };
    let hc = |prefix: &str, map: &mut HashMap<String, HostTensor>| -> Result<HcParams> {
        Ok(HcParams {
            hc_fn: take_f32(map, &format!("{prefix}_fn"), 24 * cfg.hc_mult * dim)?,
            hc_base: take_f32(map, &format!("{prefix}_base"), 24)?,
            hc_scale: {
                let v = take_f32(map, &format!("{prefix}_scale"), 3)?;
                [v[0], v[1], v[2]]
            },
        })
    };
    let hc_attn = hc("hc_attn", &mut map)?;
    let hc_ffn = hc("hc_ffn", &mut map)?;
    let moe = MoeWeights {
        gate_w: take_bf16_as_f32(&mut map, "ffn.gate.weight", ne * dim)?,
        gate_bias: opt_f32(&mut map, "ffn.gate.bias", ne)?,
        tid2eid: {
            let key = "ffn.gate.tid2eid";
            if map.contains_key(key) {
                Some(take_i32(&mut map, key, cfg.vocab_size * cfg.n_activated_experts)?)
            } else {
                None
            }
        },
        shared: ExpertF32 {
            w1: take_f32(&mut map, "ffn.shared_experts.w1.weight", inter * dim)?,
            w2: take_f32(&mut map, "ffn.shared_experts.w2.weight", dim * inter)?,
            w3: take_f32(&mut map, "ffn.shared_experts.w3.weight", inter * dim)?,
        },
        experts: ExpertBank::from_nvfp4(experts.0, experts.1, experts.2),
    };
    Ok(CpuLayer {
        kind,
        attn,
        attn_norm: take_f32(&mut map, "attn_norm.weight", dim)?,
        ffn_norm: take_f32(&mut map, "ffn_norm.weight", dim)?,
        hc_attn,
        hc_ffn,
        moe,
        main_proj: None,
        main_norm: None,
        norm: None,
        hc_head: None,
        markov_w1: None,
        markov_w2: None,
        confidence: None,
    })
}

/// Build a DSpark stage CpuLayer from Lane A's `load_mtp_stage` output
/// (mtp.{S}.* — a full Block minus compressor/indexer, plus stage extras).
pub fn cpu_stage_from_dsv4(layer: Dsv4Layer, cfg: &Dsv4Config, stage: usize) -> Result<CpuLayer> {
    let Dsv4Layer { mut tensors, experts_w1, experts_w2, experts_w3 } = layer;
    let mut base = cpu_layer_core(&mut tensors, (experts_w1, experts_w2, experts_w3), cfg, LayerKind::Swa)?;
    let dim = cfg.dim;
    if stage == 0 {
        base.main_proj = Some(take_f32(&mut tensors, "main_proj.weight", dim * 3 * dim)?);
        base.main_norm = Some(take_f32(&mut tensors, "main_norm.weight", dim)?);
    }
    if stage == cfg.n_mtp_layers - 1 {
        let rank = cfg.dspark_markov_rank;
        base.norm = Some(take_f32(&mut tensors, "norm.weight", dim)?);
        base.hc_head = Some(HcHeadParams {
            hc_fn: take_f32(&mut tensors, "hc_head_fn", cfg.hc_mult * cfg.hc_mult * dim)?,
            hc_base: take_f32(&mut tensors, "hc_head_base", cfg.hc_mult)?,
            hc_scale: take_f32(&mut tensors, "hc_head_scale", 1)?[0],
        });
        base.markov_w1 = Some(take_bf16_as_f32(&mut tensors, "markov_head.markov_w1.weight", cfg.vocab_size * rank)?);
        base.markov_w2 = Some(take_f32(&mut tensors, "markov_head.markov_w2.weight", cfg.vocab_size * rank)?);
        base.confidence = Some(take_f32(&mut tensors, "confidence_head.proj.weight", dim + rank)?);
    }
    Ok(base)
}

/// Trunk top level (embed/norm/head/hc_head) from `load_trunk_top`.
pub struct TrunkTop {
    pub embed: Vec<f32>, // [vocab, dim] bf16-valued
    pub norm: Vec<f32>,
    pub head: Vec<f32>, // fp32
    pub hc_head: HcHeadParams,
}

pub fn trunk_top_from(mut map: HashMap<String, HostTensor>, cfg: &Dsv4Config) -> Result<TrunkTop> {
    let dim = cfg.dim;
    let vocab = cfg.vocab_size;
    Ok(TrunkTop {
        embed: take_bf16_as_f32(&mut map, "embed.weight", vocab * dim)?,
        norm: take_f32(&mut map, "norm.weight", dim)?,
        head: take_f32(&mut map, "head.weight", vocab * dim)?,
        hc_head: HcHeadParams {
            hc_fn: take_f32(&mut map, "hc_head_fn", cfg.hc_mult * cfg.hc_mult * dim)?,
            hc_base: take_f32(&mut map, "hc_head_base", cfg.hc_mult)?,
            hc_scale: take_f32(&mut map, "hc_head_scale", 1)?[0],
        },
    })
}

// ---------------------------------------------------------------------------
// R5 dequant oracle helpers (single source of truth for the packed-cache gates +
// the future packed gather oracle). These mirror the device decoders EXACTLY
// (gpu_dsv4.cu §C.2 / gpu_dsv4_comp.cu §5): dequant(pack(v)) == the QAT-simmed
// bf16 value bit-for-bit — that identity is the whole R5 premise.
// ---------------------------------------------------------------------------

/// FP4 E2M1 decode (device: `dsv4_fp4_to_f32`). sign<<3 | exp<<1 | man.
pub fn fp4_e2m1_to_f32(c: u8) -> f32 {
    let e = (c >> 1) & 3;
    let m = c & 1;
    let mag = if e == 0 {
        if m == 1 { 0.5 } else { 0.0 }
    } else {
        (1.0 + 0.5 * m as f32) * (1u32 << (e - 1)) as f32
    };
    if c & 8 != 0 { -mag } else { mag }
}

/// FP8 E4M3 decode (device: `__nv_cvt_fp8_to_halfraw`). sign<<7 | exp(4,bias 7)<<3 | man(3).
pub fn fp8_e4m3_to_f32(c: u8) -> f32 {
    let sign = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((c >> 3) & 0xF) as i32;
    let m = (c & 0x7) as f32;
    let mag = if e == 0 { (m / 8.0) * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    sign * mag
}

/// UE8M0 scale byte → f32 (e8m0 byte = biased pow2 exponent; b >= 1 by the amax floors).
pub fn e8m0_to_f32(b: u8) -> f32 {
    ((b as i32 - 127) as f32).exp2()
}

/// Unpack one FP4-g32 row (the indexer cache form): n dims from n/2 code bytes + n/32
/// scale bytes — returns the bf16 values the QAT-sim cache holds (bf16(fp4*sc)).
pub fn unpack_fp4_g32_row(codes: &[u8], scales: &[u8], n: usize) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(n);
    for g in 0..(n / 32) {
        let sc = e8m0_to_f32(scales[g]);
        for k in 0..32 {
            let byte = codes[g * 16 + (k >> 1)];
            let nib = if k & 1 == 0 { byte & 0xF } else { byte >> 4 };
            out.push(half::bf16::from_f32(fp4_e2m1_to_f32(nib) * sc));
        }
    }
    out
}

/// Unpack one FP8-g64 span (the attn cache's nope form): n dims from n code bytes +
/// n/64 scale bytes — bf16(fp8*sc).
pub fn unpack_fp8_g64_span(codes: &[u8], scales: &[u8], n: usize) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(n);
    for g in 0..(n / 64) {
        let sc = e8m0_to_f32(scales[g]);
        for k in 0..64 {
            out.push(half::bf16::from_f32(fp8_e4m3_to_f32(codes[g * 64 + k]) * sc));
        }
    }
    out
}
