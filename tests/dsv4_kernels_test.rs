//! Gate G1 (LANE D): CUDA kernel prototypes for DeepSeek-V4-Flash-DSpark vs CPU references.
//!
//! Bit-exact asserted for: dsv4_topk, dsv4_act_quant, dsv4_fp4_act_quant (all ops exactly
//! replicable on CPU: max, fp32 mul/div RN, IEEE bit trick, RNE casts). hc_split_sinkhorn is
//! asserted bit-exact against the *tilelang ground-truth vectors* (DSV4_TL_VECTORS env-gated);
//! vs the in-test CPU ref it is ulp-bounded because libdevice expf ≠ glibc expf by <=1 ulp.
//! <=1e-3 asserted for dsv4_fwht_rotate (bf16) and dsv4_gather_attn (bf16 P·V rounding effects).
//!
//! Run: cargo test --release --test dsv4_kernels_test -- --nocapture
//! Ground-truth vectors: DSV4_TL_VECTORS=/tmp/dsv4_tl_vectors cargo test --release \
//!     --test dsv4_kernels_test ground_truth -- --nocapture

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// exact conversion helpers (RNE everywhere, matching cvt.rn)
// ---------------------------------------------------------------------------

pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7FC0; // quiet NaN
    }
    let lsb = (b >> 16) & 1;
    let rounded = b.wrapping_add(0x7FFF + lsb as u32);
    (rounded >> 16) as u16
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// All 128 non-negative FP8-E4M3 codes decoded to f32 (index = code).
fn fp8_pos_table() -> [f32; 128] {
    let mut t = [0.0f32; 128];
    for c in 0..128usize {
        let e = (c >> 3) & 0xF;
        let m = (c & 7) as f32;
        t[c] = if e == 0 {
            m * 2f32.powi(-9)
        } else {
            (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7)
        };
    }
    t
}

/// fp32 -> FP8-E4M3, round-to-nearest-even (= cvt.rn.satfinite.e4m3 on pre-clamped input).
fn f32_to_fp8_rne(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let t = fp8_pos_table();
    let neg = x.is_sign_negative();
    let ax = x.abs().min(448.0); // satfinite (caller pre-clamps anyway)
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (c, &tv) in t.iter().enumerate() {
        let d = (ax - tv).abs();
        if d < bd {
            best = c;
            bd = d;
        } else if d == bd && (c & 1) == 0 {
            best = c; // exact tie -> even code
        }
    }
    (if neg { 0x80 } else { 0 }) | best as u8
}

/// fp32 -> FP4-E2M1 magnitude code (0..7), RNE (= cvt.rn.satfinite.e2m1 on pre-clamped input).
fn f32_to_fp4_rne(x: f32) -> u8 {
    const T: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let neg = x.is_sign_negative();
    let ax = x.abs().min(6.0);
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (c, &tv) in T.iter().enumerate() {
        let d = (ax - tv).abs();
        if d < bd {
            best = c;
            bd = d;
        } else if d == bd && (c & 1) == 0 {
            best = c; // exact tie -> even code
        }
    }
    (if neg { 8 } else { 0 }) | best as u8
}

fn fp4_decode(c: u8) -> f32 {
    const T: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = T[(c & 7) as usize];
    if c & 8 != 0 {
        -v
    } else {
        v
    }
}

fn fp8_decode(c: u8) -> f32 {
    let t = fp8_pos_table();
    let v = t[(c & 0x7F) as usize];
    if c & 0x80 != 0 {
        -v
    } else {
        v
    }
}

/// UE8M0 scale: s = 2^ceil(log2(amax*inv)) via the IEEE bit trick (kernel.py:22-37).
/// Returns (s, e8m0_byte). s is an exact power of two so the e8m0 byte is its exponent.
fn round_scale_pow2(amax: f32, inv: f32) -> (f32, u8) {
    let v = amax * inv;
    let b = v.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + if b & 0x7FFFFF != 0 { 1 } else { 0 };
    let s = f32::from_bits(((e + 127) as u32) << 23);
    (s, (e + 127) as u8)
}

// ---------------------------------------------------------------------------
// CPU references (kernel.py semantics, exact op order)
// ---------------------------------------------------------------------------

/// act_quant (FP8 UE8M0): returns (codes, scale bytes, sim round-trip bf16 bits).
fn cpu_act_quant(x: &[u16], n: usize, group: usize) -> (Vec<u8>, Vec<u8>, Vec<u16>) {
    let rows = x.len() / n;
    let groups_per_row = n / group;
    let mut y = vec![0u8; x.len()];
    let mut s = vec![0u8; rows * groups_per_row];
    let mut sim = vec![0u16; x.len()];
    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * n + g * group;
            let mut amax = 0.0f32;
            for i in 0..group {
                amax = amax.max(bf16_bits_to_f32(x[base + i]).abs());
            }
            amax = amax.max(1e-4);
            let (sc, sb) = round_scale_pow2(amax, 1.0 / 448.0);
            s[r * groups_per_row + g] = sb;
            for i in 0..group {
                let v = bf16_bits_to_f32(x[base + i]);
                let q = (v / sc).clamp(-448.0, 448.0);
                let c = f32_to_fp8_rne(q);
                y[base + i] = c;
                sim[base + i] = f32_to_bf16_bits(fp8_decode(c) * sc);
            }
        }
    }
    (y, s, sim)
}

/// fp4_act_quant (group 32, UE8M0): packed y (low nibble = even K), scale bytes, sim.
fn cpu_fp4_act_quant(x: &[u16], n: usize) -> (Vec<u8>, Vec<u8>, Vec<u16>) {
    let rows = x.len() / n;
    let groups_per_row = n / 32;
    let mut y = vec![0u8; rows * (n / 2)];
    let mut s = vec![0u8; rows * groups_per_row];
    let mut sim = vec![0u16; x.len()];
    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * n + g * 32;
            let mut amax = 0.0f32;
            for i in 0..32 {
                amax = amax.max(bf16_bits_to_f32(x[base + i]).abs());
            }
            amax = amax.max(6.0 * 2f32.powi(-126));
            let (sc, sb) = round_scale_pow2(amax, 1.0 / 6.0);
            s[r * groups_per_row + g] = sb;
            for i in 0..32 {
                let v = bf16_bits_to_f32(x[base + i]);
                let q = (v / sc).clamp(-6.0, 6.0);
                let c = f32_to_fp4_rne(q);
                if i % 2 == 0 {
                    y[r * (n / 2) + g * 16 + i / 2] |= c;
                } else {
                    y[r * (n / 2) + g * 16 + i / 2] |= c << 4;
                }
                sim[base + i] = f32_to_bf16_bits(fp4_decode(c) * sc);
            }
        }
    }
    (y, s, sim)
}

/// hc_split_sinkhorn (kernel.py:371-438 exact sequence; butterfly tree sums (e0+e2)+(e1+e3)).
fn cpu_hc_split_sinkhorn(
    mixes: &[f32],
    scale: &[f32; 3],
    base: &[f32; 24],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let eps = 1e-6f32;
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let n = mixes.len() / 24;
    let mut pre = vec![0f32; n * 4];
    let mut post = vec![0f32; n * 4];
    let mut comb = vec![0f32; n * 16];
    for i in 0..n {
        let m = &mixes[i * 24..i * 24 + 24];
        for j in 0..4 {
            pre[i * 4 + j] = sig(m[j].mul_add(scale[0], base[j])) + eps;
        }
        for j in 0..4 {
            post[i * 4 + j] = 2.0 * sig(m[4 + j].mul_add(scale[1], base[4 + j]));
        }
        let mut c = [[0f32; 4]; 4];
        for j in 0..4 {
            for k in 0..4 {
                c[j][k] = m[8 + 4 * j + k].mul_add(scale[2], base[8 + 4 * j + k]);
            }
        }
        // row softmax + eps
        for j in 0..4 {
            let mx = c[j][0].max(c[j][1]).max(c[j][2]).max(c[j][3]);
            for k in 0..4 {
                c[j][k] = (c[j][k] - mx).exp();
            }
            let rs = (c[j][0] + c[j][2]) + (c[j][1] + c[j][3]);
            for k in 0..4 {
                c[j][k] = c[j][k] / rs + eps;
            }
        }
        // col norm
        for k in 0..4 {
            let cs = (c[0][k] + c[2][k]) + (c[1][k] + c[3][k]) + eps;
            for j in 0..4 {
                c[j][k] /= cs;
            }
        }
        // 19x (row, col)
        for _ in 0..19 {
            for j in 0..4 {
                let rs = (c[j][0] + c[j][2]) + (c[j][1] + c[j][3]) + eps;
                for k in 0..4 {
                    c[j][k] /= rs;
                }
            }
            for k in 0..4 {
                let cs = (c[0][k] + c[2][k]) + (c[1][k] + c[3][k]) + eps;
                for j in 0..4 {
                    c[j][k] /= cs;
                }
            }
        }
        for j in 0..4 {
            for k in 0..4 {
                comb[(i * 4 + j) * 4 + k] = c[j][k];
            }
        }
    }
    (pre, post, comb)
}

/// CPU top-k: total order (value desc, index asc), exact same comparator as the kernel.
fn cpu_topk(scores: &[f32], rows: usize, t: usize, k: usize) -> Vec<i32> {
    let mut out = vec![0i32; rows * k];
    for r in 0..rows {
        let row = &scores[r * t..r * t + t];
        let mut selected = vec![false; t];
        for round in 0..k {
            let mut bv = f32::NEG_INFINITY;
            let mut bi = i32::MAX;
            for (i, &v) in row.iter().enumerate() {
                if selected[i] {
                    continue;
                }
                if v > bv || (v == bv && (i as i32) < bi) {
                    bv = v;
                    bi = i as i32;
                }
            }
            if bi == i32::MAX {
                out[r * k + round] = -1;
            } else {
                out[r * k + round] = bi;
                selected[bi as usize] = true;
            }
        }
    }
    out
}

/// FWHT butterfly identical to the GPU kernel (fp32, ascending h), bf16 in/out.
fn cpu_fwht_row(x: &[u16]) -> Vec<u16> {
    let scale = f32::from_bits(0x3DB504F3); // 0x1.6a09e6p-4
    let mut v: Vec<f32> = x.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let mut h = 1usize;
    while h < 128 {
        let mut i = 0;
        while i < 128 {
            for j in i..i + h {
                let a = v[j];
                let b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h <<= 1;
    }
    v.iter().map(|&e| f32_to_bf16_bits(e * scale)).collect()
}

/// sparse_attn emu port (dsv4_ref.py): fp32 scores, single global max, bf16-rounded
/// probabilities for the P·V numerator only, denominator-only sink, -1 masking.
#[allow(clippy::too_many_arguments)]
fn cpu_gather_attn(
    q: &[u16],  // [b,m,64,512] bf16 bits
    kv: &[u16], // [b,n,512] bf16 bits
    sink: &[f32],
    idx: &[i32], // [b,m,topk]
    b: usize,
    m: usize,
    n: usize,
    topk: usize,
    scale: f32,
) -> Vec<u16> {
    let h = 64usize;
    let d = 512usize;
    let mut o = vec![0u16; b * m * h * d];
    for bi in 0..b {
        for mi in 0..m {
            let ids: Vec<i32> = idx[(bi * m + mi) * topk..(bi * m + mi + 1) * topk].to_vec();
            // gathered rows (zero at masked slots)
            let rows: Vec<Vec<f32>> = ids
                .iter()
                .map(|&ix| {
                    if ix == -1 {
                        vec![0.0f32; d]
                    } else {
                        let b0 = (bi * n + ix as usize) * d;
                        kv[b0..b0 + d].iter().map(|&v| bf16_bits_to_f32(v)).collect()
                    }
                })
                .collect();
            let valid: Vec<bool> = ids.iter().map(|&ix| ix != -1).collect();
            for hi in 0..h {
                let qb = ((bi * m + mi) * h + hi) * d;
                let mut scores: Vec<f32> = (0..topk)
                    .map(|j| {
                        let mut acc = 0.0f32;
                        for dd in 0..d {
                            acc += bf16_bits_to_f32(q[qb + dd]) * rows[j][dd];
                        }
                        acc * scale
                    })
                    .collect();
                for j in 0..topk {
                    if !valid[j] {
                        scores[j] = f32::NEG_INFINITY;
                    }
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let p: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
                let denom: f32 = p.iter().sum::<f32>() + (sink[hi] - mx).exp();
                let mut acc = vec![0f32; d];
                for j in 0..topk {
                    let pb = bf16_bits_to_f32(f32_to_bf16_bits(p[j]));
                    for dd in 0..d {
                        acc[dd] += pb * rows[j][dd];
                    }
                }
                let ob = ((bi * m + mi) * h + hi) * d;
                for dd in 0..d {
                    o[ob + dd] = f32_to_bf16_bits(acc[dd] / denom);
                }
            }
        }
    }
    o
}


/// One bf16 ulp at magnitude v (v need not be bf16-representable).
fn bf16_ulp(v: f32) -> f32 {
    let a = v.abs();
    if a == 0.0 || !a.is_finite() {
        return 2f32.powi(-133);
    }
    let e = a.log2().floor() as i32;
    2f32.powi(e - 7)
}

/// gather_attn tolerance: the emu and the kernel accumulate fp32 in different orders,
/// so pre-rounding values sit ~1e-6 relative apart and outputs can differ by ONE bf16
/// ulp near rounding boundaries. Contract per element: |d| <= max(1e-3, 1 bf16 ulp(e)),
/// i.e. the task's <=1e-3 holds wherever 1 ulp <= 1e-3 (|e| <= 0.125), and where 1 ulp
/// exceeds 1e-3 the bound is exactly one ulp. Anything >= 2 ulp fails; e=0 gets the
/// 1e-3 absolute floor.
fn assert_bf16_close(got: &[u16], exp: &[u16], tag: &str) {
    assert_eq!(got.len(), exp.len());
    let mut max_d = 0.0f32;
    let mut n_diff = 0usize;
    let mut max_ulp_big = 0.0f32; // ulp distance where 1 ulp > 1e-3
    for i in 0..got.len() {
        let g = bf16_bits_to_f32(got[i]);
        let e = bf16_bits_to_f32(exp[i]);
        let d = (g - e).abs();
        max_d = max_d.max(d);
        if d > 0.0 {
            n_diff += 1;
        }
        let u = bf16_ulp(e);
        let tol = f32::max(1e-3, u);
        assert!(d <= tol, "{tag}[{i}]: |{g} - {e}| = {d} exceeds max(1e-3, 1 ulp)");
        if u > 1e-3 {
            max_ulp_big = f32::max(max_ulp_big, d / u);
        }
    }
    println!(
        "{tag}: max|d| = {max_d:.3e}, differing {n_diff}/{} (max {max_ulp_big:.2} ulp where 1 ulp > 1e-3)",
        got.len()
    );
}

// ---------------------------------------------------------------------------
// RNG (deterministic, no deps)
// ---------------------------------------------------------------------------

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
        // uniform in [-1, 1)
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn bf16(&mut self) -> u16 {
        f32_to_bf16_bits(self.f32())
    }
}

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------

fn load_dsv4(dev: &Arc<CudaDevice>) -> Vec<CudaFunction> {
    let names = [
        "dsv4_topk",
        "dsv4_fwht_rotate",
        "dsv4_act_quant_g64",
        "dsv4_act_quant_g128",
        "dsv4_act_quant_sim_g64",
        "dsv4_act_quant_sim_g128",
        "dsv4_fp4_act_quant",
        "dsv4_fp4_act_quant_sim",
        "dsv4_hc_split_sinkhorn",
        "dsv4_gather_attn",
    ];
    let ptx = Ptx::from_src(
        std::fs::read_to_string("src/ptx/gpu_dsv4.ptx").expect("src/ptx/gpu_dsv4.ptx (cargo build first)"),
    );
    dev.load_ptx(ptx, "gpu_dsv4", &names).expect("load_ptx gpu_dsv4");
    names
        .iter()
        .map(|n| dev.get_func("gpu_dsv4", n).unwrap_or_else(|| panic!("missing kernel {n}")))
        .collect()
}

fn ulp_diff(a: f32, b: f32) -> i64 {
    if a == b {
        return 0;
    }
    let ai = a.to_bits() as i32 as i64;
    let bi = b.to_bits() as i32 as i64;
    (ai - bi).abs()
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
fn test_dsv4_topk_bitexact() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_dsv4(&dev);
    let topk_fn = &f[0];

    let mut rng = XorShift(0xD5E4_2024_0001);
    // rows × T with adversarial content: exact ties, -inf, duplicates
    let (rows, t, k) = (37usize, 2053usize, 131usize);
    let mut scores: Vec<f32> = (0..rows * t).map(|_| rng.f32() * 100.0).collect();
    // plant exact ties and -inf rows
    scores[0 * t + 5] = 7.5;
    scores[0 * t + 9] = 7.5;
    scores[0 * t + 9 + 700] = 7.5;
    for i in 0..t {
        scores[1 * t + i] = f32::NEG_INFINITY;
    }
    scores[2 * t + 0] = -1.0;
    scores[2 * t + 1] = -1.0;
    let big = (2 * t..3 * t).step_by(3);
    for i in big {
        scores[i] = 42.0; // mass tie
    }

    let expected = cpu_topk(&scores, rows, t, k);

    let s_dev = dev.htod_sync_copy(&scores).unwrap();
    let mut o_dev = dev.alloc_zeros::<i32>(rows * k).unwrap();
    dev.synchronize().unwrap();
    unsafe {
        topk_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&s_dev, &mut o_dev, rows as i32, t as i32, k as i32),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    let got = dev.dtoh_sync_copy(&o_dev).unwrap();
    assert_eq!(got, expected, "dsv4_topk mismatch vs exact-comparator CPU ref");

    // batch-invariance: row 3 alone must equal row 3 in the full batch
    let row3: Vec<f32> = scores[3 * t..4 * t].to_vec();
    let s1 = dev.htod_sync_copy(&row3).unwrap();
    let mut o1 = dev.alloc_zeros::<i32>(k).unwrap();
    dev.synchronize().unwrap();
    unsafe {
        topk_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&s1, &mut o1, 1i32, t as i32, k as i32),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    let got1 = dev.dtoh_sync_copy(&o1).unwrap();
    assert_eq!(got1, expected[3 * k..4 * k].to_vec(), "batch-invariance violation");

    // k=512, T=16384 envelope check
    let (r2, t2, k2) = (4usize, 16384usize, 512usize);
    let scores2: Vec<f32> = (0..r2 * t2).map(|_| rng.f32()).collect();
    let exp2 = cpu_topk(&scores2, r2, t2, k2);
    let s2 = dev.htod_sync_copy(&scores2).unwrap();
    let mut o2 = dev.alloc_zeros::<i32>(r2 * k2).unwrap();
    dev.synchronize().unwrap();
    unsafe {
        topk_fn
            .clone()
            .launch(
                LaunchConfig {
                    grid_dim: (r2 as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&s2, &mut o2, r2 as i32, t2 as i32, k2 as i32),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    assert_eq!(dev.dtoh_sync_copy(&o2).unwrap(), exp2, "topk k=512/T=16384");
    println!("topk: bit-exact (rows={rows} T={t} k={k} + ties/-inf; k=512/T=16384; batch-invariant)");
}

#[test]
fn test_dsv4_act_quant_bitexact() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_dsv4(&dev);
    let (q64, q128, sim64, sim128) = (&f[2], &f[3], &f[4], &f[5]);

    let mut rng = XorShift(0xD5E4_2024_0002);
    let (rows, n) = (23usize, 896usize); // 7 groups of 128, 14 of 64
    let mut x: Vec<u16> = (0..rows * n).map(|_| rng.bf16()).collect();
    // adversarial: all-zero groups (amax floor), huge/tiny magnitudes, exact powers of 2
    for i in 0..64 {
        x[0 * n + i] = 0;
    }
    x[1 * n + 3] = f32_to_bf16_bits(1e-5);
    x[1 * n + 4] = f32_to_bf16_bits(300.0);
    x[2 * n + 65] = f32_to_bf16_bits(2f32.powi(-20));
    x[3 * n + 129] = f32_to_bf16_bits(-2f32.powi(10));

    for (group, qfn, sfn) in [(64usize, q64, sim64), (128usize, q128, sim128)] {
        let (ey, es, esim) = cpu_act_quant(&x, n, group);
        let x_dev = dev.htod_sync_copy(&x).unwrap();
        let mut y_dev = dev.alloc_zeros::<u8>(rows * n).unwrap();
        let mut s_dev = dev.alloc_zeros::<u8>(rows * (n / group)).unwrap();
        dev.synchronize().unwrap();
        unsafe {
            qfn.clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (((rows * (n / group) + 7) / 8) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&x_dev, &mut y_dev, &mut s_dev, rows as i32, n as i32),
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let gy = dev.dtoh_sync_copy(&y_dev).unwrap();
        let gs = dev.dtoh_sync_copy(&s_dev).unwrap();
        assert_eq!(gy, ey, "act_quant g{group} codes mismatch");
        assert_eq!(gs, es, "act_quant g{group} scales mismatch");

        // inplace sim
        let mut xs_dev = dev.htod_sync_copy(&x).unwrap();
        let mut s2_dev = dev.alloc_zeros::<u8>(rows * (n / group)).unwrap();
        dev.synchronize().unwrap();
        unsafe {
            sfn.clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (((rows * (n / group) + 7) / 8) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut xs_dev, &mut s2_dev, rows as i32, n as i32),
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let gsim = dev.dtoh_sync_copy(&xs_dev).unwrap();
        let gs2 = dev.dtoh_sync_copy(&s2_dev).unwrap();
        assert_eq!(gsim, esim, "act_quant_sim g{group} round-trip mismatch");
        assert_eq!(gs2, es, "act_quant_sim g{group} scales mismatch");
        println!("act_quant g{group}: codes+scales+sim bit-exact ({rows}x{n})");
    }
}

#[test]
fn test_dsv4_fp4_act_quant_bitexact() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_dsv4(&dev);
    let (qfn, sfn) = (&f[6], &f[7]);

    let mut rng = XorShift(0xD5E4_2024_0003);
    let (rows, n) = (17usize, 256usize);
    let mut x: Vec<u16> = (0..rows * n).map(|_| rng.bf16()).collect();
    for i in 0..32 {
        x[0 * n + i] = 0; // all-zero group -> floor
    }
    x[1 * n + 33] = f32_to_bf16_bits(2f32.powi(-127)); // below floor
    x[2 * n + 65] = f32_to_bf16_bits(5.5); // near max
    x[3 * n + 96] = f32_to_bf16_bits(-6.0);

    let (ey, es, esim) = cpu_fp4_act_quant(&x, n);
    let x_dev = dev.htod_sync_copy(&x).unwrap();
    let mut y_dev = dev.alloc_zeros::<u8>(rows * (n / 2)).unwrap();
    let mut s_dev = dev.alloc_zeros::<u8>(rows * (n / 32)).unwrap();
    dev.synchronize().unwrap();
    unsafe {
        qfn.clone()
            .launch(
                LaunchConfig {
                    grid_dim: (((rows * (n / 32) + 7) / 8) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&x_dev, &mut y_dev, &mut s_dev, rows as i32, n as i32),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    let gy = dev.dtoh_sync_copy(&y_dev).unwrap();
    let gs = dev.dtoh_sync_copy(&s_dev).unwrap();
    assert_eq!(gy, ey, "fp4_act_quant packed codes mismatch (low nibble = even K)");
    assert_eq!(gs, es, "fp4_act_quant scales mismatch");

    let mut xs_dev = dev.htod_sync_copy(&x).unwrap();
    let mut s2_dev = dev.alloc_zeros::<u8>(rows * (n / 32)).unwrap();
    dev.synchronize().unwrap();
    unsafe {
        sfn.clone()
            .launch(
                LaunchConfig {
                    grid_dim: (((rows * (n / 32) + 7) / 8) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut xs_dev, &mut s2_dev, rows as i32, n as i32),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    let gsim = dev.dtoh_sync_copy(&xs_dev).unwrap();
    let gs2 = dev.dtoh_sync_copy(&s2_dev).unwrap();
    assert_eq!(gsim, esim, "fp4_act_quant_sim round-trip mismatch");
    assert_eq!(gs2, es, "fp4_act_quant_sim scales mismatch");
    println!("fp4_act_quant: packed codes+scales+sim bit-exact ({rows}x{n})");

    // exhaustive RNE sweep: every bf16 value in a dense range through one group
    let mut sweep: Vec<u16> = Vec::new();
    for bits in (0u32..65536).step_by(61) {
        let v = bf16_bits_to_f32(bits as u16);
        if v.is_finite() && v.abs() < 100.0 {
            sweep.push(bits as u16);
        }
    }
    while sweep.len() % 32 != 0 {
        sweep.push(0);
    }
    let sn = 32;
    let srows = sweep.len() / sn;
    let (sey, ses, _) = cpu_fp4_act_quant(&sweep, sn);
    let sw_dev = dev.htod_sync_copy(&sweep).unwrap();
    let mut sy_dev = dev.alloc_zeros::<u8>(srows * (sn / 2)).unwrap();
    let mut ss_dev = dev.alloc_zeros::<u8>(srows).unwrap();
    dev.synchronize().unwrap();
    unsafe {
        qfn.clone()
            .launch(
                LaunchConfig {
                    grid_dim: (((srows + 7) / 8) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&sw_dev, &mut sy_dev, &mut ss_dev, srows as i32, sn as i32),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    assert_eq!(dev.dtoh_sync_copy(&sy_dev).unwrap(), sey, "fp4 sweep codes");
    assert_eq!(dev.dtoh_sync_copy(&ss_dev).unwrap(), ses, "fp4 sweep scales");
    println!("fp4_act_quant: exhaustive bf16 sweep bit-exact ({} values)", sweep.len());
}

#[test]
fn test_dsv4_hc_split_sinkhorn_vs_cpu() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_dsv4(&dev);
    let skfn = &f[8];

    let mut rng = XorShift(0xD5E4_2024_0004);
    let n = 1024usize;
    let mixes: Vec<f32> = (0..n * 24).map(|_| rng.f32() * 4.0).collect();
    let scale: [f32; 3] = [rng.f32(), rng.f32(), rng.f32()];
    let base: [f32; 24] = std::array::from_fn(|_| rng.f32());

    let (epre, epost, ecomb) = cpu_hc_split_sinkhorn(&mixes, &scale, &base);

    let m_dev = dev.htod_sync_copy(&mixes).unwrap();
    let s_dev = dev.htod_sync_copy(&scale).unwrap();
    let b_dev = dev.htod_sync_copy(&base).unwrap();
    let mut pre_dev = dev.alloc_zeros::<f32>(n * 4).unwrap();
    let mut post_dev = dev.alloc_zeros::<f32>(n * 4).unwrap();
    let mut comb_dev = dev.alloc_zeros::<f32>(n * 16).unwrap();
    dev.synchronize().unwrap();
    unsafe {
        skfn.clone()
            .launch(
                LaunchConfig {
                    grid_dim: (((n + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &m_dev, &s_dev, &b_dev, &mut pre_dev, &mut post_dev, &mut comb_dev, n as i32,
                ),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    let gpre = dev.dtoh_sync_copy(&pre_dev).unwrap();
    let gpost = dev.dtoh_sync_copy(&post_dev).unwrap();
    let gcomb = dev.dtoh_sync_copy(&comb_dev).unwrap();

    // libdevice expf vs glibc expf may differ by <=1 ulp at input; after 40 norm
    // rounds the compounding stays tiny. Measure and bound tightly (structural
    // errors — wrong eps placement/iteration order — are >> ulp).
    let mut m_pre = 0i64;
    let mut m_post = 0i64;
    let mut m_comb = 0i64;
    let mut bitexact = 0usize;
    for i in 0..n * 4 {
        m_pre = m_pre.max(ulp_diff(gpre[i], epre[i]));
        m_post = m_post.max(ulp_diff(gpost[i], epost[i]));
        bitexact += (gpre[i] == epre[i]) as usize + (gpost[i] == epost[i]) as usize;
    }
    for i in 0..n * 16 {
        m_comb = m_comb.max(ulp_diff(gcomb[i], ecomb[i]));
        bitexact += (gcomb[i] == ecomb[i]) as usize;
    }
    let total = n * 24;
    println!(
        "sinkhorn vs in-test CPU ref: max ulp pre={m_pre} post={m_post} comb={m_comb}; \
         bit-identical {bitexact}/{total}"
    );
    assert!(m_pre <= 2 && m_post <= 2 && m_comb <= 16, "sinkhorn ulp bound exceeded");
    // NOTE: the BIT-EXACT gate for this kernel is test_ground_truth_tilelang_vectors
    // (GPU vs the actual tilelang kernel on this GB10 — same libdevice expf). The in-test
    // CPU ref cannot be bit-exact: glibc exp ≠ libdevice expf by <=2 ulp, compounding to
    // ~10 ulp over the 40 normalization rounds. Measured bounds above (2/2/16) still kill
    // any structural error (eps misplacement, wrong iteration order/count => >> 1e3 ulp).
    // structural invariant: comb columns ~1 after final column normalization
    for i in 0..4 {
        for k in 0..4 {
            let cs: f32 = (0..4).map(|j| gcomb[(i * 4 + j) * 4 + k]).sum();
            assert!((cs - 1.0).abs() < 1e-4, "comb col sum {cs}");
        }
    }
}

#[test]
fn test_dsv4_fwht_rotate() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_dsv4(&dev);
    let fwfn = &f[1];

    let mut rng = XorShift(0xD5E4_2024_0005);
    let rows = 1024usize;
    let x: Vec<u16> = (0..rows * 128).map(|_| rng.bf16()).collect();

    let x_dev = dev.htod_sync_copy(&x).unwrap();
    let mut y_dev = dev.alloc_zeros::<u16>(rows * 128).unwrap();
    dev.synchronize().unwrap();
    unsafe {
        fwfn.clone()
            .launch(
                LaunchConfig {
                    grid_dim: (((rows + 7) / 8) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&x_dev, &mut y_dev, rows as i32),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    let gy = dev.dtoh_sync_copy(&y_dev).unwrap();

    // 1) bit-exact vs the identical butterfly order on CPU
    let mut all_eq = true;
    let mut max_d_vs_matmul = 0.0f32;
    let mut max_d_vs_exact = 0.0f32;
    for r in 0..rows {
        let exp = cpu_fwht_row(&x[r * 128..r * 128 + 128]);
        if gy[r * 128..r * 128 + 128] != exp[..] {
            all_eq = false;
            break;
        }
    }
    assert!(all_eq, "fwht not bit-exact vs same-order CPU butterfly");

    // 2) tolerance vs the emu semantics (fp32 Hadamard matmul with H*scale, then bf16)
    //    and vs the exact (f64) rotation — demonstrates the <=1e-3 contract.
    let scale64 = 128f64.powf(-0.5);
    let mut hmat = vec![vec![1f64; 128]; 128];
    // build Sylvester matrix
    let mut size = 1;
    while size < 128 {
        for i in 0..size {
            for j in 0..size {
                hmat[i][j + size] = hmat[i][j];
                hmat[i + size][j] = hmat[i][j];
                hmat[i + size][j + size] = -hmat[i][j];
            }
        }
        size *= 2;
    }
    for r in (0..rows).step_by(97) {
        let xf: Vec<f64> = x[r * 128..r * 128 + 128]
            .iter()
            .map(|&b| bf16_bits_to_f32(b) as f64)
            .collect();
        for j in 0..128 {
            let mut acc32 = 0.0f32; // emu: fp32 matmul against H*scale (fp32 matrix)
            let mut acc64 = 0.0f64;
            for i in 0..128 {
                let hs = (hmat[i][j] * scale64) as f32;
                acc32 += (xf[i] as f32) * hs;
                acc64 += xf[i] * hmat[i][j] * scale64;
            }
            let emu = bf16_bits_to_f32(f32_to_bf16_bits(acc32));
            let got = bf16_bits_to_f32(gy[r * 128 + j]);
            max_d_vs_matmul = max_d_vs_matmul.max((got - emu).abs());
            max_d_vs_exact = max_d_vs_exact.max((got as f64 - acc64).abs() as f32);
        }
    }
    println!(
        "fwht: bit-exact vs CPU butterfly; max|d| vs emu fp32-matmul = {max_d_vs_matmul:.3e}; \
         max|d| vs exact rotation (incl. bf16 out rounding) = {max_d_vs_exact:.3e}"
    );
    assert!(max_d_vs_matmul <= 1e-3, "fwht vs emu semantics exceeds 1e-3");
}

#[test]
fn test_dsv4_gather_attn() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_dsv4(&dev);

    let mut rng = XorShift(0xD5E4_2024_0006);
    let (b, m, n, topk) = (1usize, 3usize, 200usize, 133usize);
    let scale = 512f32.powf(-0.5);
    let q: Vec<u16> = (0..b * m * 64 * 512).map(|_| rng.bf16()).collect();
    let kv: Vec<u16> = (0..b * n * 512).map(|_| rng.bf16()).collect();
    let sink: Vec<f32> = (0..64).map(|_| rng.f32()).collect();
    let mut idx: Vec<i32> = Vec::with_capacity(b * m * topk);
    for _ in 0..b * m {
        for j in 0..topk {
            // mostly valid, some -1 (tail like the window ring's padding)
            idx.push(if j >= topk - 5 && (rng.next() & 1) == 0 {
                -1
            } else {
                (rng.next() % n as u64) as i32
            });
        }
    }

    let expected = cpu_gather_attn(&q, &kv, &sink, &idx, b, m, n, topk, scale);

    let q_dev = dev.htod_sync_copy(&q).unwrap();
    let kv_dev = dev.htod_sync_copy(&kv).unwrap();
    let sink_dev = dev.htod_sync_copy(&sink).unwrap();
    let idx_dev = dev.htod_sync_copy(&idx).unwrap();
    let mut o_dev = dev.alloc_zeros::<u16>(b * m * 64 * 512).unwrap();
    dev.synchronize().unwrap();

    // 88320 B dynamic smem > 48 KB default: opt-in via the raw driver API.
    // cudarc 0.9 keeps CUfunction private, so re-load the module through sys and set
    // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES (what Phase-3's launcher will do).
    use cudarc::driver::{sys, DevicePtr};
    let ptx = std::fs::read_to_string("src/ptx/gpu_dsv4.ptx").unwrap();
    let ptx_c = std::ffi::CString::new(ptx).unwrap();
    unsafe {
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let r = sys::cuModuleLoadData(&mut module, ptx_c.as_ptr() as *const _);
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuModuleLoadData");
        let mut func: sys::CUfunction = std::ptr::null_mut();
        let name = std::ffi::CString::new("dsv4_gather_attn").unwrap();
        let r = sys::cuModuleGetFunction(&mut func, module, name.as_ptr());
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuModuleGetFunction");
        let r = sys::cuFuncSetAttribute(
            func,
            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            88320,
        );
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuFuncSetAttribute");
        let (topk_i, n_i) = (topk as i32, n as i32);
        let mut params: Vec<*mut std::ffi::c_void> = vec![
            q_dev.device_ptr() as *const u64 as *mut _,
            kv_dev.device_ptr() as *const u64 as *mut _,
            o_dev.device_ptr() as *const u64 as *mut _,
            sink_dev.device_ptr() as *const u64 as *mut _,
            idx_dev.device_ptr() as *const u64 as *mut _,
            &topk_i as *const i32 as *mut _,
            &n_i as *const i32 as *mut _,
            &scale as *const f32 as *mut _,
        ];
        let r = sys::cuLaunchKernel(
            func,
            m as u32,
            b as u32,
            4,
            256,
            1,
            1,
            88320,
            std::ptr::null_mut(), // NULL stream: cudarc memops are NULL-stream ordered
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuLaunchKernel gather_attn");
    }
    dev.synchronize().unwrap();
    let got = dev.dtoh_sync_copy(&o_dev).unwrap();
    assert_bf16_close(&got, &expected, "gather_attn vs emu");
}

// ---------------------------------------------------------------------------
// ground-truth vectors (env-gated): raw dumps from the ACTUAL tilelang kernels
// on this GB10 — the kernel.py-semantics bit-exactness proof for G1.
// Dir layout: manifest.json + <case>.<field>.bin (little-endian raw).
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct VecCase {
    name: String,
    kind: String,
    shapes: serde_json::Value,
}

fn read_bin_u16(dir: &std::path::Path, name: &str) -> Vec<u16> {
    std::fs::read(dir.join(name))
        .expect("read vector bin")
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn read_bin_i32(dir: &std::path::Path, name: &str) -> Vec<i32> {
    std::fs::read(dir.join(name))
        .expect("read vector bin")
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_bin_f32(dir: &std::path::Path, name: &str) -> Vec<f32> {
    std::fs::read(dir.join(name))
        .expect("read vector bin")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_bin_u8(dir: &std::path::Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).expect("read vector bin")
}

#[test]
fn test_ground_truth_tilelang_vectors() {
    let Ok(dir) = std::env::var("DSV4_TL_VECTORS") else {
        eprintln!("DSV4_TL_VECTORS not set — skipping tilelang ground-truth comparison");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let manifest: Vec<VecCase> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("manifest.json")).expect("manifest.json"),
    )
    .expect("parse manifest");
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let f = load_dsv4(&dev);

    for case in &manifest {
        let sh = &case.shapes;
        match case.kind.as_str() {
            "act_quant" => {
                let group = sh["group"].as_u64().unwrap() as usize;
                let (rows, n) = (sh["rows"].as_u64().unwrap() as usize, sh["n"].as_u64().unwrap() as usize);
                let x: Vec<u16> = read_bin_u16(&dir, &format!("{}.x.bin", case.name));
                let ey: Vec<u8> = read_bin_u8(&dir, &format!("{}.y.bin", case.name));
                let es: Vec<u8> = read_bin_u8(&dir, &format!("{}.s.bin", case.name));
                let esim: Vec<u16> = read_bin_u16(&dir, &format!("{}.sim.bin", case.name));
                let (qfn, sfn) = if group == 64 { (&f[2], &f[4]) } else { (&f[3], &f[5]) };
                let x_dev = dev.htod_sync_copy(&x).unwrap();
                let mut y_dev = dev.alloc_zeros::<u8>(rows * n).unwrap();
                let mut s_dev = dev.alloc_zeros::<u8>(rows * (n / group)).unwrap();
                dev.synchronize().unwrap();
                unsafe {
                    qfn.clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (((rows * (n / group) + 7) / 8) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&x_dev, &mut y_dev, &mut s_dev, rows as i32, n as i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                assert_eq!(dev.dtoh_sync_copy(&y_dev).unwrap(), ey, "{}: fp8 codes vs tilelang", case.name);
                assert_eq!(dev.dtoh_sync_copy(&s_dev).unwrap(), es, "{}: e8m0 scales vs tilelang", case.name);
                let mut xs_dev = dev.htod_sync_copy(&x).unwrap();
                let mut s2_dev = dev.alloc_zeros::<u8>(rows * (n / group)).unwrap();
                dev.synchronize().unwrap();
                unsafe {
                    sfn.clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (((rows * (n / group) + 7) / 8) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut xs_dev, &mut s2_dev, rows as i32, n as i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                assert_eq!(dev.dtoh_sync_copy(&xs_dev).unwrap(), esim, "{}: sim round-trip vs tilelang", case.name);
                assert_eq!(dev.dtoh_sync_copy(&s2_dev).unwrap(), es, "{}: sim scales vs tilelang", case.name);
                println!("GT {}: BIT-EXACT vs tilelang act_quant (group {group})", case.name);
            }
            "fp4_act_quant" => {
                let (rows, n) = (sh["rows"].as_u64().unwrap() as usize, sh["n"].as_u64().unwrap() as usize);
                let x: Vec<u16> = read_bin_u16(&dir, &format!("{}.x.bin", case.name));
                let ey: Vec<u8> = read_bin_u8(&dir, &format!("{}.y.bin", case.name));
                let es: Vec<u8> = read_bin_u8(&dir, &format!("{}.s.bin", case.name));
                let esim: Vec<u16> = read_bin_u16(&dir, &format!("{}.sim.bin", case.name));
                let x_dev = dev.htod_sync_copy(&x).unwrap();
                let mut y_dev = dev.alloc_zeros::<u8>(rows * (n / 2)).unwrap();
                let mut s_dev = dev.alloc_zeros::<u8>(rows * (n / 32)).unwrap();
                dev.synchronize().unwrap();
                unsafe {
                    f[6].clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (((rows * (n / 32) + 7) / 8) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&x_dev, &mut y_dev, &mut s_dev, rows as i32, n as i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                assert_eq!(dev.dtoh_sync_copy(&y_dev).unwrap(), ey, "{}: fp4 codes vs tilelang", case.name);
                assert_eq!(dev.dtoh_sync_copy(&s_dev).unwrap(), es, "{}: fp4 scales vs tilelang", case.name);
                let mut xs_dev = dev.htod_sync_copy(&x).unwrap();
                let mut s2_dev = dev.alloc_zeros::<u8>(rows * (n / 32)).unwrap();
                dev.synchronize().unwrap();
                unsafe {
                    f[7].clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (((rows * (n / 32) + 7) / 8) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&mut xs_dev, &mut s2_dev, rows as i32, n as i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                assert_eq!(dev.dtoh_sync_copy(&xs_dev).unwrap(), esim, "{}: fp4 sim vs tilelang", case.name);
                assert_eq!(dev.dtoh_sync_copy(&s2_dev).unwrap(), es, "{}: fp4 sim scales vs tilelang", case.name);
                println!("GT {}: BIT-EXACT vs tilelang fp4_act_quant", case.name);
            }
            "hc_split_sinkhorn" => {
                let n = sh["n"].as_u64().unwrap() as usize;
                let mixes: Vec<f32> = read_bin_f32(&dir, &format!("{}.mixes.bin", case.name));
                let scale: Vec<f32> = read_bin_f32(&dir, &format!("{}.scale.bin", case.name));
                let base: Vec<f32> = read_bin_f32(&dir, &format!("{}.base.bin", case.name));
                let epre: Vec<f32> = read_bin_f32(&dir, &format!("{}.pre.bin", case.name));
                let epost: Vec<f32> = read_bin_f32(&dir, &format!("{}.post.bin", case.name));
                let ecomb: Vec<f32> = read_bin_f32(&dir, &format!("{}.comb.bin", case.name));
                let m_dev = dev.htod_sync_copy(&mixes).unwrap();
                let s_dev = dev.htod_sync_copy(&scale).unwrap();
                let b_dev = dev.htod_sync_copy(&base).unwrap();
                let mut pre_dev = dev.alloc_zeros::<f32>(n * 4).unwrap();
                let mut post_dev = dev.alloc_zeros::<f32>(n * 4).unwrap();
                let mut comb_dev = dev.alloc_zeros::<f32>(n * 16).unwrap();
                dev.synchronize().unwrap();
                unsafe {
                    f[8].clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (((n + 255) / 256) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &m_dev, &s_dev, &b_dev, &mut pre_dev, &mut post_dev,
                                &mut comb_dev, n as i32,
                            ),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                assert_eq!(dev.dtoh_sync_copy(&pre_dev).unwrap(), epre, "{}: pre vs tilelang", case.name);
                assert_eq!(dev.dtoh_sync_copy(&post_dev).unwrap(), epost, "{}: post vs tilelang", case.name);
                assert_eq!(dev.dtoh_sync_copy(&comb_dev).unwrap(), ecomb, "{}: comb vs tilelang", case.name);
                println!("GT {}: BIT-EXACT vs tilelang hc_split_sinkhorn", case.name);
            }
            "sparse_attn" => {
                let b = sh["b"].as_u64().unwrap() as usize;
                let m = sh["m"].as_u64().unwrap() as usize;
                let n = sh["n"].as_u64().unwrap() as usize;
                let topk = sh["topk"].as_u64().unwrap() as usize;
                let scale = sh["scale"].as_f64().unwrap() as f32;
                let q: Vec<u16> = read_bin_u16(&dir, &format!("{}.q.bin", case.name));
                let kv: Vec<u16> = read_bin_u16(&dir, &format!("{}.kv.bin", case.name));
                let sink: Vec<f32> = read_bin_f32(&dir, &format!("{}.sink.bin", case.name));
                let idx: Vec<i32> = read_bin_i32(&dir, &format!("{}.idx.bin", case.name));
                let eo: Vec<u16> = read_bin_u16(&dir, &format!("{}.o.bin", case.name));
                let q_dev = dev.htod_sync_copy(&q).unwrap();
                let kv_dev = dev.htod_sync_copy(&kv).unwrap();
                let sink_dev = dev.htod_sync_copy(&sink).unwrap();
                let idx_dev = dev.htod_sync_copy(&idx).unwrap();
                let mut o_dev = dev.alloc_zeros::<u16>(b * m * 64 * 512).unwrap();
                dev.synchronize().unwrap();
                use cudarc::driver::{sys, DevicePtr};
                let ptx = std::fs::read_to_string("src/ptx/gpu_dsv4.ptx").unwrap();
                let ptx_c = std::ffi::CString::new(ptx).unwrap();
                unsafe {
                    let mut module: sys::CUmodule = std::ptr::null_mut();
                    assert_eq!(
                        sys::cuModuleLoadData(&mut module, ptx_c.as_ptr() as *const _),
                        sys::CUresult::CUDA_SUCCESS
                    );
                    let mut func: sys::CUfunction = std::ptr::null_mut();
                    let name = std::ffi::CString::new("dsv4_gather_attn").unwrap();
                    assert_eq!(
                        sys::cuModuleGetFunction(&mut func, module, name.as_ptr()),
                        sys::CUresult::CUDA_SUCCESS
                    );
                    assert_eq!(
                        sys::cuFuncSetAttribute(
                            func,
                            sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            88320,
                        ),
                        sys::CUresult::CUDA_SUCCESS
                    );
                    let (topk_i, n_i) = (topk as i32, n as i32);
                    let mut params: Vec<*mut std::ffi::c_void> = vec![
                        q_dev.device_ptr() as *const u64 as *mut _,
                        kv_dev.device_ptr() as *const u64 as *mut _,
                        o_dev.device_ptr() as *const u64 as *mut _,
                        sink_dev.device_ptr() as *const u64 as *mut _,
                        idx_dev.device_ptr() as *const u64 as *mut _,
                        &topk_i as *const i32 as *mut _,
                        &n_i as *const i32 as *mut _,
                        &scale as *const f32 as *mut _,
                    ];
                    assert_eq!(
                        sys::cuLaunchKernel(
                            func,
                            m as u32,
                            b as u32,
                            4,
                            256,
                            1,
                            1,
                            88320,
                            std::ptr::null_mut(),
                            params.as_mut_ptr(),
                            std::ptr::null_mut(),
                        ),
                        sys::CUresult::CUDA_SUCCESS
                    );
                }
                dev.synchronize().unwrap();
                let got = dev.dtoh_sync_copy(&o_dev).unwrap();
                assert_bf16_close(&got, &eo, &format!("GT {} gather_attn vs sparse_attn_emu", case.name));
            }
            other => panic!("unknown vector kind {other}"),
        }
    }
}
