// DeepSeek-V4-Flash-DSpark prototype kernels (Gate G1, DEEPSEEK_V4_PORT.md §12.B/§C).
// Numerics contract: bit-match the bundle's inference/kernel.py semantics, as lowered by
// tilelang on GB10 (extracted from the generated device sources + SASS, 2026-07-24):
//   - FP8/FP4 casts: cvt.rn.satfinite.e4m3x2/e2m1x2.f32 (round-to-nearest-even, pre-clamped)
//   - UE8M0 scales: s = 2^ceil(log2(amax*inv)) via the IEEE bit trick (kernel.py:22-37)
//   - sigmoid: 1/(1+expf(-x)) with libdevice expf; mixes*scale+base contracted to FFMA
//   - hc_split_sinkhorn row/col sums: XOR-butterfly tree order (e0+e2)+(e1+e3)
// All per-thread arrays are compile-time indexed (templates/constexpr + full unroll):
// ptxas must report ZERO stack frames for every kernel here (AGENTS.md §4).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

// build.rs hashes this file into KERNEL_BUILD_ID and passes the same -D as gpu_batch.cu. Phase 3
// promotes these kernels onto the serving path, so — like gpu_batch — this module NOW exposes its
// OWN stamp `dsv4_kernel_build_id`, asserted by the production loader (src/dsv4_gpu.rs) so a deploy
// that ships a fresh binary with a stale src/ptx/gpu_dsv4.ptx fails loudly (AGENTS.md §1: a deploy
// is THREE files).
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void dsv4_kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

#define DSV4_FULL_MASK 0xffffffffu

__device__ __forceinline__ float dsv4_bf16_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 dsv4_f32_to_bf16(float v) { return __float2bfloat16(v); }

// ---- UE8M0 scale from amax: s = 2^ceil(log2(amax * inv)), kernel.py fast_log2_ceil/fast_pow2 ----
__device__ __forceinline__ float dsv4_round_scale_pow2(float amax, float inv) {
    float v = amax * inv;
    uint32_t b = __float_as_uint(v);
    int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t)(e + 127) << 23);
}

__device__ __forceinline__ uint8_t dsv4_f32_to_e8m0(float s) {
    return __nv_cvt_float_to_e8m0(s, __NV_SATFINITE, cudaRoundPosInf);
}

__device__ __forceinline__ uint8_t dsv4_f32_to_fp8(float v) {
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float dsv4_fp8_to_f32(uint8_t c) {
    return __half2float(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3));
}

// FP4 E2M1 decode (exact: all 16 values representable). code = sign<<3 | exp<<1 | man.
__device__ __forceinline__ float dsv4_fp4_to_f32(uint8_t c) {
    float mag;
    int e = (c >> 1) & 3, m = c & 1;
    if (e == 0) mag = m ? 0.5f : 0.0f;
    else        mag = (1.0f + 0.5f * (float)m) * (float)(1 << (e - 1));
    return (c & 8) ? -mag : mag;
}

// ============================================================================
// 1. dsv4_topk — deterministic top-k over [rows, T] fp32 scores (§12.B.2).
//    Total order: value desc, index asc on ties. One CTA (256 threads) per row;
//    k iterative block-argmax rounds with a fixed reduction tree, selected flags in
//    smem — row results independent of grid/launch geometry (batch-invariant).
//    Contract: k <= T <= 16384 (host asserts), k <= 512.
// ============================================================================
#define DSV4_TOPK_MAXT 16384
#define DSV4_TOPK_THREADS 256

__device__ __forceinline__ bool dsv4_topk_better(float av, int ai, float bv, int bi) {
    return (av > bv) || (av == bv && ai < bi);
}

extern "C" __global__ void __launch_bounds__(DSV4_TOPK_THREADS)
dsv4_topk(const float* __restrict__ scores, int* __restrict__ out_idx,
          int rows, int T, int k) {
    __shared__ uint32_t selmask[DSV4_TOPK_MAXT / 32];
    __shared__ float vbuf[DSV4_TOPK_THREADS];
    __shared__ int ibuf[DSV4_TOPK_THREADS];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const float* srow = scores + (size_t)row * (size_t)T;

    for (int i = tid; i < (DSV4_TOPK_MAXT / 32); i += DSV4_TOPK_THREADS) selmask[i] = 0u;
    __syncthreads();

    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7FFFFFFF;
        for (int i = tid; i < T; i += DSV4_TOPK_THREADS) {
            if ((selmask[i >> 5] >> (i & 31)) & 1u) continue;
            float v = srow[i];
            if (dsv4_topk_better(v, i, bv, bi)) { bv = v; bi = i; }
        }
        vbuf[tid] = bv;
        ibuf[tid] = bi;
        __syncthreads();
        // fixed halving tree — deterministic regardless of launch geometry
        for (int s = DSV4_TOPK_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (dsv4_topk_better(vbuf[tid + s], ibuf[tid + s], vbuf[tid], ibuf[tid])) {
                    vbuf[tid] = vbuf[tid + s];
                    ibuf[tid] = ibuf[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            int sel = ibuf[0];
            if (sel == 0x7FFFFFFF) sel = -1;  // exhausted (k > T; contract forbids)
            out_idx[(size_t)row * (size_t)k + r] = sel;
            if (sel >= 0) selmask[sel >> 5] |= (1u << (sel & 31));
        }
        __syncthreads();
    }
}

// ============================================================================
// 2. dsv4_fwht_rotate — Walsh-Hadamard rotation of 128-dim rows, scale 128^-0.5
//    (§C / model.py rotate_activation; semantics = fp32 WHT matmul, bf16 in/out,
//    fp32 math inside). One warp per row; lane holds elements lane+32*j (j=0..3),
//    butterfly h=1..16 via shfl_xor, h=32/64 in registers. The ascending-h butterfly
//    computes the Sylvester natural-order Hadamard product.
// ============================================================================
#define DSV4_FWHT_SCALE 0x1.6a09e6p-4f  // fl32(128^-0.5)

extern "C" __global__ void __launch_bounds__(256)
dsv4_fwht_rotate(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int rows) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const size_t base = (size_t)row * 128;

    float r[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = dsv4_bf16_to_f32(x[base + lane + 32 * j]);

#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p = __shfl_xor_sync(DSV4_FULL_MASK, r[j], h);
            r[j] = (lane & h) ? (p - r[j]) : (r[j] + p);
        }
    }
    {   // h = 32: pairs (j, j^1); then h = 64: pairs (j, j^2)
        float a0 = r[0], a1 = r[1], a2 = r[2], a3 = r[3];
        r[0] = a0 + a1; r[1] = a0 - a1; r[2] = a2 + a3; r[3] = a2 - a3;
        a0 = r[0]; a1 = r[1]; a2 = r[2]; a3 = r[3];
        r[0] = a0 + a2; r[2] = a0 - a2; r[1] = a1 + a3; r[3] = a1 - a3;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) y[base + lane + 32 * j] = dsv4_f32_to_bf16(r[j] * DSV4_FWHT_SCALE);
}

// ============================================================================
// 3. dsv4_act_quant — FP8-E4M3 dynamic quant, UE8M0 scales (§C.1, kernel.py:40-125).
//    One warp per (row, group). Template G = group size (64 KV-sim / 128 GEMM).
//    amax = max|x| floored at 1e-4; s = 2^ceil(log2(amax/448)); y = cvt.rn(clamp(x/s, ±448)).
//    plain: y codes [rows,N] u8 + s [rows,N/G] e8m0 u8.
//    sim (inplace QAT round-trip): x <- bf16(f32(fp8(x/s)) * s); s still written.
// ============================================================================
template <int G, bool SIM>
__device__ __forceinline__ void dsv4_act_quant_body(
    __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ y, uint8_t* __restrict__ s,
    int rows, int N) {
    const int lane = threadIdx.x & 31;
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int groups_per_row = N / G;
    if (wid >= rows * groups_per_row) return;
    const int row = wid / groups_per_row;
    const int grp = wid % groups_per_row;
    const size_t gbase = (size_t)row * N + (size_t)grp * G;

    constexpr int E = G / 32;  // elements per lane (2 or 4)
    float v[E];
#pragma unroll
    for (int i = 0; i < E; ++i) v[i] = dsv4_bf16_to_f32(x[gbase + lane + 32 * i]);

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(DSV4_FULL_MASK, amax, off));
    amax = fmaxf(amax, 1e-4f);

    const float sc = dsv4_round_scale_pow2(amax, 1.0f / 448.0f);

#pragma unroll
    for (int i = 0; i < E; ++i) {
        float q = fminf(fmaxf(v[i] / sc, -448.0f), 448.0f);
        uint8_t c = dsv4_f32_to_fp8(q);
        if (SIM) {
            x[gbase + lane + 32 * i] = dsv4_f32_to_bf16(dsv4_fp8_to_f32(c) * sc);
        } else {
            y[gbase + lane + 32 * i] = c;
        }
    }
    if (lane == 0) s[(size_t)row * groups_per_row + grp] = dsv4_f32_to_e8m0(sc);
}

extern "C" {
__global__ void __launch_bounds__(256) dsv4_act_quant_g64(
    const __nv_bfloat16* x, uint8_t* y, uint8_t* s, int rows, int N) {
    dsv4_act_quant_body<64, false>(const_cast<__nv_bfloat16*>(x), y, s, rows, N);
}
__global__ void __launch_bounds__(256) dsv4_act_quant_g128(
    const __nv_bfloat16* x, uint8_t* y, uint8_t* s, int rows, int N) {
    dsv4_act_quant_body<128, false>(const_cast<__nv_bfloat16*>(x), y, s, rows, N);
}
__global__ void __launch_bounds__(256) dsv4_act_quant_sim_g64(
    __nv_bfloat16* x, uint8_t* s, int rows, int N) {
    dsv4_act_quant_body<64, true>(x, nullptr, s, rows, N);
}
__global__ void __launch_bounds__(256) dsv4_act_quant_sim_g128(
    __nv_bfloat16* x, uint8_t* s, int rows, int N) {
    dsv4_act_quant_body<128, true>(x, nullptr, s, rows, N);
}
}

// ============================================================================
// 3b. dsv4_fp4_act_quant — FP4-E2M1 dynamic quant, group 32, UE8M0 scales
//     (§C.2, kernel.py:128-200). amax floored at 6*2^-126; s = 2^ceil(log2(amax/6));
//     y = cvt.rn(clamp(x/s, ±6)); packs 2/byte along K, LOW nibble = even K.
// ============================================================================
template <bool SIM>
__device__ __forceinline__ void dsv4_fp4_act_quant_body(
    __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ y, uint8_t* __restrict__ s,
    int rows, int N) {
    const int lane = threadIdx.x & 31;
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int groups_per_row = N / 32;
    if (wid >= rows * groups_per_row) return;
    const int row = wid / groups_per_row;
    const int grp = wid % groups_per_row;
    const size_t gbase = (size_t)row * N + (size_t)grp * 32;

    float v = dsv4_bf16_to_f32(x[gbase + lane]);
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(DSV4_FULL_MASK, amax, off));
    amax = fmaxf(amax, 0x1.8p-124f);  // 6 * 2^-126

    const float sc = dsv4_round_scale_pow2(amax, 1.0f / 6.0f);

    float q = fminf(fmaxf(v / sc, -6.0f), 6.0f);
    uint8_t code = (uint8_t)__nv_cvt_float_to_fp4(q, __NV_E2M1, cudaRoundNearest) & 0xFu;
    if (SIM) {
        x[gbase + lane] = dsv4_f32_to_bf16(dsv4_fp4_to_f32(code) * sc);
    } else {
        // even lane packs (own = low nibble, odd neighbor = high nibble)
        uint8_t hi = (uint8_t)__shfl_down_sync(DSV4_FULL_MASK, code, 1);
        if ((lane & 1) == 0)
            y[(size_t)row * (N / 2) + (size_t)grp * 16 + (lane >> 1)] = code | (hi << 4);
    }
    if (lane == 0) s[(size_t)row * groups_per_row + grp] = dsv4_f32_to_e8m0(sc);
}

extern "C" {
__global__ void __launch_bounds__(256) dsv4_fp4_act_quant(
    const __nv_bfloat16* x, uint8_t* y, uint8_t* s, int rows, int N) {
    dsv4_fp4_act_quant_body<false>(const_cast<__nv_bfloat16*>(x), y, s, rows, N);
}
__global__ void __launch_bounds__(256) dsv4_fp4_act_quant_sim(
    __nv_bfloat16* x, uint8_t* s, int rows, int N) {
    dsv4_fp4_act_quant_body<true>(x, nullptr, s, rows, N);
}
}

// ============================================================================
// 4. dsv4_hc_split_sinkhorn — §B.8 exact sequence, fp32 (kernel.py:371-438).
//    One thread per row (24 mixes -> pre[4], post[4], comb[4x4]); comb arrays are
//    compile-time indexed (fully unrolled) -> registers only.
//    Row/col sums use the tilelang XOR-butterfly tree value: (e0+e2)+(e1+e3).
//    eps = 1e-6: pre += eps; comb = row_softmax + eps; /(col+eps); 19x (/(row+eps), /(col+eps)).
// ============================================================================
#define DSV4_HC_EPS 1e-6f

__device__ __forceinline__ float dsv4_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

extern "C" __global__ void __launch_bounds__(256)
dsv4_hc_split_sinkhorn(const float* __restrict__ mixes, const float* __restrict__ hc_scale,
                       const float* __restrict__ hc_base, float* __restrict__ pre,
                       float* __restrict__ post, float* __restrict__ comb, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* m = mixes + (size_t)i * 24;
    const float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

#pragma unroll
    for (int j = 0; j < 4; ++j)
        pre[(size_t)i * 4 + j] = dsv4_sigmoid(fmaf(m[j], s0, hc_base[j])) + DSV4_HC_EPS;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        post[(size_t)i * 4 + j] = 2.0f * dsv4_sigmoid(fmaf(m[4 + j], s1, hc_base[4 + j]));

    float c[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k)
            c[j][k] = fmaf(m[8 + 4 * j + k], s2, hc_base[8 + 4 * j + k]);

    // row softmax + eps (butterfly tree sums: (e0+e2)+(e1+e3))
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float mx = fmaxf(fmaxf(c[j][0], c[j][1]), fmaxf(c[j][2], c[j][3]));
#pragma unroll
        for (int k = 0; k < 4; ++k) c[j][k] = expf(c[j][k] - mx);
        float rs = (c[j][0] + c[j][2]) + (c[j][1] + c[j][3]);
#pragma unroll
        for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / rs + DSV4_HC_EPS;
    }
    // col norm
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float cs = (c[0][k] + c[2][k]) + (c[1][k] + c[3][k]) + DSV4_HC_EPS;
#pragma unroll
        for (int j = 0; j < 4; ++j) c[j][k] = c[j][k] / cs;
    }
    // 19x (row norm, col norm)
#pragma unroll
    for (int it = 0; it < 19; ++it) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float rs = (c[j][0] + c[j][2]) + (c[j][1] + c[j][3]) + DSV4_HC_EPS;
#pragma unroll
            for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / rs;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float cs = (c[0][k] + c[2][k]) + (c[1][k] + c[3][k]) + DSV4_HC_EPS;
#pragma unroll
            for (int j = 0; j < 4; ++j) c[j][k] = c[j][k] / cs;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) comb[((size_t)i * 4 + j) * 4 + k] = c[j][k];
}

// ============================================================================
// 5. dsv4_gather_attn — gather FlashAttention prototype (§12.B.1, §B.7).
//    MQA: 64 heads share one 512-dim KV latent (K ≡ V). Head-block HB=16 per CTA,
//    KV tiles of KB=64 gathered rows; each gathered row read ONCE for all 16 heads.
//    fp32 per-tile-max online softmax; probabilities rounded to bf16 before P·V;
//    denominator-only attention sink; -1 index masking only; caller scale = 512^-0.5.
//    Dynamic smem: q 16 KB + kv 64 KB + scores 4 KB + pcast 2 KB + idx 256 B = 88320 B
//    (>48 KB: caller must set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES).
//    Grid (m, b, 64/HB), 256 threads (8 warps, 2 heads per warp).
// ============================================================================
#define DSV4_GA_HB 16
#define DSV4_GA_KB 64
#define DSV4_GA_D 512
#define DSV4_GA_SMEM ((DSV4_GA_HB * DSV4_GA_D + DSV4_GA_KB * DSV4_GA_D) * 2 \
                      + DSV4_GA_HB * DSV4_GA_KB * 4 + DSV4_GA_HB * DSV4_GA_KB * 2 + DSV4_GA_KB * 4)

extern "C" __global__ void __launch_bounds__(256, 1)
dsv4_gather_attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
                 __nv_bfloat16* __restrict__ o, const float* __restrict__ attn_sink,
                 const int* __restrict__ topk_idxs, int topk, int n, float scale) {
    extern __shared__ __align__(16) uint8_t dsmem[];
    __nv_bfloat16* q_s = (__nv_bfloat16*)dsmem;                                   // [HB][D]
    __nv_bfloat16* kv_s = (__nv_bfloat16*)(dsmem + DSV4_GA_HB * DSV4_GA_D * 2);   // [KB][D]
    float* sc_s = (float*)(kv_s + DSV4_GA_KB * DSV4_GA_D);                        // [HB][KB]
    __nv_bfloat16* pc_s = (__nv_bfloat16*)(sc_s + DSV4_GA_HB * DSV4_GA_KB);       // [HB][KB]
    int* idx_s = (int*)(pc_s + DSV4_GA_HB * DSV4_GA_KB);                          // [KB]

    const int bx = blockIdx.x;              // query row within batch
    const int by = blockIdx.y;              // batch
    const int hb = blockIdx.z;              // head block (16 heads)
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m = gridDim.x;

    // load q [HB][D] (contiguous)
    const size_t q_base = (((size_t)by * m + bx) * 64 + hb * DSV4_GA_HB) * DSV4_GA_D;
    for (int i = tid; i < DSV4_GA_HB * DSV4_GA_D; i += 256) q_s[i] = q[q_base + i];

    const int h0 = 2 * warp, h1 = 2 * warp + 1;  // this warp's heads (local)
    float m_run0 = -INFINITY, m_run1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;
    float acc0[16], acc1[16];  // lane owns dims lane*16 .. lane*16+15 of each of its heads
#pragma unroll
    for (int dd = 0; dd < 16; ++dd) { acc0[dd] = 0.0f; acc1[dd] = 0.0f; }

    const int n_tiles = (topk + DSV4_GA_KB - 1) / DSV4_GA_KB;
    const int* idx_row = topk_idxs + ((size_t)by * m + bx) * (size_t)topk;

    for (int t = 0; t < n_tiles; ++t) {
        __syncthreads();  // protect smem overwrite vs previous iteration's reads
        if (tid < DSV4_GA_KB) {
            int g = t * DSV4_GA_KB + tid;
            idx_s[tid] = (g < topk) ? idx_row[g] : -1;
        }
        __syncthreads();
        // gather kv tile: each row read once, all 16 heads reuse from smem
        for (int i = tid; i < DSV4_GA_KB * DSV4_GA_D; i += 256) {
            int row = i >> 9, col = i & 511;
            int idx = idx_s[row];
            kv_s[i] = (idx >= 0) ? kv[((size_t)by * n + idx) * DSV4_GA_D + col]
                                 : __nv_bfloat16(0.0f);
        }
        __syncthreads();

        // scores: warp-dot per (head, row) cell; masked rows -> -inf
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            for (int j = 0; j < DSV4_GA_KB; ++j) {
                float part = 0.0f;
#pragma unroll
                for (int dd = 0; dd < 16; ++dd)
                    part = fmaf(dsv4_bf16_to_f32(q_s[h * DSV4_GA_D + lane * 16 + dd]),
                                dsv4_bf16_to_f32(kv_s[j * DSV4_GA_D + lane * 16 + dd]), part);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    part += __shfl_xor_sync(DSV4_FULL_MASK, part, off);
                float sc = part * scale;
                if (idx_s[j] < 0) sc = -INFINITY;
                if (lane == 0) sc_s[h * DSV4_GA_KB + j] = sc;
            }
        }
        __syncthreads();

        // per head: tile max, online-softmax rescale (applied to acc below), p -> bf16,
        // fp32 sum of UNROUNDED p (reference reduces the fp32 fragment)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            float m_run = hh == 0 ? m_run0 : m_run1;
            float m_tile = fmaxf(sc_s[h * DSV4_GA_KB + lane], sc_s[h * DSV4_GA_KB + lane + 32]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m_tile = fmaxf(m_tile, __shfl_xor_sync(DSV4_FULL_MASK, m_tile, off));
            float m_new = fmaxf(m_run, m_tile);
            float rescale = expf(m_run - m_new);
            float p_a = expf(sc_s[h * DSV4_GA_KB + lane] - m_new);
            float p_b = expf(sc_s[h * DSV4_GA_KB + lane + 32] - m_new);
            pc_s[h * DSV4_GA_KB + lane] = dsv4_f32_to_bf16(p_a);
            pc_s[h * DSV4_GA_KB + lane + 32] = dsv4_f32_to_bf16(p_b);
            float tsum = p_a + p_b;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                tsum += __shfl_xor_sync(DSV4_FULL_MASK, tsum, off);
            // acc *= rescale (compile-time array selection: hh is unrolled)
#pragma unroll
            for (int dd = 0; dd < 16; ++dd) {
                if (hh == 0) acc0[dd] *= rescale; else acc1[dd] *= rescale;
            }
            if (hh == 0) { m_run0 = m_new; l0 = l0 * rescale + tsum; }
            else         { m_run1 = m_new; l1 = l1 * rescale + tsum; }
        }
        __syncthreads();

        // acc += P_bf16 · V (ascending-j accumulation, fp32)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            for (int j = 0; j < DSV4_GA_KB; ++j) {
                float p = dsv4_bf16_to_f32(pc_s[h * DSV4_GA_KB + j]);
#pragma unroll
                for (int dd = 0; dd < 16; ++dd) {
                    float kv = dsv4_bf16_to_f32(kv_s[j * DSV4_GA_D + lane * 16 + dd]);
                    if (hh == 0) acc0[dd] = fmaf(p, kv, acc0[dd]);
                    else         acc1[dd] = fmaf(p, kv, acc1[dd]);
                }
            }
        }
    }

    // epilogue: denominator-only sink, o = acc / sum_exp -> bf16
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int hg = hb * DSV4_GA_HB + (hh == 0 ? h0 : h1);
        float m = hh == 0 ? m_run0 : m_run1;
        float l = (hh == 0 ? l0 : l1) + expf(attn_sink[hg] - m);
        const size_t o_base = (((size_t)by * m + bx) * 64 + hg) * DSV4_GA_D;
#pragma unroll
        for (int dd = 0; dd < 16; ++dd) {
            float a = hh == 0 ? acc0[dd] : acc1[dd];
            o[o_base + lane * 16 + dd] = dsv4_f32_to_bf16(a / l);
        }
    }
}


// ============================================================================
// 5b. dsv4_fused_gather_b — queue #5 (session 10): the verify/decode sparse gather as ONE
//     launch, replacing the assembly
//       dsv4_window_idxs_verify_b / dsv4_window_idxs_b / dsv4_compress_idxs_b /
//       dsv4_idxs_place_b  +  the unified-scratch d2d copies  +  dsv4_gather_attn
//     for the verify (s=6) and decode (s=1) shapes at any start_pos. Per-(query row,
//     head) chains are IDENTICAL to dsv4_gather_attn (same 64-key tiles over the same
//     key ORDER, same online-max/rescale sequence, same ascending-j P·V fma chain, same
//     masked-entry no-ops — p=0 into max/tsum, fma(0,0,acc) in P·V — and the same sink
//     epilogue), so it is BITWISE == the assembly by construction (gated in tests).
//     The three index-list kernels are replaced by in-kernel integer index math; the
//     unified scratch is replaced by direct reads of the three physical sources:
//       ring   (kv_cache[0..win], physical slots — position-ordered prefix),
//       kv_new (this layer's batched [s,hd] projections — the verify's own rows),
//       tail   (the compressor cache alias at kv_cache[win..] — committed comp rows).
//     Per query row r (grid.x), list entry j (tile t covers [64t, 64t+64)):
//       window segment (j < win), matching the assembly's list values:
//         full row (start_pos+r >= win-1):  j < win-1-r  -> ring slot (sp+r+1+j)%win
//                                          j >= win-1-r  -> kv_new row j-(win-1-r)
//           (== dsv4_window_idxs_verify_b's scratch rows [r+1 .. r+win]: prefix rows
//            [r+1..win) map to ring[(sp+m)%win]; new rows map to kv_new[m-win]).
//         early row (start_pos+r < win-1):  j <= start_pos+r -> j < start_pos ? ring[j]
//                                              : kv_new[j-start_pos];  else masked
//           (== dsv4_window_idxs_b branch-3: slots [0..start_pos+r], -1-padded).
//       compressed segment (jj = j-win, count t_comp_max, masks beyond the row limit):
//         HCA:  jj < (start_pos+r+1)/ratio -> tail row jj; else masked
//               (== dsv4_compress_idxs_b values offset+jj; physical row jj).
//         CSA:  idx_csa[r][jj] >= 0 -> tail row idx_csa[r][jj]-comp_off; else masked
//               (== dsv4_idxs_place_b of the remasked idx_dev; comp_off = win+s).
//         SWA:  none.
//     Masked entries keep the tile positions (bitwise no-ops), so the tile boundaries
//     and online-softmax chain are identical to the assembly in every regime.
//     Vectorized kv gather (float4 = 8 bf16 per thread per round) — load-width is free
//     under the contract (values identical). Grid (s, 1, nh/16), 256 threads,
//     dynamic smem identical to dsv4_gather_attn (88320 B).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256, 1)
dsv4_fused_gather_b(const __nv_bfloat16* __restrict__ q,
                    const __nv_bfloat16* __restrict__ ring,
                    const __nv_bfloat16* __restrict__ kv_new,
                    const __nv_bfloat16* __restrict__ comp_tail,
                    const int* __restrict__ idx_csa,
                    __nv_bfloat16* __restrict__ o,
                    const float* __restrict__ attn_sink,
                    int s, int start_pos, int win, int ratio,
                    int comp_off, int t_comp_max, int k_csa, float scale) {
    extern __shared__ __align__(16) uint8_t dsmem[];
    __nv_bfloat16* q_s = (__nv_bfloat16*)dsmem;                                   // [HB][D]
    __nv_bfloat16* kv_s = (__nv_bfloat16*)(dsmem + DSV4_GA_HB * DSV4_GA_D * 2);   // [KB][D]
    float* sc_s = (float*)(kv_s + DSV4_GA_KB * DSV4_GA_D);                        // [HB][KB]
    __nv_bfloat16* pc_s = (__nv_bfloat16*)(sc_s + DSV4_GA_HB * DSV4_GA_KB);       // [HB][KB]
    int* idx_s = (int*)(pc_s + DSV4_GA_HB * DSV4_GA_KB);                          // [KB]

    const int bx = blockIdx.x;              // query row
    const int hb = blockIdx.z;              // head block
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int r = bx;
    const int sp = start_pos % win;
    const bool full_row = (start_pos + r >= win - 1);
    const int wb = full_row ? (win - 1 - r) : start_pos;  // window ring-prefix count
    const int c_lim = (ratio > 0) ? ((start_pos + r + 1) / ratio) : 0;
    const int t_total = win + t_comp_max;

    // load q [HB][D] (contiguous)
    const size_t q_base = ((size_t)bx * 64 + hb * DSV4_GA_HB) * DSV4_GA_D;
    for (int i = tid; i < DSV4_GA_HB * DSV4_GA_D; i += 256) q_s[i] = q[q_base + i];

    const int h0 = 2 * warp, h1 = 2 * warp + 1;  // this warp's heads (local)
    float m_run0 = -INFINITY, m_run1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;
    float acc0[16], acc1[16];  // lane owns dims lane*16 .. lane*16+15 of each of its heads
#pragma unroll
    for (int dd = 0; dd < 16; ++dd) { acc0[dd] = 0.0f; acc1[dd] = 0.0f; }

    const int n_tiles = (t_total + DSV4_GA_KB - 1) / DSV4_GA_KB;

    for (int t = 0; t < n_tiles; ++t) {
        __syncthreads();  // protect smem overwrite vs previous iteration's reads
        if (tid < DSV4_GA_KB) {
            // ---- in-kernel index math (replaces dsv4_window_idxs_* / compress / place) ----
            const int j = t * DSV4_GA_KB + tid;
            int v = -1;
            if (j < win) {
                if (full_row) {
                    if (j < wb) v = (sp + r + 1 + j) % win;   // ring slot (prefix)
                    else        v = j - wb;                    // kv_new row (new)
                } else if (j <= sp + r) {
                    v = (j < start_pos) ? j : (j - start_pos); // ring row / kv_new row
                }
            } else {
                const int jj = j - win;
                if (ratio > 0) {                               // HCA
                    if (jj < c_lim) v = jj;                    // tail row
                } else if (idx_csa != 0 && jj < t_comp_max) {  // CSA (pad-guarded: the
                    const int x = idx_csa[(size_t)r * k_csa + jj];  // last tile's padded
                    if (x >= 0) v = x - comp_off;              // entries must stay masked)
                }
            }
            idx_s[tid] = v;
        }
        __syncthreads();
        // gather kv tile: each row read once (float4 = 8 bf16/thread/round), all 16 heads
        // reuse from smem; masked rows -> zeros (the p=0 / fma(0,0) no-ops, as the
        // assembly's zero-gathered rows)
        for (int i = tid * 8; i < DSV4_GA_KB * DSV4_GA_D; i += 256 * 8) {
            const int row = i >> 9;
            const int col = i & 511;
            const int idx = idx_s[row];
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (idx >= 0) {
                const int j = t * DSV4_GA_KB + row;
                const __nv_bfloat16* base =
                    (j < win) ? ((j < wb) ? ring : kv_new) : comp_tail;
                v = *(const float4*)(base + (size_t)idx * DSV4_GA_D + col);
            }
            *(float4*)(kv_s + i) = v;
        }
        __syncthreads();

        // scores: warp-dot per (head, row) cell; masked rows -> -inf
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            for (int j = 0; j < DSV4_GA_KB; ++j) {
                float part = 0.0f;
#pragma unroll
                for (int dd = 0; dd < 16; ++dd)
                    part = fmaf(dsv4_bf16_to_f32(q_s[h * DSV4_GA_D + lane * 16 + dd]),
                                dsv4_bf16_to_f32(kv_s[j * DSV4_GA_D + lane * 16 + dd]), part);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    part += __shfl_xor_sync(DSV4_FULL_MASK, part, off);
                float sc = part * scale;
                if (idx_s[j] < 0) sc = -INFINITY;
                if (lane == 0) sc_s[h * DSV4_GA_KB + j] = sc;
            }
        }
        __syncthreads();

        // per head: tile max, online-softmax rescale (applied to acc below), p -> bf16,
        // fp32 sum of UNROUNDED p (reference reduces the fp32 fragment)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            float m_run = hh == 0 ? m_run0 : m_run1;
            float m_tile = fmaxf(sc_s[h * DSV4_GA_KB + lane], sc_s[h * DSV4_GA_KB + lane + 32]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m_tile = fmaxf(m_tile, __shfl_xor_sync(DSV4_FULL_MASK, m_tile, off));
            float m_new = fmaxf(m_run, m_tile);
            float rescale = expf(m_run - m_new);
            float p_a = expf(sc_s[h * DSV4_GA_KB + lane] - m_new);
            float p_b = expf(sc_s[h * DSV4_GA_KB + lane + 32] - m_new);
            pc_s[h * DSV4_GA_KB + lane] = dsv4_f32_to_bf16(p_a);
            pc_s[h * DSV4_GA_KB + lane + 32] = dsv4_f32_to_bf16(p_b);
            float tsum = p_a + p_b;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                tsum += __shfl_xor_sync(DSV4_FULL_MASK, tsum, off);
            // acc *= rescale (compile-time array selection: hh is unrolled)
#pragma unroll
            for (int dd = 0; dd < 16; ++dd) {
                if (hh == 0) acc0[dd] *= rescale; else acc1[dd] *= rescale;
            }
            if (hh == 0) { m_run0 = m_new; l0 = l0 * rescale + tsum; }
            else         { m_run1 = m_new; l1 = l1 * rescale + tsum; }
        }
        __syncthreads();

        // acc += P_bf16 · V (ascending-j accumulation, fp32)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = hh == 0 ? h0 : h1;
            for (int j = 0; j < DSV4_GA_KB; ++j) {
                float p = dsv4_bf16_to_f32(pc_s[h * DSV4_GA_KB + j]);
#pragma unroll
                for (int dd = 0; dd < 16; ++dd) {
                    float kv = dsv4_bf16_to_f32(kv_s[j * DSV4_GA_D + lane * 16 + dd]);
                    if (hh == 0) acc0[dd] = fmaf(p, kv, acc0[dd]);
                    else         acc1[dd] = fmaf(p, kv, acc1[dd]);
                }
            }
        }
    }

    // epilogue: denominator-only sink, o = acc / sum_exp -> bf16
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int hg = hb * DSV4_GA_HB + (hh == 0 ? h0 : h1);
        float m = hh == 0 ? m_run0 : m_run1;
        float l = (hh == 0 ? l0 : l1) + expf(attn_sink[hg] - m);
        const size_t o_base = ((size_t)bx * 64 + hg) * DSV4_GA_D;
#pragma unroll
        for (int dd = 0; dd < 16; ++dd) {
            float a = hh == 0 ? acc0[dd] : acc1[dd];
            o[o_base + lane * 16 + dd] = dsv4_f32_to_bf16(a / l);
        }
    }
}

// ============================================================================
// 6. dsv4_swiglu_clamp — DSV4 expert SwiGLU (§B.9 exact; swiglu_limit = 10).
//    Reads gu [BK, 2I] bf16 (rows 0..I = gate/w1, rows I..2I = up/w3 — the
//    moe_silu_bf16_b convention, gpu_batch.cu:3763), fp32 inner math:
//      up   = clamp(up, -L, +L);  gate = min(gate, L)   (NO lower bound on gate)
//      h    = silu(gate) * up     fp32, op order = the G1 CPU ref ((g*sigmoid(g))*up,
//                                    sigmoid = 1/(1+expf(-g)) with libdevice expf)
//      h   *= rw[row]             iff ROUTED (routing weight folded BEFORE w2, §B.9;
//                                    the shared-expert variant omits it)
//    Out bf16 [BK, I] — the down-GEMM input (w2's §C.1 act-sim runs as a separate pass).
//    Elementwise, one thread per output element; no per-thread arrays (zero stack).
// ============================================================================
template <bool ROUTED>
__device__ __forceinline__ void dsv4_swiglu_clamp_body(
    __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ gu,
    const float* __restrict__ rw, float limit, int I, int BK) {
    const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)BK * I) return;
    const int bk = (int)(idx / I), r = (int)(idx % I);
    float g = dsv4_bf16_to_f32(gu[(long)bk * (2 * I) + r]);
    float u = dsv4_bf16_to_f32(gu[(long)bk * (2 * I) + I + r]);
    u = fminf(fmaxf(u, -limit), limit);   // clamp(up, -L, +L)
    g = fminf(g, limit);                  // min(gate, L) — no lower bound
    const float sg = 1.0f / (1.0f + expf(-g));
    float v = (g * sg) * u;
    if (ROUTED) v *= rw[bk];
    h[idx] = dsv4_f32_to_bf16(v);
}

extern "C" {
__global__ void __launch_bounds__(256) dsv4_swiglu_clamp(
    __nv_bfloat16* h, const __nv_bfloat16* gu, const float* rw, float limit, int I, int BK) {
    dsv4_swiglu_clamp_body<true>(h, gu, rw, limit, I, BK);
}
__global__ void __launch_bounds__(256) dsv4_swiglu_clamp_shared(
    __nv_bfloat16* h, const __nv_bfloat16* gu, float limit, int I, int BK) {
    dsv4_swiglu_clamp_body<false>(h, gu, nullptr, limit, I, BK);
}
}

// ============================================================================
// 7. dsv4_rope_last_b — RoPE on the LAST rd dims of each bf16 row (DSV4 §B.1.3:
//    q[..., -64:] and kv[..., -64:], plus inverse-RoPE de-rotation of the attn output).
//    The engine's rope_b rotates the FIRST rdim (Qwen/hy_v3 convention); DSV4 rotates the
//    tail, so it needs this kernel. Complex-adjacent-pair rotation in fp32, bf16 write-back
//    (matches dsv4_cpu::apply_rope: re' = bf(re*c - im*s), im' = bf(re*s + im*c)).
//    cos/sin [max_pos, rd/2] f32; pos[rows] i32 = absolute position per row (caller-expanded;
//    for q[s,nh,hd] every head of token t shares position start_pos+t). One warp (32 lanes =
//    rd/2=32 pairs) per row, 8 warps/block. inverse=1 -> de-rotation (negate sin). 0 stack frames.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_rope_last_b(__nv_bfloat16* __restrict__ x, const float* __restrict__ cos,
                 const float* __restrict__ sin, const int* __restrict__ pos,
                 int rows, int dim, int rd, int inverse) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const int half = rd >> 1;                 // 32 for rd=64
    if (lane >= half) return;
    const int off = dim - rd;                 // rotate dims [off, off+rd)
    const int p = pos[row];
    const float c = cos[(size_t)p * half + lane];
    const float s = inverse ? -sin[(size_t)p * half + lane] : sin[(size_t)p * half + lane];
    const size_t a = (size_t)row * dim + off + lane * 2;
    const float re = dsv4_bf16_to_f32(x[a]);
    const float im = dsv4_bf16_to_f32(x[a + 1]);
    x[a]     = dsv4_f32_to_bf16(re * c - im * s);
    x[a + 1] = dsv4_f32_to_bf16(re * s + im * c);
}

// ============================================================================
// 8. dsv4_rmsnorm_b — DSV4 RMSNorm (model.py:189-202): fp32 sumsq, rsqrt(ss/dim + eps), w·(x·inv)
//    -> bf16. eps = norm_eps (1e-6). One 256-thread block per row; each thread strides over dim,
//    then a shared-mem tree reduces the 256 partials. The reduction order differs from the CPU's
//    whole-vector pairwise tree (-> tolerance-level, the G1 reduction-order class), but the math
//    is identical. Handles dim 128..4096 (compressor/indexer norms through attn/ffn norms).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_rmsnorm_b(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x,
               const float* __restrict__ w, int rows, int dim, float eps) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* __restrict__ xr = x + (size_t)row * dim;
    __nv_bfloat16* __restrict__ yr = y + (size_t)row * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float v = dsv4_bf16_to_f32(xr[i]);
        ss = fmaf(v, v, ss);
    }
    __shared__ float sm[256];
    sm[tid] = ss;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sm[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += 256) {
        float v = dsv4_bf16_to_f32(xr[i]);
        yr[i] = dsv4_f32_to_bf16(w[i] * (v * inv));
    }
}

// 8b. dsv4_rmsnorm_pair_b — R3A.1 E2 rung 2: TWO independent RMSNorms in ONE launch
//     (blockIdx.y selects the tensor). Each block reproduces dsv4_rmsnorm_b EXACTLY (same
//     256-strided partials, same tree, same elementwise form) — bitwise vs two singles.
//     Guard per side: blocks with row >= that side's rows exit.
extern "C" __global__ void __launch_bounds__(256)
dsv4_rmsnorm_pair_b(__nv_bfloat16* __restrict__ y0, const __nv_bfloat16* __restrict__ x0,
                    const float* __restrict__ w0, int rows0, int dim0,
                    __nv_bfloat16* __restrict__ y1, const __nv_bfloat16* __restrict__ x1,
                    const float* __restrict__ w1, int rows1, int dim1, float eps) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    __nv_bfloat16* __restrict__ y;
    const __nv_bfloat16* __restrict__ x;
    const float* __restrict__ w;
    int dim;
    if (blockIdx.y == 0) {
        if (row >= rows0) return;
        y = y0; x = x0; w = w0; dim = dim0;
    } else {
        if (row >= rows1) return;
        y = y1; x = x1; w = w1; dim = dim1;
    }
    const __nv_bfloat16* __restrict__ xr = x + (size_t)row * dim;
    __nv_bfloat16* __restrict__ yr = y + (size_t)row * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float v = dsv4_bf16_to_f32(xr[i]);
        ss = fmaf(v, v, ss);
    }
    __shared__ float sm[256];
    sm[tid] = ss;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sm[0] / (float)dim + eps);
    for (int i = tid; i < dim; i += 256) {
        float v = dsv4_bf16_to_f32(xr[i]);
        yr[i] = dsv4_f32_to_bf16(w[i] * (v * inv));
    }
}

// 7b. dsv4_rope_pair_b — R3A.1 E2 rung 2: the q and kv RoPEs of one attention layer in ONE
//     launch, positions INLINE (q: p = start_pos + row/nh; kv: p = start_pos + row — identical
//     integers to the iota_positions arrays they replace). blockIdx.y = 0 → q (rows0 = s*nh),
//     1 → kv (rows1 = s). Same cos/sin tables, same rd, inverse = 0 (de-rotation sites keep
//     dsv4_rope_last_b). Per-element math identical to dsv4_rope_last_b — bitwise vs
//     iota+rope pairs. One warp per row, 8 warps/block.
extern "C" __global__ void __launch_bounds__(256)
dsv4_rope_pair_b(__nv_bfloat16* __restrict__ x0, __nv_bfloat16* __restrict__ x1,
                 const float* __restrict__ cos, const float* __restrict__ sin,
                 int start_pos, int nh, int rows0, int rows1, int dim, int rd) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    __nv_bfloat16* __restrict__ x;
    int p;
    if (blockIdx.y == 0) {
        if (row >= rows0) return;
        x = x0; p = start_pos + row / nh;
    } else {
        if (row >= rows1) return;
        x = x1; p = start_pos + row;
    }
    const int half = rd >> 1;                 // 32 for rd=64
    if (lane >= half) return;
    const int off = dim - rd;                 // rotate dims [off, off+rd)
    const float c = cos[(size_t)p * half + lane];
    const float s = sin[(size_t)p * half + lane];
    const size_t a = (size_t)row * dim + off + lane * 2;
    const float re = dsv4_bf16_to_f32(x[a]);
    const float im = dsv4_bf16_to_f32(x[a + 1]);
    x[a] = dsv4_f32_to_bf16(re * c - im * s);
    x[a + 1] = dsv4_f32_to_bf16(re * s + im * c);
}

// 7c. dsv4_rope_q_inline_b — q-side rope with INLINE positions (p = start_pos + row/nh —
//     identical integers to the iota_positions array it replaces) + a runtime inverse flag
//     (the attention-output de-rotation passes inverse=1). Per-element math identical to
//     dsv4_rope_last_b incl. the inverse sin negation — bitwise vs iota + rope_last.
extern "C" __global__ void __launch_bounds__(256)
dsv4_rope_q_inline_b(__nv_bfloat16* __restrict__ x, const float* __restrict__ cos,
                     const float* __restrict__ sin, int start_pos, int nh,
                     int rows, int dim, int rd, int inverse) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const int half = rd >> 1;                 // 32 for rd=64
    if (lane >= half) return;
    const int off = dim - rd;                 // rotate dims [off, off+rd)
    const int p = start_pos + row / nh;
    const float c = cos[(size_t)p * half + lane];
    const float s = inverse ? -sin[(size_t)p * half + lane] : sin[(size_t)p * half + lane];
    const size_t a = (size_t)row * dim + off + lane * 2;
    const float re = dsv4_bf16_to_f32(x[a]);
    const float im = dsv4_bf16_to_f32(x[a + 1]);
    x[a] = dsv4_f32_to_bf16(re * c - im * s);
    x[a + 1] = dsv4_f32_to_bf16(re * s + im * c);
}

// ============================================================================
// 9. mHC 4-stream mixing (§B.8) — hc_pre / hc_post wrap every sublayer. hc_pre = RMS over the
//    flattened 4-stream [s, hc*dim] -> rsqrt; mixes = hc_fn[24, hc*dim] @ xf * rsqrt; Sinkhorn ->
//    pre/post/comb; collapse y = Σ_h pre[h]·x[h]. hc_post = post[k]·sublayer_out + Σ_j comb[j,k]·resid.
//    All fp32 inner, bf16 in/out. One block per token (decode/prefill/verify uniform); reductions
//    are block-trees (tolerance-level vs the CPU pairwise tree, the G1 class).
// ============================================================================

// 9a. per-token rsqrt( mean(xf²) + eps ) over the flattened hc*dim streams -> rsqrt[s].
extern "C" __global__ void __launch_bounds__(256)
dsv4_hc_pre_rsqrt_b(float* __restrict__ rsqrt, const __nv_bfloat16* __restrict__ x,
                    int s, int hcdim, float eps) {
    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (t >= s) return;
    const __nv_bfloat16* __restrict__ xr = x + (size_t)t * hcdim;
    float ss = 0.0f;
    for (int i = tid; i < hcdim; i += 256) { float v = dsv4_bf16_to_f32(xr[i]); ss = fmaf(v, v, ss); }
    __shared__ float sm[256];
    sm[tid] = ss; __syncthreads();
    for (int f = 128; f > 0; f >>= 1) { if (tid < f) sm[tid] += sm[tid + f]; __syncthreads(); }
    if (tid == 0) rsqrt[t] = rsqrtf(sm[0] / (float)hcdim + eps);
}

// 9b. mixes[t, m] = (Σ_d hc_fn[m, d] · xf[t, d]) · rsqrt[t], fp32. Grid (24, s); one block per (m, t).
extern "C" __global__ void __launch_bounds__(256)
dsv4_hc_mixes_b(float* __restrict__ mixes, const float* __restrict__ hc_fn,
                const __nv_bfloat16* __restrict__ x, const float* __restrict__ rsqrt,
                int s, int hcdim) {
    const int tid = threadIdx.x;
    const int m = blockIdx.x;   // 0..23
    const int t = blockIdx.y;   // 0..s-1
    if (m >= 24 || t >= s) return;
    const __nv_bfloat16* __restrict__ xr = x + (size_t)t * hcdim;
    const float* __restrict__ fr = hc_fn + (size_t)m * hcdim;
    float acc = 0.0f;
    for (int i = tid; i < hcdim; i += 256) acc = fmaf(fr[i], dsv4_bf16_to_f32(xr[i]), acc);
    __shared__ float sm[256];
    sm[tid] = acc; __syncthreads();
    for (int f = 128; f > 0; f >>= 1) { if (tid < f) sm[tid] += sm[tid + f]; __syncthreads(); }
    if (tid == 0) mixes[(size_t)t * 24 + m] = sm[0] * rsqrt[t];
}

// 9c. hc_pre collapse: y[t, d] = Σ_h pre[t, h] · xf[t, h*dim + d] -> bf16. Grid (s,); elementwise dim.
extern "C" __global__ void __launch_bounds__(256)
dsv4_hc_collapse_b(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x,
                   const float* __restrict__ pre, int s, int dim, int hc) {
    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (t >= s) return;
    const size_t base = (size_t)t * hc * dim;
    for (int d = tid; d < dim; d += 256) {
        float acc = 0.0f;
        for (int h = 0; h < hc; ++h) acc += pre[t * hc + h] * dsv4_bf16_to_f32(x[base + (size_t)h * dim + d]);
        y[(size_t)t * dim + d] = dsv4_f32_to_bf16(acc);
    }
}

// 9e. FUSED hc_pre (T6 pp-tail): replaces the 4-launch chain (9a rsqrt + 9b mixes +
//     4. sinkhorn + 9c collapse) with ONE launch, bitwise-identical per element.
//     Three tokens per 768-thread block: token group g = tid/256, lane lt = tid%256.
//     Phase 1: per-lane partial sums over the token's hcdim (ascending i = lt, lt+256, ...)
//     — ss (rsqrt chain, 9a) + 24 mix dots (9b chains) in one pass, hc_fn rows SHARED across
//     the block's 3 tokens (L2 traffic ÷3 vs the (24, s) grid's per-token re-reads).
//     Phase 2: 25 stride-halving trees (f = 128..1, one __syncthreads per f — the 9a/9b
//     two-sync form) run INTERLEAVED over sm[k][g*256 + lt]; each array's add sequence is
//     identical to the separate kernels' -> bitwise.
//     Phase 3: lane 0 writes rsqrt + mixes (mixes = tree * rsqrt, the 9b store) and runs the
//     sinkhorn (kernel 4's single-thread sequence) -> pre/post/comb.
//     Phase 4: collapse (9c) re-reads x (L2-hot) with pre from the pre buffer.
//     Grid (ceil(s/3),); dynamic smem 25 * 768 * 4 = 76800 B (opt-in, GB10 cap 99 KB).
extern "C" __global__ void __launch_bounds__(768)
dsv4_hc_pre_fused_b(float* __restrict__ rsqrt, float* __restrict__ mixes,
                    float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
                    __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x,
                    const float* __restrict__ hc_fn, const float* __restrict__ hc_scale,
                    const float* __restrict__ hc_base, int s, int hcdim, int dim, int hc,
                    float norm_eps) {
    extern __shared__ float sm[];          // [25][768]
    const int tid = threadIdx.x;
    const int g = tid >> 8;                // token within block (0..2)
    const int lt = tid & 255;              // lane within the token's 256
    const int t = blockIdx.x * 3 + g;
    const bool valid = t < s;
    const size_t tbase = (size_t)t * hcdim;
    float ss = 0.0f;
    float acc[24];
#pragma unroll
    for (int m = 0; m < 24; ++m) acc[m] = 0.0f;
    if (valid) {
        const __nv_bfloat16* __restrict__ xr = x + tbase;
        // hc_fn rows are shared across the block's 3 tokens (same hcdim layout).
        for (int i = lt; i < hcdim; i += 256) {
            const float v = dsv4_bf16_to_f32(xr[i]);
            ss = fmaf(v, v, ss);
#pragma unroll
            for (int m = 0; m < 24; ++m) acc[m] = fmaf(hc_fn[(size_t)m * hcdim + i], v, acc[m]);
        }
    }
    float* __restrict__ smg = sm + g * 256;
    smg[lt] = ss;
#pragma unroll
    for (int m = 0; m < 24; ++m) smg[(m + 1) * 768 + lt] = acc[m];
    __syncthreads();
    for (int f = 128; f > 0; f >>= 1) {
        if (lt < f) {
            smg[lt] += smg[lt + f];
#pragma unroll
            for (int m = 0; m < 24; ++m) smg[(m + 1) * 768 + lt] += smg[(m + 1) * 768 + lt + f];
        }
        __syncthreads();
    }
    if (valid && lt == 0) {
        const float rinv = rsqrtf(smg[0] / (float)hcdim + norm_eps);
        rsqrt[t] = rinv;
        const float* m24 = smg + 768;
        const float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];
#pragma unroll
        for (int m = 0; m < 24; ++m) {
            const float mv = m24[m * 768] * rinv;
            mixes[(size_t)t * 24 + m] = mv;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            pre[(size_t)t * 4 + j] = dsv4_sigmoid(fmaf(m24[j * 768] * rinv, s0, hc_base[j])) + DSV4_HC_EPS;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            post[(size_t)t * 4 + j] = 2.0f * dsv4_sigmoid(fmaf(m24[(4 + j) * 768] * rinv, s1, hc_base[4 + j]));
        float c[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k)
                c[j][k] = fmaf(m24[(8 + 4 * j + k) * 768] * rinv, s2, hc_base[8 + 4 * j + k]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float mx = fmaxf(fmaxf(c[j][0], c[j][1]), fmaxf(c[j][2], c[j][3]));
#pragma unroll
            for (int k = 0; k < 4; ++k) c[j][k] = expf(c[j][k] - mx);
            float rs = (c[j][0] + c[j][2]) + (c[j][1] + c[j][3]);
#pragma unroll
            for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / rs + DSV4_HC_EPS;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float cs = (c[0][k] + c[2][k]) + (c[1][k] + c[3][k]) + DSV4_HC_EPS;
#pragma unroll
            for (int j = 0; j < 4; ++j) c[j][k] = c[j][k] / cs;
        }
#pragma unroll
        for (int it = 0; it < 19; ++it) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float rs = (c[j][0] + c[j][2]) + (c[j][1] + c[j][3]) + DSV4_HC_EPS;
#pragma unroll
                for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / rs;
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float cs = (c[0][k] + c[2][k]) + (c[1][k] + c[3][k]) + DSV4_HC_EPS;
#pragma unroll
                for (int j = 0; j < 4; ++j) c[j][k] = c[j][k] / cs;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) comb[((size_t)t * 4 + j) * 4 + k] = c[j][k];
    }
    __syncthreads();
    if (valid) {
        const size_t base = tbase;
        for (int d = lt; d < dim; d += 256) {
            float a = 0.0f;
            for (int h = 0; h < hc; ++h)
                a += pre[(size_t)t * hc + h] * dsv4_bf16_to_f32(x[base + (size_t)h * dim + d]);
            y[(size_t)t * dim + d] = dsv4_f32_to_bf16(a);
        }
    }
}

// 9d. hc_post: out[t, k*dim+d] = post[t,k]·sublayer_out[t,d] + Σ_j comb[t, j*hc+k]·resid[t, j*dim+d] -> bf16.
extern "C" __global__ void __launch_bounds__(256)
dsv4_hc_post_b(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ sub_out,
               const __nv_bfloat16* __restrict__ resid, const float* __restrict__ post,
               const float* __restrict__ comb, int s, int dim, int hc) {
    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (t >= s) return;
    const int hcd = hc * dim;
    const size_t tbase = (size_t)t * hcd;
    for (int i = tid; i < hcd; i += 256) {
        const int k = i / dim;
        const int d = i - k * dim;
        float acc = post[t * hc + k] * dsv4_bf16_to_f32(sub_out[(size_t)t * dim + d]);
        for (int j = 0; j < hc; ++j)
            acc += comb[((size_t)t * hc + j) * hc + k] * dsv4_bf16_to_f32(resid[tbase + (size_t)j * dim + d]);
        out[tbase + i] = dsv4_f32_to_bf16(acc);
    }
}

// ============================================================================
// 10. dsv4_router_score_b — MoE router score (§B.9): sqrtsoftplus(gate_w[e] · x[t]) fp32.
//     gate_w [n_exp, dim] fp32 (bf16 upcast at load); x [s, dim] bf16-valued; out scores [s, n_exp]
//     fp32. sqrtsoftplus = sqrt( s>20 ? s : log1p(exp(s)) ) (PyTorch softplus threshold-20). Grid
//     (n_exp, s); one block per (expert, token), 256-thread dot over dim, tree reduce. The SELECTION
//     (bias-add + topk, or tid2eid gather for hash layers) + weight renorm are separate steps.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_router_score_b(float* __restrict__ scores, const float* __restrict__ gate_w,
                    const __nv_bfloat16* __restrict__ x, int s, int dim, int n_exp) {
    const int tid = threadIdx.x;
    const int e = blockIdx.x;   // 0..n_exp-1
    const int t = blockIdx.y;   // 0..s-1
    if (e >= n_exp || t >= s) return;
    const float* __restrict__ wr = gate_w + (size_t)e * dim;
    const __nv_bfloat16* __restrict__ xr = x + (size_t)t * dim;
    float acc = 0.0f;
    for (int i = tid; i < dim; i += 256) acc = fmaf(wr[i], dsv4_bf16_to_f32(xr[i]), acc);
    __shared__ float sm[256];
    sm[tid] = acc; __syncthreads();
    for (int f = 128; f > 0; f >>= 1) { if (tid < f) sm[tid] += sm[tid + f]; __syncthreads(); }
    float v = sm[0];
    float sp = (v > 20.0f) ? v : log1pf(expf(v));
    if (tid == 0) scores[(size_t)t * n_exp + e] = sqrtf(sp);
}

// 10b. bias-add for SELECTION only (§B.9): biased[t,e] = scores[t,e] + bias[e]. The combine
//      weights still come from the UN-biased scores (kernel 10d). Elementwise; hash layers skip this.
extern "C" __global__ void __launch_bounds__(256)
dsv4_router_bias_add_b(float* __restrict__ biased, const float* __restrict__ scores,
                       const float* __restrict__ bias, int s, int n_exp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)s * n_exp) return;
    biased[idx] = scores[idx] + bias[idx % n_exp];
}

// 10c. hash-layer selection (layers 0-2): sel[t,j] = tid2eid[ids[t]*topk + j]. Elementwise gather.
extern "C" __global__ void __launch_bounds__(256)
dsv4_router_tid2eid_b(int* __restrict__ sel, const int* __restrict__ tid2eid,
                      const int* __restrict__ ids, int s, int topk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)s * topk) return;
    sel[idx] = tid2eid[(long)ids[idx / topk] * topk + (idx % topk)];
}

// 10d. weight gather + renorm (§B.9): weights[t,j] = scores[t, sel[t,j]] / wsum * route_scale,
//      wsum = Σ_j scores[t, sel[t,j]] (the UN-biased scores). One block per token, topk threads.
extern "C" __global__ void __launch_bounds__(256)
dsv4_router_weights_b(float* __restrict__ weights, const float* __restrict__ scores,
                      const int* __restrict__ sel, int s, int n_exp, int topk, float route_scale) {
    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (t >= s) return;
    __shared__ float ws[16];
    float w = 0.0f;
    if (tid < topk) {
        const int e = sel[t * topk + tid];
        w = scores[(size_t)t * n_exp + e];
        ws[tid] = w;
    }
    __syncthreads();
    float wsum = 0.0f;
    for (int j = 0; j < topk; ++j) wsum += ws[j];
    if (tid < topk) weights[t * topk + tid] = (wsum > 0.0f) ? (w / wsum * route_scale) : 0.0f;
}

// ============================================================================
// 11. dsv4_hc_head_b — final trunk collapse (§B.8 hc_head, model.py:709-716). Sigmoid-only:
//     NO post/comb/Sinkhorn (unlike hc_pre). rsqrt over the flattened hc*dim streams;
//     mixes[h] = (hc_fn[h] · xf) · rsqrt; pre[h] = sigmoid(mixes·scale + base[h]) + hc_eps;
//     y[d] = Σ_h pre[h] · xf[h*dim + d] -> bf16 (one round). hc_fn is [hc, hc*dim] fp32.
//     One block per token; reduction trees match dsv4_hc_pre_rsqrt_b / dsv4_hc_mixes_b
//     (stride-halving two-sync form — safe; tolerance-level vs the CPU pairwise tree, G1 class).
//     hc is runtime but DSV4-hardcoded to 4 (the G1 amendment; hc_fn[4,16384], base[4], scale[1]).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_hc_head_b(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x,
               const float* __restrict__ hc_fn, const float* __restrict__ hc_base,
               const float* __restrict__ hc_scale, int s, int dim, int hc, float norm_eps, float hc_eps) {
    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (t >= s) return;
    const int hcd = hc * dim;
    const __nv_bfloat16* __restrict__ xr = x + (size_t)t * hcd;
    const float scale = hc_scale[0];
    __shared__ float sm[256];
    __shared__ float pre_s[8];   // hc <= 8 (DSV4: 4); sized for the unroll below
    // rsqrt over the flattened hc*dim streams.
    float ss = 0.0f;
    for (int i = tid; i < hcd; i += 256) { float v = dsv4_bf16_to_f32(xr[i]); ss = fmaf(v, v, ss); }
    sm[tid] = ss; __syncthreads();
    for (int f = 128; f > 0; f >>= 1) { if (tid < f) sm[tid] += sm[tid + f]; __syncthreads(); }
    const float inv = rsqrtf(sm[0] / (float)hcd + norm_eps);
    // hc dot products (hc=4 compile-time-known; loop body is invariant -> unrolled by ptxas).
    for (int h = 0; h < hc; ++h) {
        const float* __restrict__ fr = hc_fn + (size_t)h * hcd;
        float acc = 0.0f;
        for (int i = tid; i < hcd; i += 256) acc = fmaf(fr[i], dsv4_bf16_to_f32(xr[i]), acc);
        sm[tid] = acc; __syncthreads();
        for (int f = 128; f > 0; f >>= 1) { if (tid < f) sm[tid] += sm[tid + f]; __syncthreads(); }
        if (tid == 0) pre_s[h] = dsv4_sigmoid(fmaf(sm[0] * inv, scale, hc_base[h])) + hc_eps;
        __syncthreads();
    }
    // collapse y[t,d] = Σ_h pre_s[h] · xf[h*dim + d] -> bf16 (one round at store).
    for (int d = tid; d < dim; d += 256) {
        float acc = 0.0f;
        for (int h = 0; h < hc; ++h) acc += pre_s[h] * dsv4_bf16_to_f32(xr[(size_t)h * dim + d]);
        y[(size_t)t * dim + d] = dsv4_f32_to_bf16(acc);
    }
}

// ============================================================================
// 12. dsv4_embed_b — input embedding gather + replicate to hc streams (model.py:916
//     forward_embed: h = embed(ids).view(B,S,1,dim).repeat(1,1,hc,1)). out [s, hc, dim]
//     bf16 = embed[ids[t], :] copied into every stream. Embed is stored bf16 -> gather is a
//     straight copy (no rounding). One block per token; 256 threads stride over hc*dim.
//     ids are int32 (the loader's tid2eid is i32; the I64 npz narrows at the replay boundary).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_embed_b(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ embed,
             const int* __restrict__ ids, int s, int dim, int hc) {
    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (t >= s) return;
    const int id = ids[t];
    const __nv_bfloat16* __restrict__ src = embed + (size_t)id * dim;
    __nv_bfloat16* __restrict__ dst = out + (size_t)t * hc * dim;
    for (int i = tid; i < hc * dim; i += 256) dst[i] = src[i % dim];
}

// ============================================================================
// 13. dsv4_main_hidden_b — DSpark trunk interface (§B.10): collect h.mean(streams) after
//     layers 40/41/42 and concatenate → main_hidden [s, 3*dim]. Each layer's post-block
//     streams are [s, hc*dim]; mean over the hc axis gives [s, dim]; concat 3 → [s, 3*dim].
//     out[t, li*dim + d] = (1/hc) Σ_h x{li}[t, h*dim + d]. li ∈ {0,1,2} ↔ layers 40/41/42.
//     Grid over s*3*dim threads (one output element each); hc=4 read loop is unrolled.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_main_hidden_b(__nv_bfloat16* __restrict__ out,
                   const __nv_bfloat16* __restrict__ x40,
                   const __nv_bfloat16* __restrict__ x41,
                   const __nv_bfloat16* __restrict__ x42,
                   int s, int hc, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (long)s * 3 * dim;
    if (idx >= total) return;
    const int d = idx % dim;
    const int li = (idx / dim) % 3;
    const int t = idx / (3 * dim);
    const __nv_bfloat16* __restrict__ x = (li == 0) ? x40 : (li == 1) ? x41 : x42;
    const size_t base = (size_t)t * hc * dim;
    float sum = 0.0f;
    for (int h = 0; h < hc; ++h) sum += dsv4_bf16_to_f32(x[base + (size_t)h * dim + d]);
    out[(size_t)t * 3 * dim + (size_t)li * dim + d] = dsv4_f32_to_bf16(sum / (float)hc);
}

// ============================================================================
// 14. dsv4_markov_gather_b — DSpark Markov bigram embedding gather: out[i, rank] =
//     markov_w1[output_ids[i], rank] (bf16-valued). markov_w1 is [vocab, rank] bf16.
//     block i = 5 rows; this gather feeds the sequential Markov chain (§B.10 forward_head).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_markov_gather_b(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ w1,
                     const int* __restrict__ ids, int block, int rank) {
    const int tid = threadIdx.x;
    const int i = blockIdx.x;
    if (i >= block) return;
    const int id = ids[i];
    const __nv_bfloat16* __restrict__ src = w1 + (size_t)id * rank;
    __nv_bfloat16* __restrict__ dst = out + (size_t)i * rank;
    for (int j = tid; j < rank; j += 256) dst[j] = src[j];
}

// ============================================================================
// dsv4_iota_b — device-side arithmetic positions (R3A.4 P3): out[i] = start + (i*mul)/div
// (i32 truncating division, exactly the host expressions start+i/nh, start+i, start+b*ratio).
// Replaces the per-layer host Vec + htod_sync_copy position uploads (each a full
// cuCtxSynchronize — ~167 syncs per prefill chunk).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_iota_b(int* __restrict__ out, int start, int mul, int div, int n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = start + (int)((i * mul) / div);
}
