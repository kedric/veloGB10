// DSV4 Phase-3 LANE 3B module: the §B.5 compressor (CSA overlap + HCA) and the §B.6
// indexer machinery. Semantics target: the G1-proven CPU reference src/dsv4_cpu.rs
// (Compressor::forward / indexer_forward), replicated operation-for-operation:
//   - fp32 wkv/wgate GEMM: per-element rounded product + pairwise-tree sum — the
//     EXACT dsv4_cpu::dot_tree order (double-buffered smem tree), so the GEMM is
//     bit-exact vs the CPU reference, not just tolerance-level.
//   - softmax-pool over ratio tokens: CPU's serial ascending-j order (max, z, then
//     p_j = e_j/z, acc += kv_j·p_j) with __fadd_rn/__fmul_rn (no FMA contraction —
//     Rust never contracts) and exp computed in double then rounded (matches the
//     host's near-correctly-rounded expf except on a ~1e-3 fraction of 1-ulp glibc
//     misses; the bf16/fp8-sim downstream snaps those).
//   - decode state machine incl. the CSA slots-4..7 write + 8-tap cat/shift and the
//     HCA slot start_pos%128 — one block, exact CPU order.
//   - indexer score chain: dot8 replica (__fmul_rn/__fadd_rn, 8 accumulators + the
//     ((0+1)+(2+3))+((4+5)+(6+7)) final tree), bf16 einsum-out rounding, exact relu,
//     bf16 ×weights products, ascending-head fp32 sum, bf16 store — bit-exact vs
//     dsv4_cpu's indexer scoring given identical q/kv/weights.
// All per-thread arrays are compile-time indexed (full unroll): ptxas must report
// ZERO stack frames for every kernel here (AGENTS.md §4).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <mma.h>

// build.rs hashes this file into KERNEL_BUILD_ID and passes the same -D as the other
// DSV4 modules; the production loader (src/dsv4_gpu.rs) asserts the stamp per module.
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void dsv4_kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

#define DSV4C_FULL_MASK 0xffffffffu

__device__ __forceinline__ float dsv4c_bf16_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 dsv4c_f32_to_bf16(float v) { return __float2bfloat16(v); }

// dtype-generic load/store for the templated GEMM (float or __nv_bfloat16 operands)
__device__ __forceinline__ float dsv4c_to_f32(float v) { return v; }
__device__ __forceinline__ float dsv4c_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void dsv4c_store(float& dst, float v) { dst = v; }
__device__ __forceinline__ void dsv4c_store(__nv_bfloat16& dst, float v) { dst = __float2bfloat16(v); }

// bf() replica: f32 -> bf16 (RNE) -> f32 (dsv4_cpu::bf).
__device__ __forceinline__ float dsv4c_bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

// exp matching the host's near-correctly-rounded expf: compute in double, round to float.
__device__ __forceinline__ float dsv4c_exp(float v) { return (float)exp((double)v); }

// ---- UE8M0 pow2 scale bit trick + FP8 RNE cast (mirrors gpu_dsv4.cu exactly) ----
__device__ __forceinline__ float dsv4c_round_scale_pow2(float amax, float inv) {
    float v = amax * inv;
    uint32_t b = __float_as_uint(v);
    int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t)(e + 127) << 23);
}
__device__ __forceinline__ uint8_t dsv4c_f32_to_e8m0(float s) {
    return __nv_cvt_float_to_e8m0(s, __NV_SATFINITE, cudaRoundPosInf);
}
__device__ __forceinline__ uint8_t dsv4c_f32_to_fp8(float v) {
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float dsv4c_fp8_to_f32(uint8_t c) {
    return __half2float(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3));
}

// ============================================================================
// 1. dsv4_comp_state_init_b — frontier state init (§B.5: kv_state zeros, score_state
//    −inf). n = coff·ratio · coff·head_dim elements per tensor.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_state_init_b(float* __restrict__ kv_state, float* __restrict__ score_state, long n) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    kv_state[i] = 0.0f;
    score_state[i] = -INFINITY;
}

// ============================================================================
// 2. dsv4_comp_gemm_tree_b — fp32 GEMM y[s,n] = x[s,k]·w[n,k]ᵀ replicating
//    dsv4_cpu::dot_tree EXACTLY: per-element f32-rounded product, then the
//    pairwise halving tree (double-buffered smem; the CPU's in-place tree is a
//    pure function of the array). One 256-thread block per output element.
//    WT = weight element type (float for compressor wkv/wgate — bf16-sourced but
//    used fp32; __nv_bfloat16 for indexer weights_proj), YT = output (float or
//    __nv_bfloat16 — bf16-out replicates gemm_bf16's final rounding).
//    Contract: k <= 4096 (smem 2×4096×4 B), any s/n. Batch-invariant by
//    construction: each output element is one fixed-order reduction.
// ============================================================================
#define DSV4C_GEMM_KMAX 4096

template <typename WT, typename YT>
__device__ __forceinline__ void dsv4c_gemm_tree_body(
    const __nv_bfloat16* __restrict__ x, const WT* __restrict__ w,
    YT* __restrict__ y, int s, int k, int n) {
    __shared__ float bufa[DSV4C_GEMM_KMAX];
    __shared__ float bufb[DSV4C_GEMM_KMAX];
    const int col = blockIdx.x, row = blockIdx.y;
    if (col >= n || row >= s) return;
    const int tid = threadIdx.x;
    const __nv_bfloat16* __restrict__ xr = x + (size_t)row * (size_t)k;
    const WT* __restrict__ wr = w + (size_t)col * (size_t)k;
    for (int j = tid; j < k; j += 256) {
        float xv = dsv4c_bf16_to_f32(xr[j]);
        float wv = dsv4c_to_f32(wr[j]);
        bufa[j] = __fmul_rn(xv, wv); // per-element rounded product (dot_tree's buf)
    }
    __syncthreads();
    float* a = bufa;
    float* b = bufb;
    int len = k;
    while (len > 1) {
        int w = len >> 1;
        for (int i = tid; i < w; i += 256) b[i] = __fadd_rn(a[2 * i], a[2 * i + 1]);
        if (len & 1) {
            if (tid == 0) b[w] = a[len - 1];
            w += 1;
        }
        __syncthreads();
        float* t = a; a = b; b = t;
        len = w;
    }
    if (tid == 0) {
        dsv4c_store(y[(size_t)row * n + col], a[0]);
    }
}

extern "C" {
// compressor wkv/wgate: fp32 weights, fp32 out
__global__ void __launch_bounds__(256) dsv4_comp_gemm_tree_f32w_b(
    const __nv_bfloat16* x, const float* w, float* y, int s, int k, int n) {
    dsv4c_gemm_tree_body<float, float>(x, w, y, s, k, n);
}
// indexer weights_proj: bf16 weights, bf16 out (gemm_bf16)
__global__ void __launch_bounds__(256) dsv4_comp_gemm_tree_bf16w_bf16out_b(
    const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, int s, int k, int n) {
    dsv4c_gemm_tree_body<__nv_bfloat16, __nv_bfloat16>(x, w, y, s, k, n);
}
}

// ============================================================================
// 3. dsv4_comp_prefill_pool_b — §B.5 prefill assembly + softmax-pool.
//    grid = nb + 1 blocks of head_dim threads:
//      blocks b < nb  — pool block b: serial ascending-j over nrow rows per column,
//                       the exact dsv4_cpu order; overlap rows (j < ratio) read the
//                       PREVIOUS block's dims :d (block 0 at start_pos==0: zeros/−inf;
//                       at start_pos>0 with carry: the frontier state rows [0..ratio)),
//                       rows ratio..2·ratio the current block's dims d:.
//      block nb       — frontier stash: LAST FULL block → state rows [0..ratio)
//                       (overlap only, when cutoff ≥ ratio) and the remainder rows →
//                       state [(overlap?ratio:0) .. +rem), scores with ape added.
//    kv_full/score_full [s, coff·d] fp32 (kernel 2's output), pooled [nb, d] fp32.
//    `carry` (0/1): when set AND coff==2, block 0's overlap rows (j<ratio) read from
//    kv_state/score_state[0..ratio) — the frontier carried from the previous chunk —
//    instead of zeros/−inf. score_state already carries score+ape (the stash adds it),
//    so NO ape re-add for the carry arm. This makes chunk-2+ ride the batched pool
//    bitwise-identical to the one-shot prefill (§12.B.5 — the frontier IS the previous
//    block's overlap rows). HCA (coff==1) has no overlap → carry is a no-op there.
// ============================================================================
__device__ __forceinline__ void dsv4c_prefill_row(
    int b, int j, int c, int ratio, int d, int coff, int carry,
    const float* __restrict__ kv_full, const float* __restrict__ score_full,
    const float* __restrict__ ape,
    const float* __restrict__ kv_state, const float* __restrict__ score_state,
    float& kv_out, float& sc_out) {
    const int cd = coff * d;
    if (coff == 2) {
        if (j < ratio) {
            if (b == 0) {
                if (carry) {
                    // frontier carry from the previous chunk: kv_state/score_state[0..ratio)
                    // hold the previous block's overlap rows (score already +ape from the stash).
                    kv_out = kv_state[j * cd + c];
                    sc_out = score_state[j * cd + c];
                } else {
                    kv_out = 0.0f;
                    sc_out = -INFINITY;
                }
                return;
            }
            const int src = ((b - 1) * ratio + j) * cd + c;
            kv_out = kv_full[src];
            sc_out = __fadd_rn(score_full[src], ape[j * cd + c]);
        } else {
            const int src = (b * ratio + (j - ratio)) * cd + d + c;
            kv_out = kv_full[src];
            sc_out = __fadd_rn(score_full[src], ape[(j - ratio) * cd + d + c]);
        }
    } else {
        const int src = (b * ratio + j) * cd + c;
        kv_out = kv_full[src];
        sc_out = __fadd_rn(score_full[src], ape[j * cd + c]);
    }
}

extern "C" __global__ void __launch_bounds__(512)
dsv4_comp_prefill_pool_b(const float* __restrict__ kv_full, const float* __restrict__ score_full,
                         const float* __restrict__ ape, float* __restrict__ pooled,
                         float* __restrict__ kv_state, float* __restrict__ score_state,
                         int s, int cutoff, int ratio, int d, int coff, int nb, int carry) {
    const int c = threadIdx.x;
    const int cd = coff * d;
    const int nrow = (coff == 2) ? 2 * ratio : ratio;
    const int b = blockIdx.x;
    if (b >= nb) return;
    // ---- pool one block (thread c = column dd) ----
    if (c < d) {
        float mx = -INFINITY;
        for (int j = 0; j < nrow; ++j) {
            float kvj, scj;
            dsv4c_prefill_row(b, j, c, ratio, d, coff, carry, kv_full, score_full, ape, kv_state, score_state, kvj, scj);
            mx = fmaxf(mx, scj);
        }
        float z = 0.0f;
        for (int j = 0; j < nrow; ++j) {
            float kvj, scj;
            dsv4c_prefill_row(b, j, c, ratio, d, coff, carry, kv_full, score_full, ape, kv_state, score_state, kvj, scj);
            z = __fadd_rn(z, dsv4c_exp(scj - mx));
        }
        float acc = 0.0f;
        for (int j = 0; j < nrow; ++j) {
            float kvj, scj;
            dsv4c_prefill_row(b, j, c, ratio, d, coff, carry, kv_full, score_full, ape, kv_state, score_state, kvj, scj);
            float p = dsv4c_exp(scj - mx) / z; // div.rn (nvcc default prec-div)
            acc = __fadd_rn(acc, __fmul_rn(kvj, p));
        }
        pooled[(size_t)b * d + c] = acc;
    }
}

// ============================================================================
// 3b. dsv4_comp_prefill_stash_b — frontier stash (split from the pool kernel to avoid a
//     race: the pool's block 0 READS kv_state[0..ratio*cd) for the carry, while the stash
//     WRITES kv_state[0..ratio*cd) — concurrent CTAs in one launch would race. Two
//     sequential launches on the blocking stream eliminate the hazard).
//     LAST FULL block → state rows [0..ratio) (overlap only, when cutoff ≥ ratio) and
//     the remainder rows → state [(overlap?ratio:0) .. +rem), scores with ape added.
// ============================================================================
extern "C" __global__ void __launch_bounds__(512)
dsv4_comp_prefill_stash_b(const float* __restrict__ kv_full, const float* __restrict__ score_full,
                          const float* __restrict__ ape,
                          float* __restrict__ kv_state, float* __restrict__ score_state,
                          int s, int cutoff, int ratio, int d, int coff, int do_stash) {
    const int c = threadIdx.x;
    const int cd = coff * d;
    // ---- frontier stash (single block) ----
    const int rem = s - cutoff;
    if (do_stash) {
        // LAST FULL block rows j = 0..ratio-1 → state rows [0..ratio), score + ape[j]
        for (int idx = c; idx < ratio * cd; idx += blockDim.x) {
            const int j = idx / cd, cc = idx - j * cd;
            const int src = (cutoff - ratio + j) * cd + cc;
            kv_state[j * cd + cc] = kv_full[src];
            score_state[j * cd + cc] = __fadd_rn(score_full[src], ape[j * cd + cc]);
        }
    }
    if (rem > 0) {
        const int offset = (coff == 2) ? ratio : 0;
        for (int idx = c; idx < rem * cd; idx += blockDim.x) {
            const int j = idx / cd, cc = idx - j * cd;
            const int dst = (offset + j) * cd + cc;
            const int src = (cutoff + j) * cd + cc;
            kv_state[dst] = kv_full[src];
            score_state[dst] = __fadd_rn(score_full[src], ape[j * cd + cc]);
        }
    }
}

// ============================================================================
// 4. dsv4_comp_decode_b — §B.5 decode state machine, ONE block of coff·d threads.
//    Per token at start_pos: score += ape[start_pos%ratio]; write state slot
//    (overlap: ratio + start_pos%ratio ∈ 4..7; else start_pos%ratio); on fire
//    ((start_pos+1)%ratio == 0) softmax-pool (CPU serial order) → pooled[d] fp32,
//    then the CSA shift state[0..ratio) ← state[ratio..2·ratio). fire flag → host.
// ============================================================================
extern "C" __global__ void __launch_bounds__(1024)
dsv4_comp_decode_b(const float* __restrict__ kv, const float* __restrict__ score,
                   const float* __restrict__ ape,
                   float* __restrict__ kv_state, float* __restrict__ score_state,
                   float* __restrict__ pooled, unsigned int* __restrict__ fire,
                   int start_pos, int ratio, int d, int coff) {
    const int c = threadIdx.x;
    const int cd = coff * d;
    const bool do_fire = ((start_pos + 1) % ratio) == 0;
    if (c < cd) {
        const int slot = (coff == 2) ? (ratio + start_pos % ratio) : (start_pos % ratio);
        kv_state[(size_t)slot * cd + c] = kv[c];
        score_state[(size_t)slot * cd + c] = __fadd_rn(score[c], ape[(start_pos % ratio) * cd + c]);
    }
    if (c == 0) *fire = do_fire ? 1u : 0u;
    if (!do_fire) return;
    __syncthreads();
    const int nrow = (coff == 2) ? 2 * ratio : ratio;
    if (c < d) {
        float mx = -INFINITY;
        for (int j = 0; j < nrow; ++j) {
            const int col = (coff == 2 && j >= ratio) ? (d + c) : c;
            mx = fmaxf(mx, score_state[(size_t)j * cd + col]);
        }
        float z = 0.0f;
        for (int j = 0; j < nrow; ++j) {
            const int col = (coff == 2 && j >= ratio) ? (d + c) : c;
            z = __fadd_rn(z, dsv4c_exp(score_state[(size_t)j * cd + col] - mx));
        }
        float acc = 0.0f;
        for (int j = 0; j < nrow; ++j) {
            const int col = (coff == 2 && j >= ratio) ? (d + c) : c;
            float p = dsv4c_exp(score_state[(size_t)j * cd + col] - mx) / z;
            acc = __fadd_rn(acc, __fmul_rn(kv_state[(size_t)j * cd + col], p));
        }
        pooled[c] = acc;
    }
    __syncthreads();
    if (coff == 2 && c < cd) {
        // CSA shift: state rows [0..ratio) ← [ratio..2·ratio)
        for (int j = 0; j < ratio; ++j) {
            kv_state[(size_t)j * cd + c] = kv_state[(size_t)(ratio + j) * cd + c];
            score_state[(size_t)j * cd + c] = score_state[(size_t)(ratio + j) * cd + c];
        }
    }
}

// ============================================================================
// 5. Post-pool helpers.
// ============================================================================

// 5a. dsv4_comp_round_bf16_b — pooled fp32 → bf16 (the reference's kv.to(bf16)).
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_round_bf16_b(__nv_bfloat16* __restrict__ y, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dsv4c_f32_to_bf16(x[i]);
}

// 5b. dsv4_comp_act_quant_sim_g64s_b — FP8 QAT-sim, group 64, ROW-STRIDED variant of
//     gpu_dsv4.cu's dsv4_act_quant_sim_g64 (identical math): the attention
//     compressor sims kv[..., :448] inside 512-stride rows. One warp per (row,
//     64-wide group); amax butterfly, floor 1e-4, UE8M0 pow2 scale, RNE codes.
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_act_quant_sim_g64s_b(__nv_bfloat16* __restrict__ x, uint8_t* __restrict__ s,
                               int rows, int n, int ld) {
    const int lane = threadIdx.x & 31;
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int groups_per_row = n / 64;
    if (wid >= rows * groups_per_row) return;
    const int row = wid / groups_per_row;
    const int grp = wid % groups_per_row;
    const size_t gbase = (size_t)row * ld + (size_t)grp * 64;

    float v0 = dsv4c_bf16_to_f32(x[gbase + lane]);
    float v1 = dsv4c_bf16_to_f32(x[gbase + lane + 32]);
    float amax = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(DSV4C_FULL_MASK, amax, off));
    amax = fmaxf(amax, 1e-4f);
    const float sc = dsv4c_round_scale_pow2(amax, 1.0f / 448.0f);

    float q0 = fminf(fmaxf(v0 / sc, -448.0f), 448.0f);
    float q1 = fminf(fmaxf(v1 / sc, -448.0f), 448.0f);
    uint8_t c0 = dsv4c_f32_to_fp8(q0);
    uint8_t c1 = dsv4c_f32_to_fp8(q1);
    x[gbase + lane] = dsv4c_f32_to_bf16(dsv4c_fp8_to_f32(c0) * sc);
    x[gbase + lane + 32] = dsv4c_f32_to_bf16(dsv4c_fp8_to_f32(c1) * sc);
    if (lane == 0) s[(size_t)row * groups_per_row + grp] = dsv4c_f32_to_e8m0(sc);
}

// 5b-codes. dsv4_comp_act_quant_g64s_b — R5b: the CODES variant of the g64s FP8 QAT-sim
//     (identical body, const row-strided input): writes the e4m3 codes DENSE [rows, n] +
//     UE8M0 scales [rows, n/64] instead of rounding x in place. dequant(packed) == the
//     sim's bf16 output bit-for-bit (same amax, same pow2 scale, same RNE code, and the
//     reader's bf16(fp8*sc) reproduces the sim's writeback exactly). Called BEFORE the
//     in-place sim in the epilogue — identical inputs to both (the R5b-1 gate asserts it).
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_act_quant_g64s_b(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ y,
                           uint8_t* __restrict__ s, int rows, int n, int ld) {
    const int lane = threadIdx.x & 31;
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int groups_per_row = n / 64;
    if (wid >= rows * groups_per_row) return;
    const int row = wid / groups_per_row;
    const int grp = wid % groups_per_row;
    const size_t gbase = (size_t)row * ld + (size_t)grp * 64;

    float v0 = dsv4c_bf16_to_f32(x[gbase + lane]);
    float v1 = dsv4c_bf16_to_f32(x[gbase + lane + 32]);
    float amax = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(DSV4C_FULL_MASK, amax, off));
    amax = fmaxf(amax, 1e-4f);
    const float sc = dsv4c_round_scale_pow2(amax, 1.0f / 448.0f);

    float q0 = fminf(fmaxf(v0 / sc, -448.0f), 448.0f);
    float q1 = fminf(fmaxf(v1 / sc, -448.0f), 448.0f);
    y[(size_t)row * n + grp * 64 + lane] = dsv4c_f32_to_fp8(q0);
    y[(size_t)row * n + grp * 64 + lane + 32] = dsv4c_f32_to_fp8(q1);
    if (lane == 0) s[(size_t)row * groups_per_row + grp] = dsv4c_f32_to_e8m0(sc);
}

// 5c. dsv4_comp_copy_rows_b — pooled bf16 rows → cache rows [row0, row0+rows).
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_copy_rows_b(__nv_bfloat16* __restrict__ dst, int dst_row0,
                      const __nv_bfloat16* __restrict__ src, int rows, int d) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)rows * d) return;
    dst[((long)dst_row0 * d) + i] = src[i];
}

// 5d. dsv4_comp_wscale_b — indexer head weights: w ← bf16(f32(w)·scale) (bf16 mul,
//     replicates dsv4_cpu's bf(v * wscale) after the bf16 GEMM).
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_wscale_b(__nv_bfloat16* __restrict__ w, float scale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = dsv4c_f32_to_bf16(__fmul_rn(dsv4c_bf16_to_f32(w[i]), scale));
}

// ============================================================================
// 6. dsv4_comp_index_score_b — §B.6 step 5: index_score[i, t] =
//    Σ_h relu(bf16(q[i,h]·kv[t])) ·bf16 weights[i,h] (fp32 ascending-h sum, bf16
//    store), block-causal −inf mask in prefill. BIT-EXACT vs dsv4_cpu given
//    identical inputs: the per-head dot replicates dot8 (8 accumulators, mul+add
//    separated, the ((0+1)+(2+3))+((4+5)+(6+7)) final tree); products of
//    bf16-valued fp32s are exact so __fmul_rn/__fadd_rn equal Rust's f32 ops.
//    Grid: (s, grid_y). One CTA per (token, tile). 256 threads per CTA use a
//    grid-stride loop over t: `t = blockIdx.y*256 + tid; t < nblocks; t += 256*gridDim.y`.
//    Each block t is scored by exactly one thread using the same dot8 chain —
//    bitwise-identical to grid_y=1 regardless of the tiling. The grid-stride
//    eliminates the single-CTA bottleneck at large nblocks (1M decode lever).
//    Shapes are the DSV4 indexer constants: NH=64 heads, HD=128.
// ============================================================================
#define DSV4C_IDX_NH 64
#define DSV4C_IDX_HD 128

extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_index_score_b(const __nv_bfloat16* __restrict__ q,
                        const __nv_bfloat16* __restrict__ kv_cache,
                        const __nv_bfloat16* __restrict__ weights,
                        float* __restrict__ scores,
                        int nblocks, int start_pos, int ratio) {
    __shared__ float qs[DSV4C_IDX_NH * DSV4C_IDX_HD];
    __shared__ float ws[DSV4C_IDX_NH];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    for (int idx = tid; idx < DSV4C_IDX_NH * DSV4C_IDX_HD; idx += 256)
        qs[idx] = dsv4c_bf16_to_f32(q[(size_t)i * (DSV4C_IDX_NH * DSV4C_IDX_HD) + idx]);
    for (int h = tid; h < DSV4C_IDX_NH; h += 256)
        ws[h] = dsv4c_bf16_to_f32(weights[(size_t)i * DSV4C_IDX_NH + h]);
    __syncthreads();
    // block-causal at ABSOLUTE position p = start_pos + i: token p sees block t ⟺
    // t < (p+1)//ratio — only fully-elapsed blocks (§B.6 step 6, generalized to the
    // verify-width case the reference never exercises: prefill start_pos==0 gives
    // (i+1)//ratio, decode s==1 gives nblocks — both reduce to this rule; a verify
    // row therefore can never see a block containing a FUTURE token, which is what
    // makes selections a pure function of the committed prefix (§12.B.2).
    const int lim = min(nblocks, (start_pos + i + 1) / ratio);
    // grid-stride loop: grid.y > 1 parallelizes the block axis across CTAs.
    for (int t = blockIdx.y * 256 + tid; t < nblocks; t += 256 * gridDim.y) {
        if (t >= lim) {
            scores[(size_t)i * nblocks + t] = -INFINITY;
            continue;
        }
        const __nv_bfloat16* __restrict__ kvr = kv_cache + (size_t)t * DSV4C_IDX_HD;
        float acc = 0.0f;
        for (int h = 0; h < DSV4C_IDX_NH; ++h) {
            const float* __restrict__ qh = qs + h * DSV4C_IDX_HD;
            float a8[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) a8[l] = 0.0f;
#pragma unroll
            for (int cc = 0; cc < DSV4C_IDX_HD / 8; ++cc) {
#pragma unroll
                for (int l = 0; l < 8; ++l)
                    a8[l] = __fadd_rn(a8[l], __fmul_rn(qh[cc * 8 + l], dsv4c_bf16_to_f32(kvr[cc * 8 + l])));
            }
            const float dot = __fadd_rn(
                __fadd_rn(__fadd_rn(a8[0], a8[1]), __fadd_rn(a8[2], a8[3])),
                __fadd_rn(__fadd_rn(a8[4], a8[5]), __fadd_rn(a8[6], a8[7])));
            const float dotbf = dsv4c_bf(dot);                    // einsum out bf16
            const float rel = fmaxf(dotbf, 0.0f);                 // relu_ exact
            acc = __fadd_rn(acc, dsv4c_bf(__fmul_rn(rel, ws[h]))); // bf16 mul, fp32 sum
        }
        scores[(size_t)i * nblocks + t] = dsv4c_bf(acc);
    }
}

// FP4 E2M1 decode + UE8M0 scale (bit-exact mirrors of gpu_dsv4.cu's helpers, §C.2).
__device__ __forceinline__ float dsv4c_fp4_to_f32(uint8_t c) {
    float mag;
    int e = (c >> 1) & 3, m = c & 1;
    if (e == 0) mag = m ? 0.5f : 0.0f;
    else        mag = (1.0f + 0.5f * (float)m) * (float)(1 << (e - 1));
    return (c & 8) ? -mag : mag;
}
__device__ __forceinline__ float dsv4c_e8m0_to_f32(uint8_t b) {
    return __uint_as_float((uint32_t)b << 23); // e8m0 byte = biased pow2 exponent (b ≥ 1 by the sim's amax floor)
}
// dequant == the simmed bf16 value, bit-for-bit (the epilogue's sim wrote exactly
// dsv4_f32_to_bf16(fp4_to_f32(code) * sc) — reproduce it, then upcast like the bf16 reader).
__device__ __forceinline__ float dsv4c_fp4_deq_bf16(uint8_t nib, float sc) {
    return dsv4c_bf16_to_f32(dsv4c_f32_to_bf16(dsv4c_fp4_to_f32(nib) * sc));
}

// 6c. dsv4_comp_index_score_fp4_b — R5a-2 packed-cache score reader. The indexer cache
//     rows as FP4-e2m1 codes (g32, UE8M0 scales) — the SAME VALUES as the bf16 QAT-sim
//     rows (the epilogue writes them losslessly; dsv4c_fp4_deq_bf16 reproduces them
//     bit-for-bit). HEAD-GROUPED (8 groups × 8 heads): the 8 elements of each cc step are
//     dequantized ONCE per head group (the bf16 reader re-upcasts per element per head).
//     Per-head dot8 chains, the ((0+1)+(2+3))+((4+5)+(6+7)) final trees, relu, the bf16
//     weight products and the ascending-head acc are UNCHANGED ⇒ bitwise ==
//     dsv4_comp_index_score_b given the same cache contents (the R5a-2 gate asserts it).
//     4× fewer cache bytes AND ~7× fewer per-element ops per block. Stack audit (AGENTS §4):
//     all hot-path arrays are named scalars / compile-time indexed — ptxas reports a 16 B
//     frame that is PROLOGUE-COLD (one pointer spilled across the smem-fill barrier, ~6
//     instructions per CTA; the per-block loop is stack-free — SASS-checked 2026-07-30).
//     If the R5a-2 bench regresses, this kernel is reverted (the mHC rule).
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_index_score_fp4_b(const __nv_bfloat16* __restrict__ q,
                            const uint8_t* __restrict__ codes,
                            const uint8_t* __restrict__ scales,
                            const __nv_bfloat16* __restrict__ weights,
                            float* __restrict__ scores,
                            int nblocks, int start_pos, int ratio) {
    __shared__ float qs[DSV4C_IDX_NH * DSV4C_IDX_HD];
    __shared__ float ws[DSV4C_IDX_NH];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    for (int idx = tid; idx < DSV4C_IDX_NH * DSV4C_IDX_HD; idx += 256)
        qs[idx] = dsv4c_bf16_to_f32(q[(size_t)i * (DSV4C_IDX_NH * DSV4C_IDX_HD) + idx]);
    for (int h = tid; h < DSV4C_IDX_NH; h += 256)
        ws[h] = dsv4c_bf16_to_f32(weights[(size_t)i * DSV4C_IDX_NH + h]);
    __syncthreads();
    const int lim = min(nblocks, (start_pos + i + 1) / ratio);
    for (int t = blockIdx.y * 256 + tid; t < nblocks; t += 256 * gridDim.y) {
        if (t >= lim) {
            scores[(size_t)i * nblocks + t] = -INFINITY;
            continue;
        }
        const uint8_t* __restrict__ cods = codes + (size_t)t * (DSV4C_IDX_HD / 2);
        const uint8_t* __restrict__ scls = scales + (size_t)t * (DSV4C_IDX_HD / 32);
        float acc = 0.0f;
// 4-head groups (a8[4][8] = 32 accumulator registers; partial unrolls — the arrays are
        // (hh,l)-indexed only, so hg/cc unroll factors don't create stack frames)
#pragma unroll 2
        for (int hg = 0; hg < DSV4C_IDX_NH / 4; ++hg) {
            float a8[4][8];
#pragma unroll
            for (int hh = 0; hh < 4; ++hh)
#pragma unroll
                for (int l = 0; l < 8; ++l) a8[hh][l] = 0.0f;
#pragma unroll 4
            for (int cc = 0; cc < DSV4C_IDX_HD / 8; ++cc) {
                const int gbase = cc >> 2;            // one 32-wide FP4 group = 4 cc steps
                const int nb0 = (cc & 3) * 4;         // first packed byte of this cc step
                const float sc = dsv4c_e8m0_to_f32(scls[gbase]);
                // ONE u32 load for the 8 nibbles of this cc step (contiguous bytes).
                const uint32_t pack = *(const uint32_t*)(cods + gbase * 16 + nb0);
                const float v0 = dsv4c_fp4_deq_bf16(pack & 0xFu, sc);
                const float v1 = dsv4c_fp4_deq_bf16((pack >> 4) & 0xFu, sc);
                const float v2 = dsv4c_fp4_deq_bf16((pack >> 8) & 0xFu, sc);
                const float v3 = dsv4c_fp4_deq_bf16((pack >> 12) & 0xFu, sc);
                const float v4 = dsv4c_fp4_deq_bf16((pack >> 16) & 0xFu, sc);
                const float v5 = dsv4c_fp4_deq_bf16((pack >> 20) & 0xFu, sc);
                const float v6 = dsv4c_fp4_deq_bf16((pack >> 24) & 0xFu, sc);
                const float v7 = dsv4c_fp4_deq_bf16((pack >> 28) & 0xFu, sc);
#pragma unroll
                for (int hh = 0; hh < 4; ++hh) {
                    const float* __restrict__ qh = qs + (hg * 4 + hh) * DSV4C_IDX_HD + cc * 8;
                    a8[hh][0] = __fadd_rn(a8[hh][0], __fmul_rn(qh[0], v0));
                    a8[hh][1] = __fadd_rn(a8[hh][1], __fmul_rn(qh[1], v1));
                    a8[hh][2] = __fadd_rn(a8[hh][2], __fmul_rn(qh[2], v2));
                    a8[hh][3] = __fadd_rn(a8[hh][3], __fmul_rn(qh[3], v3));
                    a8[hh][4] = __fadd_rn(a8[hh][4], __fmul_rn(qh[4], v4));
                    a8[hh][5] = __fadd_rn(a8[hh][5], __fmul_rn(qh[5], v5));
                    a8[hh][6] = __fadd_rn(a8[hh][6], __fmul_rn(qh[6], v6));
                    a8[hh][7] = __fadd_rn(a8[hh][7], __fmul_rn(qh[7], v7));
                }
            }
#pragma unroll
            for (int hh = 0; hh < 4; ++hh) {
                const float dot = __fadd_rn(
                    __fadd_rn(__fadd_rn(a8[hh][0], a8[hh][1]), __fadd_rn(a8[hh][2], a8[hh][3])),
                    __fadd_rn(__fadd_rn(a8[hh][4], a8[hh][5]), __fadd_rn(a8[hh][6], a8[hh][7])));
                const float dotbf = dsv4c_bf(dot);                    // einsum out bf16
                const float rel = fmaxf(dotbf, 0.0f);                 // relu_ exact
                acc = __fadd_rn(acc, dsv4c_bf(__fmul_rn(rel, ws[hg * 4 + hh])));
            }
        }
        scores[(size_t)i * nblocks + t] = dsv4c_bf(acc);
    }
}

// ============================================================================
// 7. dsv4_comp_idx_remask_b — §B.6 step 7: selected block ≥ the row's block-causal
//    limit (start_pos+i+1)//ratio → −1, else +offset (see kernel 6's note: prefill
//    and decode are the two cases the reference defines; the per-row limit is the
//    verify-width generalization that keeps selections prefix-pure). Elementwise [s,k].
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_idx_remask_b(int* __restrict__ idx, int s, int k, int start_pos, int ratio, int nblocks, int offset) {
    const long f = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (f >= (long)s * k) return;
    const int v = idx[f];
    const int lim = min(nblocks, (int)(start_pos + (int)(f / k) + 1) / ratio);
    idx[f] = (v >= lim) ? -1 : (v + offset);
}

// ============================================================================
// 8. dsv4_comp_gemm_tc_b — compressor wkv/wgate GEMM via WMMA tensor cores.
//    Replaces the scalar tree GEMM for the ATTENTION compressor (rotate=false).
//    bf16 inputs (weights cast from bf16-valued f32 at upload), fp32 accumulate.
// ============================================================================
extern "C" __global__ void __launch_bounds__(32)
dsv4_comp_gemm_tc_b(float* __restrict__ out, const __nv_bfloat16* __restrict__ x,
                    const __nv_bfloat16* __restrict__ w, int s, int k, int n) {
    using namespace nvcuda;
    const int M = 16, N = 16, K = 16;
    const int mb = blockIdx.y, nb = blockIdx.x;
    const int row_base = mb * 16, col_base = nb * 16;
    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int kk = 0; kk < k; kk += K) {
        if (row_base + 15 < s) {
            wmma::load_matrix_sync(a_frag, x + (size_t)row_base * k + kk, k);
        } else {
            __shared__ __nv_bfloat16 a_pad[M * K];
            for (int i = threadIdx.x; i < M * K; i += 32) {
                int m = i / K, ki = i % K, row = row_base + m;
                a_pad[i] = (row < s) ? x[(size_t)row * k + kk + ki] : (__nv_bfloat16)0;
            }
            wmma::load_matrix_sync(a_frag, a_pad, K);
        }
        wmma::load_matrix_sync(b_frag, w + (size_t)col_base * k + kk, k);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __shared__ float smem[M * N];
    wmma::store_matrix_sync(smem, c_frag, N, wmma::mem_row_major);
    for (int i = threadIdx.x; i < M * N; i += 32) {
        int mj = i / N, nj = i % N, row = row_base + mj, col = col_base + nj;
        if (row < s && col < n) out[(size_t)row * n + col] = smem[mj * N + nj];
    }
}

// ============================================================================
// 9-11. Hierarchical top-k helpers for >64K context (dsv4_topk covers T≤16384).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_score_gather_b(float* __restrict__ gathered, const float* __restrict__ scores,
                    const int* __restrict__ idx, int s, int m, int nb) {
    const long f = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (f >= (long)s * m) return;
    const int row = (int)(f / m), j = (int)(f % m), gi = idx[f];
    gathered[f] = (gi >= 0 && gi < nb) ? scores[(size_t)row * nb + gi] : -INFINITY;
}

extern "C" __global__ void __launch_bounds__(256)
dsv4_idx_remap_b(int* __restrict__ out, const int* __restrict__ stage2_idx,
                const int* __restrict__ lookup, int s, int k, int m) {
    const long f = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (f >= (long)s * k) return;
    const int j = (int)(f % k), si = stage2_idx[f];
    out[f] = (si >= 0 && si < m) ? lookup[(size_t)(f / k) * m + si] : -1;
}

extern "C" __global__ void __launch_bounds__(256)
dsv4_idx_offset_place_b(int* __restrict__ dst, const int* __restrict__ src,
                        int s, int k, int m, int col_offset, int offset) {
    const long f = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (f >= (long)s * k) return;
    const int j = (int)(f % k), v = src[f];
    dst[(size_t)(f / k) * m + col_offset + j] = (v >= 0) ? (v + offset) : -1;
}

// ============================================================================
// 12. dsv4_comp_index_score_tile_b — stripe variant of kernel 6 for the STREAMING
//     top-k (DSV4_LONG_CONTEXT_1M §4): scores blocks [t0, t0+tc) into stripe[s, tc]
//     (dense, tc-strided) instead of the full [s, nblocks] matrix — peak memory
//     s·tc·4, independent of context length. The dot8 chain is copied byte-for-byte
//     from kernel 6 (same order ⇒ bitwise-same scores); the block-causal limit stays
//     ABSOLUTE: global block t = t0+j is visible to row i ⟺ t < min(nblocks,
//     (start_pos+i+1)/ratio) — identical to kernel 6's mask for t0==0.
//     Grid: (s, grid_y) — grid-stride loop over j (same pattern as kernel 6).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_index_score_tile_b(const __nv_bfloat16* __restrict__ q,
                             const __nv_bfloat16* __restrict__ kv_cache,
                             const __nv_bfloat16* __restrict__ weights,
                             float* __restrict__ stripe,
                             int t0, int tc, int nblocks, int start_pos, int ratio) {
    __shared__ float qs[DSV4C_IDX_NH * DSV4C_IDX_HD];
    __shared__ float ws[DSV4C_IDX_NH];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    for (int idx = tid; idx < DSV4C_IDX_NH * DSV4C_IDX_HD; idx += 256)
        qs[idx] = dsv4c_bf16_to_f32(q[(size_t)i * (DSV4C_IDX_NH * DSV4C_IDX_HD) + idx]);
    for (int h = tid; h < DSV4C_IDX_NH; h += 256)
        ws[h] = dsv4c_bf16_to_f32(weights[(size_t)i * DSV4C_IDX_NH + h]);
    __syncthreads();
    const int lim = min(nblocks, (start_pos + i + 1) / ratio);
    for (int j = blockIdx.y * 256 + tid; j < tc; j += 256 * gridDim.y) {
        const int t = t0 + j;
        if (t >= lim) {
            stripe[(size_t)i * tc + j] = -INFINITY;
            continue;
        }
        const __nv_bfloat16* __restrict__ kvr = kv_cache + (size_t)t * DSV4C_IDX_HD;
        float acc = 0.0f;
        for (int h = 0; h < DSV4C_IDX_NH; ++h) {
            const float* __restrict__ qh = qs + h * DSV4C_IDX_HD;
            float a8[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) a8[l] = 0.0f;
#pragma unroll
            for (int cc = 0; cc < DSV4C_IDX_HD / 8; ++cc) {
#pragma unroll
                for (int l = 0; l < 8; ++l)
                    a8[l] = __fadd_rn(a8[l], __fmul_rn(qh[cc * 8 + l], dsv4c_bf16_to_f32(kvr[cc * 8 + l])));
            }
            const float dot = __fadd_rn(
                __fadd_rn(__fadd_rn(a8[0], a8[1]), __fadd_rn(a8[2], a8[3])),
                __fadd_rn(__fadd_rn(a8[4], a8[5]), __fadd_rn(a8[6], a8[7])));
            const float dotbf = dsv4c_bf(dot);                    // einsum out bf16
            const float rel = fmaxf(dotbf, 0.0f);                 // relu_ exact
            acc = __fadd_rn(acc, dsv4c_bf(__fmul_rn(rel, ws[h]))); // bf16 mul, fp32 sum
        }
        stripe[(size_t)i * tc + j] = dsv4c_bf(acc);
    }
}

// ============================================================================
// 13-14. Streaming top-k merge helpers: strided f32 place, and gather+place, into
//     the wider [s, m] merge buffer (the deterministic dsv4_topk then reduces it).
//     i32 has dsv4_idx_offset_place_b (kernel 11); these are the f32 twins.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_f32_place_b(float* __restrict__ dst, const float* __restrict__ src,
                 int s, int k, int m, int col_offset) {
    const long f = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (f >= (long)s * k) return;
    dst[(size_t)(f / k) * m + col_offset + (int)(f % k)] = src[f];
}

extern "C" __global__ void __launch_bounds__(256)
dsv4_f32_gather_place_b(float* __restrict__ dst, const float* __restrict__ scores,
                        const int* __restrict__ idx, int s, int k, int m, int col_offset, int nb) {
    const long f = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (f >= (long)s * k) return;
    const int gi = idx[f];
    dst[(size_t)(f / k) * m + col_offset + (int)(f % k)] =
        (gi >= 0 && gi < nb) ? scores[(size_t)(f / k) * nb + gi] : -INFINITY;
}


// ============================================================================
// 8d. dsv4_comp_gemm_tc_pair_b — fused wkv+wgate compressor GEMM (R3.3, bitwise-safe).
// ONE launch computes both GEMMs of gemm_pair: tiles [0, tiles_n) → out0 (wkv),
// tiles [tiles_n, 2*tiles_n) → out1 (wgate). Each tile's k-loop is IDENTICAL to
// dsv4_comp_gemm_tc_b's (same WMMA instruction sequence per element) — the fusion changes
// only the launch geometry, never the reduction order, so the §12.B.5 cross-width bitwise
// contract holds exactly. Motivation: at decode (s=1) the two sequential 32-CTA launches pay
// full one-warp-tile latency twice (R3.0 census: 415 µs × 92/token); one 64-CTA launch pays
// it once and covers 64/48 SMs instead of 32/48.
//
// R3A.1 (A-path, load schedule only — bitwise-identical tiles): the E0c microbenchmark
// isolated the decode-width latency to the per-K-tile a_pad smem detour (77-82 GB/s with
// it — even restructured — vs 262 GB/s with a plain direct A load at 128 CTAs). The detour
// is now gone entirely: partial-row tiles read a caller-filled 16-row x_pad panel
// (rows < rc from x, rows >= rc exact +0) through the SAME direct wmma.load the full tiles
// use. The A tile the mma sees is bit-identical to the old a_pad contents (verified vs the
// old path on hardware at s in {1,2,6,13,15,16}); the per-element mma chain, B path, and
// epilogue are untouched. x_pad is filled by dsv4_comp_pad16_b once per gemm_pair call
// (only when s % 16 != 0), never inside the K-loop.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_comp_pad16_b(__nv_bfloat16* __restrict__ dst,
                  const __nv_bfloat16* __restrict__ src, int row0, int rc, int k) {
    // dst[16, k]: rows < rc from src[row0 .. row0+rc) (the partial tile's real rows), else zero.
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k) dst[(size_t)row * k + col] =
        (row < rc) ? src[(size_t)(row0 + row) * k + col] : (__nv_bfloat16)0;
}

extern "C" __global__ void __launch_bounds__(32)
dsv4_comp_gemm_tc_pair_b(float* __restrict__ out0, float* __restrict__ out1,
                         const __nv_bfloat16* __restrict__ x,
                         const __nv_bfloat16* __restrict__ x_pad,
                         const __nv_bfloat16* __restrict__ w0, const __nv_bfloat16* __restrict__ w1,
                         int s, int k, int n) {
    using namespace nvcuda;
    const int M = 16, N = 16, K = 16;
    const int tiles_n = (n + N - 1) / N;
    const int which = blockIdx.x >= tiles_n ? 1 : 0;
    const int nb = which ? (blockIdx.x - tiles_n) : blockIdx.x;
    const int mb = blockIdx.y;
    const int row_base = mb * 16, col_base = nb * 16;
    float* out = which ? out1 : out0;
    const __nv_bfloat16* w = which ? w1 : w0;
    // Partial-row tile? Read the pre-padded panel (bit-identical to the old a_pad tile).
    const __nv_bfloat16* xa = (row_base + M > s) ? x_pad : (x + (size_t)row_base * k);
    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int kk = 0; kk < k; kk += K) {
        wmma::load_matrix_sync(a_frag, xa + kk, k);
        wmma::load_matrix_sync(b_frag, w + (size_t)col_base * k + kk, k);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __shared__ float smem[M * N];
    wmma::store_matrix_sync(smem, c_frag, N, wmma::mem_row_major);
    for (int i = threadIdx.x; i < M * N; i += 32) {
        int mj = i / N, nj = i % N, row = row_base + mj, col = col_base + nj;
        if (row < s && col < n) out[(size_t)row * n + col] = smem[mj * N + nj];
    }
}

// ============================================================================
// 8e. dsv4_comp_gemm_fast_pair_b — item 2.5 fast path: the fused wkv+wgate compressor
// pair as a big-tile bf16→fp32 GEMM (their cuBLAS `torch.mm out_dtype=fp32` class).
// TOLERANCE-CLASS by contract (item 2.5 / §6-a): SAME inputs as dsv4_comp_gemm_tc_pair_b
// (bf16 weights + bf16 activations, fp32 out) — the per-element wmma m16n16k16 chain over
// ascending K is unchanged, but the warp/tile ownership is scheduler-chosen: FOUR warps
// per CTA each own a [16,16] n-subtile of a [16,64] tile, no cross-warp reduce. The
// difference vs the exact kernel is therefore pure reduction-order class (rel-L2 ~1e-7),
// NOT quant noise. Only the attention compressor (rotate=false) uses it; the indexer
// (rotate, topk-precision-sensitive) keeps the scalar tree GEMMs.
// Grid: (2*tiles64, tiles_m), 128 threads. Requires n % 64 == 0 (asserted host-side;
// production CSA n=2048, HCA n=512).
// ============================================================================
extern "C" __global__ void __launch_bounds__(128)
dsv4_comp_gemm_fast_pair_b(float* __restrict__ out0, float* __restrict__ out1,
                           const __nv_bfloat16* __restrict__ x,
                           const __nv_bfloat16* __restrict__ x_pad,
                           const __nv_bfloat16* __restrict__ w0, const __nv_bfloat16* __restrict__ w1,
                           int s, int k, int n) {
    using namespace nvcuda;
    const int M = 16, N = 16, K = 16;
    const int tiles64 = n / 64;
    const int which = blockIdx.x >= tiles64 ? 1 : 0;
    const int nb = which ? (blockIdx.x - tiles64) : blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int row_base = blockIdx.y * M, col_base = nb * 64 + warp * N;
    float* out = which ? out1 : out0;
    const __nv_bfloat16* w = which ? w1 : w0;
    // Partial-row tile? Read the pre-padded panel (same convention as tc_pair).
    const __nv_bfloat16* xa = (row_base + M > s) ? x_pad : (x + (size_t)row_base * k);
    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    for (int kk = 0; kk < k; kk += K) {
        wmma::load_matrix_sync(a_frag, xa + kk, k);
        wmma::load_matrix_sync(b_frag, w + (size_t)col_base * k + kk, k);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __shared__ float smem[4][M * N];   // per-warp slot: the 4 warps' store_matrix_sync
                                        // would clobber a single shared buffer (no barrier
                                        // between the fragment store and the read-back).
    wmma::store_matrix_sync(smem[warp], c_frag, N, wmma::mem_row_major);
    const int lane = threadIdx.x & 31;
    for (int i = lane; i < M * N; i += 32) {   // warp-uniform (the tc_pair epilogue shape)
        int mj = i / N, nj = i % N, row = row_base + mj, col = col_base + nj;
        if (row < s && col < n) out[(size_t)row * n + col] = smem[warp][mj * N + nj];
    }
}
