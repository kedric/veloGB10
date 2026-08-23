// DSV4 Phase-3 lane 3A module — SWA-128 ring attention kernels (DEEPSEEK_V4_PORT.md §B.1–B.2).
// Build-id stamp per AGENTS.md §1. Compiled to src/ptx/gpu_dsv4_attn.ptx by build.rs (nvcc sm_121),
// loaded via dsv4_gpu::Dsv4Kernels::load_module and launched on the blocking compute stream.
//
// Kernels (semantics target = the G1-proven CPU reference src/dsv4_cpu.rs, NOT a re-derivation):
//   1. dsv4_attn_rescale_b       §B.1.1 per-head weight-free RMS rescale — BIT-EXACT target.
//   2. dsv4_kv_sim_g64_strided   §B.1.2 KV QAT-sim on kv[..., :448] of a [rows,512] tensor —
//                                  BIT-EXACT (same warp-group math as dsv4_act_quant_sim_g64,
//                                  row-stride parameter added; CPU ref sims a contiguous copy).
//   3. dsv4_ring_write_b         §B.2 KV ring write (write-before-attention, slot pos%128,
//                                  rotated last-128 write at prefill S>128) — pure copy, bit-exact.
//   4. dsv4_window_idxs_b        §B.2 window index lists (mirror dsv4_cpu::window_topk_idxs) —
//                                  integer math, bit-exact.
//   5. dsv4_olo_proj_b           §B.1.4 grouped-LoRA O first GEMM (bf16, wo_a dequantized at
//                                  load per §F.2/§12.A.3) — tolerance-level (fp32 block-tree
//                                  reduce vs CPU pairwise tree), batch-invariant per output.
#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <mma.h>

extern "C" __global__ void dsv4_kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

#define DSV4A_FULL_MASK 0xffffffffu

__device__ __forceinline__ float dsv4a_bf16_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 dsv4a_f32_to_bf16(float v) { return __float2bfloat16(v); }

// ---- UE8M0 scale from amax: s = 2^ceil(log2(amax * inv)), kernel.py fast_log2_ceil/fast_pow2 ----
// (byte-identical algorithm to gpu_dsv4.cu's copy — PTX modules cannot share device functions)
__device__ __forceinline__ float dsv4a_round_scale_pow2(float amax, float inv) {
    float v = amax * inv;
    uint32_t b = __float_as_uint(v);
    int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t)(e + 127) << 23);
}

__device__ __forceinline__ uint8_t dsv4a_f32_to_fp8(float v) {
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float dsv4a_fp8_to_f32(uint8_t c) {
    return __half2float(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3));
}

// ============================================================================
// 1. dsv4_attn_rescale_b — §B.1.1 weight-free per-head RMS rescale (model.py:506-507):
//      q *= rsqrt(mean(q², dim=-1) + 1e-6), no learned gain, in-place bf16.
//    BIT-EXACT vs dsv4_cpu::attn_qkv's rescale loop — the torch bf16 op-by-op sequence:
//      sq[j]  = bf16(v·v)                     (per-op rounding of the materialized .square())
//      ss     = pairwise_sum(sq)              (adjacent-pair tree, f32 — NOT a halving reduce)
//      mean   = bf16(ss / 512); arg = bf16(mean + eps)
//      r      = bf16(1 / sqrt(arg))           (correctly-rounded sqrt + div, then RNE)
//      v[j]   = bf16(v · r)
//    One block per (token, head) row; 256 threads, dim = head_dim = 512 (locked, §A.2).
//    The pairwise tree: level 1 in registers (adjacent pairs), then sm[t] = sm[2t]+sm[2t+1]
//    halving in smem — exactly pairwise_sum's association (a stride-256 tree would NOT match).
//    No runtime-indexed arrays, no FMA-contractible pairs — zero stack, no contraction risk.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_attn_rescale_b(__nv_bfloat16* __restrict__ q, int rows, int dim, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    if (dim != 512) return; // host contract: head_dim = 512 (asserted host-side)
    const int tid = threadIdx.x;
    __nv_bfloat16* __restrict__ r = q + (size_t)row * 512;
    __shared__ float sm[256];
    __shared__ float rbc;

    const float v0 = dsv4a_bf16_to_f32(r[2 * tid]);
    const float v1 = dsv4a_bf16_to_f32(r[2 * tid + 1]);
    const float s0 = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(v0 * v0)); // bf16-rounded square
    const float s1 = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(v1 * v1));
    sm[tid] = s0 + s1; // pairwise_sum level 1 (adjacent pairs)
    __syncthreads();
    // In-place compaction sm[t] = sm[2t]+sm[2t+1] is RACY within a level (thread t writes a
    // slot thread t/2 reads in the same phase), so: read into a register, sync, store, sync.
#pragma unroll
    for (int n = 256; n > 1; n >>= 1) {
        float v = 0.0f;
        if (tid < (n >> 1)) v = sm[2 * tid] + sm[2 * tid + 1];
        __syncthreads();
        if (tid < (n >> 1)) sm[tid] = v;
        __syncthreads();
    }
    if (tid == 0) {
        const float mean = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(sm[0] / 512.0f));
        const float arg = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(mean + eps));
        rbc = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(1.0f / __fsqrt_rn(arg)));
    }
    __syncthreads();
    const float rr = rbc;
    r[2 * tid] = dsv4a_f32_to_bf16(v0 * rr);
    r[2 * tid + 1] = dsv4a_f32_to_bf16(v1 * rr);
}

// ============================================================================
// 2. dsv4_kv_sim_g64_strided — §B.1.2 KV QAT-sim: FP8-E4M3 dynamic round-trip, group 64,
//    UE8M0 scales, IN PLACE on the [.., :448] view of a [rows, stride] bf16 tensor
//    (rope dims [448:512] untouched — they stay bf16 by contract).
//    Identical per-(row,group) math to dsv4_act_quant_body<64, SIM=true> (one warp per
//    group, 2 elems/lane, XOR-butterfly amax, floor 1e-4, pow2 ceil, RNE cvt) — the CPU ref
//    (dsv4_cpu::attn_qkv) sims a contiguous [rows,448] copy; a strided in-place pass over the
//    same groups is value-identical. BIT-EXACT target. No scale output (the simmed bf16
//    values carry the codes' content exactly — consumers never read ue8m0 here).
//    Grid: ceil(rows * (448/64) * 32 / 256) blocks × 256 threads (one warp per (row, group)).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_kv_sim_g64_strided(__nv_bfloat16* __restrict__ x, int rows, int stride, int n) {
    const int G = 64;
    const int lane = threadIdx.x & 31;
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int groups_per_row = n / G; // 7 for n=448
    if (wid >= rows * groups_per_row) return;
    const int row = wid / groups_per_row;
    const int grp = wid % groups_per_row;
    const size_t gbase = (size_t)row * (size_t)stride + (size_t)grp * G;

    float v0 = dsv4a_bf16_to_f32(x[gbase + lane]);
    float v1 = dsv4a_bf16_to_f32(x[gbase + 32 + lane]);
    float amax = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(DSV4A_FULL_MASK, amax, off));
    amax = fmaxf(amax, 1e-4f);

    const float sc = dsv4a_round_scale_pow2(amax, 1.0f / 448.0f);
    const float q0 = fminf(fmaxf(v0 / sc, -448.0f), 448.0f);
    const float q1 = fminf(fmaxf(v1 / sc, -448.0f), 448.0f);
    x[gbase + lane] = dsv4a_f32_to_bf16(dsv4a_fp8_to_f32(dsv4a_f32_to_fp8(q0)) * sc);
    x[gbase + 32 + lane] = dsv4a_f32_to_bf16(dsv4a_fp8_to_f32(dsv4a_f32_to_fp8(q1)) * sc);
}

// ============================================================================
// 2b. dsv4_rescale_rope_sim_b — E2 rung 5 (Tier 1.2): the attention-front TAIL in ONE
//     launch, replacing dsv4_attn_rescale_b + dsv4_rope_pair_b + dsv4_kv_sim_g64_strided
//     (3 launches -> 1) at decode/verify widths (s <= 16; prefill keeps the pair arm).
//       q rows  (blockIdx.x <  rows_q): rescale (verbatim §1 body) -> q rope (verbatim
//                                       dsv4_rope_pair_b blockIdx.y==0 arm, warp 0)
//       kv rows (blockIdx.x >= rows_q): kv rope (verbatim y==1 arm, warp 0) -> KV QAT-sim
//                                       (verbatim §2 body, one warp per 64-group)
//     Row-independent ops: the per-row fused order (rescale->rope on a q row, rope->sim on
//     a kv row) IS the global 3-launch sequence; every reduction tree and per-element chain
//     is copied verbatim -> BITWISE vs the three separate launches (gate:
//     tests/dsv4_spine_test.rs::rescale_rope_sim_fused_bitwise_match_separate).
//     start_pos is a SCALAR kernel arg (inline positions, rope_pair-style) — the CUDA-graph
//     policy table patches slot 4 per replay (the rope_q_inline_b bug class: EVERY
//     position-dependent kernel needs a policy entry).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_rescale_rope_sim_b(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kv,
                        const float* __restrict__ cosp, const float* __restrict__ sinp,
                        int start_pos, int nh, int rows_q, int s, int dim, int rd, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int half = rd >> 1;                 // 32 for rd=64
    const int off = dim - rd;                 // rotate dims [off, off+rd)
    if (row < rows_q) {
        // ---- q row: §1 rescale, verbatim (one block per row, 256 threads, dim = 512) ----
        if (dim != 512) return;
        __nv_bfloat16* __restrict__ r = q + (size_t)row * 512;
        __shared__ float sm[256];
        __shared__ float rbc;
        const float v0 = dsv4a_bf16_to_f32(r[2 * tid]);
        const float v1 = dsv4a_bf16_to_f32(r[2 * tid + 1]);
        const float s0 = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(v0 * v0));
        const float s1 = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(v1 * v1));
        sm[tid] = s0 + s1;
        __syncthreads();
#pragma unroll
        for (int n = 256; n > 1; n >>= 1) {
            float v = 0.0f;
            if (tid < (n >> 1)) v = sm[2 * tid] + sm[2 * tid + 1];
            __syncthreads();
            if (tid < (n >> 1)) sm[tid] = v;
            __syncthreads();
        }
        if (tid == 0) {
            const float mean = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(sm[0] / 512.0f));
            const float arg = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(mean + eps));
            rbc = dsv4a_bf16_to_f32(dsv4a_f32_to_bf16(1.0f / __fsqrt_rn(arg)));
        }
        __syncthreads();
        const float rr = rbc;
        r[2 * tid] = dsv4a_f32_to_bf16(v0 * rr);
        r[2 * tid + 1] = dsv4a_f32_to_bf16(v1 * rr);
        __syncthreads();
        // ---- q rope (verbatim rope_pair y=0 arm; p = start_pos + row/nh, warp 0) ----
        if (tid < half) {
            const int p = start_pos + row / nh;
            const float c = cosp[(size_t)p * half + tid];
            const float sn = sinp[(size_t)p * half + tid];
            const size_t a = (size_t)row * dim + off + tid * 2;
            const float re = dsv4a_bf16_to_f32(q[a]);
            const float im = dsv4a_bf16_to_f32(q[a + 1]);
            q[a] = dsv4a_f32_to_bf16(re * c - im * sn);
            q[a + 1] = dsv4a_f32_to_bf16(re * sn + im * c);
        }
        return;
    }
    // ---- kv row: rope (warp 0) then §2 sim (one warp per 64-group over [0, dim-rd)) ----
    const int krow = row - rows_q;
    if (krow >= s) return;
    if (tid < half) {
        const int p = start_pos + krow;
        const float c = cosp[(size_t)p * half + tid];
        const float sn = sinp[(size_t)p * half + tid];
        const size_t a = (size_t)krow * dim + off + tid * 2;
        const float re = dsv4a_bf16_to_f32(kv[a]);
        const float im = dsv4a_f32_to_bf16(kv[a + 1]);
        kv[a] = dsv4a_f32_to_bf16(re * c - im * sn);
        kv[a + 1] = dsv4a_f32_to_bf16(re * sn + im * c);
    }
    __syncthreads();
    const int G = 64;
    const int lane = tid & 31;
    const int wid = tid >> 5;                 // warp in block
    const int groups_per_row = (dim - rd) / G;  // 7 for dim=512, rd=64
    if (wid >= groups_per_row) return;
    const size_t gbase = (size_t)krow * (size_t)dim + (size_t)wid * G;
    float v0 = dsv4a_bf16_to_f32(kv[gbase + lane]);
    float v1 = dsv4a_bf16_to_f32(kv[gbase + 32 + lane]);
    float amax = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(DSV4A_FULL_MASK, amax, o));
    amax = fmaxf(amax, 1e-4f);
    const float sc = dsv4a_round_scale_pow2(amax, 1.0f / 448.0f);
    const float q0 = fminf(fmaxf(v0 / sc, -448.0f), 448.0f);
    const float q1 = fminf(fmaxf(v1 / sc, -448.0f), 448.0f);
    kv[gbase + lane] = dsv4a_f32_to_bf16(dsv4a_fp8_to_f32(dsv4a_f32_to_fp8(q0)) * sc);
    kv[gbase + 32 + lane] = dsv4a_f32_to_bf16(dsv4a_fp8_to_f32(dsv4a_f32_to_fp8(q1)) * sc);
}

// ============================================================================
// 3. dsv4_ring_write_b — §B.2 SWA KV ring write. Physical slot = pos % win; the write
//    happens BEFORE attention (the current token attends to itself). Unified prefill/decode:
//      rows r ∈ [lo, s) of kv are copied to cache[(start_pos + r) % win],
//      lo = (s > win) ? s - win : 0   (when s > win, only the LAST win rows are written —
//      earlier rows wrap and would race with later rows at the same physical slot. This is
//      safe because the gather reads from a scratch buffer, NOT the ring; the ring only needs
//      the correct final state for the next chunk/decode. For start_pos==0 this is the
//      reference's "rotated write"; for R2.1 batched continuation it prevents the race that
//      corrupted chunk 2+ ring state when s=4096 >> win=128).
//    Grid = s - lo blocks (one per kv row written), 256 threads × 2 bf16 each (dim = 512).
//    Pure copy — bit-exact by construction.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_ring_write_b(__nv_bfloat16* __restrict__ cache, const __nv_bfloat16* __restrict__ kv,
                  int s, int start_pos, int win, int dim) {
    const int lo = (s > win) ? s - win : 0;
    const int r = lo + blockIdx.x;
    if (r >= s) return;
    const int slot = (start_pos + r) % win;
    const size_t src = (size_t)r * dim;
    const size_t dst = (size_t)slot * dim;
    for (int i = threadIdx.x; i < dim; i += 256) cache[dst + i] = kv[src + i];
}

// ============================================================================
// 4. dsv4_window_idxs_b — §B.2 SWA window index lists, int32 [s, t], −1 = masked.
//    Mirrors dsv4_cpu::window_topk_idxs (model.py:260-271) exactly:
//      prefill (start_pos == 0): t = min(s, win); row i: base = max(0, i−(win−1));
//          entry j = base+j if base+j ≤ i else −1.
//      decode (start_pos ≥ win−1): t = win; sp = start_pos % win; every row =
//          cat([sp+1 .. win−1], [0 .. sp]) — all 128 physical slots, oldest→newest.
//      early decode (0 < start_pos < win−1): t = win; row = [0..start_pos], −1-padded.
//    One thread per (row, entry). Integer math — bit-exact.
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_window_idxs_b(int* __restrict__ out, int s, int start_pos, int win, int t) {
    const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)s * t) return;
    const int row = (int)(idx / t);
    const int j = (int)(idx % t);
    int v;
    if (start_pos >= win - 1) {
        const int sp = start_pos % win;
        const int tail = win - 1 - sp;      // entries from [sp+1 .. win−1]
        v = (j < tail) ? (sp + 1 + j) : (j - tail);
    } else if (start_pos > 0) {
        v = (j <= start_pos) ? j : -1;
    } else {
        const int base = (row >= win - 1) ? (row - (win - 1)) : 0;
        const int cand = base + j;
        v = (cand <= row) ? cand : -1;
    }
    out[idx] = v;
}

// ============================================================================
// 5. dsv4_olo_proj_b — §B.1.4 grouped-LoRA O projection, first GEMM (bf16):
//      o [s, 64, 512] viewed as [s, G=8, 4096]; wo_a [G*R, 4096] = view(8, 1024, 4096);
//      out[t, g*R + r] = bf16( Σ_d o[t, g, d] · wo_a[g*R + r, d] )   (einsum bsgd,grd->bsgr).
//    wo_a is dequantized to bf16 at load (§F.2/§12.A.3) — this is a bf16 GEMM, NOT fp8_bsb.
//    One block per (output column, token row): grid (G*R, s), 256 threads, strided fmaf
//    chain + fixed halving tree (the dsv4_router_score_b pattern) — every output element is
//    computed by the same instruction sequence regardless of s (batch-invariant). fp32
//    accumulate, single bf16 round on store. Tolerance-level vs the CPU's pairwise-tree
//    gemm_bf16 (reduction-order class, rel-L2 ~1e-7..1e-5). Zero stack frames.
//    Weight row for output column c: wo_a + c*gd; activation: o + t*(G*NH_local*hd) +
//    (c/R)*gd where the o row stride is nh*hd = 32768 (G*R*... derived from params).
// ============================================================================
extern "C" __global__ void __launch_bounds__(256)
dsv4_olo_proj_b(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ o,
                const __nv_bfloat16* __restrict__ wo_a, int s, int g, int r, int gd,
                int o_row_stride) {
    const int c = blockIdx.x;  // 0 .. g*r-1 (output column, = group-major g*r + rr)
    const int t = blockIdx.y;  // token row
    if (c >= g * r || t >= s) return;
    const int tid = threadIdx.x;
    const __nv_bfloat16* __restrict__ xr = o + (size_t)t * (size_t)o_row_stride
                                           + (size_t)(c / r) * (size_t)gd;
    const __nv_bfloat16* __restrict__ wr = wo_a + (size_t)c * (size_t)gd;
    float acc = 0.0f;
    for (int i = tid; i < gd; i += 256)
        acc = fmaf(dsv4a_bf16_to_f32(wr[i]), dsv4a_bf16_to_f32(xr[i]), acc);
    __shared__ float sm[256];
    sm[tid] = acc;
    __syncthreads();
#pragma unroll
    for (int f = 128; f > 0; f >>= 1) {
        if (tid < f) sm[tid] += sm[tid + f];
        __syncthreads();
    }
    if (tid == 0) out[(size_t)t * (size_t)(g * r) + c] = dsv4a_f32_to_bf16(sm[0]);
}

// ============================================================================
// Lane 3C — CSA/HCA index-list construction (integer-only, bit-exact targets).
//
// The trunk attention index list (§B.3/§B.4) is `cat([window_idxs, compress_idxs])`
// per query row, laid out row-major as `[s, t_total]` int32 with −1 = masked. The
// gather kernel (`dsv4_gather_attn` in the spine) reads this buffer directly, so
// both sub-lists must land in one contiguous buffer at their respective column
// offsets. The SWA lane (kernel #4 above) writes window idxs into a `[s, t_win]`
// buffer where stride == count; CSA/HCA need stride (`t_total`) != count (`t_win`),
// and a second writer for the compress part. Three small integer kernels:
//   6. dsv4_window_idxs_strided_b — window idxs into a strided [s, t_stride] buffer
//                                  at column 0 (CSA/HCA path; logic identical to #4).
//   7. dsv4_compress_idxs_b      — §B.4 HCA compress idxs (all completed 128-blocks,
//                                  no top-k) into [s, t_stride] at column col_offset.
//                                  Mirrors dsv4_cpu::compress_topk_idxs exactly.
//   8. dsv4_idxs_place_b         — generic strided placement: copy src[s, t_count]
//                                  into dst[s, t_stride] at column col_offset. Used
//                                  to place the CSA indexer's [s, k] output (the
//                                  indexer writes its own contiguous buffer; this
//                                  copies it into the unified list at col t_win).
// All three: one thread per (row, entry); pure integer math; bit-exact.
// ============================================================================

// 6. dsv4_window_idxs_strided_b — §B.2 window idxs with separate count/stride.
//    Identical branching to dsv4_window_idxs_b (kernel #4); writes into the FIRST
//    t_count columns of each row (col_offset = 0). Entries at columns ≥ t_count are
//    NOT touched (the host fills the compress part via kernel #7 or #8). For SWA,
//    pass t_count == t_stride and this is equivalent to kernel #4.
//    R2.1 batched-continuation regime (start_pos > 0 && s > 1): the host builds a
//    unified scratch = [prefix(win) | new(s) | comp(nb)]; the window for row r is
//    scratch rows [r+1 .. r+win] (prefix fills the early entries — no masking needed).
//    This is bitwise-identical to one-shot (same positions, same order: oldest→newest).
extern "C" __global__ void __launch_bounds__(256)
dsv4_window_idxs_strided_b(int* __restrict__ out, int s, int start_pos, int win,
                           int t_count, int t_stride) {
    const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)s * t_count) return;
    const int row = (int)(idx / t_count);
    const int j = (int)(idx % t_count);
    int v;
    if (start_pos > 0 && s >= t_count) {
        // R2.1 batched continuation chunk: unified scratch [prefix(win) | new(s) | comp].
        // window for row r = scratch rows [r+1 .. r+win]; always valid (no masking).
        // Gate: s >= t_count (== win) distinguishes from verify (s ≤ ~6 < 128).
        v = row + 1 + j;
    } else if (start_pos >= win - 1) {
        const int sp = start_pos % win;
        const int tail = win - 1 - sp;
        v = (j < tail) ? (sp + 1 + j) : (j - tail);
    } else if (start_pos > 0) {
        v = (j <= start_pos) ? j : -1;
    } else {
        const int base = (row >= win - 1) ? (row - (win - 1)) : 0;
        const int cand = base + j;
        v = (cand <= row) ? cand : -1;
    }
    out[(size_t)row * (size_t)t_stride + j] = v;
}

// 6b. dsv4_window_idxs_verify_b — R4 verify-width batched arm. The host builds a unified
//    scratch = [prefix(win), POSITION-ORDERED via a rotated ring copy | new(s) | comp(nb)];
//    for row r the window is scratch rows [r+1 .. r+win] — the same mapping as the R2.1
//    continuation branch (kernel #6 branch 1), valid for ANY start_pos >= win because the
//    rotated copy position-orders the prefix (the R2.1 branch needs start_pos%win==0 for
//    that; the rotation generalizes it). Integer math — bit-exact.
extern "C" __global__ void __launch_bounds__(256)
dsv4_window_idxs_verify_b(int* __restrict__ out, int s, int win, int t_count, int t_stride) {
    const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)s * t_count) return;
    const int row = (int)(idx / t_count);
    const int j = (int)(idx % t_count);
    out[(size_t)row * (size_t)t_stride + j] = row + 1 + j;
}

// 6c. dsv4_dspark_draft_idxs_b — DSpark draft attention's non-causal index list,
//     device-side (was a host Vec build + htod per draft step — a full sync each). Every
//     row is identical: [0..t_win) ++ [win .. win+block). Integer math — bit-exact.
extern "C" __global__ void __launch_bounds__(256)
dsv4_dspark_draft_idxs_b(int* __restrict__ out, int block, int t_win, int t, int win) {
    const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)block * t) return;
    const int j = (int)(idx % t);
    out[idx] = (j < t_win) ? j : win + (j - t_win);
}

// 7. dsv4_compress_idxs_b — §B.4 HCA compress idxs (model.py:274-282). Every
//    completed 128-token block, no top-k. Mirrors dsv4_cpu::compress_topk_idxs:
//      decode (start_pos > 0, s == 1): every row = [offset, offset+1, ..., offset+((start_pos+1)/ratio)-1].
//      prefill (start_pos == 0): row i sees block j ⟺ j < (i+1)/ratio; nb = s/ratio
//                                 total entries per row (−1-masked beyond the causal limit).
//      R2.1 continuation (start_pos > 0, s > 1): same as prefill but lim includes start_pos:
//                                 lim = (start_pos + row + 1) / ratio (the batched-chunk
//                                 generalization — bitwise-identical to one-shot).
//    `t_count` = entries per row (decode: (start_pos+1)/ratio; prefill/cont: nb = (start_pos+s)/ratio).
//    `col_offset` = where in the unified [s, t_stride] buffer these land (= t_win).
//    Integer math — bit-exact.
extern "C" __global__ void __launch_bounds__(256)
dsv4_compress_idxs_b(int* __restrict__ out, int s, int start_pos, int ratio, int offset,
                     int t_count, int t_stride, int col_offset) {
    const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)s * t_count) return;
    const int row = (int)(idx / t_count);
    const int j = (int)(idx % t_count);
    int v;
    if (start_pos > 0 && s == 1) {
        v = offset + j;   // decode: arange(0, (start_pos+1)/ratio) + offset
    } else {
        // prefill (start_pos==0) OR R2.1 continuation (start_pos>0, s>1):
        // per-row block-causal limit with start_pos offset (bit-exact vs one-shot).
        const int lim = (start_pos + row + 1) / ratio;
        v = (j < lim) ? (offset + j) : -1;
    }
    out[(size_t)row * (size_t)t_stride + col_offset + j] = v;
}

// 8. dsv4_idxs_place_b — generic strided placement: dst[row, col_offset + j] = src[row, j].
//    Copies a contiguous [s, t_count] int32 buffer into a strided [s, t_stride] buffer
//    at the given column offset. Used to place the CSA indexer's [s, k] output into the
//    unified index list. No masking — the source already carries −1 for masked entries
//    (the indexer's `dsv4_comp_idx_remask_b` kernel applies the offset + masking).
extern "C" __global__ void __launch_bounds__(256)
dsv4_idxs_place_b(int* __restrict__ dst, const int* __restrict__ src,
                  int s, int t_count, int t_stride, int col_offset) {
    const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)s * t_count) return;
    const int row = (int)(idx / t_count);
    const int j = (int)(idx % t_count);
    dst[(size_t)row * (size_t)t_stride + col_offset + j] = src[idx];
}

// ============================================================================
// 9. dsv4_olo_proj_tc_b — §B.1.4 grouped-LoRA O projection via WMMA tensor cores.
//    Replaces the scalar dsv4_olo_proj_b (#5) for production prefill speed: the scalar
//    kernel is ~30% of the CSA long8k replay (2.17s at S=8192). Both inputs are already
//    bf16 device buffers — no weight reformat needed.
//
//    Math (identical to #5): for each group grp ∈ 0..g-1:
//      out[t, grp*r + n] = bf16( Σ_d o[t, grp*gd + d] · wo_a[grp*r + n, d] )
//    WMMA m16n16k16 bf16 fragments, fp32 accumulate, single bf16 RNE on store.
//
//    Grid: (ceil(r/16)*g, ceil(s/16), 1), 32 threads (1 warp) per CTA. One CTA computes
//    one [16, 16] output tile for one group. Bounds-checked on store for non-multiple-of-16 s.
//    Tolerance-level vs the scalar kernel's halving tree (reduction-order class, rel-L2 ~1e-7).
//    (Launcher note: grid.x MUST equal this kernel's tiles_n*g — the old (r+7)/8 launch
//    over-provisioned 2x and half the CTAs exited at the grp >= g guard; R3A.1 fixed.)
// ============================================================================
extern "C" __global__ void __launch_bounds__(32)
dsv4_olo_proj_tc_b(__nv_bfloat16* __restrict__ out,
                   const __nv_bfloat16* __restrict__ o,
                   const __nv_bfloat16* __restrict__ wo_a,
                   int s, int g, int r, int gd, int o_row_stride)
{
    using namespace nvcuda;
    const int M = 16, N = 16, K = 16;
    const int tiles_n = (r + N - 1) / N;
    const int grp = blockIdx.x / tiles_n;
    const int nb = blockIdx.x % tiles_n;
    const int mb = blockIdx.y;
    if (grp >= g) return;

    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const __nv_bfloat16* a_base = o + (size_t)mb * 16 * o_row_stride + (size_t)grp * gd;
    const __nv_bfloat16* b_base = wo_a + (size_t)grp * r * gd + (size_t)nb * 16 * gd;

    for (int k = 0; k < gd; k += K) {
        wmma::load_matrix_sync(a_frag, a_base + k, o_row_stride);
        wmma::load_matrix_sync(b_frag, b_base + k, gd);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ float smem[M * N];
    wmma::store_matrix_sync(smem, c_frag, N, wmma::mem_row_major);

    const int row_base = mb * 16;
    const int col_base = grp * r + nb * 16;
    const int out_stride = g * r;
    for (int i = threadIdx.x; i < M * N; i += 32) {
        const int m = i / N;
        const int n = i % N;
        const int row = row_base + m;
        const int col = col_base + n;
        if (row < s && col < grp * r + r) {
            out[(size_t)row * out_stride + col] = __float2bfloat16(smem[m * N + n]);
        }
    }
}

// ============================================================================
// 9b. dsv4_olo_proj_tc4_b — C=4 n-tile-packed variant (R3A.4 P5).
// Same math, same per-element contract as dsv4_olo_proj_tc_b: each output element's chain
// is 256 sequential wmma.mma over ascending K with +0 init and the same bf16 RNE epilogue.
// Each CTA computes FOUR adjacent n-tiles of the same group: ONE shared A-fragment load
// per K-tile feeds 4 independent accumulator chains (the contract-sanctioned ILP). The
// [16,gd] activation slab is read 16x per (grp,mb) instead of 64x — at prefill width that
// slab re-read was the prefill's biggest single kernel item (nsys: 3.57 s of ~11.3 s).
// Grid: ((r/16/4)*g, ceil(s/16), 1) = (16*g, tiles_m). Requires r % 64 == 0 (host asserts;
// r=1024 in production).
// ============================================================================
extern "C" __global__ void __launch_bounds__(32)
dsv4_olo_proj_tc4_b(__nv_bfloat16* __restrict__ out,
                    const __nv_bfloat16* __restrict__ o,
                    const __nv_bfloat16* __restrict__ wo_a,
                    int s, int g, int r, int gd, int o_row_stride)
{
    using namespace nvcuda;
    const int M = 16, N = 16, K = 16, C = 4;
    const int tiles_n = (r + N - 1) / N;             // 64 unpackaged n-tiles per group
    const int packs_n = tiles_n / C;                 // 16 packs per group
    const int grp = blockIdx.x / packs_n;
    const int nb0 = (blockIdx.x % packs_n) * C;
    const int mb = blockIdx.y;
    if (grp >= g) return;

    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major> b_frag[C];
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag[C];
    #pragma unroll
    for (int c = 0; c < C; c++) wmma::fill_fragment(c_frag[c], 0.0f);

    const __nv_bfloat16* a_base = o + (size_t)mb * 16 * o_row_stride + (size_t)grp * gd;
    const __nv_bfloat16* b_base = wo_a + (size_t)grp * r * gd + (size_t)nb0 * 16 * gd;

    for (int k = 0; k < gd; k += K) {
        wmma::load_matrix_sync(a_frag, a_base + k, o_row_stride);
        #pragma unroll
        for (int c = 0; c < C; c++) {
            wmma::load_matrix_sync(b_frag[c], b_base + (size_t)c * 16 * gd + k, gd);
            wmma::mma_sync(c_frag[c], a_frag, b_frag[c], c_frag[c]);
        }
    }

    __shared__ float smem[C][M * N];
    const int row_base = mb * 16;
    const int out_stride = g * r;
    #pragma unroll
    for (int c = 0; c < C; c++) {
        wmma::store_matrix_sync(smem[c], c_frag[c], N, wmma::mem_row_major);
        const int col_base = grp * r + (nb0 + c) * 16;
        for (int i = threadIdx.x; i < M * N; i += 32) {
            const int m = i / N;
            const int n = i % N;
            const int row = row_base + m;
            const int col = col_base + n;
            if (row < s && col < grp * r + r) {
                out[(size_t)row * out_stride + col] = __float2bfloat16(smem[c][m * N + n]);
            }
        }
    }
}

