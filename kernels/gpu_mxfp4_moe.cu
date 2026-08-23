// gpu_mxfp4_moe.cu — MXFP4-NATIVE MoE expert GEMMs (qwen3.5-122B MoE, --mxfp4=on), sm_121a.
//
// Native (OMMA) counterparts of the two NVFP4 MoE expert GEMMs in kernels/gpu_batch.cu
// (gemm_moe_grouped_mma_fp4 :5026 and gemm_moe_mma_fp4 :4893), plus the multi-group
// activation quant+pack (mxfp4_quant_pack_ng_b) that feeds them:
//
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
//       .f32.e2m1.e2m1.f32.ue4m3      (SASS: OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X)
//   d<4>, a<4>, b<2>, c<4>, sfa, {0,0}, sfb, {0,0}
//
// All fragment/scale layouts are the ones EMPIRICALLY VERIFIED on GB10 (probe 1899/1899,
// kernels/mxfp4_bench.cu header comment, 2026-08-06) — do NOT re-derive:
//
//   A fragment (lane g = lane>>2, t = lane&3; reg a_r, nibble j = LSB-first):
//     a0: row g     k 8t..8t+7      a1: row g+8   k 8t..8t+7
//     a2: row g     k 8t+32..8t+39  a3: row g+8   k 8t+32..8t+39
//   B fragment (b_r, nibble j): token n = g, k = 8t + j + 32*r   (ONE token row per lane)
//   C fragment (reg i): row g + 8*(i>=2), col 2t + (i&1)
//   SFA (lane (g,t), byte v): row g + 8*(t&1), kblock v  -- lanes t in {2,3} ignored
//   SFB (lane (g,t), byte v): token g, kblock v          -- only lanes t == 0 read
//   kblock v covers k in [16v, 16v+16) of the instruction's 64-wide K.
//
// Repacked weight layouts (per 16-row tile mt, 64-K kstep ks; expert tiles are stacked
// contiguously: tile index mt_g = e*ntm_per_expert + mt for LOCAL expert e):
//   Aimg[(mt_g*nks+ks)*128 + lane*4 + r]   u32, 512 B per (tile,kstep), lane-major
//   SFAw[(mt_g*nks+ks)*16  + g*2 + (t&1)]  u32, 64 B per (tile,kstep), t<=1 valid
//   gs[mt_g]                                f32 per 16-row tile (applied once in the epilogue)
// Activation layouts (8-token group g per kstep; N groups are packed in ONE launch):
//   Bp[((g*nks+ks)*32 + lane)*2 + r]       u32, r=0 -> b0 (k 8t..8t+7), r=1 -> b1 (k+32)
//   SFB[(g*nks+ks)*32 + lane]              u32, byte v = scale(token g, kblock ks*4+v), t==0 only
//
// e2m1 nibbles are NOT shifted (the mxf4nvf4 path needs none of the mxf8f6f4 padding);
// ue4m3 sign bit ignored; 1.0 = 0x38.  Quant recipe (identical to the dense twin
// gpu_mxfp4.cu mxfp4_quant_pack_b): per-16-block amax -> e4m3_ceil(amax/6) -> cvt.rn.
// satfinite.e2m1x2.  Padded token rows (>= N real rows) are ZEROED with scale 0 so the
// OMMA columns they feed are exactly 0 and never stored.
//
// Schedule contracts (mirror the bf16 kernels exactly): k-visit order FIXED (warp w takes
// ksteps w, w+8, ...), the cross-warp reduction is the FIXED-order shared-memory merge
// (sh[i*256 + warp*32 + lane], ascending warp index), gs applied exactly once per tile,
// NO atomics anywhere.  Every per-thread array is compile-time indexed: ptxas must report
// a ZERO stack frame (hard rule).
//
// COMPILE: sm_121a family-specific features (the mma and cvt.e2m1x2 reject plain sm_121);
// build.rs must target this file with --gpu-architecture=sm_121a (the serving manifest for
// other files stays sm_121).  For a standalone executable use the -gencode form
// (arch=compute_121a,code=sm_121a), exactly like kernels/mxfp4_bench.cu.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifdef KERNEL_BUILD_ID
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }
#else
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = 0; }
#endif

typedef __nv_bfloat16 bf16;

#define MMA_NW 8                       // warps per block (split the kstep chain, fixed order)
#define MMA_SMEM (8 * 256)             // [8 acc slots (2 groups x 4)][8 warps][32 lanes] f32

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

// Two f32 -> one byte of two e2m1 nibbles (low nibble = src[0]), round-to-nearest-even with
// satfinite, the hardware instruction (sm_121a).  Verbatim from gpu_mxfp4.cu.
__device__ __forceinline__ unsigned char cvt_e2m1x2(float lo, float hi) {
    unsigned tmp;
    asm volatile(
        "{\n.reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n}"
        : "=r"(tmp) : "f"(lo), "f"(hi));
    return (unsigned char)(tmp & 0xff);
}

// ue4m3 encode of |x| rounded UP (sign bit 0 — the OMMA ignores it). Mirrors the quantizer.
__device__ __host__ __forceinline__ unsigned char e4m3_ceil(float x) {
    if (!(x > 0.f)) return 0x00;
    if (x >= 448.0f) return 0x7F;
    int e;
    float m = frexpf(x, &e);
    int e4 = e + 6;
    int mant = (int)ceilf((m - 0.5f) * 16.0f);
    if (mant >= 8) { mant = 0; e4++; }
    if (e4 < 0) {
        int sm = (int)ceilf(x * 512.0f);
        return (unsigned char)(sm > 7 ? 7 : sm);
    }
    if (e4 > 14) return 0x7F;
    return (unsigned char)((e4 << 3) | mant);
}

__device__ __host__ __forceinline__ float ue4m3_f(unsigned char s) {
    int e = (s >> 3) & 0xF, m = s & 7;
    if (e == 0) return (float)m * 0.001953125f;
    return (1.0f + m / 8.0f) * exp2f((float)e - 7);
}

// ---------------------------------------------------------------------------
// Activation quant + pack for N 8-token groups in ONE launch.  Group g (g = blockIdx.x/nks)
// quantizes the 8 token rows X[(g*row_stride + n)*K .. ]: row n in {0..7}, N = number of REAL
// rows per group (N=8 for the grouped MoE path where all 8 rows are in-bounds padded data,
// row_stride=8; N=1 for the slot path where only row 0 is real — row_stride=1 maps group g to
// token row g — and rows 1..7 are ZEROED, their scales come out 0, so the OMMA columns they
// feed are exactly 0).  Writes Bp[g*nks*64 + ks*64 ...] and SFB[g*nks*32 + ks*32 ...]: the same
// layouts the GEMMs consume, offset by g*nks.  One block per (group, 64-K kstep); warp n =
// token n.  Shared-memory staging + two __syncthreads, same structure as gpu_mxfp4.cu's
// mxfp4_quant_pack_b.
// ---------------------------------------------------------------------------
extern "C" __global__ void mxfp4_quant_pack_ng_b(const bf16* __restrict__ X, int K, int ngroups,
                                                 int nks, int N, int row_stride,
                                                 uint32_t* __restrict__ Bp,
                                                 uint32_t* __restrict__ SFB,
                                                 const int* __restrict__ poff, int ne) {
    const int grp = blockIdx.x / nks;    // 8-token group
    const int ks = blockIdx.x % nks;     // 64-K kstep
    // BS(b) group-bound exit (EXPERT_BATCH_SCALING §5b): the grouped GEMM early-exits on the SAME
    // device bound (nt*16 >= poff[ne]) — quantizing the fully-padded groups past the real total is
    // the F12 static-grid waste this kills. NULL poff = unbounded (the slot arm / dense refs).
    if (poff && grp * 8 >= poff[ne]) return;
    const int n = threadIdx.x >> 5;      // token 0..7 (>= N: zero row)
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    __shared__ float sh[8][64];
    if (n < N) {
        const bf16* row = X + (long long)(grp * row_stride + n) * K + (long long)ks * 64;
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = b2f(row[lane * 2 + i]);
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = 0.0f;
    }
    __syncthreads();

    // per-token per-kblock scales (4 kblocks of 16 per token)
    __shared__ float sc[8][4];
    if (lane < 4) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(sh[n][lane * 16 + i]));
        sc[n][lane] = e4m3_ceil(amax / 6.0f);
    }
    __syncthreads();

    // pack b0/b1: nibble j of b_r = code of X[token g][ks*64 + 8t + j + 32r]
    uint32_t b0 = 0, b1 = 0;
    const float inv0 = sc[g][t >> 1] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)sc[g][t >> 1]);
    const float inv1 = sc[g][2 + (t >> 1)] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)sc[g][2 + (t >> 1)]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned cb = (unsigned)cvt_e2m1x2(sh[g][8 * t + 2 * j] * inv0, sh[g][8 * t + 2 * j + 1] * inv0);
        b0 |= cb << (8 * j);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned cb = (unsigned)cvt_e2m1x2(sh[g][32 + 8 * t + 2 * j] * inv1, sh[g][32 + 8 * t + 2 * j + 1] * inv1);
        b1 |= cb << (8 * j);
    }
    const size_t bbase = ((size_t)grp * nks + ks) * 32 + lane;
    Bp[bbase * 2 + 0] = b0;
    Bp[bbase * 2 + 1] = b1;

    // SFB: u32 per lane; byte v = scale(token g, kblock ks*4 + v). Only t == 0 is read.
    uint32_t sfb = 0;
    if (t == 0) {
        unsigned char b0c = (unsigned char)sc[g][0], b1c = (unsigned char)sc[g][1];
        unsigned char b2c = (unsigned char)sc[g][2], b3c = (unsigned char)sc[g][3];
        sfb = (uint32_t)b0c | ((uint32_t)b1c << 8) | ((uint32_t)b2c << 16) | ((uint32_t)b3c << 24);
    }
    SFB[bbase] = sfb;
}

// ---------------------------------------------------------------------------
// Grouped MoE GEMV (16 tokens per expert-tile block) — OMMA port of gemm_moe_grouped_mma_fp4
// (gpu_batch.cu:5026).  grid (M/16, ngroups); block (mt, nt) computes tile mt of the expert
// owning 16-token group nt.  The 16 tokens are the Bp groups 2*nt (tokens 0..7) and 2*nt+1
// (tokens 8..15); the expert's weight tile is read ONCE and fed to both OMMAs (acc[0] = group
// 0, acc[1] = group 1; OMMA q's C fragment covers output columns q*8..q*8+7).  The launch
// grid.y is a static upper bound: blocks with nt*16 >= poff[ne] exit against the DEVICE-side
// padded total (no host readback).  ALL 16 output columns are stored (the caller's C buffer
// is padded), no column guard; fixed k-visit order, fixed-order cross-warp reduce, no atomics.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 6) void mxfp4_gemm_moe_grouped_native_b(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const uint32_t* __restrict__ Bp,
    const uint32_t* __restrict__ SFBw, int ntm_per_expert, int nks,
    int expert_base, const int* __restrict__ tile_e, const int* __restrict__ poff, int ne)
{
    const int M = ntm_per_expert * 16;
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 16 >= poff[ne]) return;                 // E16: static-grid bound, device-side total
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int e = tile_e[nt * 2] - expert_base;      // LOCAL expert id
    const long long mt_g = (long long)e * ntm_per_expert + mt;
    __shared__ float sh[MMA_SMEM];

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint32_t* wt = reinterpret_cast<const uint32_t*>(Wt) + (size_t)(mt_g * nks) * 128;
    const uint32_t* sfa_base = reinterpret_cast<const uint32_t*>(SFAw) + (size_t)(mt_g * nks) * 16;

    // Fixed k-visit order: warp w takes ksteps w, w+8, ... (N-independent, no atomics).
    for (int ks = warp; ks < nks; ks += MMA_NW) {
        const uint32_t a0 = wt[(size_t)ks * 128 + lane * 4 + 0];
        const uint32_t a1 = wt[(size_t)ks * 128 + lane * 4 + 1];
        const uint32_t a2 = wt[(size_t)ks * 128 + lane * 4 + 2];
        const uint32_t a3 = wt[(size_t)ks * 128 + lane * 4 + 3];
        const uint32_t sfa = (t <= 1) ? sfa_base[(size_t)ks * 16 + g * 2 + t] : 0u;
#pragma unroll
        for (int q = 0; q < 2; q++) {
            const int gg = 2 * nt + q;               // Bp group: tokens q*8..q*8+7
            const uint32_t b0 = Bp[(((size_t)gg * nks + ks) * 32 + lane) * 2 + 0];
            const uint32_t b1 = Bp[(((size_t)gg * nks + ks) * 32 + lane) * 2 + 1];
            const uint32_t sfb = SFBw[((size_t)gg * nks + ks) * 32 + lane];
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                : "+f"(acc[q][0]), "+f"(acc[q][1]), "+f"(acc[q][2]), "+f"(acc[q][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
        }
    }

    // Cross-warp fixed-order reduction (mirrors mma_warp_reduce; no atomics).
#pragma unroll
    for (int q = 0; q < 2; q++)
#pragma unroll
        for (int i = 0; i < 4; i++) sh[(q * 4 + i) * 256 + warp * 32 + lane] = acc[q][i];
    __syncthreads();
    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    if (rslot < 8) {                                  // 2 groups x 4 acc slots = all 16 cols
        const int q = rslot >> 2, i = rslot & 3;
        const int col = q * 8 + 2 * t + (i & 1);
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];  // FIXED order
        const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
        C[(long long)(nt * 16 + col) * M + m] = f2b(v * gs[mt_g]);  // gs applied exactly once
    }
}

// ---------------------------------------------------------------------------
// Per-slot MoE GEMV (N=1 token per slot) — OMMA port of gemm_moe_mma_fp4 (gpu_batch.cu:4893,
// the SINGLE-tile kernel).  grid (M/16, bslot); bslot = (token, expert-pair) slot index.
// Remote experts (e < 0 or e >= e_span) contribute an exact zero.  The slot's B side comes
// from Bp group xrow (xrow = bslot if x_by_slot, else bslot/Kslots) — the caller quantized
// ONE 8-token group per distinct token with N=1 (row 0 real, rows 1..7 zeroed).  ONE OMMA per
// kstep; epilogue stores only column 0 (N=1), gs[mt_g] once.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 6) void mxfp4_gemm_moe_slot_native_b(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const uint32_t* __restrict__ Bp,
    const uint32_t* __restrict__ SFBw, const int* __restrict__ ids, int nks,
    int Kslots, int expert_base, int e_span, int ntm_per_expert)
{
    const int M = ntm_per_expert * 16;
    const int x_by_slot = (Kslots == 0);               // Kslots=0: slot-major rows (down proj)
    const int kslots = x_by_slot ? 1 : Kslots;         // avoid the div-by-zero in the xrow map
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int e = ids[bslot] - expert_base;            // LOCAL expert id
    bf16* Cb = C + (long long)bslot * M;
    if (e < 0 || e >= e_span) {                        // remote expert: contribute 0
        if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
        return;
    }
    const int xrow = x_by_slot ? bslot : (bslot / kslots);
    const long long mt_g = (long long)e * ntm_per_expert + mt;
    __shared__ float sh[MMA_SMEM];

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t* wt = reinterpret_cast<const uint32_t*>(Wt) + (size_t)(mt_g * nks) * 128;
    const uint32_t* sfa_base = reinterpret_cast<const uint32_t*>(SFAw) + (size_t)(mt_g * nks) * 16;

    // Fixed k-visit order: warp w takes ksteps w, w+8, ...
    for (int ks = warp; ks < nks; ks += MMA_NW) {
        const uint32_t a0 = wt[(size_t)ks * 128 + lane * 4 + 0];
        const uint32_t a1 = wt[(size_t)ks * 128 + lane * 4 + 1];
        const uint32_t a2 = wt[(size_t)ks * 128 + lane * 4 + 2];
        const uint32_t a3 = wt[(size_t)ks * 128 + lane * 4 + 3];
        const uint32_t sfa = (t <= 1) ? sfa_base[(size_t)ks * 16 + g * 2 + t] : 0u;
        const uint32_t b0 = Bp[(((size_t)xrow * nks + ks) * 32 + lane) * 2 + 0];
        const uint32_t b1 = Bp[(((size_t)xrow * nks + ks) * 32 + lane) * 2 + 1];
        const uint32_t sfb = SFBw[((size_t)xrow * nks + ks) * 32 + lane];
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
              "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
    }

    // Cross-warp fixed-order reduction (mirrors mma_warp_reduce; no atomics).
#pragma unroll
    for (int i = 0; i < 4; i++) sh[i * 256 + warp * 32 + lane] = acc[i];
    __syncthreads();
    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    if (rslot < 4) {                                   // N=1: only column 0 is stored
        const int i = rslot;
        const int col = 2 * t + (i & 1);
        if (col == 0) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];  // FIXED order
            const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
            if (m < M) Cb[m] = f2b(v * gs[mt_g]);      // gs applied exactly once
        }
    }
}

// ---------------------------------------------------------------------------
// FUSED per-slot MoE GEMV (EXPERT_FUSED_QUANT_RESPONSE.md §4.1, F2) — replaces the
// mxfp4_quant_pack_ng_b + mxfp4_gemm_moe_slot_native_b pair on the fused dispatch path
// (GB10_MXFP4_FUSED, default ON; GB10_MXFP4_FUSED=0 keeps today's separate launches).
// Per 64-K kstep each warp quantizes the N=1 group it is about to feed to the OMMA: the
// slot's X row (xrow), same op order as the standalone quant (§3) — byte-identical
// fragments. Lanes with g > 0 emit b0=b1=sfb=0 directly (== the standalone kernel's zeroed
// rows 1..7). XSILU=1 (the down side): the stage computes silu(g,u) from the interleaved
// GU row with moe_silu_bf16_b's EXACT math (g/(1+__expf(-g)))*u, bf16 in and out — the
// silu launch and the h_s buffer disappear (§6). Grid/block, the remote-expert exact-zero
// early-out, the xrow map, k-visit order, and the fixed-order cross-warp reduction are
// verbatim from mxfp4_gemm_moe_slot_native_b. K = the GEMM's K (h for gu, mi for dn);
// XSILU=1 reads the GU row at stride 2*K (g/u interleaved).
// ---------------------------------------------------------------------------
template <bool XSILU>
__device__ __forceinline__ void moe_slot_fused_body(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, const int* __restrict__ ids,
    int nks, int K, int Kslots, int expert_base, int e_span, int ntm_per_expert)
{
    const int M = ntm_per_expert * 16;
    const int x_by_slot = (Kslots == 0);               // Kslots=0: slot-major rows (down proj)
    const int kslots = x_by_slot ? 1 : Kslots;         // avoid the div-by-zero in the xrow map
    const int mt = blockIdx.x, bslot = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int e = ids[bslot] - expert_base;            // LOCAL expert id
    bf16* Cb = C + (long long)bslot * M;
    if (e < 0 || e >= e_span) {                        // remote expert: contribute 0
        if (threadIdx.x < 16) Cb[mt * 16 + threadIdx.x] = f2b(0.f);
        return;
    }
    const int xrow = x_by_slot ? bslot : (bslot / kslots);
    const long long mt_g = (long long)e * ntm_per_expert + mt;
    __shared__ float sh[MMA_SMEM];
    __shared__ bf16 shw[MMA_NW][8][64];
    __shared__ float scw[MMA_NW][8][4];

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t* wt = reinterpret_cast<const uint32_t*>(Wt) + (size_t)(mt_g * nks) * 128;
    const uint32_t* sfa_base = reinterpret_cast<const uint32_t*>(SFAw) + (size_t)(mt_g * nks) * 16;

    // Fixed k-visit order: warp w takes ksteps w, w+8, ...
    for (int ks = warp; ks < nks; ks += MMA_NW) {
        const uint32_t a0 = wt[(size_t)ks * 128 + lane * 4 + 0];
        const uint32_t a1 = wt[(size_t)ks * 128 + lane * 4 + 1];
        const uint32_t a2 = wt[(size_t)ks * 128 + lane * 4 + 2];
        const uint32_t a3 = wt[(size_t)ks * 128 + lane * 4 + 3];
        const uint32_t sfa = (t <= 1) ? sfa_base[(size_t)ks * 16 + g * 2 + t] : 0u;
        // STAGE: the slot's 64-K window (N=1). One uniform code path: row 0 is the slot's X
        // row (xrow), rows 1..7 are zero-staged, so dead rows go through the standalone
        // kernel's exact inv-0 path (scale 0 -> inv 0 -> +0 codes) and the fragments are
        // byte-identical to mxfp4_quant_pack_ng_b's zeroed rows. XSILU: silu(gu[k])*gu[K+k]
        // with moe_silu_bf16_b's exact math, staged as bf16 (bit-exact on the b2f read).
        {
            const bf16* row = X + (size_t)xrow * (XSILU ? 2 * K : K) + (size_t)ks * 64 + lane * 2;
            if (XSILU) {
                const float g0 = b2f(row[0]), u0 = b2f(row[K + 0]);
                const float g1 = b2f(row[1]), u1 = b2f(row[K + 1]);
                shw[warp][0][lane * 2 + 0] = f2b((g0 / (1.f + __expf(-g0))) * u0);
                shw[warp][0][lane * 2 + 1] = f2b((g1 / (1.f + __expf(-g1))) * u1);
            } else {
                shw[warp][0][lane * 2 + 0] = row[0];
                shw[warp][0][lane * 2 + 1] = row[1];
            }
#pragma unroll
            for (int n = 1; n < 8; n++) {
                shw[warp][n][lane * 2 + 0] = f2b(0.f);
                shw[warp][n][lane * 2 + 1] = f2b(0.f);
            }
        }
        __syncwarp();
        // SCALE: lane (n,v) = (lane>>2, lane&3) — same ascending amax chain, amax/6.0f,
        // e4m3_ceil (dead rows -> amax 0 -> scale byte 0).
        {
            const int n = lane >> 2, v = lane & 3;
            float amax = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(b2f(shw[warp][n][v * 16 + i])));
            scw[warp][n][v] = e4m3_ceil(amax / 6.0f);
        }
        __syncwarp();
        // PACK: all lanes active; lane (g,t) packs token g's fragment (rows 1..7 -> zeros).
        const float inv0 = scw[warp][g][t >> 1] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)scw[warp][g][t >> 1]);
        const float inv1 = scw[warp][g][2 + (t >> 1)] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)scw[warp][g][2 + (t >> 1)]);
        uint32_t b0 = 0, b1 = 0, sfb = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned cb = (unsigned)cvt_e2m1x2(b2f(shw[warp][g][8 * t + 2 * j]) * inv0, b2f(shw[warp][g][8 * t + 2 * j + 1]) * inv0);
            b0 |= cb << (8 * j);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned cb = (unsigned)cvt_e2m1x2(b2f(shw[warp][g][32 + 8 * t + 2 * j]) * inv1, b2f(shw[warp][g][32 + 8 * t + 2 * j + 1]) * inv1);
            b1 |= cb << (8 * j);
        }
        if (t == 0) {
            unsigned char b0c = (unsigned char)scw[warp][g][0], b1c = (unsigned char)scw[warp][g][1];
            unsigned char b2c = (unsigned char)scw[warp][g][2], b3c = (unsigned char)scw[warp][g][3];
            sfb = (uint32_t)b0c | ((uint32_t)b1c << 8) | ((uint32_t)b2c << 16) | ((uint32_t)b3c << 24);
        }
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
              "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
    }

    // Cross-warp fixed-order reduction (mirrors mma_warp_reduce; no atomics).
#pragma unroll
    for (int i = 0; i < 4; i++) sh[i * 256 + warp * 32 + lane] = acc[i];
    __syncthreads();
    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    if (rslot < 4) {                                   // N=1: only column 0 is stored
        const int i = rslot;
        const int col = 2 * t + (i & 1);
        if (col == 0) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];  // FIXED order
            const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
            if (m < M) Cb[m] = f2b(v * gs[mt_g]);      // gs applied exactly once
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 5) void mxfp4_gemm_moe_slot_fused_b0(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, const int* __restrict__ ids,
    int nks, int K, int Kslots, int expert_base, int e_span, int ntm_per_expert)
{
    moe_slot_fused_body<false>(C, Wt, SFAw, gs, X, ids, nks, K, Kslots, expert_base, e_span, ntm_per_expert);
}

extern "C" __global__ __launch_bounds__(256, 6) void mxfp4_gemm_moe_slot_fused_b1(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, const int* __restrict__ ids,
    int nks, int K, int Kslots, int expert_base, int e_span, int ntm_per_expert)
{
    moe_slot_fused_body<true>(C, Wt, SFAw, gs, X, ids, nks, K, Kslots, expert_base, e_span, ntm_per_expert);
}

// ---------------------------------------------------------------------------
// FUSED grouped MoE GEMV (EXPERT_FUSED_QUANT_RESPONSE.md §5, F3) — replaces the
// mxfp4_quant_pack_ng_b + mxfp4_gemm_moe_grouped_native_b pair (verify + prefill windows;
// the prefill arm sits behind the GB10_MXFP4_FUSED_PREFILL=0 escape). Per 64-K kstep each
// warp runs TWO 8-token group passes (q = 0,1; groups 2nt, 2nt+1) against ONE per-warp
// staging buffer: stage -> scale -> pack -> the q-th mma into acc[q], with __syncwarp()
// between passes. The A fragment is loaded once per kstep and shared by both mmas, exactly
// as in the unfused kernel. All 8 rows of a group are real (the gather wrote exact +0 for
// padding rows), so every lane packs; the poff[ne] device-side early-exit is kept as the
// FIRST statement — exited blocks never stage nor quantize (the F12 padded-grid kill).
// XSILU=1 (dn): the stage computes silu(g,u) from the interleaved GU row with
// moe_silu_bf16_b's exact math; the silu launch and the h_p buffer disappear (§6).
// Grid/block, tile map, k-visit order, and the fixed-order cross-warp reduction are
// verbatim from mxfp4_gemm_moe_grouped_native_b.
// ---------------------------------------------------------------------------
template <bool XSILU>
__device__ __forceinline__ void moe_grouped_fused_body(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, int ntm_per_expert, int nks,
    int K, int expert_base, const int* __restrict__ tile_e, const int* __restrict__ poff, int ne)
{
    const int M = ntm_per_expert * 16;
    const int mt = blockIdx.x, nt = blockIdx.y;
    if (nt * 16 >= poff[ne]) return;                 // E16: static-grid bound, device-side total
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int e = tile_e[nt * 2] - expert_base;      // LOCAL expert id
    const long long mt_g = (long long)e * ntm_per_expert + mt;
    __shared__ float sh[MMA_SMEM];
    __shared__ bf16 shw[MMA_NW][8][64];
    __shared__ float scw[MMA_NW][8][4];

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint32_t* wt = reinterpret_cast<const uint32_t*>(Wt) + (size_t)(mt_g * nks) * 128;
    const uint32_t* sfa_base = reinterpret_cast<const uint32_t*>(SFAw) + (size_t)(mt_g * nks) * 16;

    // Fixed k-visit order: warp w takes ksteps w, w+8, ... (N-independent, no atomics).
    for (int ks = warp; ks < nks; ks += MMA_NW) {
#pragma unroll
        for (int q = 0; q < 2; q++) {
            const int gg = 2 * nt + q;               // Bp group: tokens q*8..q*8+7
            // (a) A-side loads FIRST per pass (the quant ALU hides under the weight-load
            // latency; a per-pass lifetime avoids holding 5 registers across both passes).
            const uint32_t a0 = wt[(size_t)ks * 128 + lane * 4 + 0];
            const uint32_t a1 = wt[(size_t)ks * 128 + lane * 4 + 1];
            const uint32_t a2 = wt[(size_t)ks * 128 + lane * 4 + 2];
            const uint32_t a3 = wt[(size_t)ks * 128 + lane * 4 + 3];
            const uint32_t sfa = (t <= 1) ? sfa_base[(size_t)ks * 16 + g * 2 + t] : 0u;
            // (b) STAGE the 8 rows of group gg (64 elems/row, 2/lane/row). XSILU: silu from
            // the interleaved GU row, moe_silu_bf16_b's exact math, staged bf16 (bit-exact).
#pragma unroll 4
            for (int n = 0; n < 8; n++) {
                const bf16* row = X + (size_t)(gg * 8 + n) * (XSILU ? 2 * K : K) + (size_t)ks * 64 + lane * 2;
                if (XSILU) {
                    const float g0 = b2f(row[0]), u0 = b2f(row[K + 0]);
                    const float g1 = b2f(row[1]), u1 = b2f(row[K + 1]);
                    shw[warp][n][lane * 2 + 0] = f2b((g0 / (1.f + __expf(-g0))) * u0);
                    shw[warp][n][lane * 2 + 1] = f2b((g1 / (1.f + __expf(-g1))) * u1);
                } else {
                    shw[warp][n][lane * 2 + 0] = row[0];
                    shw[warp][n][lane * 2 + 1] = row[1];
                }
            }
            __syncwarp();
            // SCALE: lane (n,v) = (lane>>2, lane&3); same ascending amax chain, amax/6.0f.
            {
                const int n = lane >> 2, v = lane & 3;
                float amax = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(b2f(shw[warp][n][v * 16 + i])));
                scw[warp][n][v] = e4m3_ceil(amax / 6.0f);
            }
            __syncwarp();
            // PACK: all lanes active (all 8 rows real in the grouped path).
            const float inv0 = scw[warp][g][t >> 1] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)scw[warp][g][t >> 1]);
            const float inv1 = scw[warp][g][2 + (t >> 1)] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)scw[warp][g][2 + (t >> 1)]);
            uint32_t b0 = 0, b1 = 0, sfb = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                unsigned cb = (unsigned)cvt_e2m1x2(b2f(shw[warp][g][8 * t + 2 * j]) * inv0, b2f(shw[warp][g][8 * t + 2 * j + 1]) * inv0);
                b0 |= cb << (8 * j);
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                unsigned cb = (unsigned)cvt_e2m1x2(b2f(shw[warp][g][32 + 8 * t + 2 * j]) * inv1, b2f(shw[warp][g][32 + 8 * t + 2 * j + 1]) * inv1);
                b1 |= cb << (8 * j);
            }
            if (t == 0) {
                unsigned char b0c = (unsigned char)scw[warp][g][0], b1c = (unsigned char)scw[warp][g][1];
                unsigned char b2c = (unsigned char)scw[warp][g][2], b3c = (unsigned char)scw[warp][g][3];
                sfb = (uint32_t)b0c | ((uint32_t)b1c << 8) | ((uint32_t)b2c << 16) | ((uint32_t)b3c << 24);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                : "+f"(acc[q][0]), "+f"(acc[q][1]), "+f"(acc[q][2]), "+f"(acc[q][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
            __syncwarp();   // pass boundary: next pass's stage writes wait for this pack's reads
        }
    }

    // Cross-warp fixed-order reduction (mirrors mma_warp_reduce; no atomics).
#pragma unroll
    for (int q = 0; q < 2; q++)
#pragma unroll
        for (int i = 0; i < 4; i++) sh[(q * 4 + i) * 256 + warp * 32 + lane] = acc[q][i];
    __syncthreads();
    const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;
    if (rslot < 8) {                                  // 2 groups x 4 acc slots = all 16 cols
        const int q = rslot >> 2, i = rslot & 3;
        const int col = q * 8 + 2 * t + (i & 1);
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];  // FIXED order
        const int m = mt * 16 + g + ((i >= 2) ? 8 : 0);
        C[(long long)(nt * 16 + col) * M + m] = f2b(v * gs[mt_g]);  // gs applied exactly once
    }
}

extern "C" __global__ __launch_bounds__(256, 5) void mxfp4_gemm_moe_grouped_fused_b0(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, int ntm_per_expert, int nks,
    int K, int expert_base, const int* __restrict__ tile_e, const int* __restrict__ poff, int ne)
{
    moe_grouped_fused_body<false>(C, Wt, SFAw, gs, X, ntm_per_expert, nks, K, expert_base, tile_e, poff, ne);
}

extern "C" __global__ __launch_bounds__(256, 5) void mxfp4_gemm_moe_grouped_fused_b1(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const bf16* __restrict__ X, int ntm_per_expert, int nks,
    int K, int expert_base, const int* __restrict__ tile_e, const int* __restrict__ poff, int ne)
{
    moe_grouped_fused_body<true>(C, Wt, SFAw, gs, X, ntm_per_expert, nks, K, expert_base, tile_e, poff, ne);
}
