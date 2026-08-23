// mxfp4_bench.cu — Standalone persistent-grid GEMV benchmark + validation harness for the
// sm_121a native block-scaled FP4 warp instruction (SASS: OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X).
//
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
//       .f32.e2m1.e2m1.f32.ue4m3
//   d<4>, a<4>, b<2>, c<4>, sfa, {0,0}, sfb, {0,0}
//
// SEMANTICS:  D[16x8] += A[16x64] * B[8x64]^T  (B fragment is K-major: per lane (g,t),
//   reg b_r, nibble j holds X[token g][k = 8t + j + 32r]) with per-(row|token, 16-K-block)
//   ue4m3 scales.  A = weights (16 rows x 64 K), B = activations (8 tokens x 64 K).
//   fp32 accumulate; e2m1 inputs; ue4m3 (e4m3) 8-bit scales, one per 16 K elements ("4X").
//
// All fragment/scale layouts below were EMPIRICALLY VERIFIED on this machine by
// /tmp/opencode/mxfp4_probe/probe.cu (1899/1899 checks pass, GB10 cc 12.1, sm_121a):
//
//   A fragment (lane g = lane>>2, t = lane&3; reg a_r, nibble j = LSB-first):
//     a0: row g     k 8t..8t+7      a1: row g+8   k 8t..8t+7
//     a2: row g     k 8t+32..8t+39  a3: row g+8   k 8t+32..8t+39
//     row = g + 8*(r&1),  k = 8t + j + 32*(r>>1)
//   B fragment (b_r, nibble j): token n = g, k = 8t + j + 32*r
//   C/D fragment (reg i): row g + 8*(i>=2), col 2t + (i&1)   [one f32 per reg]
//   SFA (lane (g,t), byte v, v = LSB first): row g + 8*(t&1), kblock v  -- lanes t in {2,3}
//     are IGNORED by hardware (their sfa register is not read)
//   SFB (lane (g,t), byte v): token g, kblock v  -- only lanes t == 0 are read
//   kblock v covers k in [16v, 16v+16) of the 64-wide instruction K.
//   ue4m3 decode: sign bit IGNORED; value = (1 + mant/8) * 2^(exp-7) from bits [7]=sign(skip),
//     [6:3]=exp, [2:0]=mant.  1.0 = 0x38, 2.0 = 0x40, 4.0 = 0x48.
//   e2m1 nibble: bits [3:0], no shift (mxf4nvf4 needs NO nibble<<2 padding; that shift is
//     only for the mxf8f6f4 f8f6f4 path). 1.0 = 0b0010, 1.5 = 0b0011, 4.0 = 0b0110.
//
// ============================ HOST-REPACKED WEIGHT LAYOUT ============================
// Input: bf16 W[M][K], M % 16 == 0, K % 64 == 0.  Quantized per (row, 16-K-block):
//   e2m1 code c = e2m1_rn(x / scale),  scale = ue4m3(amax/6 rounded UP) (magnitude byte).
//
// Per (16-row tile mt, 64-K kstep ks) the host writes:
//
//   Aimg[(mt*nks + ks)*128 + lane*4 + r]   (u32, 512 B per (tile,kstep), lane-major)
//     reg a_r nibble j (bits 4j..4j+3) = code of W[mt*16 + g + 8*(r&1)][ks*64 + 8t + j + 32*(r>>1)]
//
//   SFAimg[(mt*nks + ks)*16 + g*2 + (t&1)] (u32, 64 B per (tile,kstep))
//     byte v of the u32 = ue4m3 scale of row (mt*16 + g + 8*(t&1)), kblock (ks*4 + v).
//     Valid for lanes t in {0,1} only; the kernel supplies 0 for t in {2,3}.
//
//   gs[mt]                                  (f32, per 16-row tile tensor scale, applied once
//     in the epilogue -- mirrors the production gemm_mma_fp4_b contract)
//
// ============================ PRE-PACKED B (ACTIVATION) LAYOUT =======================
// Input: bf16 X[8][K].  Quantized per (token, 16-K-block) the same way.
//
//   Bp[(ks*32 + lane)*2 + r]                (u32, 256 B per kstep; r = 0 -> b0, r = 1 -> b1)
//     b0 nibble j = code of X[g][ks*64 + 8t + j]
//     b1 nibble j = code of X[g][ks*64 + 32 + 8t + j]
//
//   SFB[ks*32 + lane]                       (u32, 128 B per kstep; one per lane)
//     byte v = ue4m3 scale of token g, kblock (ks*4 + v).  Only lanes t == 0 are read;
//     the host fills the other lanes' words with 0 (they are ignored by hardware).
//
// The same 8 tokens' B fragments feed EVERY warp and EVERY tile (activations are shared),
// exactly like the production m16n8k16 kernel where B = X is shared across warps.
//
// ============================ KERNEL SCHEDULE ========================================
// Mirrors kernels/gpu_batch.cu gemm_mma_fp4_b (the serving GEMM), with the OMMA in place
// of the bf16 m16n8k16 sequence: 256 threads / 8 warps, __launch_bounds__(256, 6),
// grid = min(ntm, 288) persistent blocks; block b computes tiles b, b+gridDim.x, ...
// Warp w walks the tile's ksteps in FIXED order w, w+8, w+16, ...; the 8 warps' partial
// accumulators are merged in FIXED warp order through shared memory (sh[i*256+w*32+lane],
// ascending warp index -- no atomics), then the epilogue multiplies by gs[mt] once and
// stores bf16 to C[n*M + m].  Every per-thread array is compile-time indexed: ptxas must
// report a ZERO stack frame (hard rule).
//
// Build (GB10-only; the mxf4nvf4 mma and cvt.e2m1x2 require the 'a' target):
//   /usr/local/cuda/bin/nvcc -O3 -gencode arch=compute_121a,code=sm_121a mxfp4_bench.cu \
//       -o mxfp4_bench
//   NOTE: plain -arch=sm_121a embeds compute_121 PTX in the fatbin for executable builds
//   and ptxas rejects the mma; use the -gencode form.  sm_121 and sm_121f-family notes:
//   sm_121 REJECTS the instruction; sm_121f and sm_121a accept it (CUDA 13.0.88).
//
// Run:
//   ./mxfp4_bench --verify --M 8192 --K 4096 --iters 200     # correctness + timing
//   ./mxfp4_bench --quant-self-test                           # quant/pack round trip
//   ./mxfp4_bench --help

#include <cstdio>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CHECK_CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

typedef __nv_bfloat16 bf16;

#define MMA_NW 8          // warps per block
#define MMA_SMEM (4 * 256)  // [4 acc slots][8 warps][32 lanes] f32 (GEMV: single 16x8 subtile)

// ---------------------------------------------------------------------------
// Format helpers (self-contained; no gpu_batch includes)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

// Two f32 -> one byte of two e2m1 nibbles (low nibble = src[0], high = src[1]),
// round-to-nearest-even with satfinite, exact hardware path (CUDA 13.0.88, sm_121a).
__device__ __forceinline__ unsigned char cvt_e2m1x2(float lo, float hi) {
    unsigned tmp;
    asm volatile(
        "{\n.reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n}"
        : "=r"(tmp) : "f"(lo), "f"(hi));
    return (unsigned char)(tmp & 0xff);
}

// e4m3 (ue4m3) encode of |x| rounded UP (smallest representable magnitude >= |x|).
// Sign bit always written 0 -- the OMMA ignores the ue4m3 sign bit (verified).
// 0.0 -> 0x00; clamps to 0x7F (448) at the top; subnormals allowed below exp 1.
__device__ __host__ __forceinline__ unsigned char e4m3_ceil(float x) {
    if (!(x > 0.f)) return 0x00;               // x <= 0 or NaN -> 0
    if (x >= 448.0f) return 0x7F;
    int e;
    float m = frexpf(x, &e);                   // m in [0.5, 1), x = m * 2^e
    int e4 = e + 6;                            // value = (1 + mant/8) * 2^(e4-7): e4-7 = e-1
    int mant = (int)ceilf((m - 0.5f) * 16.0f); // mant bits of (1.mant): m-0.5 in [0, 0.5)
    if (mant >= 8) { mant = 0; e4++; }
    if (e4 < 0) {                               // subnormal: value = mant * 2^-9 (standard fp8)
        int sm = (int)ceilf(x * 512.0f);
        return (unsigned char)(sm > 7 ? 7 : sm);
    }
    if (e4 > 14) return 0x7F;
    return (unsigned char)((e4 << 3) | mant);
}

// ue4m3 decode (verified against hardware incl. subnormals: exp==0 -> mant*2^-9, and the
// sign bit is ignored).
__device__ __host__ __forceinline__ float ue4m3_f(unsigned char s) {
    int e = (s >> 3) & 0xF, m = s & 7;
    if (e == 0) return (float)m * 0.001953125f;          // m * 2^-9
    return (1.0f + m / 8.0f) * exp2f((float)e - 7);
}

// One 16-element bf16 block -> (ue4m3 scale, 8 nibble codes low..high).
// scale = ue4m3(amax/6 rounded UP); code = e2m1_rn(x * (1/scale)) via cvt.
__device__ __forceinline__ void quantize_bf16_block(const bf16* p, unsigned char& scale,
                                                    unsigned& nibbles /* 8 codes, code i at bits 4i */) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(b2f(p[i])));
    scale = e4m3_ceil(amax / 6.0f);
    float inv = (scale == 0x00) ? 0.f : 1.0f / ue4m3_f(scale);
    nibbles = 0u;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned char byte = cvt_e2m1x2(b2f(p[2 * i]) * inv, b2f(p[2 * i + 1]) * inv);
        nibbles |= (unsigned)byte << (8 * i);
    }
}

// Full B-side quant + pack: bf16 X[8][K] -> Bp/SFB (the layouts above). One block per
// 64-K kstep; 8 warps = 8 tokens.  Not on the timed path (the bench consumes pre-packed
// input) -- exists so the quant pipeline is complete, correct, and self-testable.
__global__ void mxfp4_quant_pack_b(const bf16* __restrict__ X, int K, int nks,
                                   uint32_t* __restrict__ Bp, uint32_t* __restrict__ SFB) {
    const int ks = blockIdx.x;
    const int n = threadIdx.x >> 5;          // token 0..7
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    __shared__ float sh[8][64];              // bf16 staged as f32
    const bf16* row = X + (long long)n * K + (long long)ks * 64;
#pragma unroll
    for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = b2f(row[lane * 2 + i]);
    __syncthreads();

    // scales: 4 kblocks of 16 per token; one thread per (token, kblock)
    __shared__ float sc[8][4];
    if (lane < 4) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(sh[n][lane * 16 + i]));
        sc[n][lane] = e4m3_ceil(amax / 6.0f);
    }
    __syncthreads();

    // pack b0/b1 for this lane: nibble j of b_r = code of X[g][ks*64 + 8t + j + 32r]
    // (the B fragment slot (g,t) holds TOKEN g's row -- verified layout)
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
    Bp[((long long)ks * 32 + lane) * 2 + 0] = b0;
    Bp[((long long)ks * 32 + lane) * 2 + 1] = b1;

    // SFB: u32 per lane; byte v = scale(token g, kblock ks*4 + v). Only t == 0 is read.
    uint32_t sfb = 0;
    if (t == 0) {
        unsigned char b0c = (unsigned char)sc[g][0], b1c = (unsigned char)sc[g][1];
        unsigned char b2c = (unsigned char)sc[g][2], b3c = (unsigned char)sc[g][3];
        sfb = (uint32_t)b0c | ((uint32_t)b1c << 8) | ((uint32_t)b2c << 16) | ((uint32_t)b3c << 24);
    }
    SFB[(long long)ks * 32 + lane] = sfb;
}

// ---------------------------------------------------------------------------
// The benchmarked GEMV kernel
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 6) void mxfp4_gemv_bench(
    bf16* __restrict__ C, const uint8_t* __restrict__ Wt, const uint8_t* __restrict__ SFAw,
    const float* __restrict__ gs, const uint32_t* __restrict__ Bp,
    const uint32_t* __restrict__ SFBw, int ntm, int nks, int M)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    __shared__ float sh[MMA_SMEM];

    // Persistent tiles: block b computes tiles b, b+gridDim.x, ... (production schedule).
    for (int mt = blockIdx.x; mt < ntm; mt += gridDim.x) {
        __syncthreads();                     // sh reuse barrier (write->read->next write)
        float acc[4] = {0.f, 0.f, 0.f, 0.f};

        const uint32_t* wt = reinterpret_cast<const uint32_t*>(Wt) + (size_t)mt * nks * 128;
        const uint32_t* sfa_base = reinterpret_cast<const uint32_t*>(SFAw) + (size_t)mt * nks * 16;

        // Fixed k-visit order per tile: warp w takes ksteps w, w+8, ... (N-independent).
        for (int ks = warp; ks < nks; ks += MMA_NW) {
            const uint32_t a0 = wt[(size_t)ks * 128 + lane * 4 + 0];
            const uint32_t a1 = wt[(size_t)ks * 128 + lane * 4 + 1];
            const uint32_t a2 = wt[(size_t)ks * 128 + lane * 4 + 2];
            const uint32_t a3 = wt[(size_t)ks * 128 + lane * 4 + 3];
            // SFA: lanes t in {0,1} carry the row (g + 8*(t&1)) scales; t in {2,3} ignored.
            const uint32_t sfa = (t <= 1) ? sfa_base[(size_t)ks * 16 + g * 2 + t] : 0u;
            const uint32_t b0 = Bp[((size_t)ks * 32 + lane) * 2 + 0];
            const uint32_t b1 = Bp[((size_t)ks * 32 + lane) * 2 + 1];
            const uint32_t sfb = SFBw[(size_t)ks * 32 + lane];

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
        const int rlane = threadIdx.x & 31, rslot = threadIdx.x >> 5;   // 256 (slot, lane) pairs
        float v = 0.0f;
        if (rslot < 4) {
#pragma unroll
            for (int w = 0; w < MMA_NW; w++) v += sh[rslot * 256 + w * 32 + rlane];  // FIXED order
            // Invert the D fragment map: reg i -> (row g + 8*(i>=2), col 2t + (i&1)).
            const int m = mt * 16 + g + ((rslot >= 2) ? 8 : 0);
            const int n = 2 * t + (rslot & 1);
            C[(long long)n * M + m] = f2b(v * gs[mt]);   // gs applied exactly once
        }
    }
}

// ---------------------------------------------------------------------------
// Host: reference quantization + repack (mirror of the device math, exact formulas)
// ---------------------------------------------------------------------------
static bf16 f2b_h(float x) { return __float2bfloat16(x); }
static float b2f_h(bf16 x) { return __bfloat162float(x); }

static unsigned char e4m3_ceil_h(float x) {
    if (!(x > 0.f)) return 0x00;
    if (x >= 448.0f) return 0x7F;
    int e;
    float m = frexpf(x, &e);
    int e4 = e + 6;
    int mant = (int)ceilf((m - 0.5f) * 16.0f);
    if (mant >= 8) { mant = 0; e4++; }
    if (e4 < 0) {
        int shift = -e4;
        int sm = (mant | 8) >> shift;
        if ((mant | 8) & ((1 << shift) - 1)) sm++;
        if (sm >= 8) return 0x08;
        return (unsigned char)sm;
    }
    if (e4 > 14) return 0x7F;
    return (unsigned char)((e4 << 3) | mant);
}

// host mirror of cvt.rn.satfinite.e2m1x2 for a single float (round to nearest even,
// saturate at 6.0; the 16 e2m1 values with their bit patterns).
static unsigned char e2m1_rn_h(float x) {
    const float vals[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float ax = fabsf(x);
    int best = 0;
    float bd = 1e30f;
    for (int c = 0; c < 8; c++) {
        float d = fabsf(ax - vals[c]);
        if (d < bd - 1e-7f) { bd = d; best = c; }
        else if (fabsf(d - bd) <= 1e-7f) { if (best & 1) best = c; }  // tie -> even code
    }
    if (x < 0) best |= 8;
    return (unsigned char)best;
}
static void quant_block_h(const bf16* p, unsigned char& scale, unsigned char* codes /* 16 */) {
    float amax = 0.f;
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(b2f_h(p[i])));
    scale = e4m3_ceil_h(amax / 6.0f);
    float inv = (scale == 0x00) ? 0.f : 1.0f / ue4m3_f(scale);
    for (int i = 0; i < 16; i++) codes[i] = e2m1_rn_h(b2f_h(p[i]) * inv);
}

// Host weight repack: bf16 W[M][K] -> Aimg/SFAimg/gs (layouts in the header comment).
static void repack_weights_h(const std::vector<bf16>& W, int M, int K,
                             std::vector<uint8_t>& Aimg, std::vector<uint8_t>& SFAw,
                             std::vector<float>& gs) {
    int ntm = M / 16, nks = K / 64;
    Aimg.assign((size_t)ntm * nks * 512, 0);
    SFAw.assign((size_t)ntm * nks * 64, 0);
    gs.assign(ntm, 1.0f);
    for (int mt = 0; mt < ntm; mt++) {
        gs[mt] = (mt & 1) ? 0.5f : 2.0f;   // exercise the epilogue multiply
        for (int ks = 0; ks < nks; ks++) {
            // scales per (row, kblock)
            unsigned char scale16[16][4];
            unsigned char codes[16][64];
            for (int r = 0; r < 16; r++) {
                for (int kb = 0; kb < 4; kb++) {
                    quant_block_h(&W[((size_t)mt * 16 + r) * K + (size_t)ks * 64 + kb * 16],
                                  scale16[r][kb], &codes[r][kb * 16]);
                }
            }
            for (int lane = 0; lane < 32; lane++) {
                int g = lane >> 2, t = lane & 3;
                for (int r = 0; r < 4; r++) {
                    int row = g + 8 * (r & 1);
                    uint32_t v = 0;
                    for (int j = 0; j < 8; j++) {
                        int k = 8 * t + j + 32 * (r >> 1);
                        v |= (uint32_t)codes[row][k] << (4 * j);
                    }
                    ((uint32_t*)Aimg.data())[((size_t)mt * nks + ks) * 128 + lane * 4 + r] = v;
                }
                if (t <= 1) {
                    uint32_t v = 0;
                    for (int b = 0; b < 4; b++)
                        v |= (uint32_t)scale16[g + 8 * t][b] << (8 * b);
                    ((uint32_t*)SFAw.data())[((size_t)mt * nks + ks) * 16 + g * 2 + t] = v;
                }
            }
        }
    }
}

// Host B pack: bf16 X[8][K] -> Bp/SFB (header comment layouts).
static void pack_b_h(const std::vector<bf16>& X, int K,
                     std::vector<uint32_t>& Bp, std::vector<uint32_t>& SFB) {
    int nks = K / 64;
    Bp.assign((size_t)nks * 64, 0);
    SFB.assign((size_t)nks * 32, 0);
    for (int ks = 0; ks < nks; ks++) {
        unsigned char scale8[8][4];
        unsigned char codes[8][64];
        for (int n = 0; n < 8; n++) {
            for (int kb = 0; kb < 4; kb++) {
                quant_block_h(&X[(size_t)n * K + (size_t)ks * 64 + kb * 16],
                              scale8[n][kb], &codes[n][kb * 16]);
            }
        }
        for (int lane = 0; lane < 32; lane++) {
            int g = lane >> 2, t = lane & 3;
            uint32_t b0 = 0, b1 = 0;
            for (int j = 0; j < 8; j++) {
                b0 |= (uint32_t)codes[g][8 * t + j] << (4 * j);
                b1 |= (uint32_t)codes[g][32 + 8 * t + j] << (4 * j);
            }
            Bp[((size_t)ks * 32 + lane) * 2 + 0] = b0;
            Bp[((size_t)ks * 32 + lane) * 2 + 1] = b1;
            if (t == 0) {
                SFB[(size_t)ks * 32 + lane] = (uint32_t)scale8[g][0] | ((uint32_t)scale8[g][1] << 8) |
                                              ((uint32_t)scale8[g][2] << 16) | ((uint32_t)scale8[g][3] << 24);
            }
        }
    }
}

// CPU reference GEMV from the quantized values (exact fp32; summation order is
// deliberately sequential over k so hardware fp32 reassociation stays within tolerance).
static std::vector<float> ref_gemv(const std::vector<bf16>& W, const std::vector<bf16>& X,
                                   int M, int K, const std::vector<float>& gs) {
    std::vector<float> D((size_t)8 * M, 0.f);
    for (int mt = 0; mt < M / 16; mt++) {
        for (int n = 0; n < 8; n++) {
            for (int m = 0; m < 16; m++) {
                double s = 0.0;
                for (int k = 0; k < K; k++) {
                    // quantize both operands the same way the host repack did
                    int kb = k / 16;
                    unsigned char sca, scb;
                    unsigned char ca[16], cb[16];
                    quant_block_h(&W[((size_t)mt * 16 + m) * K + kb * 16], sca, ca);
                    quant_block_h(&X[(size_t)n * K + kb * 16], scb, cb);
                    const float va[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                          -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
                    float fa = va[ca[k % 16]], fb = va[cb[k % 16]];
                    float sa = ue4m3_f(sca);
                    float sb = ue4m3_f(scb);
                    s += (double)fa * sa * (double)fb * sb;
                }
                D[(size_t)n * M + mt * 16 + m] = (float)s * gs[mt];
            }
        }
    }
    return D;
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
static void usage() {
    printf("mxfp4_bench [--verify] [--quant-self-test] [--M N] [--K N] [--iters N] [--seed N]\n");
}

int main(int argc, char** argv) {
    int M = 8192, K = 4096, iters = 200;
    unsigned seed = 12345;
    bool verify = false, quant_self_test = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--M") M = atoi(argv[++i]);
        else if (a == "--K") K = atoi(argv[++i]);
        else if (a == "--iters") iters = atoi(argv[++i]);
        else if (a == "--seed") seed = (unsigned)atoi(argv[++i]);
        else if (a == "--verify") verify = true;
        else if (a == "--quant-self-test") quant_self_test = true;
        else if (a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 1; }
    }
    if (M % 16 || K % 64) { fprintf(stderr, "M %% 16 and K %% 64 required (got %dx%d)\n", M, K); return 1; }

    cudaDeviceProp p;
    CHECK_CU(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s cc %d.%d SMs %d L2 %d KiB\n", p.name, p.major, p.minor,
           p.multiProcessorCount, (int)(p.l2CacheSize / 1024));
    if (p.major * 10 + p.minor != 121) { fprintf(stderr, "expected cc 12.1\n"); return 1; }

    int ntm = M / 16, nks = K / 64;
    printf("M=%d K=%d  ntm=%d nks=%d grid=%d\n", M, K, ntm, nks, std::min(ntm, 288));

    // deterministic data
    std::vector<bf16> W((size_t)M * K), X(8 * (size_t)K);
    {
        uint32_t s = seed;
        auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (int)(s >> 8) % 2001 - 1000; };
        for (auto& v : W) v = f2b_h((float)rnd() / 128.0f);
        for (auto& v : X) v = f2b_h((float)rnd() / 128.0f);
    }

    // ---- quant self test: device quant+pack vs host quant+pack ----
    if (quant_self_test) {
        std::vector<uint32_t> Bp_h, SFB_h;
        pack_b_h(X, K, Bp_h, SFB_h);
        uint32_t *dBp, *dSFB; bf16* dX;
        CHECK_CU(cudaMalloc(&dBp, Bp_h.size() * 4));
        CHECK_CU(cudaMalloc(&dSFB, SFB_h.size() * 4));
        CHECK_CU(cudaMalloc(&dX, X.size() * 2));
        CHECK_CU(cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice));
        mxfp4_quant_pack_b<<<nks, 256>>>(dX, K, nks, dBp, dSFB);
        CHECK_CU(cudaGetLastError());
        CHECK_CU(cudaDeviceSynchronize());
        std::vector<uint32_t> Bp_d(Bp_h.size()), SFB_d(SFB_h.size());
        CHECK_CU(cudaMemcpy(Bp_d.data(), dBp, Bp_h.size() * 4, cudaMemcpyDeviceToHost));
        CHECK_CU(cudaMemcpy(SFB_d.data(), dSFB, SFB_h.size() * 4, cudaMemcpyDeviceToHost));
        int diff = 0;
        for (size_t i = 0; i < Bp_h.size(); i++) if (Bp_d[i] != Bp_h[i]) diff++;
        for (size_t i = 0; i < SFB_h.size(); i++) if (SFB_d[i] != SFB_h[i]) diff++;
        printf("quant self test: %zu Bp + %zu SFB words, %d mismatches -> %s\n",
               Bp_h.size(), SFB_h.size(), diff, diff ? "FAIL" : "PASS");

        CHECK_CU(cudaFree(dBp)); CHECK_CU(cudaFree(dSFB)); CHECK_CU(cudaFree(dX));
        if (diff) return 1;
    }

    // ---- repack + pack ----
    std::vector<uint8_t> Aimg, SFAw;
    std::vector<float> gs;
    std::vector<uint32_t> Bp, SFB;
    repack_weights_h(W, M, K, Aimg, SFAw, gs);
    pack_b_h(X, K, Bp, SFB);

    bf16 *dC; uint8_t *dWt, *dSFA; float *dgs; uint32_t *dBp, *dSFB;
    CHECK_CU(cudaMalloc(&dC, 8 * (size_t)M * 2));
    CHECK_CU(cudaMalloc(&dWt, Aimg.size()));
    CHECK_CU(cudaMalloc(&dSFA, SFAw.size()));
    CHECK_CU(cudaMalloc(&dgs, gs.size() * 4));
    CHECK_CU(cudaMalloc(&dBp, Bp.size() * 4));
    CHECK_CU(cudaMalloc(&dSFB, SFB.size() * 4));
    CHECK_CU(cudaMemcpy(dWt, Aimg.data(), Aimg.size(), cudaMemcpyHostToDevice));
    CHECK_CU(cudaMemcpy(dSFA, SFAw.data(), SFAw.size(), cudaMemcpyHostToDevice));
    CHECK_CU(cudaMemcpy(dgs, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CU(cudaMemcpy(dBp, Bp.data(), Bp.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CU(cudaMemcpy(dSFB, SFB.data(), SFB.size() * 4, cudaMemcpyHostToDevice));

    int grid = std::min(ntm, 288);
    auto launch = [&]() {
        mxfp4_gemv_bench<<<grid, 256>>>(dC, dWt, dSFA, dgs, dBp, dSFB, ntm, nks, M);
    };

    // ---- validation vs CPU reference ----
    // The kernel output is bf16; the reference is computed in double from the SAME host
    // quantization.  Compare against the ref rounded to bf16 (kernel IS bf16) with a
    // 5e-3 relative bound: bf16 half-ulp is <= 3.9e-3, so any mismatch above that is a bug.
    if (verify) {
        launch();
        CHECK_CU(cudaGetLastError());
        CHECK_CU(cudaDeviceSynchronize());
        std::vector<bf16> out(8 * (size_t)M);
        CHECK_CU(cudaMemcpy(out.data(), dC, out.size() * 2, cudaMemcpyDeviceToHost));
        auto ref = ref_gemv(W, X, M, K, gs);
        double maxrel = 0, maxabs = 0;
        int bad = 0;
        for (size_t i = 0; i < out.size(); i++) {
            float r = b2f_h(f2b_h(ref[i]));   // ref rounded to bf16
            float o = b2f_h(out[i]);
            double rel = fabs((double)o - r) / fmax(1.0, fabs(r));
            maxrel = fmax(maxrel, rel);
            maxabs = fmax(maxabs, fabs((double)o - r));
            if (rel > 5e-3) bad++;
        }
        printf("verify: %d/%zu mismatches, maxrel=%.2e maxabs=%.2e -> %s\n",
               bad, out.size(), maxrel, maxabs, (bad == 0) ? "PASS (bit-exact after bf16)" : "FAIL");
        if (bad) return 1;
    }

    // ---- timing ----
    cudaEvent_t t0, t1;
    CHECK_CU(cudaEventCreate(&t0)); CHECK_CU(cudaEventCreate(&t1));
    launch(); CHECK_CU(cudaDeviceSynchronize());          // warmup
    CHECK_CU(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) launch();
    CHECK_CU(cudaEventRecord(t1));
    CHECK_CU(cudaEventSynchronize(t1));
    float ms = 0; CHECK_CU(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;
    double useful_bytes = (double)M * K / 2.0 + 8.0 * K / 2.0 + 8.0 * M * 2.0;  // e2m1 W + B + out
    double dram_bytes = (double)M * K / 2.0 + 8.0 * K / 2.0;                    // weights + activations
    printf("kernel: %.3f ms/iter  useful %.1f GB/s (273 GB/s ceiling -> %.1f%%), DRAM-only %.1f GB/s\n",
           ms, useful_bytes / ms / 1e6, 100.0 * useful_bytes / ms / 2.73e8,
           dram_bytes / ms / 1e6);
    return 0;
}
