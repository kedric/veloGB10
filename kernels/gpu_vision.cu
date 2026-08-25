// Vision-tower kernels for the Qwen3-VL visual trunk (SM12x, GB10). FP32 throughout.
//
// These implement the GPU half of the vision tower port (PLAN/W2_GPU_PORT_HANDOFF.md). The tower
// runs in FP32 to match the CPU reference (src/vision_encoder.rs::forward_cpu), which loads the
// on-disk BF16 weights to FP32 and does all matmuls in FP32; the binding cross-chain envelope
// (PLAN/W2_PREPROC_SPEC.md §10, rel-L2 ~2.9e-5) can only be met at FP32 precision. This path NEVER
// touches the NVFP4/FP8 dequant chain (AGENTS §2.4 / §6) — it is the plain BF16-weight -> FP32
// compute class of the vision tower.
//
// head_dim = 72 is parameterized as a template parameter `HD` (AGENTS §7: never hardcode a
// head_dim into a kernel; the text kernels' 128..512 envelope gets a sub-128 instance by
// parameterization, not a special case). The extern "C" wrapper dispatches the runtime head_dim to
// the matching compile-time instantiation, with a generic scalar fallback for any other value.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void vision_kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

__device__ __forceinline__ float v_gelu_tanh_f(float x) {
    const float K = 0.7978845608028654f;   // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(K * (x + 0.044715f * x * x * x)));
}

// nn.GELU() (default erf form) as used by the merger. `erff` from libdevice matches the CPU
// reference's window to ~1e-7, far inside the 2.9e-5 envelope.
__device__ __forceinline__ float v_gelu_f(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));  // x/sqrt(2)
}

__device__ __forceinline__ float v_rot_half(const float* q, int d, int hd) {
    // rotate_half: first hd/2 entries are the second half (negated); the rest re-enter unwrapped.
    return (d < hd / 2) ? -q[hd / 2 + d] : q[d - hd / 2];
}

// Warp-wide sum via butterfly shuffle (all 32 lanes active; full mask).
__device__ __forceinline__ float v_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// ---- LayerNorm (vision), NOT RMSNorm: out = (x - mean) * rsqrt(var + eps) * w + b ----
// one block per row; D = HIDDEN (1152) so blockDim must work for D <= ~1024+ (use 256, grid-stride).
extern "C" __global__ void vision_layernorm(float* out, const float* x, const float* w,
                                            const float* b, int N, int D, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float s_ln[];
    float* rmean = s_ln;            // [bdim]
    float* rvar = s_ln + bdim;      // [bdim]
    const float* xr = x + (long)row * D;
    float* yr = out + (long)row * D;

    float msum = 0.0f;
    for (int i = tid; i < D; i += bdim) msum += xr[i];
    rmean[tid] = msum;
    __syncthreads();
    for (int st = bdim / 2; st > 0; st >>= 1) {
        if (tid < st) rmean[tid] += rmean[tid + st];
        __syncthreads();
    }
    float mean = rmean[0] / (float)D;
    __syncthreads();               // all threads must read mean before rvar reuses the other half

    float vsum = 0.0f;
    for (int i = tid; i < D; i += bdim) { float d = xr[i] - mean; vsum += d * d; }
    rvar[tid] = vsum;
    __syncthreads();
    for (int st = bdim / 2; st > 0; st >>= 1) {
        if (tid < st) rvar[tid] += rvar[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(rvar[0] / (float)D + eps);
    for (int i = tid; i < D; i += bdim) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// ---- out[i] += src[i] (elementwise, full length n) ----
extern "C" __global__ void vision_add_inplace(float* out, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += src[i];
}

// ---- out[i] += bias[i % outn]  (in-place bias-broadcast for a [rows, outn] row-major activation) ----
extern "C" __global__ void vision_bias_add(float* out, const float* bias, int rows, int outn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * outn) out[i] += bias[i % outn];
}

// ---- gelu (pytorch_tanh) in-place ----
extern "C" __global__ void vision_gelu_tanh(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v_gelu_tanh_f(out[i]);
}

// ---- gelu (erf, the merger) in-place ----
extern "C" __global__ void vision_gelu(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v_gelu_f(out[i]);
}

// ---- Vision FULL self-attention (not causal), over N tokens, one packed chunk, no KV cache. ----
//
// Warp-per-query flash attention: each warp handles one (token, head) and streams keys with an
// online softmax, so the working set is O(HD) registers, NOT O(N) shared memory. The previous
// per-(token,head)-block kernel materialized all N scores in __shared__ (scores[N] @ 4 B) — fine at
// N<=4096, but a high-res image (e.g. 1946x1628 -> 12,444 patches) needs 49.9 KiB > the 48 KiB
// static cap and failed with CUDA_ERROR_INVALID_VALUE. This version is scalable to any N.
//
// qkv is [N, 3*hidden], hidden = heads*hd: each token's row is [q | k | v] (head-major, hd each).
// cos/sin are [N, hd]. out is [N, hidden]. Grid = ceil(N/nwarps)*heads; blockDim = 128 (4 warps).
template <int HD>
__device__ void vision_attn_impl(const float* qkv, const float* cos, const float* sin,
                                 float* out, int N, int heads, int hidden, float scale) {
    const int hd = HD;
    constexpr int DIPL = (HD + 31) / 32;   // dims owned per lane
    int qb = blockIdx.x / heads;
    int h = blockIdx.x % heads;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int nwarps = blockDim.x / 32;
    long q = (long)qb * nwarps + warp;
    if (q >= N) return;

    const float* qrow = qkv + (long)q * 3 * hidden + (long)h * hd;
    float qo[DIPL];
    #pragma unroll
    for (int c = 0; c < DIPL; ++c) {
        int d = lane + c * 32;
        if (d < HD) qo[c] = qrow[d] * cos[(long)q * hd + d] + v_rot_half(qrow, d, hd) * sin[(long)q * hd + d];
        else qo[c] = 0.0f;
    }
    float oa[DIPL];
    #pragma unroll
    for (int c = 0; c < DIPL; ++c) oa[c] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int j = 0; j < N; ++j) {
        const float* kbase = qkv + (long)j * 3 * hidden + (long)h * hd;
        float partial = 0.0f;
        #pragma unroll
        for (int c = 0; c < DIPL; ++c) {
            int d = lane + c * 32;
            if (d < HD) {
                float kv = kbase[hidden + d];
                float kr = kv * cos[(long)j * hd + d] + v_rot_half(kbase + hidden, d, hd) * sin[(long)j * hd + d];
                partial += qo[c] * kr;
            }
        }
        float s = v_warp_sum(partial) * scale;
        float m_new = fmaxf(m, s);
        float rescale = __expf(m - m_new);          // exp(m - m_new); first key: exp(-inf)=0
        float e = __expf(s - m_new);
        l = l * rescale + e;
        #pragma unroll
        for (int c = 0; c < DIPL; ++c) {
            int d = lane + c * 32;
            if (d < HD) oa[c] = oa[c] * rescale + e * kbase[2 * hidden + d];
        }
        m = m_new;
    }
    if (l <= 0.0f) l = 1.0f;
    #pragma unroll
    for (int c = 0; c < DIPL; ++c) {
        int d = lane + c * 32;
        if (d < HD) out[(long)q * heads * hd + (long)h * hd + d] = oa[c] / l;
    }
}

// Generic out-of-envelope head_dim fallback (correct, not tuned): the same warp-per-query flash, but
// a runtime hd. The vision tower's config head_dim is 72, so this never runs for this model — it
// keeps the kernel exhaustive so a wrong head_dim yields a correct-but-slow result, never a
// wrong-layout one.
#define MAXVH 512
#define GDIPL ((MAXVH + 31) / 32)
__device__ void vision_attn_generic_dev(const float* qkv, const float* cos, const float* sin,
                                        float* out, int N, int heads, int hidden,
                                        float scale, int hd) {
    int qb = blockIdx.x / heads;
    int h = blockIdx.x % heads;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int nwarps = blockDim.x / 32;
    long q = (long)qb * nwarps + warp;
    if (q >= N) return;

    const float* qrow = qkv + (long)q * 3 * hidden + (long)h * hd;
    float qo[GDIPL];
    for (int c = 0; c < GDIPL; ++c) {
        int d = lane + c * 32;
        if (d < hd) qo[c] = qrow[d] * cos[(long)q * hd + d] + v_rot_half(qrow, d, hd) * sin[(long)q * hd + d];
        else qo[c] = 0.0f;
    }
    float oa[GDIPL];
    for (int c = 0; c < GDIPL; ++c) oa[c] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int j = 0; j < N; ++j) {
        const float* kbase = qkv + (long)j * 3 * hidden + (long)h * hd;
        float partial = 0.0f;
        for (int c = 0; c < GDIPL; ++c) {
            int d = lane + c * 32;
            if (d < hd) {
                float kv = kbase[hidden + d];
                float kr = kv * cos[(long)j * hd + d] + v_rot_half(kbase + hidden, d, hd) * sin[(long)j * hd + d];
                partial += qo[c] * kr;
            }
        }
        float s = v_warp_sum(partial) * scale;
        float m_new = fmaxf(m, s);
        float rescale = __expf(m - m_new);
        float e = __expf(s - m_new);
        l = l * rescale + e;
        for (int c = 0; c < GDIPL; ++c) {
            int d = lane + c * 32;
            if (d < hd) oa[c] = oa[c] * rescale + e * kbase[2 * hidden + d];
        }
        m = m_new;
    }
    if (l <= 0.0f) l = 1.0f;
    for (int c = 0; c < GDIPL; ++c) {
        int d = lane + c * 32;
        if (d < hd) out[(long)q * heads * hd + (long)h * hd + d] = oa[c] / l;
    }
}

// Unmangled dispatch for the model's fixed head_dim (72) — the hot path. `hd` is NOT taken here;
// the Rust caller asserts config head_dim == HEAD_DIM and launches this symbol only for 72 (it picks
// `vision_attn_generic` for any other value). Keeping the generic out of this kernel's image keeps
// its register/stack budget at the HD=72 instantiation (AGENTS §4).
extern "C" __global__ void vision_attn(const float* qkv, const float* cos, const float* sin,
                                       float* out, int N, int heads, int hidden, float scale) {
    vision_attn_impl<72>(qkv, cos, sin, out, N, heads, hidden, scale);
}

// Generic out-of-envelope head_dim fallback (correct, not tuned); the vision tower never takes it
// for this model (head_dim fixed 72), but it keeps the kernel exhaustive.
extern "C" __global__ void vision_attn_generic(const float* qkv, const float* cos, const float* sin,
                                               float* out, int N, int heads, int hidden,
                                               float scale, int hd) {
    vision_attn_generic_dev(qkv, cos, sin, out, N, heads, hidden, scale, hd);
}

// ---- cuBLAS-attention support: transpose qkv [N, 3*hidden] (token-major) into per-head contiguous
// [N, hd] row-major q/k/v buffers (applying vision RoPE to q/k). Output qo/ko/vo are [heads, N, hd]
// = h*N*hd + tq*hd + d. The per-head [N, hd] row-major layout is what the cuBLAS QK/PV GEMMs below
// read (transa=T / transb=T). Kept out of the text attention path (AGENTS §2 — vision-only).
extern "C" __global__ void vision_rope_split_transpose(const float* qkv, const float* cos,
                                                       const float* sin,
                                                       float* qo, float* ko, float* vo,
                                                       int N, int heads, int hidden, int hd) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * heads * hd;
    if (i >= total) return;
    int d = (int)(i % hd);
    long r = i / hd;
    int tq = (int)(r % N);
    int h = (int)(r / N);
    const float* row = qkv + (long)tq * 3 * hidden + (long)h * hd;
    float qv = row[d];
    qo[i] = qv * cos[(long)tq * hd + d] + v_rot_half(row, d, hd) * sin[(long)tq * hd + d];
    float kv = row[hidden + d];
    ko[i] = kv * cos[(long)tq * hd + d] + v_rot_half(row + hidden, d, hd) * sin[(long)tq * hd + d];
    vo[i] = row[2 * hidden + d];
}

// In-place softmax over the KEY dim of a [N, N] column-major score matrix (s[tq + tk*N]).
// Threads = queries (tq is the fast index, so adjacent tq read contiguous memory — coalesced).
extern "C" __global__ void vision_softmax_rows(float* s, int N) {
    int tq = blockIdx.x * blockDim.x + threadIdx.x;
    if (tq >= N) return;
    float m = -INFINITY;
    for (int tk = 0; tk < N; ++tk) m = fmaxf(m, s[tq + (long)tk * N]);
    float l = 0.0f;
    for (int tk = 0; tk < N; ++tk) { float e = __expf(s[tq + (long)tk * N] - m); s[tq + (long)tk * N] = e; l += e; }
    float inv = (l > 0.0f) ? 1.0f / l : 1.0f;
    for (int tk = 0; tk < N; ++tk) s[tq + (long)tk * N] *= inv;
}

// Write one head's attention output O [N, hd] column-major (o[tq + d*N]) into the token-major
// `out` [N, hidden] (out[tq*hidden + h*hd + d]).
extern "C" __global__ void vision_o_write(const float* o, float* out, int N, int hidden, int hd, int h) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * hd;
    if (i >= total) return;
    int tq = (int)(i % N);
    int d = (int)(i / N);
    out[(long)tq * hidden + (long)h * hd + d] = o[i];
}
