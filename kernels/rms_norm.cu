#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cstdint>

// =============================================================================
// RMS Normalization Kernel for sm_121
// out[i] = weight * x[i] / sqrt(mean(x^2) + eps)
// =============================================================================
extern "C" __global__ void rms_norm_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    int n,
    float eps
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Cooperative reduction
    float sum_sq = 0.0f;
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        float val = __half2float(input[i]);
        sum_sq += val * val;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float mean_sq = sdata[0] / (float)n;
    float rms = rsqrtf(mean_sq + eps);

    // Normalize
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        float val = __half2float(input[i]) * rms;
        float w = __half2float(weight[i % n]); // weight size matches n for simplicity
        output[i] = __float2half(val * w);
    }
}

// =============================================================================
// SiLU + Gate Multiplication Kernel
// out = x * silu(gate) = x * (gate / (1 + exp(-gate)))
// =============================================================================
extern "C" __global__ void silu_gate_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float g = __half2float(gate[gid]);
        float u = __half2float(up[gid]);
        float silu = g / (1.0f + __expf(-g));
        output[gid] = __float2half(silu * u);
    }
}

// =============================================================================
// Residual Addition Kernel
// out = a + b
// =============================================================================
extern "C" __global__ void add_residual_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ output,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float av = __half2float(a[gid]);
        float bv = __half2float(b[gid]);
        output[gid] = __float2half(av + bv);
    }
}

// =============================================================================
// RoPE (Rotary Position Embedding) Application Kernel
// =============================================================================
__device__ float compute_rope_cos(int pos, int dim, float base, int head_dim) {
    float theta = powf(base, -((float)(2 * (dim / 2)) / (float)head_dim));
    return cosf((float)pos * theta);
}

__device__ float compute_rope_sin(int pos, int dim, float base, int head_dim) {
    float theta = powf(base, -((float)(2 * (dim / 2)) / (float)head_dim));
    return sinf((float)pos * theta);
}

extern "C" __global__ void apply_rope_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int pos,
    int num_heads,
    int head_dim,
    float base
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * head_dim;

    if (gid < total) {
        int head = gid / head_dim;
        int dim = gid % head_dim;

        // Only apply to even dimensions
        if (dim % 2 == 0) {
            float val = __half2float(input[gid]);
            float next_val = __half2float(input[gid + 1]);

            float theta = powf(base, -((float)dim / (float)head_dim));
            float cos_val = cosf((float)pos * theta);
            float sin_val = sinf((float)pos * theta);

            output[gid] = __float2half(val * cos_val - next_val * sin_val);
            output[gid + 1] = __float2half(val * sin_val + next_val * cos_val);
        }
    }
}
