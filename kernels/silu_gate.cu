#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cstdint>

// Re-declare silu_gate_mul if not in separate file
__global__ void silu_gate_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int n
);

// =============================================================================
// SiLU Gate Multiplication Kernel
// =============================================================================
__global__ void silu_gate_mul_kernel(
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

// Re-declare add_residual
__global__ void add_residual_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ output,
    int n
);

// =============================================================================
// Residual Addition Kernel
// =============================================================================
__global__ void add_residual_kernel(
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
