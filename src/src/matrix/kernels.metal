#include <metal_stdlib>
using namespace metal;

// Tiled Matrix Multiplication Kernel for M2
// Optimized for Half-Precision (float16)
kernel void tiled_matmul_f16(
    const device half* A [[ buffer(0) ]],
    const device half* B [[ buffer(1) ]],
    device half* C       [[ buffer(2) ]],
    const device uint3* dims [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint M = dims[0].x;
    uint N = dims[0].y;
    uint K = dims[0].z;

    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += (float)A[gid.y * K + k] * (float)B[k * N + gid.x];
    }

    C[gid.y * N + gid.x] = (half)sum;
}
