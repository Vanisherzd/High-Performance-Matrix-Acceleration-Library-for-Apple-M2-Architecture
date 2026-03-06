#include <metal_stdlib>
using namespace metal;

// M2-optimized Tiled Matrix Multiplication with Mixed-Precision
// TILE_SIZE is chosen based on SIMD width (32) and threadgroup memory limits
#define TILE_SIZE 32

/**
 * @brief Reliability-First Tiled MatMul Kernel with Mixed-Precision.
 * 
 * Performs accumulation in float (32-bit) for numerical stability 
 * before casting back to half (16-bit) for memory storage.
 */
kernel void tiled_matmul_f16(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]], // [M, N, K]
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    threadgroup half shared_A[TILE_SIZE][TILE_SIZE] [[threadgroup(0)]],
    threadgroup half shared_B[TILE_SIZE][TILE_SIZE] [[threadgroup(1)]]
) {
    const uint M = dims.x;
    const uint N = dims.y;
    const uint K = dims.z;
    
    const uint row = gid.y;
    const uint col = gid.x;
    
    // MIXED-PRECISION ACCUMULATION: Using float (32-bit) for dot product
    float acc = 0.0f;
    
    // Iterate over tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into threadgroup memory (Coalesced memory access)
        if (row < M && (t * TILE_SIZE + tid.x) < K) {
            shared_A[tid.y][tid.x] = A[row * K + (t * TILE_SIZE + tid.x)];
        } else {
            shared_A[tid.y][tid.x] = 0.0h;
        }
        
        if (col < N && (t * TILE_SIZE + tid.y) < K) {
            shared_B[tid.y][tid.x] = B[(t * TILE_SIZE + tid.y) * N + col];
        } else {
            shared_B[tid.y][tid.x] = 0.0h;
        }
        
        // Barrier to ensure all threads in group have loaded the tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product from this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            // Mixed-precision: Casting to float before multiply
            acc += (float)shared_A[tid.y][k] * (float)shared_B[k][tid.x];
        }
        
        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result: Cast back to half for memory write
    if (row < M && col < N) {
        C[row * N + col] = (half)acc;
    }
}

// SIMD-group optimized kernel for M2 with Mixed-Precision
kernel void simd_matmul_f16(
    device const half *A [[buffer(0)]],
    device const half *B [[buffer(1)]],
    device half *C [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint M = dims.x;
    const uint N = dims.y;
    const uint K = dims.z;
    
    const uint row = gid.y;
    const uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    // Mixed-precision: Accumulate in float
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += (float)A[row * K + k] * (float)B[k * N + col];
    }
    
    // Cast back to half for final output
    C[row * N + col] = (half)acc;
}
