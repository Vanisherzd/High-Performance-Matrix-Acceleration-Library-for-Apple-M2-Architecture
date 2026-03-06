#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>
#include "float16_matrix.hpp"

namespace MetalFloat16Accelerator {

class CPUOptimizer {
public:
    // NEON-vectorized matrix multiply (8 elements per loop)
    static void neon_matmul_f16(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C) {
        const uint32_t M = A.rows();
        const uint32_t N = B.cols();
        const uint32_t K = A.cols();
        
        // Transpose B for better memory access patterns if it's large
        // For simplicity here, we assume B is already laid out row-major but we access it column-wise
        // A better optimization would be a blocked NEON approach
        
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                float16x8_t acc_vec = vdupq_n_f16(0.0h);
                float acc = 0.0f;
                
                uint32_t k = 0;
                for (; k + 7 < K; k += 8) {
                    // Load 8 half-precision floats from A and B
                    // Note: B is row-major, so this access is suboptimal.
                    // A truly optimized version would transpose B first.
                    float16x8_t a_vec = vld1q_f16(&A(i, k));
                    
                    // Manual fetch for B (non-contiguous in memory)
                    float16_t b_vals[8] = {
                        B(k, j), B(k+1, j), B(k+2, j), B(k+3, j),
                        B(k+4, j), B(k+5, j), B(k+6, j), B(k+7, j)
                    };
                    float16x8_t b_vec = vld1q_f16(b_vals);
                    
                    // Vector Multiply-Accumulate: acc = acc + (a * b)
                    acc_vec = vfmaq_f16(acc_vec, a_vec, b_vec);
                }
                
                // Reduce vector to scalar sum
                acc = (float)vaddvq_f16(acc_vec);
                
                // Process remaining elements
                for (; k < K; ++k) {
                    acc += (float)A(i, k) * (float)B(k, j);
                }
                
                C(i, j) = (half)acc;
            }
        }
    }
    
    // Multi-threaded CPU MatMul
    static void multithreaded_matmul(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C) {
        const uint32_t M = A.rows();
        const uint32_t num_threads = std::thread::hardware_concurrency();
        const uint32_t rows_per_thread = (M + num_threads - 1) / num_threads;
        
        std::vector<std::jthread> threads;
        for (uint32_t t = 0; t < num_threads; ++t) {
            uint32_t start_row = t * rows_per_thread;
            uint32_t end_row = std::min(start_row + rows_per_thread, M);
            
            if (start_row >= end_row) break;
            
            threads.emplace_back([&A, &B, &C, start_row, end_row]() {
                const uint32_t N = B.cols();
                const uint32_t K = A.cols();
                
                for (uint32_t i = start_row; i < end_row; ++i) {
                    for (uint32_t j = 0; j < N; ++j) {
                        float acc = 0.0f;
                        for (uint32_t k = 0; k < K; ++k) {
                            acc += (float)A(i, k) * (float)B(k, j);
                        }
                        C(i, j) = (half)acc;
                    }
                }
            });
        }
    }
};

} // namespace MetalFloat16Accelerator
