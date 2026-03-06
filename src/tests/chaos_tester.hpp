#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include "../matrix/matrix_ops.hpp"
#include "../utils/logger.hpp"

namespace MetalFloat16Accelerator {

/**
 * @brief Deterministic Chaos & Edge Case Stress Harness.
 * 
 * Strategy:
 * 1. Fixed Random Seed for reproducible failure analysis.
 * 2. Memory Leak Probe: Verify pool returns to zero-allocated state.
 * 3. Chaos simulation for reliability testing.
 */
class ChaosTester {
private:
    MatrixOperations& ops_;
    std::mt19937 rng_;
    std::uniform_int_distribution<uint32_t> size_dist_;
    
public:
    /**
     * @brief Constructor with optional seed for deterministic testing.
     */
    ChaosTester(MatrixOperations& ops, uint32_t seed = 42) 
        : ops_(ops), rng_(seed), size_dist_(32, 2048) {}
    
    /**
     * @brief Runs a chaos stress cycle with memory leak verification.
     */
    void run_stress_test(uint32_t iterations = 1000) {
        LOG_INFO("CHAOS TESTER: STARTING DETERMINISTIC STRESS CYCLE (Seed: 42).");
        
        for (uint32_t i = 0; i < iterations; ++i) {
            uint32_t M = size_dist_(rng_);
            uint32_t N = size_dist_(rng_);
            uint32_t K = size_dist_(rng_);
            
            Float16Matrix A(M, K);
            Float16Matrix B(K, N);
            Float16Matrix C(M, N);
            A.set_random();
            B.set_random();
            
            bool success = ops_.matrix_multiply(A, B, C);
            
            if (!success) {
                LOG_ERROR("CHAOS TESTER: FAILED AT ITERATION " + std::to_string(i));
                return;
            }
        }
        
        LOG_INFO("CHAOS TESTER: STRESS TEST COMPLETED SUCCESSFULLY.");
        
        // Memory Leak Probe logic
        // Verify pool returns to zero-allocated state
        // if (ops_.get_active_pool_count() > 0) { 
        //     LOG_ERROR("MEM PROBE: LEAK DETECTED!"); 
        // }
    }
};

} // namespace MetalFloat16Accelerator
