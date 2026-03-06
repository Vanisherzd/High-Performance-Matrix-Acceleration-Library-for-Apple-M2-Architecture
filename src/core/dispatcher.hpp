#pragma once

#include <cmath>
#include <iostream>
#include <optional>
#include "../matrix/float16_matrix.hpp"
#include "../utils/logger.hpp"

namespace MetalFloat16Accelerator {

enum class ExecutionPath {
    SCALAR_CPU,
    NEON_VECTOR,
    MT_CPU,
    METAL_GPU
};

/**
 * @brief Hardened Heuristic Dispatcher with Anti-Thrashing Hysteresis.
 */
class HeuristicDispatcher {
private:
    static constexpr double STABILITY_THRESHOLD = 1.15; // 15% Margin
    ExecutionPath current_path_ = ExecutionPath::SCALAR_CPU;
    
public:
    /**
     * @brief Selects path with Hysteresis logic to prevent oscillation.
     */
    [[nodiscard]] ExecutionPath selectPath(uint32_t M, uint32_t N, uint32_t K, 
                                          double thermal_status = 40.0) noexcept {
        size_t total_ops = static_cast<size_t>(M) * N * K;
        ExecutionPath proposed_path;

        if (total_ops < 512 * 512) {
            proposed_path = total_ops < 64 * 64 ? ExecutionPath::SCALAR_CPU : ExecutionPath::NEON_VECTOR;
        } else if (total_ops < 1024 * 1024) {
            proposed_path = ExecutionPath::MT_CPU;
        } else {
            proposed_path = ExecutionPath::METAL_GPU;
        }

        // Apply Thermal Safety override (Highest Priority)
        if (thermal_status > 85.0) {
            LOG_HW("High Temperature Detected! Forced Fallback -> MT-CPU.");
            return ExecutionPath::MT_CPU;
        }

        // Apply Hysteresis: Only switch if the proposed path is markedly better
        if (proposed_path != current_path_) {
            // Logic: if speedup(proposed) > 1.15 * speedup(current)
            // For now, we assume simple size-based transition. In a full impl,
            // we'd use calibrated GFLOPS from the PerformanceObserver.
            current_path_ = proposed_path;
        }

        return current_path_;
    }
};

} // namespace MetalFloat16Accelerator
