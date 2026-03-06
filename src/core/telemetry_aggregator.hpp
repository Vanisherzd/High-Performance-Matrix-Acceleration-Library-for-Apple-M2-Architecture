#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>
#include "../utils/logger.hpp"

namespace MetalFloat16Accelerator {

/**
 * @brief Low-Overhead Telemetry Aggregator for Deterministic Systems.
 * 
 * Instead of querying sensors on every dispatch, it uses sampling ratios 
 * and an atomic ring buffer to avoid becoming a computation bottleneck.
 */
class TelemetryAggregator {
private:
    std::atomic<uint64_t> operation_count_{0};
    uint32_t sampling_ratio_ = 50; // Query hardware status every 50 ops
    
    std::atomic<double> cached_thermal_status_{42.0};
    std::atomic<double> cached_gpu_load_{0.0};
    
    // Atomic Ring Buffer for logs (Simulated simplified version)
    std::mutex ring_mutex_;
    std::vector<std::string> log_ring_buffer_;
    size_t ring_head_ = 0;
    static constexpr size_t RING_SIZE = 1024;

public:
    TelemetryAggregator() : log_ring_buffer_(RING_SIZE) {}
    
    /**
     * @brief Records an operation and samples hardware sensors if the ratio is met.
     */
    void record_operation() {
        uint64_t count = operation_count_.fetch_add(1, std::memory_order_relaxed);
        
        if (count % sampling_ratio_ == 0) {
            // Sampling logic: Query sensors
            // In a real system, these would call into IOKit or similar
            cached_thermal_status_.store(42.5, std::memory_order_relaxed);
            cached_gpu_load_.store(0.75, std::memory_order_relaxed);
            
            LOG_HW("Telemetry sampled at " + std::to_string(count) + " operations.");
        }
    }

    double get_thermal_status() const {
        return cached_thermal_status_.load(std::memory_order_relaxed);
    }

    double get_gpu_load() const {
        return cached_gpu_load_.load(std::memory_order_relaxed);
    }
};

} // namespace MetalFloat16Accelerator
