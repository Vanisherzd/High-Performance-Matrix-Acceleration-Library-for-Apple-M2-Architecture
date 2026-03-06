#pragma once

#include <atomic>
#include <chrono>
#include <thread>
#include <stop_token>
#include <iostream>
#include <vector>
#include <deque>
#include <numeric>
#include "../utils/logger.hpp"

namespace MetalFloat16Accelerator {

/**
 * @brief Hardened Background Watchdog with Telemetry Debouncing.
 * 
 * Strategy:
 * 1. Ignore transient spikes (< 50ms) using a Moving Average Filter.
 * 2. Asynchronous out-of-band monitoring using std::jthread.
 */
class BackgroundWatchdog {
private:
    std::jthread monitor_thread_;
    std::atomic<double> current_thermal_status_{42.0};
    std::atomic<double> gpu_occupancy_{0.0};
    
    // Telemetry Debouncing: Sliding window of 5 samples
    std::deque<double> thermal_samples_;
    static constexpr size_t DEBOUNCE_WINDOW = 5;
    static constexpr uint32_t SAMPLING_INTERVAL_MS = 20; // 5 x 20ms = 100ms window

public:
    BackgroundWatchdog() {
        monitor_thread_ = std::jthread([this](std::stop_token stop_token) {
            LOG_INFO("Hardened Watchdog: Monitoring Thread STARTED.");
            
            while (!stop_token.stop_requested()) {
                double raw_thermal = 42.5; // Simulated query
                double raw_occupancy = 0.75; // Simulated query
                
                // Debouncing logic: Moving average to filter noise
                thermal_samples_.push_back(raw_thermal);
                if (thermal_samples_.size() > DEBOUNCE_WINDOW) {
                    thermal_samples_.pop_front();
                }
                
                double avg_thermal = std::accumulate(thermal_samples_.begin(), 
                                                     thermal_samples_.end(), 0.0) / 
                                     thermal_samples_.size();
                
                current_thermal_status_.store(avg_thermal, std::memory_order_relaxed);
                gpu_occupancy_.store(raw_occupancy, std::memory_order_relaxed);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(SAMPLING_INTERVAL_MS));
            }
            
            LOG_INFO("Hardened Watchdog: Monitoring Thread SHUT DOWN.");
        });
    }

    [[nodiscard]] double get_thermal_status() const noexcept {
        return current_thermal_status_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] double get_gpu_occupancy() const noexcept {
        return gpu_occupancy_.load(std::memory_order_relaxed);
    }
};

} // namespace MetalFloat16Accelerator
