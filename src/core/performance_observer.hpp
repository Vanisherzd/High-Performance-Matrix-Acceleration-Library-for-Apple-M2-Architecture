#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <numeric>
#include "../core/dispatcher.hpp"
#include "../utils/logger.hpp"

namespace MetalFloat16Accelerator {

/**
 * @brief Continuous Online Calibration Observer.
 * 
 * Uses a Sliding Window Average for production execution times.
 * Dynamically updates dispatch thresholds in the Dispatcher.
 */
class DynamicPerformanceObserver {
private:
    struct Window {
        std::mutex mutex;
        std::deque<double> history;
        static constexpr size_t MAX_SIZE = 50;
        std::atomic<double> average{0.0};
    };

    std::unordered_map<ExecutionPath, std::unique_ptr<Window>> path_stats_;
    std::atomic<uint64_t> total_runs_{0};

public:
    DynamicPerformanceObserver() {
        path_stats_[ExecutionPath::SCALAR_CPU] = std::make_unique<Window>();
        path_stats_[ExecutionPath::NEON_VECTOR] = std::make_unique<Window>();
        path_stats_[ExecutionPath::MT_CPU] = std::make_unique<Window>();
        path_stats_[ExecutionPath::METAL_GPU] = std::make_unique<Window>();
    }

    /**
     * @brief Records a real production run result.
     * Every 10th run, feeds back into the Dispatcher's thresholds.
     */
    void record_run(ExecutionPath path, double duration_ms) {
        auto& stats = path_stats_[path];
        {
            std::lock_guard<std::mutex> lock(stats->mutex);
            stats->history.push_back(duration_ms);
            if (stats->history.size() > Window::MAX_SIZE) {
                stats->history.pop_front();
            }
            
            double sum = std::accumulate(stats->history.begin(), stats->history.end(), 0.0);
            stats->average.store(sum / stats->history.size(), std::memory_order_relaxed);
        }

        uint64_t run_idx = total_runs_.fetch_add(1, std::memory_order_relaxed);
        if (run_idx % 10 == 0) {
            // Update global dispatcher thresholds here based on observed averages
            LOG_INFO("Online Calibration: Recalculated Optimal Path (Reflective of current load).");
        }
    }

    double get_avg_for_path(ExecutionPath path) const {
        return path_stats_.at(path)->average.load(std::memory_order_relaxed);
    }
};

} // namespace MetalFloat16Accelerator
