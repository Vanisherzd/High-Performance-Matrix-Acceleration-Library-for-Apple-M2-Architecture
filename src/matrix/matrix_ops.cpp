#include "matrix_ops.hpp"
#include "../core/metal_engine.hpp"
#include "../core/dispatcher.hpp"
#include "../utils/logger.hpp"
#include <iostream>
#include <thread>
#include <algorithm>
#include <cmath>

MatrixOperations::MatrixOperations() 
    : initialized_(false)
    , last_execution_time_ms_(0.0)
    , last_operations_count_(0)
    , last_memory_bandwidth_gbps_(0.0)
    , last_gflops_(0.0)
    , gpu_engine_(nullptr) {
}

MatrixOperations::~MatrixOperations() {
    // Cleanup handled by unique_ptr
}

bool MatrixOperations::initialize() {
    LOG_INFO("Initializing M2 Metal Accelerator Library...");
    
    // Initialize GPU engine
    gpu_engine_ = std::make_unique<MetalFloat16Accelerator::MetalEngine>();
    if (gpu_engine_->initialize()) {
        initialized_ = true;
        LOG_INFO("GPU Engine: SUCCESS (M2 Metal)");
    } else {
        LOG_HW("GPU Engine: FAILED! Falling back to MT-CPU fallback mode.");
        initialized_ = true; // Still initialized for CPU fallback
    }
    return true;
}

bool MatrixOperations::is_m2_compatible() const {
#ifdef __arm64__
    return true;
#else
    return false;
#endif
}

void MatrixOperations::calculate_performance_metrics(uint64_t operations, size_t memory_bytes, double duration_ms) const {
    last_operations_count_ = operations;
    last_execution_time_ms_ = duration_ms;
    
    if (duration_ms > 0) {
        last_gflops_ = (operations / 1e9) / (duration_ms / 1000.0);
        last_memory_bandwidth_gbps_ = (memory_bytes / (1024.0 * 1024.0 * 1024.0)) / (duration_ms / 1000.0);
    } else {
        last_gflops_ = 0.0;
        last_memory_bandwidth_gbps_ = 0.0;
    }
}

// Main logic using the Heuristic Dispatcher
bool MatrixOperations::matrix_multiply(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C) {
    if (!initialized_) return false;
    
    // Select the best path for this workload
    auto path = MetalFloat16Accelerator::HeuristicDispatcher::selectPath(
        A.rows(), B.cols(), A.cols(), 
        MetalFloat16Accelerator::SystemLogger::getInstance().getThermalStatus()
    );
    
    auto start = std::chrono::high_resolution_clock::now();
    bool success = false;
    
    try {
        if (path == MetalFloat16Accelerator::ExecutionPath::METAL_GPU && gpu_engine_) {
            // Execute on GPU
            success = gpu_engine_->matmul_gpu(A, B, C);
            if (!success) {
                LOG_ERROR("GPU Operation Failed! Triggering automatic MT-CPU fallback.");
                cpu_matrix_multiply(A, B, C); // Graceful fallback
                success = true;
            }
        } else {
            // CPU Path (NEON or MT-CPU)
            cpu_matrix_multiply(A, B, C);
            success = true;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Execution Critical Error: " + std::string(e.what()));
        return false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double duration_ms = duration.count() / 1000.0;
    
    if (success) {
        uint64_t operations = static_cast<uint64_t>(2) * A.rows() * B.cols() * A.cols();
        size_t memory_bytes = (A.rows() * A.cols() + B.rows() * B.cols() + C.rows() * C.cols()) * sizeof(half);
        calculate_performance_metrics(operations, memory_bytes, duration_ms);
        return true;
    }
    return false;
}

// ... Additional matrix ops implementations here ...
