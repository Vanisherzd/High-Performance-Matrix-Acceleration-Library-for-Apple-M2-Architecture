#pragma once

#include "float16_matrix.hpp"
#include <chrono>
#include <memory>

namespace MetalFloat16Accelerator {
    class MetalEngine; // Forward declaration
}

// Simple matrix operations for M2 optimization (CPU & GPU)
class MatrixOperations {
private:
    bool initialized_;
    mutable double last_execution_time_ms_;
    mutable uint64_t last_operations_count_;
    mutable double last_memory_bandwidth_gbps_;
    mutable double last_gflops_;
    
    // GPU Engine
    std::unique_ptr<MetalFloat16Accelerator::MetalEngine> gpu_engine_;
    
    // CPU matrix multiplication (optimized for M2)
    void cpu_matrix_multiply(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    void cpu_matrix_add(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    void cpu_matrix_subtract(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    void cpu_matrix_transpose(const Float16Matrix& A, Float16Matrix& C);
    void cpu_matrix_scale(const Float16Matrix& A, float scalar, Float16Matrix& C);
    
    // Performance measurement
    double get_time_ms() const;
    void calculate_performance_metrics(uint64_t operations, size_t memory_bytes, double duration_ms) const;
    
public:
    MatrixOperations();
    ~MatrixOperations();
    
    // Initialization
    bool initialize();
    bool is_initialized() const { return initialized_; }
    bool is_m2_compatible() const;
    
    // Matrix operations
    bool matrix_multiply(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    bool matrix_add(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    bool matrix_subtract(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    bool matrix_transpose(const Float16Matrix& A, Float16Matrix& C);
    bool matrix_scale(const Float16Matrix& A, float scalar, Float16Matrix& C);
    
    // Performance monitoring
    double get_last_execution_time_ms() const { return last_execution_time_ms_; }
    uint64_t get_last_operations_count() const { return last_operations_count_; }
    double get_last_memory_bandwidth_gbps() const { return last_memory_bandwidth_gbps_; }
    double get_last_gflops() const { return last_gflops_; }
    
    void reset_performance_counters();
    void print_performance_stats() const;
    
    // Device information
    void print_device_info() const;
    const char* get_device_name() const;
};