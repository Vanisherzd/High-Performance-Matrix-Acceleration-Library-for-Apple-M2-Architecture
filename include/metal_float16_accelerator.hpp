#pragma once

#include <memory>
#include <cstdint>
#include "../src/matrix/float16_matrix.hpp"
#include "../src/matrix/matrix_ops.hpp"
#include "../src/core/metal_device.hpp"

namespace MetalFloat16Accelerator {

// M2-specific configuration
struct M2Config {
    static constexpr uint32_t SIMD_WIDTH = 32;
    static constexpr uint32_t TILE_SIZE = 32;
    static constexpr uint32_t THREADGROUP_SIZE = 256;
    static constexpr uint32_t MAX_THREADS_PER_BLOCK = 1024;
    static constexpr uint32_t SHARED_MEMORY_SIZE = 32 * 1024;
};

// Performance metrics
struct PerformanceMetrics {
    double execution_time_ms;
    double memory_bandwidth_gbps;
    uint64_t operations_count;
    double gflops;
    
    PerformanceMetrics() 
        : execution_time_ms(0.0)
        , memory_bandwidth_gbps(0.0)
        , operations_count(0)
        , gflops(0.0) {}
};

// Main accelerator class
class Accelerator {
private:
    MatrixOperations* matrix_ops_;
    
public:
    Accelerator();
    ~Accelerator();
    
    // Initialization
    bool initialize();
    bool is_m2_compatible() const;
    
    // Matrix operations
    bool matrix_multiply(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    bool matrix_add(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    bool matrix_subtract(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    bool matrix_transpose(const Float16Matrix& A, Float16Matrix& C);
    bool matrix_scale(const Float16Matrix& A, float scalar, Float16Matrix& C);
    
    // Performance monitoring
    PerformanceMetrics get_last_performance_metrics() const;
    void reset_performance_counters();
    
    // Device information
    void print_device_info() const;
    const char* get_device_name() const;
};

// Utility functions
bool validate_m2_compatibility();
const char* get_metal_error_string(uint32_t error_code);

} // namespace MetalFloat16Accelerator