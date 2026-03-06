#include "../include/metal_float16_accelerator.hpp"
#include "matrix/float16_matrix.hpp"
#include "matrix/matrix_ops.hpp"
#include <iostream>
#include <stdexcept>
#include <new>

namespace MetalFloat16Accelerator {

Accelerator::Accelerator() 
    : matrix_ops_(nullptr) {
    matrix_ops_ = new(std::nothrow) MatrixOperations();
    if (!matrix_ops_) {
        throw std::bad_alloc();
    }
}

Accelerator::~Accelerator() {
    delete matrix_ops_;
    matrix_ops_ = nullptr;
}

bool Accelerator::initialize() {
    if (!matrix_ops_) {
        printf("Error: Matrix operations not initialized\n");
        return false;
    }
    
    if (!matrix_ops_->initialize()) {
        printf("Error: Failed to initialize matrix operations\n");
        return false;
    }
    
    printf("M2 Metal Float16 Matrix Accelerator initialized\n");
    print_device_info();
    
    return true;
}

bool Accelerator::is_m2_compatible() const {
    return matrix_ops_ ? matrix_ops_->is_m2_compatible() : false;
}

bool Accelerator::matrix_multiply(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C) {
    if (!matrix_ops_) {
        printf("Error: Matrix operations not initialized\n");
        return false;
    }
    
    // Validate dimensions
    if (A.cols() != B.rows()) {
        printf("Error: Matrix dimensions incompatible for multiplication: A(%dx%d) * B(%dx%d)\n",
               A.rows(), A.cols(), B.rows(), B.cols());
        return false;
    }
    
    // Ensure output matrix has correct dimensions
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        printf("Error: Output matrix dimensions incorrect: expected (%dx%d), got (%dx%d)\n",
               A.rows(), B.cols(), C.rows(), C.cols());
        return false;
    }
    
    return matrix_ops_->matrix_multiply(A, B, C);
}

bool Accelerator::matrix_add(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C) {
    if (!matrix_ops_) {
        printf("Error: Matrix operations not initialized\n");
        return false;
    }
    
    // Validate dimensions
    if (!A.is_compatible_with(B)) {
        printf("Error: Matrix dimensions incompatible for addition: A(%dx%d) + B(%dx%d)\n",
               A.rows(), A.cols(), B.rows(), B.cols());
        return false;
    }
    
    // Ensure output matrix has correct dimensions
    if (!C.is_compatible_with(A)) {
        printf("Error: Output matrix dimensions incorrect: expected (%dx%d), got (%dx%d)\n",
               A.rows(), A.cols(), C.rows(), C.cols());
        return false;
    }
    
    return matrix_ops_->matrix_add(A, B, C);
}

bool Accelerator::matrix_subtract(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C) {
    if (!matrix_ops_) {
        printf("Error: Matrix operations not initialized\n");
        return false;
    }
    
    // Validate dimensions
    if (!A.is_compatible_with(B)) {
        printf("Error: Matrix dimensions incompatible for subtraction: A(%dx%d) - B(%dx%d)\n",
               A.rows(), A.cols(), B.rows(), B.cols());
        return false;
    }
    
    // Ensure output matrix has correct dimensions
    if (!C.is_compatible_with(A)) {
        printf("Error: Output matrix dimensions incorrect: expected (%dx%d), got (%dx%d)\n",
               A.rows(), A.cols(), C.rows(), C.cols());
        return false;
    }
    
    return matrix_ops_->matrix_subtract(A, B, C);
}

bool Accelerator::matrix_transpose(const Float16Matrix& A, Float16Matrix& C) {
    if (!matrix_ops_) {
        printf("Error: Matrix operations not initialized\n");
        return false;
    }
    
    // Validate dimensions
    if (C.rows() != A.cols() || C.cols() != A.rows()) {
        printf("Error: Output matrix dimensions incorrect for transpose: A(%dx%d) -> C(%dx%d)\n",
               A.rows(), A.cols(), C.rows(), C.cols());
        return false;
    }
    
    return matrix_ops_->matrix_transpose(A, C);
}

bool Accelerator::matrix_scale(const Float16Matrix& A, float scalar, Float16Matrix& C) {
    if (!matrix_ops_) {
        printf("Error: Matrix operations not initialized\n");
        return false;
    }
    
    // Validate dimensions
    if (!C.is_compatible_with(A)) {
        printf("Error: Output matrix dimensions incorrect for scaling: A(%dx%d) -> C(%dx%d)\n",
               A.rows(), A.cols(), C.rows(), C.cols());
        return false;
    }
    
    return matrix_ops_->matrix_scale(A, scalar, C);
}

MetalFloat16Accelerator::PerformanceMetrics Accelerator::get_last_performance_metrics() const {
    PerformanceMetrics metrics;
    
    if (matrix_ops_) {
        metrics.execution_time_ms = matrix_ops_->get_last_execution_time_ms();
        metrics.operations_count = matrix_ops_->get_last_operations_count();
        metrics.memory_bandwidth_gbps = matrix_ops_->get_last_memory_bandwidth_gbps();
        metrics.gflops = matrix_ops_->get_last_gflops();
    }
    
    return metrics;
}

void Accelerator::reset_performance_counters() {
    if (matrix_ops_) {
        matrix_ops_->reset_performance_counters();
    }
}

void Accelerator::print_device_info() const {
    if (matrix_ops_) {
        matrix_ops_->print_device_info();
    } else {
        printf("Error: Matrix operations not initialized\n");
    }
}

const char* Accelerator::get_device_name() const {
    return matrix_ops_ ? matrix_ops_->get_device_name() : "Unknown";
}

// Utility functions
bool validate_m2_compatibility() {
#ifdef __arm64__
    // Check if running on Apple Silicon
    // In a full implementation, you would check specific M2 identifiers
    return true;
#else
    printf("Warning: Not running on Apple Silicon\n");
    return false;
#endif
}

const char* get_metal_error_string(uint32_t error_code) {
    // Simple error mapping (would expand in full implementation)
    switch (error_code) {
        case 0: return "Success";
        case 1: return "Device not found";
        case 2: return "Invalid operation";
        default: return "Unknown error";
    }
}

} // namespace MetalFloat16Accelerator
