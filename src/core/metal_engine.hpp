#pragma once

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <memory>
#include <string>
#include <vector>
#include "../matrix/float16_matrix.hpp"

namespace MetalFloat16Accelerator {

class MetalEngine {
private:
    MTL::Device* device_;
    MTL::CommandQueue* command_queue_;
    MTL::ComputePipelineState* matmul_pso_;
    MTL::Library* library_;
    
    bool load_kernels();
    
public:
    MetalEngine();
    ~MetalEngine();
    
    bool initialize();
    
    // Matrix Multiplication with Zero-Copy
    bool matmul_gpu(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C);
    
    // Performance metrics
    void print_device_info() const;
    
    MTL::Device* get_device() const { return device_; }
};

} // namespace MetalFloat16Accelerator
