#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include "metal_engine.hpp"
#include <iostream>

namespace MetalFloat16Accelerator {

MetalEngine::MetalEngine() 
    : device_(nullptr)
    , command_queue_(nullptr)
    , matmul_pso_(nullptr)
    , library_(nullptr) {
}

MetalEngine::~MetalEngine() {
    if (matmul_pso_) matmul_pso_->release();
    if (library_) library_->release();
    if (command_queue_) command_queue_->release();
    if (device_) device_->release();
}

bool MetalEngine::initialize() {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) {
        return false;
    }
    
    command_queue_ = device_->newCommandQueue();
    if (!command_queue_) {
        return false;
    }
    
    return load_kernels();
}

bool MetalEngine::load_kernels() {
    NS::Error* error = nullptr;
    
    // In a real project, you'd load the .metallib file
    // For this environment, we'll assume the library is compiled and available
    library_ = device_->newDefaultLibrary();
    if (!library_) {
        // Fallback: try to compile from source if default library not found
        // (Simplified for this task)
        printf("Error: Metal default library not found.
");
        return false;
    }
    
    auto kernel_name = NS::String::string("tiled_matmul_f16", NS::UTF8StringEncoding);
    auto matmul_fn = library_->newFunction(kernel_name);
    
    if (!matmul_fn) {
        printf("Error: Could not find kernel function.
");
        return false;
    }
    
    matmul_pso_ = device_->newComputePipelineState(matmul_fn, &error);
    matmul_fn->release();
    
    if (!matmul_pso_) {
        printf("Error: Failed to create pipeline state: %s
", 
               error->localizedDescription()->utf8String());
        return false;
    }
    
    return true;
}

bool MetalEngine::matmul_gpu(const Float16Matrix& A, const Float16Matrix& B, Float16Matrix& C) {
    if (!matmul_pso_) return false;
    
    const uint32_t M = A.rows();
    const uint32_t N = B.cols();
    const uint32_t K = A.cols();
    
    // Zero-Copy Strategy: Wrap existing CPU memory in MTL::Buffer
    // ResourceStorageModeShared ensures both CPU and GPU see the same memory
    MTL::Buffer* bufA = device_->newBuffer(A.data(), A.get_memory_size_bytes(), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufB = device_->newBuffer(B.data(), B.get_memory_size_bytes(), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufC = device_->newBuffer(C.data(), C.get_memory_size_bytes(), MTL::ResourceStorageModeShared);
    
    uint32_t dims[3] = {M, N, K};
    MTL::Buffer* bufDims = device_->newBuffer(dims, sizeof(dims), MTL::ResourceStorageModeShared);
    
    MTL::CommandBuffer* commandBuffer = command_queue_->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(matmul_pso_);
    encoder->setBuffer(bufA, 0, 0);
    encoder->setBuffer(bufB, 0, 1);
    encoder->setBuffer(bufC, 0, 2);
    encoder->setBuffer(bufDims, 0, 3);
    
    // Configure Grid and Threadgroup
    // M2 SIMD width is 32, so 32x32 is a common tile size for MatMul
    MTL::Size threadgroupSize = MTL::Size::size(32, 8, 1); // 256 threads
    MTL::Size gridSize = MTL::Size::size((N + 31) / 32 * 32, (M + 7) / 8 * 8, 1);
    
    encoder->dispatchThreads(gridSize, threadgroupSize);
    encoder->endEncoding();
    
    commandBuffer->commit();
    
    // CRITICAL: Precise hardware timing requires waiting for completion
    commandBuffer->waitUntilCompleted();
    
    // Cleanup temporary buffers (since we are zero-copying from raw pointers each time)
    // Optimization: In production, you would reuse these buffers.
    bufA->release();
    bufB->release();
    bufC->release();
    bufDims->release();
    
    return true;
}

void MetalEngine::print_device_info() const {
    if (device_) {
        printf("  GPU Device: %s
", device_->name()->utf8String());
        printf("  Unified Memory: %s
", device_->hasUnifiedMemory() ? "Yes" : "No");
        printf("  Max Threads per Threadgroup: %llu
", device_->maxThreadsPerThreadgroup().width);
    }
}

} // namespace MetalFloat16Accelerator
