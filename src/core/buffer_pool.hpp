#pragma once

#include <Metal/Metal.hpp>
#include <map>
#include <vector>
#include <mutex>
#include <optional>
#include "../utils/logger.hpp"

namespace MetalFloat16Accelerator {

/**
 * @brief Production-ready BufferPool for M2 Unified Memory.
 * 
 * Manages memory segments to avoid expensive allocation/deallocation calls
 * during frequent matrix operations. Ensures 64-byte cache-line alignment.
 */
class BufferPool {
private:
    MTL::Device* device_;
    std::mutex mutex_;
    
    // Map of size to list of available buffers
    std::map<size_t, std::vector<MTL::Buffer*>> free_buffers_;
    std::vector<MTL::Buffer*> all_buffers_;
    
    static constexpr size_t ALIGNMENT = 64; // Cache-line alignment
    static constexpr size_t MAX_POOL_SIZE_MB = 2048; // 2GB Limit
    size_t current_pool_size_ = 0;

public:
    BufferPool(MTL::Device* device) : device_(device) {
        if (device_) device_->retain();
    }
    
    ~BufferPool() {
        for (auto* buf : all_buffers_) {
            buf->release();
        }
        if (device_) device_->release();
    }

    /**
     * @brief Acquires a buffer of specific size from the pool or allocates a new one.
     */
    MTL::Buffer* acquire(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Align size to cache line
        size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        
        auto it = free_buffers_.find(size);
        if (it != free_buffers_.end() && !it->second.empty()) {
            MTL::Buffer* buf = it->second.back();
            it->second.pop_back();
            return buf;
        }
        
        // Allocate new if pool not full
        if (current_pool_size_ + size > MAX_POOL_SIZE_MB * 1024 * 1024) {
            LOG_ERROR("BufferPool: Memory Limit Exceeded! Refusing allocation.");
            return nullptr;
        }
        
        MTL::Buffer* new_buf = device_->newBuffer(size, MTL::ResourceStorageModeShared);
        if (new_buf) {
            all_buffers_.push_back(new_buf);
            current_pool_size_ += size;
            LOG_HW("Allocated new MTL::Buffer (Shared): " + std::to_string(size / 1024) + " KB");
        }
        return new_buf;
    }

    /**
     * @brief Returns a buffer to the pool for reuse.
     */
    void release_buffer(MTL::Buffer* buffer) {
        if (!buffer) return;
        std::lock_guard<std::mutex> lock(mutex_);
        free_buffers_[buffer->length()].push_back(buffer);
    }
};

} // namespace MetalFloat16Accelerator
