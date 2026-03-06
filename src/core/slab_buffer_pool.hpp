#pragma once

#include <Metal/Metal.hpp>
#include <shared_mutex>
#include <vector>
#include <atomic>
#include <list>
#include "../utils/logger.hpp"

namespace MetalFloat16Accelerator {

/**
 * @brief Final Hardened Slab Allocator (Hardware-Aligned).
 * 
 * Strategy:
 * 1. Page-Aligned: All buckets are multiples of 16KB (M2 Page Size).
 * 2. Adaptive Locking: std::shared_mutex for registry + Atomic fast-path.
 * 3. DMA-Optimized: Uses MTL::Device alignment requirements.
 */
class alignas(64) HardenedSlabPool {
private:
    static constexpr size_t PAGE_SIZE = 16384; // 16KB for Apple M2
    
    struct alignas(64) Bucket {
        mutable std::shared_mutex rw_mutex;
        std::vector<MTL::Buffer*> available;
        std::atomic<size_t> count{0}; // Lock-free availability check
    };

    MTL::Device* device_;
    std::vector<std::unique_ptr<Bucket>> registry_;
    static constexpr size_t MAX_BUCKETS = 128; 
    
    std::atomic<size_t> active_allocations_{0};

    /**
     * @brief Rounds up to nearest Page-Aligned multiple.
     */
    size_t alignToPage(size_t size) const noexcept {
        return (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    }

    size_t getBucketIndex(size_t size) const noexcept {
        return (size / PAGE_SIZE) % MAX_BUCKETS;
    }

public:
    HardenedSlabPool(MTL::Device* device) : device_(device) {
        if (device_) device_->retain();
        for (size_t i = 0; i < MAX_BUCKETS; ++i) {
            registry_.push_back(std::make_unique<Bucket>());
        }
    }

    ~HardenedSlabPool() {
        for (auto& bucket : registry_) {
            std::unique_lock lock(bucket->rw_mutex);
            for (auto* buf : bucket->available) {
                buf->release();
            }
        }
        if (device_) device_->release();
    }

    /**
     * @brief Hot Path: Acquire buffer with lock-free check.
     */
    [[nodiscard]] MTL::Buffer* acquire(size_t size) noexcept {
        const size_t aligned_size = alignToPage(size);
        const size_t idx = getBucketIndex(aligned_size);
        auto& bucket = registry_[idx];

        // 1. FAST PATH: Lock-free atomic check
        if (bucket->count.load(std::memory_order_acquire) > 0) {
            std::unique_lock lock(bucket->rw_mutex);
            if (!bucket->available.empty()) {
                MTL::Buffer* buf = bucket->available.back();
                bucket->available.pop_back();
                bucket->count.fetch_sub(1, std::memory_order_release);
                active_allocations_.fetch_add(1, std::memory_order_relaxed);
                return buf;
            }
        }

        // 2. SLOW PATH: Allocate new hardware-aligned buffer
        // Query Metal for hardware-specific alignment requirements
        MTL::SizeAndAlign res = device_->heapBufferSizeAndAlign(aligned_size, MTL::ResourceStorageModeShared);
        
        MTL::Buffer* new_buf = device_->newBuffer(res.size, MTL::ResourceStorageModeShared);
        if (new_buf) {
            active_allocations_.fetch_add(1, std::memory_order_relaxed);
            return new_buf;
        }
        return nullptr;
    }

    void release_buffer(MTL::Buffer* buffer) noexcept {
        if (!buffer) return;
        const size_t idx = getBucketIndex(buffer->length());
        auto& bucket = registry_[idx];

        std::unique_lock lock(bucket->rw_mutex);
        bucket->available.push_back(buffer);
        bucket->count.fetch_add(1, std::memory_order_release);
        active_allocations_.fetch_sub(1, std::memory_order_relaxed);
    }

    size_t get_active_count() const noexcept {
        return active_allocations_.load(std::memory_order_relaxed);
    }
};

} // namespace MetalFloat16Accelerator
