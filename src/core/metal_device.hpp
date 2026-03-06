#pragma once

#include <string>
#include <iostream>

namespace MetalFloat16Accelerator {

// Simple Metal device wrapper for now
class MetalDevice {
private:
    bool is_initialized_;
    bool is_m2_compatible_;
    std::string device_name_;
    uint32_t max_threadgroup_memory_;
    uint32_t max_threads_per_threadgroup_;
    uint32_t simd_width_;
    
public:
    MetalDevice();
    ~MetalDevice();
    
    // Lifecycle
    bool initialize();
    void shutdown();
    bool is_initialized() const { return is_initialized_; }
    
    // Device information
    const std::string& get_device_name() const { return device_name_; }
    bool is_m2_device() const { return is_m2_compatible_; }
    uint32_t get_max_threadgroup_memory() const { return max_threadgroup_memory_; }
    uint32_t get_max_threads_per_threadgroup() const { return max_threads_per_threadgroup_; }
    uint32_t get_simd_width() const { return simd_width_; }
    
    // Utility methods
    void print_device_info() const;
    bool validate_m2_compatibility();
};

} // namespace MetalFloat16Accelerator