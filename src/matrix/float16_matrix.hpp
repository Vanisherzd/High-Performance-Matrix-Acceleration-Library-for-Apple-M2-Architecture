#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>

// Use half type for float16 on Apple Silicon
#ifdef __ARM_NEON
#include <arm_neon.h>
typedef __fp16 half;
#else
typedef uint16_t half;
#endif

class Float16Matrix {
private:
    half* data_;
    uint32_t rows_;
    uint32_t cols_;
    uint32_t stride_;  // Padded for M2 cache efficiency
    bool device_allocated_;
    
    // M2 memory alignment
    static constexpr uint32_t CACHE_LINE_SIZE = 64;
    static constexpr uint32_t ALIGNMENT = 64;
    
    // Helper for aligned allocation
    static half* aligned_malloc(size_t size) {
        if (size == 0) return nullptr;
        
        void* ptr = nullptr;
        size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        
        if (posix_memalign(&ptr, ALIGNMENT, aligned_size) != 0) {
            return nullptr;
        }
        
        return static_cast<half*>(ptr);
    }
    
    static void aligned_free(half* ptr) {
        if (ptr) {
            free(ptr);
        }
    }
    
    // Helper to convert float to half
    static half float_to_half(float f) {
        return static_cast<half>(f);
    }
    
public:
    Float16Matrix() 
        : data_(nullptr), rows_(0), cols_(0), stride_(0), device_allocated_(false) {}
    
    Float16Matrix(uint32_t rows, uint32_t cols) 
        : rows_(rows), cols_(cols), stride_(cols), device_allocated_(false) {
        // M2-optimized padding for cache efficiency
        stride_ = ((cols + CACHE_LINE_SIZE/sizeof(half) - 1) & ~(CACHE_LINE_SIZE/sizeof(half) - 1));
        
        size_t total_size = stride_ * rows_ * sizeof(half);
        data_ = aligned_malloc(total_size);
        
        if (!data_) {
            throw std::bad_alloc();
        }
    }
    
    Float16Matrix(const Float16Matrix& other) 
        : rows_(other.rows_), cols_(other.cols_), stride_(other.stride_), device_allocated_(false) {
        size_t total_size = stride_ * rows_ * sizeof(half);
        data_ = aligned_malloc(total_size);
        
        if (!data_) {
            throw std::bad_alloc();
        }
        
        // Copy data with stride consideration
        for (uint32_t row = 0; row < rows_; ++row) {
            for (uint32_t col = 0; col < cols_; ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
    
    Float16Matrix(Float16Matrix&& other) noexcept 
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_), 
          stride_(other.stride_), device_allocated_(other.device_allocated_) {
        other.data_ = nullptr;
        other.rows_ = other.cols_ = other.stride_ = 0;
        other.device_allocated_ = false;
    }
    
    ~Float16Matrix() {
        if (data_ && !device_allocated_) {
            aligned_free(data_);
        }
    }
    
    // Assignment operators
    Float16Matrix& operator=(const Float16Matrix& other) {
        if (this != &other) {
            // Free existing data
            if (data_ && !device_allocated_) {
                aligned_free(data_);
            }
            
            // Copy dimensions
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;
            device_allocated_ = false;
            
            // Allocate new memory
            size_t total_size = stride_ * rows_ * sizeof(half);
            data_ = aligned_malloc(total_size);
            
            if (!data_) {
                throw std::bad_alloc();
            }
            
            // Copy data
            for (uint32_t row = 0; row < rows_; ++row) {
                for (uint32_t col = 0; col < cols_; ++col) {
                    (*this)(row, col) = other(row, col);
                }
            }
        }
        return *this;
    }
    
    Float16Matrix& operator=(Float16Matrix&& other) noexcept {
        if (this != &other) {
            // Free existing data
            if (data_ && !device_allocated_) {
                aligned_free(data_);
            }
            
            // Move data
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            stride_ = other.stride_;
            device_allocated_ = other.device_allocated_;
            
            // Clear other
            other.data_ = nullptr;
            other.rows_ = other.cols_ = other.stride_ = 0;
            other.device_allocated_ = false;
        }
        return *this;
    }
    
    // Access operators
    half& operator()(uint32_t row, uint32_t col) {
        return data_[row * stride_ + col];
    }
    
    const half& operator()(uint32_t row, uint32_t col) const {
        return data_[row * stride_ + col];
    }
    
    // Properties
    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t stride() const { return stride_; }
    half* data() { return data_; }
    const half* data() const { return data_; }
    
    // M2-optimized memory layout
    uint32_t get_padded_stride() const {
        return ((cols_ + 31) / 32) * 32;  // Align to 32-element boundaries
    }
    
    size_t get_memory_size_bytes() const {
        return stride_ * rows_ * sizeof(half);
    }
    
    // Operations
    void fill(float value) {
        half h_val = float_to_half(value);
        for (uint32_t row = 0; row < rows_; ++row) {
            for (uint32_t col = 0; col < cols_; ++col) {
                (*this)(row, col) = h_val;
            }
        }
    }
    
    void set_identity() {
        fill(0.0f);
        uint32_t min_dim = rows_ < cols_ ? rows_ : cols_;
        for (uint32_t i = 0; i < min_dim; ++i) {
            (*this)(i, i) = float_to_half(1.0f);
        }
    }
    
    void set_random() {
        // Simple random float16 generation
        for (uint32_t row = 0; row < rows_; ++row) {
            for (uint32_t col = 0; col < cols_; ++col) {
                float val = static_cast<float>(rand()) / RAND_MAX;
                (*this)(row, col) = float_to_half(val);
            }
        }
    }
    
    void copy_from(const Float16Matrix& other) {
        if (!is_compatible_with(other)) {
            throw std::invalid_argument("Matrix dimensions are not compatible");
        }
        
        for (uint32_t row = 0; row < rows_; ++row) {
            for (uint32_t col = 0; col < cols_; ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
    
    // Validation
    bool is_compatible_with(const Float16Matrix& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_;
    }
    
    bool is_square() const { 
        return rows_ == cols_; 
    }
    
    // Debug
    void print() const {
        printf("Float16Matrix(%d x %d, stride=%d):\n", rows_, cols_, stride_);
        for (uint32_t row = 0; row < (rows_ < 8 ? rows_ : 8); ++row) {
            printf("  [");
            for (uint32_t col = 0; col < (cols_ < 8 ? cols_ : 8); ++col) {
                printf("%.3f", static_cast<float>((*this)(row, col)));
                if (col < cols_ - 1) printf(", ");
            }
            if (cols_ > 8) printf("...");
            printf("]\n");
        }
        if (rows_ > 8) printf("  ...\n");
    }
    
    bool validate() const {
        if (!data_) return false;
        if (rows_ == 0 || cols_ == 0) return false;
        if (stride_ < cols_) return false;
        return true;
    }
};