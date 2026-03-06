#include "../include/metal_float16_accelerator.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "M2 Metal Float16 Matrix Accelerator - Simple Test\n";
    
    try {
        // Initialize accelerator
        MetalFloat16Accelerator::Accelerator accelerator;
        if (!accelerator.initialize()) {
            std::cerr << "Failed to initialize accelerator\n";
            return 1;
        }
        
        std::cout << "Accelerator successfully initialized!\n";
        std::cout << "Testing float16 matrix operations...\n\n";
        
        // Simple test with 64x64 matrix
        uint32_t size = 64;
        Float16Matrix A(size, size);
        Float16Matrix B(size, size);
        Float16Matrix C(size, size);
        
        // Initialize matrices
        A.set_random();
        B.set_random();
        C.fill(0.0f);
        
        std::cout << "Created " << size << "x" << size << " matrices\n";
        
        // Test matrix addition
        auto start = std::chrono::high_resolution_clock::now();
        bool success = accelerator.matrix_add(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            auto metrics = accelerator.get_last_performance_metrics();
            
            std::cout << "✅ Matrix Addition: SUCCESS\n";
            std::cout << "   Execution time: " << duration.count() << " μs\n";
            std::cout << "   GFLOPS: " << std::fixed << std::setprecision(2) << metrics.gflops << "\n";
            std::cout << "   Memory Bandwidth: " << std::fixed << std::setprecision(2) << metrics.memory_bandwidth_gbps << " GB/s\n";
        } else {
            std::cout << "❌ Matrix Addition: FAILED\n";
        }
        
        std::cout << "Test completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}