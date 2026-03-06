#include "../include/metal_float16_accelerator.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "M2 Metal Float16 Matrix Accelerator - Benchmark Suite\n";
    
    try {
        // Initialize accelerator
        MetalFloat16Accelerator::Accelerator accelerator;
        if (!accelerator.initialize()) {
            std::cerr << "Failed to initialize accelerator\n";
            return 1;
        }
        
        // Create benchmark instance
        std::cout << "Starting comprehensive benchmark suite...\n";
        accelerator.print_device_info();
        
        // Test different matrix sizes
        std::vector<uint32_t> sizes = {128, 256, 512, 1024, 2048};
        
        for (uint32_t size : sizes) {
            std::cout << "\n=== Matrix Size: " << size << "x" << size << " ===\n";
            
            // Matrix Addition Benchmark
            {
                auto start = std::chrono::high_resolution_clock::now();
                Float16Matrix A(size, size);
                Float16Matrix B(size, size);
                Float16Matrix C(size, size);
                
                A.set_random();
                B.set_random();
                C.fill(0.0f);
                
                bool success = accelerator.matrix_add(A, B, C);
                auto end = std::chrono::high_resolution_clock::now();
                
                if (success) {
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    std::cout << "Add: " << std::fixed << std::setprecision(3) << duration.count() << "μs (";
                    
                    uint64_t operations = size * size;
                    double time_seconds = duration.count() / 1000000.0;
                    double gflops = (operations * 2) / (time_seconds * 1e9);  // 2 FLOPS per element for addition
                    
                    std::cout << gflops << " GFLOPS)\n";
                } else {
                    std::cout << "Add: FAILED\n";
                }
            }
            
            // Matrix Multiplication Benchmark
            {
                auto start = std::chrono::high_resolution_clock::now();
                Float16Matrix A(size, size);
                Float16Matrix B(size, size);
                Float16Matrix C(size, size);
                
                A.set_random();
                B.set_random();
                C.fill(0.0f);
                
                bool success = accelerator.matrix_multiply(A, B, C);
                auto end = std::chrono::high_resolution_clock::now();
                
                if (success) {
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    std::cout << "Mul: " << std::fixed << std::setprecision(3) << duration.count() << "μs (";
                    
                    uint64_t operations = size * size * size;  // N*N*N for multiplication
                    double time_seconds = duration.count() / 1000000.0;
                    double gflops = (operations * 2) / (time_seconds * 1e9);  // 2 FLOPS per element
                    
                    std::cout << gflops << " GFLOPS)\n";
                } else {
                    std::cout << "Mul: FAILED\n";
                }
            }
        }
        
        std::cout << "\nBenchmark suite completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}