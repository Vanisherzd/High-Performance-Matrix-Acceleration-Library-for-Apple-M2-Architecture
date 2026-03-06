#include "../include/metal_float16_accelerator.hpp"
#include "../src/matrix/float16_matrix.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "M2 Metal Float16 Matrix Accelerator - Basic Example\n";
    
    try {
        // Initialize accelerator
        MetalFloat16Accelerator::Accelerator accelerator;
        if (!accelerator.initialize()) {
            std::cerr << "Failed to initialize accelerator\n";
            return 1;
        }
        
        std::cout << "Testing basic matrix operations...\n\n";
        
        // Test 1: Matrix Addition
        std::cout << "=== Matrix Addition Test ===\n";
        Float16Matrix A(256, 256);
        Float16Matrix B(256, 256);
        Float16Matrix C(256, 256);
        
        // Initialize matrices
        A.set_random();
        B.set_random();
        C.fill(0.0f);
        
        std::cout << "Matrix A: " << A.rows() << "x" << A.cols() << "\n";
        std::cout << "Matrix B: " << B.rows() << "x" << B.cols() << "\n";
        std::cout << "Matrix C: " << C.rows() << "x" << C.cols() << "\n";
        
        // Perform addition
        auto start = std::chrono::high_resolution_clock::now();
        bool success = accelerator.matrix_add(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Matrix addition successful!\n";
            std::cout << "Execution time: " << duration.count() << " microseconds\n";
            
            // Get performance metrics
            auto metrics = accelerator.get_last_performance_metrics();
            std::cout << "GFLOPS: " << metrics.gflops << "\n";
            std::cout << "Memory Bandwidth: " << metrics.memory_bandwidth_gbps << " GB/s\n\n";
        } else {
            std::cerr << "Matrix addition failed!\n\n";
        }
        
        // Test 2: Matrix Multiplication
        std::cout << "=== Matrix Multiplication Test ===\n";
        Float16Matrix D(128, 128);
        Float16Matrix E(128, 128);
        Float16Matrix F(128, 128);
        
        D.set_random();
        E.set_random();
        F.fill(0.0f);
        
        std::cout << "Matrix D: " << D.rows() << "x" << D.cols() << "\n";
        std::cout << "Matrix E: " << E.rows() << "x" << E.cols() << "\n";
        std::cout << "Matrix F: " << F.rows() << "x" << F.cols() << "\n";
        
        // Perform multiplication
        start = std::chrono::high_resolution_clock::now();
        success = accelerator.matrix_multiply(D, E, F);
        end = std::chrono::high_resolution_clock::now();
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Matrix multiplication successful!\n";
            std::cout << "Execution time: " << duration.count() << " microseconds\n";
            
            // Get performance metrics
            auto metrics = accelerator.get_last_performance_metrics();
            std::cout << "GFLOPS: " << metrics.gflops << "\n";
            std::cout << "Memory Bandwidth: " << metrics.memory_bandwidth_gbps << " GB/s\n\n";
        } else {
            std::cerr << "Matrix multiplication failed!\n\n";
        }
        
        // Test 3: Matrix Transpose
        std::cout << "=== Matrix Transpose Test ===\n";
        Float16Matrix G(64, 64);
        Float16Matrix H(64, 64);
        
        G.set_random();
        H.fill(0.0f);
        
        std::cout << "Matrix G: " << G.rows() << "x" << G.cols() << "\n";
        std::cout << "Matrix H: " << H.rows() << "x" << H.cols() << "\n";
        
        // Perform transpose
        start = std::chrono::high_resolution_clock::now();
        success = accelerator.matrix_transpose(G, H);
        end = std::chrono::high_resolution_clock::now();
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Matrix transpose successful!\n";
            std::cout << "Execution time: " << duration.count() << " microseconds\n";
            
            // Get performance metrics
            auto metrics = accelerator.get_last_performance_metrics();
            std::cout << "GFLOPS: " << metrics.gflops << "\n";
            std::cout << "Memory Bandwidth: " << metrics.memory_bandwidth_gbps << " GB/s\n\n";
        } else {
            std::cerr << "Matrix transpose failed!\n\n";
        }
        
        std::cout << "All tests completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}