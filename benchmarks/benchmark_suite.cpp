#include "../include/metal_float16_accelerator.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

class BenchmarkSuite {
private:
    MetalFloat16Accelerator::Accelerator* accelerator_;
    
    struct BenchmarkResult {
        std::string operation;
        uint32_t matrix_size;
        double execution_time_ms;
        double gflops;
        bool success;
    };
    
    std::vector<BenchmarkResult> results_;
    
    // Helper methods
    void run_matrix_add_benchmark(uint32_t size);
    void run_matrix_multiply_benchmark(uint32_t size);
    void run_matrix_transpose_benchmark(uint32_t size);
    void run_matrix_scale_benchmark(uint32_t size);
    
    double calculate_gflops(uint64_t operations, double time_ms) const;
    void print_results() const;
    
public:
    BenchmarkSuite();
    ~BenchmarkSuite();
    
    void run_all_benchmarks();
    void clear_results();
};

BenchmarkSuite::BenchmarkSuite() 
    : accelerator_(new MetalFloat16Accelerator::Accelerator()) {
}

BenchmarkSuite::~BenchmarkSuite() {
    delete accelerator_;
}

double BenchmarkSuite::calculate_gflops(uint64_t operations, double time_ms) const {
    if (time_ms > 0) {
        double time_seconds = time_ms / 1000.0;
        return (operations / 1e9) / time_seconds;
    }
    return 0.0;
}

void BenchmarkSuite::print_results() const {
    std::cout << "\n=== BENCHMARK RESULTS ===\n";
    std::cout << std::setw(20) << "Operation" << "Matrix Size" << "Time (ms)" << "GFLOPS" << "Status" << "\n";
    std::cout << std::setw(20) << std::setfill('-') << "" << std::setfill('-') << "" << std::setfill('-') << "\n";
    
    for (const auto& result : results_) {
        std::cout << std::setw(20) << result.operation
                  << std::setw(12) << result.matrix_size
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.execution_time_ms
                  << std::setw(10) << std::fixed << std::setprecision(2) << result.gflops
                  << std::setw(8) << (result.success ? "PASS" : "FAIL") << "\n";
    }
    
    std::cout << std::setw(20) << std::setfill('-') << "" << std::setfill('-') << "" << std::setfill('-') << "\n";
}

void BenchmarkSuite::run_matrix_add_benchmark(uint32_t size) {
    std::cout << "\nRunning Matrix Addition Benchmark (" << size << "x" << size << ")...\n";
    
    try {
        Float16Matrix A(size, size);
        Float16Matrix B(size, size);
        Float16Matrix C(size, size);
        
        A.set_random();
        B.set_random();
        C.fill(0.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = accelerator_->matrix_add(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        
        BenchmarkResult result;
        result.operation = "Matrix Add";
        result.matrix_size = size;
        result.success = success;
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            result.execution_time_ms = duration.count() / 1000.0;
            result.gflops = calculate_gflops(size * size, result.execution_time_ms);
        }
        
        results_.push_back(result);
        
    } catch (const std::exception& e) {
        std::cerr << "Matrix addition benchmark failed: " << e.what() << std::endl;
        
        BenchmarkResult result;
        result.operation = "Matrix Add";
        result.matrix_size = size;
        result.success = false;
        result.execution_time_ms = 0.0;
        result.gflops = 0.0;
        results_.push_back(result);
    }
}

void BenchmarkSuite::run_matrix_multiply_benchmark(uint32_t size) {
    std::cout << "\nRunning Matrix Multiplication Benchmark (" << size << "x" << size << ")...\n";
    
    try {
        Float16Matrix A(size, size);
        Float16Matrix B(size, size);
        Float16Matrix C(size, size);
        
        A.set_random();
        B.set_random();
        C.fill(0.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = accelerator_->matrix_multiply(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        
        BenchmarkResult result;
        result.operation = "Matrix Mul";
        result.matrix_size = size;
        result.success = success;
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            result.execution_time_ms = duration.count() / 1000.0;
            result.gflops = calculate_gflops(size * size * size, result.execution_time_ms);
        }
        
        results_.push_back(result);
        
    } catch (const std::exception& e) {
        std::cerr << "Matrix multiplication benchmark failed: " << e.what() << std::endl;
        
        BenchmarkResult result;
        result.operation = "Matrix Mul";
        result.matrix_size = size;
        result.success = false;
        result.execution_time_ms = 0.0;
        result.gflops = 0.0;
        results_.push_back(result);
    }
}

void BenchmarkSuite::run_matrix_transpose_benchmark(uint32_t size) {
    std::cout << "\nRunning Matrix Transpose Benchmark (" << size << "x" << size << ")...\n";
    
    try {
        Float16Matrix A(size, size);
        Float16Matrix C(size, size);
        
        A.set_random();
        C.fill(0.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = accelerator_->matrix_transpose(A, C);
        auto end = std::chrono::high_resolution_clock::now();
        
        BenchmarkResult result;
        result.operation = "Matrix Trans";
        result.matrix_size = size;
        result.success = success;
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            result.execution_time_ms = duration.count() / 1000.0;
            result.gflops = calculate_gflops(size * size, result.execution_time_ms);
        }
        
        results_.push_back(result);
        
    } catch (const std::exception& e) {
        std::cerr << "Matrix transpose benchmark failed: " << e.what() << std::endl;
        
        BenchmarkResult result;
        result.operation = "Matrix Trans";
        result.matrix_size = size;
        result.success = false;
        result.execution_time_ms = 0.0;
        result.gflops = 0.0;
        results_.push_back(result);
    }
}

void BenchmarkSuite::run_matrix_scale_benchmark(uint32_t size) {
    std::cout << "\nRunning Matrix Scaling Benchmark (" << size << "x" << size << ")...\n";
    
    try {
        Float16Matrix A(size, size);
        Float16Matrix C(size, size);
        
        A.set_random();
        C.fill(0.0f);
        
        float scalar = 2.0f;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = accelerator_->matrix_scale(A, scalar, C);
        auto end = std::chrono::high_resolution_clock::now();
        
        BenchmarkResult result;
        result.operation = "Matrix Scale";
        result.matrix_size = size;
        result.success = success;
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            result.execution_time_ms = duration.count() / 1000.0;
            result.gflops = calculate_gflops(size * size, result.execution_time_ms);
        }
        
        results_.push_back(result);
        
    } catch (const std::exception& e) {
        std::cerr << "Matrix scaling benchmark failed: " << e.what() << std::endl;
        
        BenchmarkResult result;
        result.operation = "Matrix Scale";
        result.matrix_size = size;
        result.success = false;
        result.execution_time_ms = 0.0;
        result.gflops = 0.0;
        results_.push_back(result);
    }
}

void BenchmarkSuite::run_all_benchmarks() {
    clear_results();
    
    std::cout << "Starting comprehensive benchmark suite...\n";
    accelerator_->print_device_info();
    
    // Test different matrix sizes
    std::vector<uint32_t> sizes = {128, 256, 512, 1024};
    
    for (uint32_t size : sizes) {
        run_matrix_add_benchmark(size);
        run_matrix_multiply_benchmark(size);
        run_matrix_transpose_benchmark(size);
        run_matrix_scale_benchmark(size);
    }
    
    print_results();
    
    std::cout << "\nBenchmark suite completed!\n";
}

void BenchmarkSuite::clear_results() {
    results_.clear();
}