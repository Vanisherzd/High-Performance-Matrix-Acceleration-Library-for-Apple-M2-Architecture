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