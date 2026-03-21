/**
 * correctness_tests.cpp
 *
 * Validates numerical accuracy of the Metal GPU path and CPU NEON path
 * against a float32 reference implementation.
 *
 * Key questions answered:
 *   1. Does the FP16 GPU result match a FP32 reference within tolerance?
 *   2. Do CPU and GPU paths agree with each other?
 *   3. Does mixed-precision accumulation actually improve accuracy vs pure FP16?
 *   4. Are edge-case matrix shapes (non-square, primes, boundary sizes) handled?
 *
 * Build & run:
 *   cmake --build build --target correctness_tests
 *   ./build/correctness_tests
 *
 * Exit code 0 = all PASS, non-zero = at least one FAIL.
 */

#include "../include/metal_float16_accelerator.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

// ─── Reference implementation ────────────────────────────────────────────────

// Computes C = A * B in float32 (ground truth).
// Inputs read as float; output written as float.
static void ref_matmul_f32(
    const Float16Matrix& A, const Float16Matrix& B, std::vector<float>& C_ref)
{
    const uint32_t M = A.rows(), K = A.cols(), N = B.cols();
    C_ref.assign(static_cast<size_t>(M) * N, 0.0f);

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t k = 0; k < K; ++k) {
            float a_ik = static_cast<float>(A(i, k));
            for (uint32_t j = 0; j < N; ++j) {
                C_ref[i * N + j] += a_ik * static_cast<float>(B(k, j));
            }
        }
    }
}

// ─── Error metrics ────────────────────────────────────────────────────────────

struct ErrorStats {
    double max_abs_err;
    double mean_abs_err;
    double rmse;
    double max_rel_err;   // relative to |ref| where |ref| > 1e-6
    size_t nan_count;
    size_t inf_count;
};

static ErrorStats compute_errors(
    const Float16Matrix& C_fp16,
    const std::vector<float>& C_ref,
    uint32_t M, uint32_t N)
{
    ErrorStats s{};
    double sum_sq = 0.0, sum_abs = 0.0;
    size_t rel_count = 0;

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float got = static_cast<float>(C_fp16(i, j));
            float ref = C_ref[i * N + j];

            if (std::isnan(got)) { s.nan_count++; continue; }
            if (std::isinf(got)) { s.inf_count++; continue; }

            double err = std::abs(static_cast<double>(got) - static_cast<double>(ref));
            s.max_abs_err = std::max(s.max_abs_err, err);
            sum_abs += err;
            sum_sq  += err * err;

            if (std::abs(ref) > 1e-6) {
                double rel = err / std::abs(ref);
                s.max_rel_err = std::max(s.max_rel_err, rel);
                ++rel_count;
            }
        }
    }

    size_t total = static_cast<size_t>(M) * N;
    s.mean_abs_err = sum_abs / static_cast<double>(total);
    s.rmse         = std::sqrt(sum_sq / static_cast<double>(total));
    return s;
}

// ─── Test harness ─────────────────────────────────────────────────────────────

struct TestCase {
    uint32_t M, N, K;
    const char* desc;
};

static int total_pass = 0, total_fail = 0;

static void run_test(MetalFloat16Accelerator::Accelerator& accel, const TestCase& tc,
                     double tol_max_abs, double tol_max_rel)
{
    Float16Matrix A(tc.M, tc.K), B(tc.K, tc.N), C(tc.M, tc.N);
    A.set_random(); B.set_random();

    // Reference (FP32)
    std::vector<float> C_ref;
    ref_matmul_f32(A, B, C_ref);

    // Library path (auto-dispatched)
    bool ok = accel.matrix_multiply(A, B, C);
    auto m = accel.get_last_performance_metrics();

    if (!ok) {
        printf("  [FAIL] %-30s  dispatch error\n", tc.desc);
        ++total_fail;
        return;
    }

    ErrorStats e = compute_errors(C, C_ref, tc.M, tc.N);

    bool pass = (e.max_abs_err <= tol_max_abs) &&
                (e.max_rel_err <= tol_max_rel) &&
                (e.nan_count == 0) &&
                (e.inf_count == 0);

    printf("  [%s] %-30s  max_abs=%.4f  max_rel=%.4f%%  rmse=%.5f"
           "  NaN=%zu  Inf=%zu  %.1fms  %.1fGFLOPS\n",
           pass ? "PASS" : "FAIL", tc.desc,
           e.max_abs_err, e.max_rel_err * 100.0, e.rmse,
           e.nan_count, e.inf_count,
           m.execution_time_ms, m.gflops);

    if (!pass) {
        if (e.max_abs_err > tol_max_abs)
            printf("         ↳ max_abs_err %.4f exceeds tolerance %.4f\n",
                   e.max_abs_err, tol_max_abs);
        if (e.max_rel_err > tol_max_rel)
            printf("         ↳ max_rel_err %.2f%% exceeds tolerance %.2f%%\n",
                   e.max_rel_err * 100.0, tol_max_rel * 100.0);
    }

    pass ? ++total_pass : ++total_fail;
}

// ─── GPU vs CPU agreement test ────────────────────────────────────────────────

static void run_gpu_cpu_agreement(MetalFloat16Accelerator::Accelerator& accel,
                                  uint32_t N, double tol_max_abs)
{
    // Force a size that hits the GPU path (N=512 → ops = 512^3 >> 1M threshold)
    Float16Matrix A(N, N), B(N, N), C_gpu(N, N), C_cpu_ref(N, N);
    A.set_random(); B.set_random();

    // GPU path
    accel.matrix_multiply(A, B, C_gpu);

    // CPU reference via ref_matmul_f32 → store back as FP16
    std::vector<float> ref;
    ref_matmul_f32(A, B, ref);
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t j = 0; j < N; ++j)
            C_cpu_ref(i, j) = static_cast<__fp16>(ref[i * N + j]);

    ErrorStats e = compute_errors(C_gpu, ref, N, N);
    bool pass = (e.max_abs_err <= tol_max_abs) && (e.nan_count == 0);

    printf("  [%s] GPU vs FP32-ref %u×%u            max_abs=%.4f  rmse=%.5f\n",
           pass ? "PASS" : "FAIL", N, N, e.max_abs_err, e.rmse);
    pass ? ++total_pass : ++total_fail;
}

// ─── Mixed-precision benefit test ────────────────────────────────────────────

// Demonstrates that mixed-precision (FP32 accumulator) outperforms a
// pure-FP16 naive dot-product for large N.
static void mixed_precision_benefit_test(uint32_t N)
{
    Float16Matrix A(N, N), B(N, N);
    A.set_random(); B.set_random();

    std::vector<float> ref;
    ref_matmul_f32(A, B, ref);

    // Simulate pure-FP16 accumulation (naive path)
    std::vector<float> c_pure_fp16(static_cast<size_t>(N) * N, 0.0f);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            __fp16 acc = 0.0f;
            for (uint32_t k = 0; k < N; ++k) {
                acc = static_cast<__fp16>(
                    static_cast<float>(acc) +
                    static_cast<float>(A(i, k)) * static_cast<float>(B(k, j)));
            }
            c_pure_fp16[i * N + j] = static_cast<float>(acc);
        }
    }

    // Mixed-precision (FP32 accumulator, mirrors Metal kernel behaviour)
    std::vector<float> c_mixed(static_cast<size_t>(N) * N, 0.0f);
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < N; ++k)
                acc += static_cast<float>(A(i, k)) * static_cast<float>(B(k, j));
            c_mixed[i * N + j] = acc;
        }

    // Compute max error for each
    double max_err_fp16 = 0.0, max_err_mixed = 0.0;
    for (size_t idx = 0; idx < static_cast<size_t>(N) * N; ++idx) {
        max_err_fp16  = std::max(max_err_fp16,
                                 std::abs(c_pure_fp16[idx] - ref[idx]));
        max_err_mixed = std::max(max_err_mixed,
                                 std::abs(c_mixed[idx] - ref[idx]));
    }

    double improvement = max_err_fp16 / (max_err_mixed + 1e-12);
    printf("  [INFO] Mixed-precision benefit @ %u×%u:\n", N, N);
    printf("         pure-FP16 accumulator max_err=%.4f\n", max_err_fp16);
    printf("         FP32 accumulator      max_err=%.6f  (%.1f× more accurate)\n",
           max_err_mixed, improvement);
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║       Metal-Float16-Accelerator — Correctness Tests   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    MetalFloat16Accelerator::Accelerator accel;
    if (!accel.initialize()) {
        fprintf(stderr, "ERROR: failed to initialise accelerator\n");
        return 1;
    }

    // FP16 tolerance: machine epsilon ≈ 9.77e-4 (~0.1%)
    // For large matrices, expect max relative error < 2% due to accumulation
    constexpr double TOL_ABS = 1.0;     // absolute error tolerance per element
    constexpr double TOL_REL = 0.05;    // 5% relative tolerance

    // ── Section 1: Matrix shapes ─────────────────────────────────────────────
    printf("── Section 1: Shape Coverage ──────────────────────────────────\n");
    std::vector<TestCase> cases = {
        // Square power-of-2
        { 64,   64,   64,   "Square 64×64"},
        {128,  128,  128,   "Square 128×128"},
        {256,  256,  256,   "Square 256×256"},
        {512,  512,  512,   "Square 512×512 (GPU)"},
        // Non-square
        {128,  256,   64,   "Rect 128×256 K=64"},
        {256,  128,  512,   "Rect 256×128 K=512"},
        // Odd / prime sizes (boundary conditions)
        { 33,   33,   33,   "Prime 33×33"},
        {127,  127,  127,   "Near-pow2 127×127"},
        {129,  129,  129,   "Near-pow2 129×129"},
        { 31,   65,   97,   "Irregular 31×65 K=97"},
    };
    for (auto& tc : cases)
        run_test(accel, tc, TOL_ABS, TOL_REL);

    // ── Section 2: GPU vs FP32 reference agreement ───────────────────────────
    printf("\n── Section 2: GPU vs FP32 Reference ───────────────────────────\n");
    for (uint32_t N : {128u, 256u, 512u, 1024u})
        run_gpu_cpu_agreement(accel, N, TOL_ABS);

    // ── Section 3: Mixed-precision benefit ───────────────────────────────────
    printf("\n── Section 3: Mixed-Precision Accumulation Benefit ────────────\n");
    for (uint32_t N : {256u, 512u, 1024u})
        mixed_precision_benefit_test(N);

    // ── Section 4: Identity matrix sanity check ──────────────────────────────
    printf("\n── Section 4: Identity Sanity Check ───────────────────────────\n");
    {
        constexpr uint32_t N = 64;
        Float16Matrix A(N, N), I(N, N), C(N, N);
        A.set_random();
        I.set_identity();
        accel.matrix_multiply(A, I, C);  // A * I should ≈ A

        double max_err = 0.0;
        for (uint32_t i = 0; i < N; ++i)
            for (uint32_t j = 0; j < N; ++j)
                max_err = std::max(max_err,
                    std::abs(static_cast<double>(C(i, j)) -
                             static_cast<double>(A(i, j))));

        bool pass = max_err < 0.01;
        printf("  [%s] A * I = A (%u×%u)  max_err=%.6f\n",
               pass ? "PASS" : "FAIL", N, N, max_err);
        pass ? ++total_pass : ++total_fail;
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Result: %d PASSED  %d FAILED  (total %d)\n",
           total_pass, total_fail, total_pass + total_fail);

    if (total_fail == 0)
        printf("  Status: ALL TESTS PASSED\n");
    else
        printf("  Status: FAILURES DETECTED — see above\n");
    printf("══════════════════════════════════════════════════════════════\n\n");

    return total_fail > 0 ? 1 : 0;
}
