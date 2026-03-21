/**
 * alphazero_inference_sim.cpp
 *
 * Simulates the matrix-multiplication core of an AlphaZero-style neural
 * network forward pass and compares throughput across execution paths.
 *
 * AlphaZero Network Approximation (Go 19×19):
 *   Input  : batch_size × feature_dim  (positions × encoded features)
 *   Backbone: N_BLOCKS residual blocks, each = 2 × GEMM(batch × dim, dim × dim)
 *   Policy head : GEMM(batch × dim, dim × POLICY_SIZE)
 *   Value  head : GEMM(batch × dim, dim × 1) → scalar evaluation
 *
 * What this measures:
 *   • End-to-end forward pass latency (ms) per batch
 *   • Throughput: positions evaluated per second
 *   • Metal GPU vs scalar CPU speedup
 *
 * Output: JSON to stdout so the Python benchmark script can parse it.
 *
 * Build (from project root after cmake):
 *   cmake --build build --target alphazero_inference_sim
 *
 * Run directly:
 *   ./build/alphazero_inference_sim
 *
 * Run via Python benchmark harness:
 *   uv run tools/alphazero_benchmark.py
 */

#include "../include/metal_float16_accelerator.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ─── AlphaZero Network Configuration ─────────────────────────────────────────

struct AlphaZeroConfig {
    // Standard AlphaZero (Go 19×19) settings
    uint32_t batch_size   = 256;   // positions evaluated per MCTS batch
    uint32_t feature_dim  = 256;   // residual tower width (channels)
    uint32_t n_blocks     = 20;    // residual blocks (each = 2 linear layers)
    uint32_t policy_size  = 362;   // 19×19 moves + 1 pass
    int      n_warmup     = 5;
    int      n_samples    = 30;
};

// ─── Single forward pass ──────────────────────────────────────────────────────

/**
 * Runs one full forward pass through the simplified AlphaZero network.
 *
 * Computation graph (linear algebra view):
 *   for each residual block i = 0..N_BLOCKS-1:
 *     H = ReLU(W1[i] * H)        // GEMM: (batch × dim) × (dim × dim)
 *     H = ReLU(W2[i] * H) + H    // GEMM + residual add
 *   policy_logits = H * W_policy  // GEMM: (batch × dim) × (dim × policy_size)
 *   value         = H * W_value   // GEMM: (batch × dim) × (dim × 1)
 *
 * Note: activations (ReLU) and batch-norm are elementwise and negligible vs GEMM.
 *       We benchmark the GEMM-dominant portion — the true bottleneck.
 */
static double forward_pass_ms(MetalFloat16Accelerator::Accelerator& accel,
                               const AlphaZeroConfig& cfg,
                               std::vector<Float16Matrix>& W1,
                               std::vector<Float16Matrix>& W2,
                               Float16Matrix& W_policy,
                               Float16Matrix& W_value)
{
    // Activations (hidden state)
    Float16Matrix H(cfg.batch_size, cfg.feature_dim);
    Float16Matrix H_new(cfg.batch_size, cfg.feature_dim);
    H.set_random();

    auto t0 = Clock::now();

    // Residual backbone
    for (uint32_t b = 0; b < cfg.n_blocks; ++b) {
        accel.matrix_multiply(H,     W1[b], H_new);   // first linear
        accel.matrix_multiply(H_new, W2[b], H);       // second linear (in-place)
    }

    // Policy head
    Float16Matrix policy_out(cfg.batch_size, cfg.policy_size);
    accel.matrix_multiply(H, W_policy, policy_out);

    // Value head
    Float16Matrix value_out(cfg.batch_size, 1);
    accel.matrix_multiply(H, W_value, value_out);

    auto t1 = Clock::now();
    return Ms(t1 - t0).count();
}

// ─── Scalar reference (no acceleration) ─────────────────────────────────────

static double scalar_forward_pass_ms(const AlphaZeroConfig& cfg) {
    // Naive triple-nested loop for one residual block GEMM as reference timing.
    // We scale to full network by linear extrapolation (avoids 10+ minute wait).

    const uint32_t M = cfg.batch_size, N = cfg.feature_dim, K = cfg.feature_dim;

    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
    for (auto& x : A) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : B) x = static_cast<float>(rand()) / RAND_MAX;

    auto t0 = Clock::now();
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t k = 0; k < K; ++k) {
            float a = A[i * K + k];
            for (uint32_t j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
    auto t1 = Clock::now();

    double single_gemm_ms = Ms(t1 - t0).count();

    // Full pass: 2 GEMMs/block × n_blocks + policy + value
    uint32_t n_gemms = cfg.n_blocks * 2 + 2;
    return single_gemm_ms * n_gemms;
}

// ─── Stats helpers ────────────────────────────────────────────────────────────

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
}

static double vec_mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    AlphaZeroConfig cfg;

    // Optionally override batch_size from CLI: ./alphazero_inference_sim 64
    if (argc > 1) cfg.batch_size = static_cast<uint32_t>(std::atoi(argv[1]));

    MetalFloat16Accelerator::Accelerator accel;
    if (!accel.initialize()) {
        fprintf(stderr, "{\"error\": \"accelerator_init_failed\"}\n");
        return 1;
    }

    // ── Allocate and initialise network weight matrices ───────────────────────
    std::vector<Float16Matrix> W1, W2;
    W1.reserve(cfg.n_blocks);
    W2.reserve(cfg.n_blocks);
    for (uint32_t b = 0; b < cfg.n_blocks; ++b) {
        W1.emplace_back(cfg.feature_dim, cfg.feature_dim);  W1.back().set_random();
        W2.emplace_back(cfg.feature_dim, cfg.feature_dim);  W2.back().set_random();
    }
    Float16Matrix W_policy(cfg.feature_dim, cfg.policy_size); W_policy.set_random();
    Float16Matrix W_value (cfg.feature_dim, 1);               W_value.set_random();

    // ── Warmup ────────────────────────────────────────────────────────────────
    for (int i = 0; i < cfg.n_warmup; ++i)
        forward_pass_ms(accel, cfg, W1, W2, W_policy, W_value);

    // ── Timed samples ─────────────────────────────────────────────────────────
    std::vector<double> times;
    times.reserve(cfg.n_samples);
    for (int i = 0; i < cfg.n_samples; ++i)
        times.push_back(forward_pass_ms(accel, cfg, W1, W2, W_policy, W_value));

    double accel_p50_ms  = median(times);
    double accel_mean_ms = vec_mean(times);
    std::sort(times.begin(), times.end());
    double accel_p99_ms  = times[static_cast<size_t>(0.99 * (times.size() - 1))];

    // ── Scalar reference (single sample, extrapolated) ────────────────────────
    double scalar_ms = scalar_forward_pass_ms(cfg);

    // ── Derived metrics ───────────────────────────────────────────────────────
    uint32_t n_gemms       = cfg.n_blocks * 2 + 2;
    uint64_t flops_per_pass =
        static_cast<uint64_t>(2) * cfg.batch_size * cfg.feature_dim * cfg.feature_dim * (cfg.n_blocks * 2)
      + static_cast<uint64_t>(2) * cfg.batch_size * cfg.feature_dim * cfg.policy_size
      + static_cast<uint64_t>(2) * cfg.batch_size * cfg.feature_dim * 1;

    double accel_gflops  = static_cast<double>(flops_per_pass) / (accel_p50_ms  * 1e6);
    double scalar_gflops = static_cast<double>(flops_per_pass) / (scalar_ms * 1e6);
    double speedup       = scalar_ms / accel_p50_ms;

    // positions/sec = batch_size / time_per_batch
    double accel_pos_per_sec  = cfg.batch_size / (accel_p50_ms  / 1000.0);
    double scalar_pos_per_sec = cfg.batch_size / (scalar_ms / 1000.0);

    // ── Human-readable summary (stderr so it doesn't pollute JSON stdout) ─────
    fprintf(stderr, "\n");
    fprintf(stderr, "  AlphaZero Inference Simulation — Apple M2 Metal FP16\n");
    fprintf(stderr, "  %-30s %u × %u   depth=%u blocks\n",
            "Network:", cfg.batch_size, cfg.feature_dim, cfg.n_blocks);
    fprintf(stderr, "  %-30s %u GEMMs per forward pass\n", "GEMM count:", n_gemms);
    fprintf(stderr, "  %-30s %.2f B FLOP\n", "FLOPs / pass:",
            static_cast<double>(flops_per_pass) / 1e9);
    fprintf(stderr, "\n");
    fprintf(stderr, "  %-30s %8.2fms  (%.1f GFLOPS)  %.0f pos/s\n",
            "Metal GPU (this lib):", accel_p50_ms, accel_gflops, accel_pos_per_sec);
    fprintf(stderr, "  %-30s %8.2fms  (%.1f GFLOPS)  %.0f pos/s\n",
            "Scalar CPU (C++ -O3):", scalar_ms, scalar_gflops, scalar_pos_per_sec);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Speedup: %.1f×    P99 latency: %.2fms\n\n",
            speedup, accel_p99_ms);

    // ── JSON output (stdout) for Python harness ───────────────────────────────
    printf("{\n");
    printf("  \"config\": {\n");
    printf("    \"batch_size\": %u,\n",    cfg.batch_size);
    printf("    \"feature_dim\": %u,\n",   cfg.feature_dim);
    printf("    \"n_blocks\": %u,\n",      cfg.n_blocks);
    printf("    \"policy_size\": %u,\n",   cfg.policy_size);
    printf("    \"n_gemms_per_pass\": %u,\n", n_gemms);
    printf("    \"flops_per_pass\": %llu\n", (unsigned long long)flops_per_pass);
    printf("  },\n");
    printf("  \"metal_gpu\": {\n");
    printf("    \"p50_ms\": %.4f,\n",   accel_p50_ms);
    printf("    \"p99_ms\": %.4f,\n",   accel_p99_ms);
    printf("    \"mean_ms\": %.4f,\n",  accel_mean_ms);
    printf("    \"gflops\": %.2f,\n",   accel_gflops);
    printf("    \"positions_per_sec\": %.0f\n", accel_pos_per_sec);
    printf("  },\n");
    printf("  \"scalar_cpu\": {\n");
    printf("    \"p50_ms\": %.4f,\n",   scalar_ms);
    printf("    \"gflops\": %.4f,\n",   scalar_gflops);
    printf("    \"positions_per_sec\": %.0f\n", scalar_pos_per_sec);
    printf("  },\n");
    printf("  \"speedup\": %.2f\n", speedup);
    printf("}\n");

    return 0;
}
