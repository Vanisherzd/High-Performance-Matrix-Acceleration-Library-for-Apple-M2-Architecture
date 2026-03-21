/**
 * latency_profile.cpp
 *
 * Statistical latency profiler — collects N=200 samples per configuration
 * and reports P50 / P95 / P99 / P99.9, jitter, and coefficient of variation.
 *
 * Why this matters (systems engineering):
 *   • Mean latency hides tail behaviour caused by thermal throttling,
 *     OS scheduling, TLB pressure, and cache evictions.
 *   • P99 / jitter directly predicts worst-case frame times in real-time
 *     systems (robotics, media pipelines, on-device inference).
 *   • CV (std/mean) measures determinism — a key property of production
 *     systems that sit on a hardware scheduler.
 *
 * Build:
 *   cmake --build build --target latency_profile
 *   ./build/latency_profile
 */

#include "../include/metal_float16_accelerator.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Dur   = std::chrono::duration<double, std::milli>;

// ─── Percentile helpers ───────────────────────────────────────────────────────

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (static_cast<double>(v.size()) - 1.0);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = lo + 1;
    if (hi >= v.size()) return v.back();
    return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

static double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double stdev(const std::vector<double>& v, double m) {
    double sq = 0.0;
    for (double x : v) sq += (x - m) * (x - m);
    return std::sqrt(sq / v.size());
}

// ─── Profiler ─────────────────────────────────────────────────────────────────

struct LatencyStats {
    double p50, p95, p99, p999;
    double mean_ms, stdev_ms, cv;
    double jitter;      // P99 - P50
    double min_ms, max_ms;
    double gflops_mean;
};

static LatencyStats profile(MetalFloat16Accelerator::Accelerator& accel,
                             uint32_t N, int n_warmup, int n_samples)
{
    Float16Matrix A(N, N), B(N, N), C(N, N);
    A.set_random(); B.set_random();

    // Warmup — not recorded
    for (int i = 0; i < n_warmup; ++i)
        accel.matrix_multiply(A, B, C);

    std::vector<double> times;
    times.reserve(n_samples);
    double gflops_sum = 0.0;

    for (int i = 0; i < n_samples; ++i) {
        auto t0 = Clock::now();
        accel.matrix_multiply(A, B, C);
        auto t1 = Clock::now();
        double ms = Dur(t1 - t0).count();
        times.push_back(ms);
        gflops_sum += accel.get_last_performance_metrics().gflops;
    }

    double m = mean(times);
    double s = stdev(times, m);

    LatencyStats st{};
    st.p50   = percentile(times, 0.50);
    st.p95   = percentile(times, 0.95);
    st.p99   = percentile(times, 0.99);
    st.p999  = percentile(times, 0.999);
    st.mean_ms   = m;
    st.stdev_ms  = s;
    st.cv        = (m > 0) ? s / m : 0.0;
    st.jitter    = st.p99 - st.p50;
    st.min_ms    = times.front();   // already sorted by percentile
    st.max_ms    = times.back();
    st.gflops_mean = gflops_sum / n_samples;
    return st;
}

// ─── Memory bandwidth analysis ────────────────────────────────────────────────

// For matrix operations, theoretical bandwidth = (bytes_read + bytes_written) / time
// This lets us tell whether we are compute-bound or memory-bandwidth-bound.
static void print_bandwidth_analysis(uint32_t N, double exec_ms) {
    // MatMul reads A(N²) + B(N²), writes C(N²) = 3N² elements × 2 bytes (FP16)
    double bytes = 3.0 * N * N * 2.0;
    double bw_gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (exec_ms / 1000.0);

    // M2 unified memory theoretical peak: ~100 GB/s
    constexpr double M2_PEAK_BW = 100.0;
    double bw_util = bw_gbps / M2_PEAK_BW * 100.0;

    // Roofline: for N×N matmul, arithmetic intensity = N/3 FLOP/byte
    double ai = static_cast<double>(N) / 3.0;
    const char* bound = (ai < 50.0) ? "memory-bw-bound" : "compute-bound";

    printf("    BW: %.1f GB/s (%.1f%% of M2 peak)  AI=%.0f FLOP/byte  → %s\n",
           bw_gbps, bw_util, ai, bound);
}

// ─── ASCII histogram ──────────────────────────────────────────────────────────

static void print_histogram(std::vector<double> v, int bins = 16) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    double lo = v.front(), hi = v.back();
    double range = hi - lo;
    if (range < 1e-9) range = 1e-9;

    std::vector<int> counts(bins, 0);
    for (double x : v) {
        int b = static_cast<int>((x - lo) / range * (bins - 1));
        b = std::min(b, bins - 1);
        counts[b]++;
    }
    int max_count = *std::max_element(counts.begin(), counts.end());

    printf("    Latency distribution (%.2f–%.2fms, %zu samples):\n",
           lo, hi, v.size());
    for (int b = 0; b < bins; ++b) {
        double bucket_ms = lo + (b + 0.5) * range / bins;
        int bar_len = (max_count > 0) ? counts[b] * 30 / max_count : 0;
        printf("    %6.2fms │%s%s (%d)\n",
               bucket_ms,
               std::string(bar_len, '█').c_str(),
               std::string(30 - bar_len, ' ').c_str(),
               counts[b]);
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("\n");
    printf("╔═════════════════════════════════════════════════════════════╗\n");
    printf("║   Metal-Float16-Accelerator — Statistical Latency Profiler  ║\n");
    printf("║              P50 / P95 / P99 / P99.9 · Jitter · BW          ║\n");
    printf("╚═════════════════════════════════════════════════════════════╝\n\n");

    MetalFloat16Accelerator::Accelerator accel;
    if (!accel.initialize()) {
        fprintf(stderr, "ERROR: failed to initialise accelerator\n");
        return 1;
    }

    constexpr int WARMUP   = 10;
    constexpr int SAMPLES  = 200;

    struct Config { uint32_t N; const char* label; };
    std::vector<Config> configs = {
        { 128, "128×128  (dispatch overhead regime)"},
        { 256, "256×256  (transition regime)"},
        { 512, "512×512  (GPU peak, compute-bound)"},
        {1024, "1024×1024 (sustained GPU throughput)"},
        {2048, "2048×2048 (large — thermal stress)"},
    };

    printf("  Configuration: %d warmup runs + %d timed samples per size\n\n",
           WARMUP, SAMPLES);

    for (auto& cfg : configs) {
        printf("── %s ─\n", cfg.label);

        // Collect raw samples for histogram
        Float16Matrix A(cfg.N, cfg.N), B(cfg.N, cfg.N), C(cfg.N, cfg.N);
        A.set_random(); B.set_random();
        for (int i = 0; i < WARMUP; ++i) accel.matrix_multiply(A, B, C);

        std::vector<double> raw;
        raw.reserve(SAMPLES);
        for (int i = 0; i < SAMPLES; ++i) {
            auto t0 = Clock::now();
            accel.matrix_multiply(A, B, C);
            auto t1 = Clock::now();
            raw.push_back(Dur(t1 - t0).count());
        }

        auto st = profile(accel, cfg.N, 0, 0);  // stats from sorted raw
        // recompute from raw directly
        double m = mean(raw);
        double s = stdev(raw, m);
        std::sort(raw.begin(), raw.end());

        auto pct = [&](double p) {
            double idx = p * (raw.size() - 1);
            size_t lo = static_cast<size_t>(idx);
            size_t hi = std::min(lo + 1, raw.size() - 1);
            return raw[lo] + (idx - lo) * (raw[hi] - raw[lo]);
        };

        double p50  = pct(0.50), p95 = pct(0.95);
        double p99  = pct(0.99), p999 = pct(0.999);
        double cv   = (m > 1e-9) ? s / m : 0.0;
        double jitter = p99 - p50;

        uint64_t ops = static_cast<uint64_t>(2) * cfg.N * cfg.N * cfg.N;
        double gflops_p50 = static_cast<double>(ops) / (p50 * 1e6);

        printf("    Mean:  %8.3fms  ± %.3fms  CV=%.2f%%\n", m, s, cv * 100.0);
        printf("    P50:   %8.3fms  → %.1f GFLOPS\n", p50, gflops_p50);
        printf("    P95:   %8.3fms  (+%.3fms vs P50)\n", p95, p95 - p50);
        printf("    P99:   %8.3fms  (+%.3fms vs P50)  Jitter=%.3fms\n",
               p99, p99 - p50, jitter);
        printf("    P99.9: %8.3fms  Min=%.3f  Max=%.3f\n", p999, raw.front(), raw.back());

        print_bandwidth_analysis(cfg.N, p50);

        // Flag determinism issues
        if (cv > 0.10)
            printf("    ⚠ WARNING: CV=%.1f%% > 10%% — high jitter (thermal throttling?)\n",
                   cv * 100.0);
        if (jitter > p50 * 0.20)
            printf("    ⚠ WARNING: Jitter %.3fms = %.1f%% of P50 — non-deterministic\n",
                   jitter, jitter / p50 * 100.0);

        print_histogram(raw);
        printf("\n");
    }

    // ── Cross-size scaling analysis ──────────────────────────────────────────
    printf("── Scaling Analysis (P50 time vs matrix size) ─────────────────\n");
    printf("  Ideal O(N³) compute-bound scaling: doubling N → 8× longer\n\n");

    struct ScalePoint { uint32_t N; double p50_ms; };
    std::vector<ScalePoint> scale_pts;

    for (uint32_t N : {128u, 256u, 512u, 1024u}) {
        Float16Matrix A(N,N), B(N,N), C(N,N);
        A.set_random(); B.set_random();
        for (int i = 0; i < 5; ++i) accel.matrix_multiply(A, B, C);
        std::vector<double> t;
        for (int i = 0; i < 20; ++i) {
            auto t0 = Clock::now();
            accel.matrix_multiply(A, B, C);
            auto t1 = Clock::now();
            t.push_back(Dur(t1 - t0).count());
        }
        std::sort(t.begin(), t.end());
        double p50_ms = t[t.size() / 2];
        scale_pts.push_back({N, p50_ms});
    }

    for (size_t i = 0; i < scale_pts.size(); ++i) {
        auto& sp = scale_pts[i];
        printf("  %4u×%4u  P50=%8.3fms", sp.N, sp.N, sp.p50_ms);
        if (i > 0) {
            double ratio = sp.p50_ms / scale_pts[i-1].p50_ms;
            printf("  (%.1f× vs prev, ideal=8.0×)", ratio);
        }
        printf("\n");
    }

    printf("\n  Note: ratio >> 8 at small sizes = dispatch overhead dominant\n");
    printf("        ratio ≈ 8 at large sizes = pure compute scaling\n\n");

    return 0;
}
