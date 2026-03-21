#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "matplotlib>=3.8",
# ]
# ///
"""
alphazero_benchmark.py
======================
Benchmarks the AlphaZero inference simulation against a NumPy (FP32) baseline
and generates a publication-quality comparison chart.

Usage (no manual venv setup required):
    uv run tools/alphazero_benchmark.py

What it does:
    1. Runs the C++ Metal FP16 simulation  → JSON timing data
    2. Runs an equivalent NumPy FP32 forward pass baseline
    3. Produces a side-by-side bar chart + latency breakdown table
    4. Saves chart to plots/alphazero_comparison.png
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
BUILD_DIR   = REPO_ROOT / "build"
BINARY      = BUILD_DIR / "alphazero_inference_sim"
PLOTS_DIR   = REPO_ROOT / "plots"


def _check_binary() -> None:
    if not BINARY.exists():
        print(f"[ERROR] Binary not found: {BINARY}")
        print("        Run:  cmake --build build --target alphazero_inference_sim")
        sys.exit(1)


# ── C++ runner ────────────────────────────────────────────────────────────────

def run_cpp_simulation(batch_size: int = 256) -> dict:
    """Run the Metal FP16 C++ binary and parse its JSON output."""
    print(f"  Running Metal FP16 simulation (batch={batch_size})…")
    result = subprocess.run(
        [str(BINARY), str(batch_size)],
        capture_output=True,
        text=True,
        check=True,
    )
    # stderr has human-readable output, stdout has JSON
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            print(f"  {line}")
    return json.loads(result.stdout)


# ── NumPy baseline ────────────────────────────────────────────────────────────

def numpy_forward_pass(batch_size: int, feature_dim: int,
                       n_blocks: int, policy_size: int,
                       n_samples: int = 10) -> dict:
    """
    NumPy FP32 AlphaZero forward pass baseline.
    Uses np.matmul which calls BLAS (single-thread on macOS = Accelerate).
    """
    rng = np.random.default_rng(42)

    # Allocate weights once (same as C++ does)
    W1 = [rng.random((feature_dim, feature_dim), dtype=np.float32) for _ in range(n_blocks)]
    W2 = [rng.random((feature_dim, feature_dim), dtype=np.float32) for _ in range(n_blocks)]
    W_policy = rng.random((feature_dim, policy_size), dtype=np.float32)
    W_value  = rng.random((feature_dim, 1),           dtype=np.float32)

    def one_pass() -> float:
        H = rng.random((batch_size, feature_dim), dtype=np.float32)
        t0 = time.perf_counter()
        for b in range(n_blocks):
            H = np.maximum(0, H @ W1[b])   # first linear + ReLU
            H = np.maximum(0, H @ W2[b])   # second linear + ReLU
        _policy = H @ W_policy
        _value  = H @ W_value
        return (time.perf_counter() - t0) * 1000.0

    # Warmup
    for _ in range(3):
        one_pass()

    times = sorted(one_pass() for _ in range(n_samples))
    p50   = times[len(times) // 2]
    p99   = times[int(0.99 * (len(times) - 1))]
    mean  = sum(times) / len(times)

    n_gemms       = n_blocks * 2 + 2
    flops_per_pass = (2 * batch_size * feature_dim * feature_dim * n_blocks * 2
                    + 2 * batch_size * feature_dim * policy_size
                    + 2 * batch_size * feature_dim * 1)
    gflops = flops_per_pass / (p50 * 1e6)
    pos_per_sec = batch_size / (p50 / 1000.0)

    return {
        "p50_ms":  p50,
        "p99_ms":  p99,
        "mean_ms": mean,
        "gflops":  gflops,
        "positions_per_sec": pos_per_sec,
    }


# ── Scalar (pure Python loop — worst case reference) ─────────────────────────

def scalar_python_extrapolated(batch_size: int, feature_dim: int,
                                n_blocks: int, policy_size: int) -> float:
    """
    Times a single 64×64 scalar matmul in Python and extrapolates.
    Represents the absolute unoptimised baseline (no SIMD, no BLAS).
    """
    N = 64
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    t0 = time.perf_counter()
    C = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for k in range(N):
            for j in range(N):
                C[i, j] += A[i, k] * B[k, j]
    single_64_ms = (time.perf_counter() - t0) * 1000.0

    # Scale: (batch_size/64)² × (feature_dim/64)³ × (2*n_blocks + 2)
    scale = ((batch_size / N) * (feature_dim / N) ** 2 * ((n_blocks * 2) + 2))
    return single_64_ms * scale


# ── Chart ─────────────────────────────────────────────────────────────────────

COLORS = {
    "metal_gpu":   "#7B2FBE",   # purple
    "numpy_blas":  "#2196F3",   # blue
    "scalar_cpp":  "#78909C",   # grey
}


def make_chart(cpp_data: dict, numpy_data: dict) -> None:
    cfg         = cpp_data["config"]
    metal       = cpp_data["metal_gpu"]
    scalar_cpp  = cpp_data["scalar_cpu"]

    batch  = cfg["batch_size"]
    dim    = cfg["feature_dim"]
    blocks = cfg["n_blocks"]

    labels  = ["Scalar C++\n(-O3, no SIMD)", "NumPy\n(BLAS, FP32)", "Metal GPU\n(FP16, this lib)"]
    p50_ms  = [scalar_cpp["p50_ms"], numpy_data["p50_ms"],  metal["p50_ms"]]
    gflops  = [scalar_cpp["gflops"], numpy_data["gflops"],  metal["gflops"]]
    pos_s   = [scalar_cpp["positions_per_sec"], numpy_data["positions_per_sec"],
               metal["positions_per_sec"]]
    colors  = [COLORS["scalar_cpp"], COLORS["numpy_blas"], COLORS["metal_gpu"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"AlphaZero Inference — batch={batch} · {dim}-dim · {blocks} residual blocks\n"
        f"Apple M2 Silicon · Metal FP16 vs NumPy FP32 vs Scalar C++",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── Panel A: Latency (lower is better) ───────────────────────────────────
    ax = axes[0]
    bars = ax.bar(labels, p50_ms, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_title("Forward Pass Latency (P50)\nLower is Better ↓", fontsize=11)
    ax.set_ylabel("Time (ms)")
    ax.set_yscale("log")
    for bar, val in zip(bars, p50_ms):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val * 1.15, f"{val:.1f}ms",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── Panel B: GFLOPS (higher is better) ───────────────────────────────────
    ax = axes[1]
    bars = ax.bar(labels, gflops, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_title("Throughput (GFLOPS)\nHigher is Better ↑", fontsize=11)
    ax.set_ylabel("GFLOPS")
    ax.set_yscale("log")
    for bar, val in zip(bars, gflops):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val * 1.15, f"{val:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── Panel C: Positions / second (higher is better) ───────────────────────
    ax = axes[2]
    bars = ax.bar(labels, pos_s, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_title("Positions Evaluated / sec\nHigher is Better ↑", fontsize=11)
    ax.set_ylabel("positions / second")
    ax.set_yscale("log")
    for bar, val in zip(bars, pos_s):
        label = f"{val/1000:.0f}K" if val >= 1000 else f"{val:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                val * 1.15, label,
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── Speedup annotations ───────────────────────────────────────────────────
    for ax_idx, (ax, vals) in enumerate(zip(axes, [p50_ms, gflops, pos_s])):
        ref = vals[0]  # scalar_cpp is baseline
        for i, val in enumerate(vals):
            su = val / ref if ax_idx > 0 else ref / val   # latency: lower ref
            if i > 0:
                ax.annotate(
                    f"{su:.0f}×",
                    xy=(i, vals[i]),
                    xytext=(i, vals[i] * (0.55 if ax_idx == 0 else 2.0)),
                    ha="center",
                    fontsize=9,
                    color=colors[i],
                    fontweight="bold",
                )

    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / "alphazero_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {out}")
    plt.close(fig)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_table(cpp_data: dict, numpy_data: dict) -> None:
    cfg    = cpp_data["config"]
    metal  = cpp_data["metal_gpu"]
    scalar = cpp_data["scalar_cpu"]

    speedup_vs_scalar = scalar["p50_ms"] / metal["p50_ms"]
    speedup_vs_numpy  = numpy_data["p50_ms"] / metal["p50_ms"]

    print()
    print("  ┌────────────────────────────────────────────────────────────────┐")
    print(f"  │  AlphaZero Inference Benchmark  "
          f"(batch={cfg['batch_size']}, dim={cfg['feature_dim']}, blocks={cfg['n_blocks']})   │")
    print("  ├──────────────────────┬──────────┬──────────┬──────────────────┤")
    print("  │ Backend              │  P50(ms) │  GFLOPS  │  pos/sec         │")
    print("  ├──────────────────────┼──────────┼──────────┼──────────────────┤")
    print(f"  │ Scalar C++ (-O3)     │ {scalar['p50_ms']:>8.1f} │ {scalar['gflops']:>8.2f} │ {scalar['positions_per_sec']:>16,.0f} │")
    print(f"  │ NumPy FP32 (BLAS)    │ {numpy_data['p50_ms']:>8.1f} │ {numpy_data['gflops']:>8.1f} │ {numpy_data['positions_per_sec']:>16,.0f} │")
    print(f"  │ Metal GPU FP16 (ours)│ {metal['p50_ms']:>8.1f} │ {metal['gflops']:>8.1f} │ {metal['positions_per_sec']:>16,.0f} │")
    print("  ├──────────────────────┴──────────┴──────────┴──────────────────┤")
    print(f"  │  Speedup vs Scalar C++:  {speedup_vs_scalar:>6.1f}×                              │")
    print(f"  │  Speedup vs NumPy BLAS:  {speedup_vs_numpy:>6.1f}×                              │")
    print(f"  │  GPU P99 latency:        {metal['p99_ms']:>6.2f}ms  (jitter = {metal['p99_ms'] - metal['p50_ms']:.2f}ms)          │")
    print("  └────────────────────────────────────────────────────────────────┘")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  AlphaZero Inference Benchmark — Metal FP16 vs NumPy FP32   ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()

    _check_binary()

    # 1. C++ Metal simulation
    cpp_data = run_cpp_simulation(batch_size=256)
    cfg = cpp_data["config"]

    # 2. NumPy baseline
    print(f"\n  Running NumPy FP32 baseline (batch={cfg['batch_size']})…")
    numpy_data = numpy_forward_pass(
        batch_size  = cfg["batch_size"],
        feature_dim = cfg["feature_dim"],
        n_blocks    = cfg["n_blocks"],
        policy_size = cfg["policy_size"],
        n_samples   = 20,
    )
    print(f"  NumPy P50: {numpy_data['p50_ms']:.2f}ms  ({numpy_data['gflops']:.1f} GFLOPS)")

    # 3. Print comparison table
    print_table(cpp_data, numpy_data)

    # 4. Generate chart
    print("  Generating comparison chart…")
    make_chart(cpp_data, numpy_data)

    print("  Done.\n")


if __name__ == "__main__":
    main()
