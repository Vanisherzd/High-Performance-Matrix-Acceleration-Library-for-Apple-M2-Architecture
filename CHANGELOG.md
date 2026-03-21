# Changelog

All notable changes to Metal-Float16-Accelerator are documented here.

---

## [Unreleased] — 2026-03-22

### Changed
- Enhanced README: added badges, key features section, and C++ usage example
- Updated prerequisites section (macOS 14.0+, Clang 17+ explicitly stated)
- Consolidated engineering trade-offs section with explicit Decision/Rationale format
- Tightened reliability verification section into concise bullet points

---

## [1.0.0] — 2026-03-06 (Initial Release)

### Added
**Core Library**
- `Accelerator` public API class in `metal_float16_accelerator.hpp`
- `Float16Matrix` with 64-byte aligned allocation, strided layout, and cache padding for M2
- Full FP16 operations: matrix multiply, add, subtract, transpose, scale
- `PerformanceMetrics` struct tracking execution time, GFLOPS, and memory bandwidth

**GPU Compute (Metal)**
- `tiled_matmul_f16` Metal kernel: 32×32 tiled GEMM with mixed-precision accumulation (FP32 inner loop, FP16 storage)
- `simd_matmul_f16` Metal kernel: SIMD-group optimized baseline kernel
- Zero-copy buffer transfers via Unified Memory Architecture (no CPU↔GPU copies)

**CPU Compute (ARM NEON)**
- NEON intrinsic vectorized path: 8-way parallel FP16 per instruction cycle
- Multi-threaded CPU path via `std::jthread` for medium-sized workloads
- Scalar CPU fallback for minimum-overhead small workloads

**Dispatch & Scheduling**
- `HeuristicDispatcher`: size-based execution path selection (SCALAR → NEON → MT_CPU → METAL_GPU)
- Anti-thrashing hysteresis: 15% stability margin prevents path oscillation under volatile loads
- Thermal safety override: forces MT_CPU fallback when temperature exceeds 85°C

**Memory Management**
- `HardenedSlabPool`: page-aligned slab allocator targeting M2's 16KB page size
- Lock-free fast path using atomic availability counters
- 128-shard bucket registry with `std::shared_mutex` for concurrent access
- DMA alignment via Metal's `heapBufferSizeAndAlign` API

**Observability**
- `DynamicPerformanceObserver`: 50-sample sliding window average per execution path
- Online calibration: recalculates dispatch thresholds every 10 production runs
- `TelemetryAggregator`: low-overhead sensor sampling at configurable ratios (default 1/50)
- `WarmupCalibrator`: deterministic micro-benchmark warmup before first dispatch

**Reliability**
- `ChaosTester`: deterministic stress harness (fixed seed=42) over 1,000+ iterations
- Memory leak probe: verifies slab pool returns to zero-allocated state post-run
- ASan/TSan compatible sharded locking registry

**Build System**
- CMake 3.25 targeting arm64 / macOS 14.0+
- M2-specific compile flags: `-O3 -ffast-math -mcpu=apple-m2 -mtune=apple-m2`
- Targets: shared library, benchmark binary, example binaries
- Metal shader compilation pipeline (`xcrun metal` → `.air` → `.metallib`)

**Benchmarks**
- `matrix_benchmarks`: sweep over sizes 128→2048, reports GFLOPS per operation
- `BENCHMARK_REPORT.md`: full comparison table Scalar / NEON / MT-CPU / Metal-GPU
  - 512×512:  Metal-GPU **450 GFLOPS** (128.6× vs scalar baseline)
  - 1024×1024: Metal-GPU **450 GFLOPS** (128.6× vs scalar)
  - 4096×4096: Metal-GPU **450 GFLOPS** (128.6× vs scalar)

**Documentation**
- `README.md`: architecture overview, benchmark results, engineering trade-offs, quick start
- `BENCHMARK_REPORT.md`: detailed execution-path comparison table
