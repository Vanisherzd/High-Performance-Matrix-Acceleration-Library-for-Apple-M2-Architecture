# Metal-Float16-Accelerator: Hardware-Software Co-Design for High-Performance Linear Algebra

## Mission Statement
The Metal-Float16-Accelerator is a production-grade system library engineered to achieve peak floating-point throughput on Apple Silicon (M2) architectures. By leveraging the Unified Memory Architecture (UMA) and specialized compute units, the library provides a deterministic, high-concurrency framework for half-precision (FP16) matrix operations. The primary objective is to bridge the gap between high-level mathematical abstractions and low-level hardware execution through rigorous hardware-software co-design.

## Core Architecture
The system is architected as a multi-layered acceleration stack, ensuring optimal resource utilization across varying workloads.

### 1. Heuristic Dispatch Plane
A deterministic dispatcher evaluates workload dimensions (M, N, K) and real-time hardware telemetry. It utilizes **Hysteresis Logic** with a 15% stability margin to prevent execution path oscillation (thrashing) under volatile system loads.

### 2. Metal Compute Engine (GPU)
Implements **Tiled Matrix Multiplication** kernels optimized for the M2 GPU.
*   **Tile size optimization:** Uses 32x32 tiles aligned with the SIMD-group width.
*   **Threadgroup memory:** Caches sub-matrices to minimize global memory bandwidth pressure.
*   **Mixed-Precision Accumulation:** Performs dot-product accumulation in 32-bit float (FP32) to maintain numerical stability before downcasting to FP16 for storage.

### 3. Vectorized CPU Engine (NEON)
A high-performance fallback and small-workload path utilizing **ARM NEON Intrinsics**. It processes 8-way parallel FP16 operations per instruction cycle, distributed across performance cores via `std::jthread`.

### 4. Memory Management Plane (Sharded Slab Pool)
A kernel-grade allocator designed for the M2's **16KB Page Size**.
*   **DMA Alignment:** All buffers are perfectly aligned to page boundaries to eliminate straddled-page penalties during Direct Memory Access.
*   **Sharded Locking:** Implements a partitioned mutex strategy (32 shards) to reduce lock contention in high-concurrency environments.
*   **Slab Bucketing:** Uses 1.5x stepping to balance internal fragmentation against allocation speed.

## Key Engineering Trade-offs

### Page-Alignment vs. Memory Density
The library prioritizes **Hardware-Aligned Bucketing (16KB)** over memory density. While this introduces internal fragmentation for sub-page allocations, it ensures zero-latency DMA transfers and minimizes Translation Lookaside Buffer (TLB) misses, which are critical for predictable high-performance execution.

### Mixed-Precision vs. Pure FP16
To adhere to industry standards for numerical accuracy, the system performs inner-loop accumulation in FP32. This trade-off incurs a minor register pressure increase but prevents catastrophic rounding errors common in deep-learning workloads exceeding 512-dimension dot products.

### Asynchronous Telemetry vs. Synchronous Overhead
Monitoring is decoupled into an **Out-of-Band Background Watchdog**. By using a low-priority thread and atomic ring buffers, the system captures thermal trends and GPU occupancy without introducing latency into the critical computation path.

## Performance Methodology

### Metrics and Calculation
Performance is quantified using GFLOPS (Giga-Floating Point Operations Per Second), calculated as:
$$GFLOPS = \frac{2 \times M \times N \times K}{t_{duration} \times 10^9}$$
All measurements reflect **actual hardware execution time** by utilizing `waitUntilCompleted` on the Metal Command Buffer, ensuring CPU dispatch latency is excluded from the kernel throughput analysis.

### Dynamic Calibration
The system employs a **Sliding Window Average** observer. It continuously monitors production runs and updates dispatch thresholds in real-time, allowing the library to adapt to thermal throttling and concurrent system processes.

## Reliability Verification

### Deterministic Chaos Testing
A specialized stress harness simulates extreme system conditions, including:
*   **Memory Pressure:** Forcing aggressive slab evictions.
*   **Thermal Spikes:** Triggering fallback heuristics.
*   **Fixed-Seed Reproducibility:** Every chaos cycle uses a deterministic seed to ensure that edge-case failures can be replicated and analyzed.

### Memory Integrity
The library implements **Memory Leak Probes** that verify the Slab Pool returns to a zero-allocated state post-execution. Integration with AddressSanitizer (ASan) and ThreadSanitizer (TSan) ensures the absence of data races and illegal memory access in the sharded locking registry.

### Telemetry Debouncing
To prevent reactive instability, the system applies a **Moving Average Filter** to all hardware sensors. Thermal spikes shorter than 50ms are suppressed, ensuring the dispatcher reacts to sustained hardware trends rather than transient electrical noise.
