# DESIGN_DECISIONS.md

## Hardware-Software Co-Design Decisions

### 1. Page-Aligned Slab Bucketing
**Decision:** All buffer allocations are multiples of **16KB (Apple M2 Page Size)**.
**Rationale:** Aligning with hardware pages is critical for **Direct Memory Access (DMA)**. Straddling page boundaries requires the hardware to perform two memory transactions instead of one, introducing significant latency in high-throughput matrix operations. While this increases "internal fragmentation" for small matrices, the gain in GPU execution predictability far outweighs the memory overhead.

### 2. Anti-Thrashing Hysteresis
**Decision:** Implementation of a **15% Stability Threshold** for execution path dispatching.
**Rationale:** In high-performance systems, the cost of switching execution paths (e.g., from CPU to GPU) includes state synchronization and context switching. If the performance gain is marginal, the system may oscillate ("thrash") between paths, causing high execution variance. Hysteresis ensures we only switch when the benefit is deterministic.

### 3. Telemetry Debouncing
**Decision:** Moving Average Filter for thermal and GPU occupancy sensors.
**Rationale:** Hardware sensors are noisy. Transient spikes in thermal readings (e.g., from a momentary burst on a neighboring CPU core) should not trigger a library-wide fallback to CPU mode. Debouncing ensures the system reacts to **trends**, not electrical noise.

### 4. Deterministic Chaos Testing
**Decision:** Fixed-seed random number generation for reliability testing.
**Rationale:** "Random" crashes are the nightmare of system software. By using a fixed seed, we can reproduce the exact sequence of matrix sizes and system stressors that led to a failure, turning "flaky" bugs into deterministic, fixable ones.

### 5. Mixed-Precision Accumulation
**Decision:** Accumulating dot products in **32-bit float** before storing in **16-bit half**.
**Rationale:** Standard half-precision accumulation leads to catastrophic rounding errors for large matrices ($N > 512$). Mixed-precision provides the best balance of speed (using FP16 memory bandwidth) and accuracy (using FP32 ALU units), matching industry standards for AI model training.
