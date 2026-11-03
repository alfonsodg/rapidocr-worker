# RapidOCR Rust Worker: Memory Optimization & Engineering Report

Comprehensive post-mortem and technical report detailing the root-cause
investigation, memory allocator mechanics, TDD implementation, and production
verification for RapidOCR memory stabilization (#19).

---

## 1. Executive Summary

In high-throughput document ingestion pipelines, the RapidOCR worker service
suffered from severe, progressive resident set size (RSS) retention:

- **Historical Peak**: Up to **38 GB of RAM** retained across 8 worker processes
  following dense engineering drawing ingestion.
- **Initial Mitigation (`--no-arena`)**: Partially lowered retention, but
  workers still leveled off at **156–302 MB RSS per worker** (`[heap]` pinned
  at 120.9–283.5 MB) after processing complex documents.
- **Root Cause**: Glibc heap fragmentation and break-pointer (`sbrk`) pinning.
  Intermediate stage allocations (bounding boxes, crops, RGB buffers) were
  allocated *above* the detection stage's transient memory peak in the
  contiguous heap, preventing `malloc_trim(0)` from returning top chunks.
- **Engineering Fix**: Strict reverse-lifetime variable scoping in
  `OcrEngine::process` (`src/engine.rs`) combined with immediate stage drops
  and an explicit `trim_allocator_heap()` invocation (`src/session.rs`).
- **Production Results**:
  - **Memory Reduction**: Bounded, flat post-load RSS of **~73.7 MB per worker**
    (589.9 MB total for all 8 workers, down from 38 GB originally and 1,352 MB
    pre-fix).
  - **Zero Latency Regression**: -0.7% to -4.8% faster execution due to zero
    syscall churn.
  - **Accuracy Parity**: 100% byte-identical text and confidence scores.

---

## 2. Problem Statement & Containment History

### 2.1 The ONNX Runtime Arena Retention Trap

RapidOCR utilizes ONNX Runtime (`libonnxruntime.so.1.30.0`) for detection
(`PP-OCRv6_small_det`), classification (`ch_ppocr_mobile_v2.0_cls`), and
recognition (`PP-OCRv6_small_rec`).

By default, ONNX Runtime enables its internal memory arena
(`OrtArenaCfg`). While efficient for repetitive fixed-size inference, variable-
resolution document processing creates an escalating memory high-water mark:

1. An A0 blueprint or dense contract produces thousands of text candidate
   boxes.
2. The arena allocator expands monotonically to allocate intermediate scratch
   buffers for detection and recognition tensors.
3. The arena never shrinks back to the operating system, permanently claiming
   2.6–10.5 GB per worker.

### 2.2 Legacy Containment Measures

Prior to this engineering fix, the production cluster relied on defensive
containment rather than a structural resolution:

- **ZRAM Swap**: 48 GB compressed swap partition on the production host.
- **Systemd Limits**: `MemoryHigh=6G` set per worker unit.
- **Nightly Restarts**: A cron job (`rapidocr-keeper`) restarted all workers at
  08:15 UTC daily.

### 2.3 Failed Alternative Mitigations

Before investigating internal lifetime scoping, alternative allocator controls
were benchmarked and rejected:

- **`MALLOC_MMAP_THRESHOLD_=131072`**:
  Forcing glibc to use `mmap` for allocations $\ge 128\text{ KiB}$ prevented
  heap retention, but degraded latency by **+74%** (4,134 ms $\to$ 7,189 ms)
  due to continuous kernel page zeroing and cross-core TLB shootdowns in inner
  operator loops.
- **`jemalloc` (`libjemalloc.so.2`)**:
  Suffered from severe thread-cache contention and extent lock contention across
  the 4 intra-op inference threads, causing a **+28%** latency regression
  (4,482 ms $\to$ 5,730 ms).

---

## 3. Root Cause: Glibc Top-Chunk Pinning

The core defect was identified in `rapidocr-worker/src/engine.rs` inside the
primary execution method:

```rust
pub fn process(&self, img_bytes: &[u8]) -> Result<OcrResult, OcrError>
```

### 3.1 Glibc Heap Mechanics

Glibc manages the program heap as a contiguous segment adjusted via the
`brk`/`sbrk` system calls. When memory is freed via `free()`, it is returned to
glibc's internal bins. However, the operating system can only reclaim virtual
memory from the top of the heap (the break pointer). If even a small 64-byte
allocation remains active at the highest address, no memory below it can be
released via `malloc_trim(0)`.

### 3.2 Overlapping Lifetimes in `engine.rs`

In the original implementation:

1. `resized` (`cv_resize_bilinear`) was kept alive while `det_tensor` was
   allocated and evaluated.
2. `det_tensor` (~15.5 MB) was declared in the outer scope and remained alive
   until `process()` exited.
3. **The Critical Pin**: `self.post.boxes_from_bitmap(...)` was executed
   **while `det_out` (`SessionOutputs`) was still in scope**. `det_out` held the
   ~200 MB intermediate execution frame from detection at the top of the heap.
   Consequently, `bx` (`Vec<BoundingBox>`), `crops`, and internal recognition
   data structures were allocated at addresses *above* the detection peak.
4. `img` (the full decoded RGB image) and `boxes` remained allocated throughout
   recognition.

Because long-lived stage outputs sat physically above the detection buffer in
the linear heap, `malloc_trim(0)` was completely ineffective, leaving 120–280
MB permanently stranded.

---

## 4. Implementation: Reverse-Lifetime Scoping

The fix was developed with strict TDD and merged into `develop`.

### 4.1 Allocator Trimming Primitive (`src/session.rs`)

A native glibc trimming function was exposed:

```rust
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn trim_allocator_heap() -> bool {
    // malloc_trim(0) returns 1 if memory was actually returned to the OS.
    unsafe { libc::malloc_trim(0) == 1 }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub fn trim_allocator_heap() -> bool {
    false
}
```

### 4.2 Stage Isolation (`src/engine.rs`)

The execution sequence was rewritten with strict reverse-lifetime blocks:

1. **Detection Preprocessing Scoping**:
   `resized` is constructed and converted into `det_tensor` inside an isolated
   scope so `resized` drops before inference runs.

2. **Detection Output Decoupling**:
   `det_tensor` and `det_out` are evaluated inside an inner block. The raw
   probability bitmap slice is copied to a compact vector, and `det_out` is
   dropped *before* `boxes_from_bitmap` allocates bounding box polygons.

3. **Early Drops of Raw Buffers**:
   Once crops are extracted from the source image, both `img` and `boxes` are
   explicitly destroyed:

   ```rust
   let crops = self.extract_crops(&img, &boxes)?;
   drop(boxes);
   drop(img);
   ```

4. **Post-Recognition Trim**:
   Once recognition finishes, `crops` is immediately dropped and
   `trim_allocator_heap()` is invoked:

   ```rust
   let rec_results = self.recognize_crops(&crops)?;
   drop(crops);
   trim_allocator_heap();
   ```

Because no downstream allocations exist above the detection high-water mark at
the moment `trim_allocator_heap()` executes, glibc safely rolls the break
pointer back to baseline.

---

## 5. Test-Driven Development (TDD) Verification

Strict TDD was enforced prior to making production code modifications.

### 5.1 Test Suite Creation

- **Python Integration Contract**:
  `tests/test_session_memory_pattern.py` validates that worker processes
  start with `--no-arena` and maintain bounded RSS across cycles.
- **Rust Unit Test**:
  `tests/arena_config_19.rs` (`trim_allocator_heap_releases_freed_top_chunk`)
  allocates a 64 MB transient buffer, frees it, calls `trim_allocator_heap()`,
  and verifies heap retraction via `/proc/self/status`.

### 5.2 Verification Pipeline

- **Local Python Suite**: `pytest tests/` $\to$ `12 passed in 0.64s`.
- **Remote Cargo Suite**:
  - `cargo test` $\to$ `81 passed` (43 unit + 38 integration), `0 failed`.
  - `cargo clippy --all-targets -- -D warnings` $\to$ `0 warnings`.
  - `cargo fmt --check` $\to$ `0 diffs`.
- **CI Pipeline**: GitHub Actions (`.github/workflows/ci.yml`) runs the same
  gates on every push (`rust-test`, `rust-lint`, `markdownlint`, `shellcheck`).

---

## 6. Production Benchmarks & Verification

The binary was deployed to a production server (2× Intel Xeon Silver 4410Y,
48 logical threads).

### 6.1 Benchmark A: 6-Page Dense Corpus (`bench19`)

Measured across 6 dense engineering pages followed by 5 trivial rounds:

| Metric | Pre-fix | Post-fix | Delta |
| :--- | :--- | :--- | :--- |
| **Retained RSS** | 156 MB (heap 121M) | **74 MB** (heap 38M) | **-52.8%** |
| **Peak (`VmHWM`)** | 805.1 MB | **733.2 MB** | **-71.9 MB** |
| **Median Latency** | 4,327.9 ms | **4,298.8 ms** | **-0.7%** |
| **Mean Latency** | 7,193.5 ms | **7,179.4 ms** | **-0.2%** |
| **Parity** | Reference | **100% Identical** | **0 delta** |

- **Pre-Fix RSS Sequence**:
  228.6 $\to$ 258.8 $\to$ 168.6 $\to$ 180.2 MB.
- **Post-Fix RSS Sequence**:
  72.8 $\to$ 73.6 $\to$ 73.6 $\to$ 75.0 MB (flat).

### 6.2 Benchmark B: Consecutive Blueprint Stress (`6x plano-clamp-1.png`)

Subjected to repeated, back-to-back heavy engineering drawing queries:

- **Pre-Fix RSS Sequence**:
  `[186.1, 145.0, 127.3, 198.9, 140.0, 177.4] MB` (Mean latency: 23,333.6 ms).
- **Post-Fix RSS Sequence**:
  `[74.8, 75.0, 75.0, 75.0, 75.0, 75.0] MB` (**Zero drift**, Mean latency:
  22,205.2 ms, **-4.8% faster**).

### 6.3 Live Production Fleet Promotion

1. Release binary installed to `/usr/local/bin/rapidocr-worker` (timestamped
   backup kept alongside it).
2. Rolling restart executed for `rapidocr-rust-worker@{0..7}.service`.
3. Concurrent production traffic (`cert-denso-1.png` + `dni-1.png`) sent to all
   8 worker sockets:
   - **8/8 workers active**: post-load RSS strictly between **73.6 and 73.9 MB**.
   - **Total 8-worker fleet footprint**: **589.9 MB** (down from 38 GB
     originally).

---

## 7. CPU Topology & Scaling Guidelines

### 7.1 Current Production Footprint

On the dual Intel Xeon Silver 4410Y host:

- **Worker Configuration**: `--threads 4` (`intra_threads = 4`,
  `inter_threads = 1`).
- **Peak CPU Load**: 8 workers $\times$ 4 threads = **32 logical threads**.
- **Headroom**: **16 logical threads** permanently available for co-located
  services outside this worker.
- **Idle State**: 0.0% CPU (blocked on UNIX domain socket `accept`).

### 7.2 Scaling Recommendations

Because RAM is bounded at ~74 MB per worker, memory is no longer the scaling
bottleneck:

1. **Option 1 (Balanced Concurrency - Retained Default)**:
   Keep 8 workers $\times$ 4 threads. Optimal single-page latency and zero
   CPU starvation for co-located models.
2. **Option 2 (High Batch Concurrency)**:
   Expand to 10 workers $\times$ 4 threads (`rapidocr-rust-worker@{0..9}`, 40
   threads total, ~740 MB total RSS).
3. **Option 3 (High-Density Multi-Worker)**:
   Reconfigure to 12 workers $\times$ 2 threads (`rapidocr-rust-worker@{0..11}`,
   24 threads total, ~890 MB total RSS) for maximum batch throughput.

---

## 8. Artifact and Issue References

- **Issue**: #19 (memory stabilization).
- **Key Files**:
  - `src/engine.rs` (Scoped stage drops)
  - `src/session.rs` (Glibc heap trimming)
  - `tests/arena_config_19.rs` (TDD regression test)
  - `tests/test_session_memory_pattern.py` (Contract test)
