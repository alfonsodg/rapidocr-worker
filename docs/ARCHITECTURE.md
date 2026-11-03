# Architecture: rapidocr-worker

`rapidocr-worker` is a high-performance, standalone OCR worker written in Rust. It eliminates Python interpreter overhead and Global Interpreter Lock (GIL) contention, providing sub-millisecond socket IPC and high-throughput document recognition on multicore CPUs.

## System Overview

```text
+---------------------+
|   Socket client     |
|   (e.g. async Py)   |
+----------+----------+
           | UNIX Domain Socket (length-prefixed framing)
           v
+-------------------------------------------------------------+
|                     rapidocr-worker (Rust)                  |
|                                                             |
|   +-----------------------------------------------------+   |
|   | 1. Image Decode & Resize (736px min side length)     |   |
|   +--------------------------+--------------------------+   |
|                              |                              |
|   +--------------------------v--------------------------+   |
|   | 2. Text Detection (DBNet ONNX Session)              |   |
|   |    - DBPostProcess: Contours + Unclip + Calipers    |   |
|   +--------------------------+--------------------------+   |
|                              |                              |
|   +--------------------------v--------------------------+   |
|   | 3. Reading Order Sorter (Top-to-Bottom, Left-to-R.) |   |
|   +--------------------------+--------------------------+   |
|                              |                              |
|   +--------------------------v--------------------------+   |
|   | 4. Perspective Crop & Orientation Classifier (CLS)  |   |
|   |    - 4-point homography warp + 180° rotation check  |   |
|   +--------------------------+--------------------------+   |
|                              |                              |
|   +--------------------------v--------------------------+   |
|   | 5. Text Recognition (PP-OCRv5 Latin ONNX Session)   |   |
|   |    - Dynamic Width Batching + OpenCV Bilinear Resize|   |
|   +--------------------------+--------------------------+   |
|                              |                              |
|   +--------------------------v--------------------------+   |
|   | 6. CTC Greedy Decoder (Lines metadata + Space token)|   |
|   +--------------------------+--------------------------+   |
|                              |                              |
|   +--------------------------v--------------------------+   |
|   | 7. Text & Confidence Aggregation (JSON response)    |   |
|   +-----------------------------------------------------+   |
+-------------------------------------------------------------+
```

## Detailed Pipeline Stages

### 1. Image Ingestion and Preprocessing

The incoming image bytes are decoded via `image::load_from_memory` into RGB format. The image is scaled proportionally so that its shorter side reaches `--limit-side-len` (default 736px, `limit_type=min` parity with Python) and both dimensions are aligned to multiples of 32 for optimal tensor stride alignment.

### 2. Text Detection (`det.rs`, `contours.rs`, `geometry.rs`)

* The normalized input tensor `[1, 3, H, W]` runs through the DBNet detection model.
* The probability map output is binarized against the detection threshold.
* Contours are extracted via single-pass union-find connected components and Moore boundary tracing (matching OpenCV `RETR_LIST` and `CHAIN_APPROX_SIMPLE`).
* Contours are expanded using parallel polygon offset (clipper distance unclip formula) and enclosed in minimum area bounding boxes using rotating calipers.

### 3. Reading Order Box Sorting (`order.rs`)

Text boxes are grouped into horizontal reading bands based on vertical overlap and threshold comparisons. Within each band, boxes are sorted left-to-right, ensuring natural multi-column document reading order and preventing paragraph scrambling.

### 4. Perspective Rectification & Direction Classification (`perspective.rs`, `cls.rs`)

* Each detected quad is perspective-transformed into a rectified rectangular image patch using 4-point perspective warping.
* Vertical crops are rotated 90° counterclockwise prior to classification.
* **Sampled Orientation Fast Path (opt-in)**: When `--cls-sampled` is enabled, the first sample chunk (6 crops after aspect-ratio sorting) is evaluated. Only if every sample is confidently upright (>0.9) do the remaining boxes bypass CLS inference.
* **Adaptive Fallback**: Any inverted or mixed evidence in the sample falls back to exhaustive per-box classification for the remaining boxes, matching the Python engine. The fast path is disabled by default (`--cls-sampled` to enable, `--cls-full` to force per-box, `--no-cls` to skip the stage).

### 5. Text Recognition with Dynamic Width & Bucketing (`rec.rs`, `batch_runner.rs`)

* **Aspect-Ratio Bucketing**: Rectified text crops are partitioned into aspect ratio buckets (compact $\le 6.67$, normal $6.67 < \text{ratio} \le 12.0$, and wide $> 12.0$) before batching.
* **Dynamic Batch Scaling**: Because compact crops clamp to the minimum tensor width ($320\text{ px}$), they are packed into larger batches (up to $2\times$ base batch size, e.g. 12), maximizing GEMM compute density without padding waste. Wide crops use base batch size (e.g. 6) to cap peak memory.
* Batch width is dynamically computed as `(REC_HEIGHT * max_ratio).ceil()`, allowing arbitrary line widths without squashing text.
* Resizing uses exact OpenCV-compatible bilinear interpolation (matching Python OpenCV resize logits and eliminating anti-aliasing blurring).

### 6. CTC Greedy Decoding (`rec.rs`)

* The raw 3D output tensor `[batch_size, timesteps, num_classes]` is decoded using CTC greedy search.
* The character dictionary is loaded from embedded ONNX metadata using line-delimited parsing (`lines()`) and decoded with a blank at index 0 plus a trailing space token, preserving exact character indices for the deployed Latin model.

## Concurrency & Threading Model

To avoid CPU oversubscription and maximize multicore throughput:

* **Inter-op parallelism**: Fixed strictly to `1` per ONNX session.
* **Intra-op parallelism**: Configurable via `--threads` (default: 4 threads per session).
* **Recognition Session Pool**: Recognition sessions are pre-allocated in a pool sized by `--rec-workers`. The intra-op thread budget is divided evenly across workers (`threads / rec_workers`, minimum 1), eliminating intra-op synchronization stalls.
* **Scoped Worker Threads**: In multi-worker mode (`--rec-workers > 1`), recognition batches are processed concurrently using scoped threads (`std::thread::scope`) with zero mutex contention.
* **Process isolation**: Each worker process runs independently under systemd (`rapidocr-rust-worker@.service`), bound to its own UNIX domain socket. No shared memory exists between worker processes.

## Security Considerations

* **AF_UNIX sockets**: Restrict access to local users via filesystem permissions (`RuntimeDirectoryMode=0770`, owned by the service user running the unit).
* **Systemd hardening**: Worker unit files enforce `NoNewPrivileges=true`, `RestrictAddressFamilies=AF_UNIX`, and `ReadWritePaths=/run/rapidocr-rust-%i`.
