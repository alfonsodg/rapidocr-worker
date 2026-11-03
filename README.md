# rapidocr-worker

[![CI](https://github.com/alfonsodg/rapidocr-worker/actions/workflows/ci.yml/badge.svg)](https://github.com/alfonsodg/rapidocr-worker/actions/workflows/ci.yml)

High-throughput, standalone text-recognition worker written in Rust using ONNX Runtime (`ort`). It listens on a UNIX domain socket, processes images through the full PP-OCR pipeline (Detection + Classification + Latin/Spanish Recognition) with strictly bounded CPU threads, and returns extracted text and confidence scores in JSON.

Runs as an isolated UNIX-socket daemon with bounded memory and no Python GIL contention, allowing sustained parallel processing across multiple worker instances. Parity and performance against the Python engine are tracked with measured evidence in issues #6 and #8.

## Core Features

- **Parity Under Measurement**: Preserves spaces, accents (`á, é, í, ó, ú`) and `ñ` with the Latin model (`ppocrv5_latin_rec_mobile.onnx`). The parity harness (`scripts/parity_harness.py`) measured a mean similarity of 0.9695 on the 22 in-scope samples (run recorded in #8) against the `>= 0.99` gate agreed in #6; the gate is not met yet.
- **Full Latin & Spanish Support**: Full preservation of spaces, accents (`á, é, í, ó, ú`), and `ñ` using `ppocrv5_latin_rec_mobile.onnx`.
- **Natural Reading Order**: Automatic horizontal band grouping and left-to-right sorting of detected text blocks.
- **Aspect-Ratio Bucketing & Adaptive Batching**: Groups crops by aspect ratio and dynamically scales compact batches up to 2x base size, minimizing zero-padding overhead.
- **Optional Sampled Direction Classification**: Opt-in page orientation consensus (disabled by default) that skips per-box CLS on confidently upright scans, with automatic per-box fallback on any inverted or mixed evidence.
- **Session-Pooled Concurrency**: Scoped worker threads distribute recognition batches across CPU cores without thread barrier contention.
- **Drop-in Protocol Compatibility**: Implements a length-prefixed socket contract that any orchestrator client can speak.

## Quick Start

### Build

```bash
# Clone the repository
git clone git@github.com:alfonsodg/rapidocr-worker.git
cd rapidocr-worker

# Build optimized release binary
cargo build --release

# Run unit tests
cargo test
```

### Run Locally

```bash
./target/release/rapidocr-worker \
  --socket /run/rapidocr-rust-1/worker.sock \
  --det-model /opt/models/ch_PP-OCRv4_det_infer.onnx \
  --cls-model /opt/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --rec-model /opt/models/ppocrv5_latin_rec_mobile.onnx \
  --limit-side-len 736 \
  --threads 4
```

## Socket Protocol

Communication runs over UNIX domain stream sockets (`AF_UNIX`).

1. **Request Frame**:
   - `>II` header (4-byte `meta_len` + 4-byte `image_len`, big-endian).
   - `meta_len` bytes: UTF-8 JSON metadata (e.g. `{"media_type":"image/png"}`).
   - `image_len` bytes: Raw image data.

2. **Response Frame**:
   - `>I` header (4-byte `payload_len`, big-endian).
   - `payload_len` bytes: UTF-8 JSON payload:

     ```json
     {"ok": true, "text": "Recognized text...", "confidence": 0.985, "elapsed_s": 0.34}
     ```

## Documentation

Detailed documentation is available in the [`docs/`](docs/) directory:

- [Architecture Guide](docs/ARCHITECTURE.md): System components, pipeline stages, and threading model.
- [Socket API Specification](docs/API.md): Binary frame format, JSON schemas, and client examples.
- [Configuration Reference](docs/CONFIGURATION.md): Command-line options, environment variables, and client integration settings.
- [Installation & Deployment Guide](docs/INSTALL.md): Prerequisites, build steps, systemd setup, and host deployment.
- [Memory Optimization Report](docs/MEMORY_OPTIMIZATION.md): Root-cause analysis, TDD verification, and production benchmarks for RSS bounding (#19).
- [Development Standards](docs/STANDARDS.md): Rust code quality, error handling, and testing policies.
- [Mermaid Diagrams](docs/DIAGRAMS.md): High-level sequence and flow diagrams.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Alfonso de la Guarda.
