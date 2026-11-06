# rapidocr-worker

[![CI](https://github.com/alfonsodg/rapidocr-worker/actions/workflows/ci.yml/badge.svg)](https://github.com/alfonsodg/rapidocr-worker/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/alfonsodg/rapidocr-worker)](https://github.com/alfonsodg/rapidocr-worker/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/alfonsodg/rapidocr-worker/blob/main/LICENSE)

High-throughput, standalone text-recognition worker written in Rust on top of
ONNX Runtime (`ort`). It listens on a UNIX domain socket, runs the full PP-OCR
pipeline — detection, direction classification and Latin/Spanish recognition —
with strictly bounded CPU threads, and returns extracted text and confidence
scores as JSON.

It runs as an isolated daemon with bounded memory and no Python GIL contention,
so multiple instances can serve sustained parallel traffic on a single host.

## Features

- **Bounded memory**: inference scratch memory is returned to the OS after each
  page; measured post-load RSS stays flat at ~74 MB per worker under dense
  workloads (see the [memory report](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/MEMORY_OPTIMIZATION.md) and #19).
- **Latin/Spanish fidelity**: preserves spaces, accents (`á, é, í, ó, ú`) and
  `ñ`. Parity against the Python engine is measured with
  `scripts/parity_harness.py`: mean similarity 0.9695 on the 22 in-scope
  samples against the `>= 0.99` gate agreed in #6 (tracked in #8; the gate is
  not met yet).
- **Natural reading order**: horizontal band grouping and left-to-right sorting
  of detected text blocks.
- **Aspect-ratio bucketing & adaptive batching**: groups crops by aspect ratio
  and scales compact batches up to 2x base size, minimizing zero-padding
  overhead.
- **Optional sampled direction classification**: opt-in page orientation
  consensus (`--cls-sampled`) that skips per-box CLS on confidently upright
  scans, with automatic per-box fallback on any inverted or mixed evidence.
- **Session-pooled concurrency**: scoped worker threads distribute recognition
  batches across CPU cores without thread-barrier contention.
- **PP-OCRv4 and PP-OCRv6 profiles**: `--det-profile`, `--det-normalization`
  and `--keys` support v6 models, including external character dictionaries.
- **Simple protocol**: length-prefixed UNIX socket framing that any orchestrator
  client can speak (see [API.md](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/API.md)).

## Requirements

- Linux x86_64 (builds with Rust 1.89+, pinned in `rust-toolchain.toml`).
- ONNX Runtime shared library (`libonnxruntime.so`), loaded dynamically at
  runtime through `ORT_DYLIB_PATH`.
- PP-OCR ONNX models: detection, direction classification and a Latin/Spanish
  recognizer (see [INSTALL.md](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/INSTALL.md) for candidates and sizes).

## Install

### Prebuilt release binary

Download the archive for your system from the
[Releases](https://github.com/alfonsodg/rapidocr-worker/releases/latest) page
and verify it against `SHA256SUMS`:

```bash
tar -xzf rapidocr-worker-v<version>-x86_64-unknown-linux-gnu.tar.gz
sudo install -m 0755 rapidocr-worker-v<version>-x86_64-unknown-linux-gnu/rapidocr-worker /usr/local/bin/
```

The archive also ships the systemd unit template (`rapidocr-worker@.service`),
the README and the license.

The same archive is available from GitHub Packages as an OCI artifact:

```bash
oras pull ghcr.io/alfonsodg/rapidocr-worker:latest
```

### Build from source

```bash
git clone git@github.com:alfonsodg/rapidocr-worker.git
cd rapidocr-worker

# Build the optimized release binary
cargo build --release

# Run the test suite
cargo test
```

## Run

```bash
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
./target/release/rapidocr-worker \
  --socket /run/rapidocr-rust-1/worker.sock \
  --det-model /opt/models/ch_PP-OCRv4_det_infer.onnx \
  --cls-model /opt/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --rec-model /opt/models/ppocrv5_latin_rec_mobile.onnx \
  --limit-side-len 736 \
  --threads 4
```

### Run as a service

A hardened systemd template unit lives in
[`configs/systemd/rapidocr-worker@.service`](https://github.com/alfonsodg/rapidocr-worker/blob/main/configs/systemd/rapidocr-worker@.service).
It runs one worker instance per `%i` index, each with its own socket under
`/run/rapidocr-rust-%i/`. Step-by-step host instructions (binary install, unit
setup, scaling across CPU cores) are in [INSTALL.md](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/INSTALL.md).

### Spanish checkpoints

The recognizer accepts any PP-OCR-compatible ONNX model. A Spanish-tuned pair —
a 116-character vocabulary plus two fine-tuned checkpoints — is published
separately and loads through the existing flags, with no code change:

```bash
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
./target/release/rapidocr-worker \
  --socket /run/rapidocr-rust-1/worker.sock \
  --det-model /opt/models/ch_PP-OCRv4_det_infer.onnx \
  --cls-model /opt/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --rec-model /opt/models/spanish_rec_v18.onnx \
  --keys /opt/models/ppocr_keys_es.txt \
  --threads 4
```

`--keys` is mandatory for this pair: the checkpoint carries no embedded
character metadata, and loading it with another vocabulary corrupts CTC decoding
silently — the output still looks like text, with shifted indices.

| asset | where |
| --- | --- |
| `spanish_rec_v18.onnx` | RapidOCR release `spanish-multidoc-v1` |
| `spanish_rec_v19.onnx` | same release (alternative for receipts and records) |
| `ppocr_keys_es.txt` | same release, and committed in the RapidOCR branch |

Release: <https://github.com/alfonsodg/RapidOCR/releases/tag/spanish-multidoc-v1>
Hugging Face: <https://huggingface.co/alfonsodlg/rapidocr-spanish>

Measured against the stock Latin checkpoint on 3,079 held-out crops (44,947
reference characters, 14 Spanish document types): CER micro 0.1476 against
0.1745, macro 0.1238 against 0.1412.

## Configuration

The most common flags; the full reference with defaults and semantics is in
[CONFIGURATION.md](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/CONFIGURATION.md).

| Flag | Default | Purpose |
|---|---|---|
| `--socket <path>` | required | UNIX domain socket to listen on |
| `--det-model`, `--cls-model`, `--rec-model` | required (det, rec) | ONNX model paths |
| `--threads <n>` | `4` | Intra-op thread budget, split across recognition sessions |
| `--limit-side-len <n>` | `736` | Shorter image side is scaled to this before detection (`limit_type=min`) |
| `--rec-batch <n>` | `6` | Base recognition batch size; compact crops scale up to 2x |
| `--rec-workers <n>` | `1` | Parallel recognition session workers (scoped threads) |
| `--no-arena` | off | Disable the ONNX CPU memory arena to bound RSS (#19) |
| `--det-profile`, `--det-normalization` | `ppocrv4`, `scalar` | Detection post-processing and normalization profiles (PP-OCRv4/v6) |
| `--keys <path>` | — | External character dictionary for models without embedded metadata |

Environment variables: `ORT_DYLIB_PATH` (required), `RUST_LOG` (optional,
default `info`).

## Socket Protocol

Communication runs over UNIX domain stream sockets (`AF_UNIX`).

1. **Request frame**:
   - `>II` header: 4-byte `meta_len` + 4-byte `image_len`, big-endian.
   - `meta_len` bytes: UTF-8 JSON metadata (e.g. `{"media_type":"image/png"}`).
   - `image_len` bytes: raw image data (PNG, JPEG, TIFF, BMP, WebP).

2. **Response frame**:
   - `>I` header: 4-byte `payload_len`, big-endian.
   - `payload_len` bytes: UTF-8 JSON payload.

   ```json
   {
     "ok": true,
     "text": "Recognized line 1\nRecognized line 2",
     "confidence": 0.985,
     "elapsed_s": 0.34,
     "det_ms": 210.4,
     "cls_ms": 12.7,
     "rec_ms": 119.0
   }
   ```

The complete framing specification, error responses and a reference Python
client are in [API.md](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/API.md).

## Performance

Measured on a dual-socket Xeon host (48 logical threads, 8 worker instances):

- Retained RSS is flat at ~73.7 MB per worker after dense engineering pages
  (589.9 MB for the whole 8-worker fleet, down from multi-GB retention before
  the arena fix); latency unchanged or slightly better, with byte-identical
  output. Full numbers: [memory report](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/MEMORY_OPTIMIZATION.md).
- Recognition dominates runtime; the worker's USPs are bounded memory,
  deterministic behavior and isolation, not raw speed versus the Python engine.

## Development

```bash
cargo test                        # Rust unit + integration tests
cargo clippy --all-targets -- -D warnings
cargo fmt --check
pytest tests/                     # Python contract tests
bash scripts/install-hooks.sh     # optional local git hooks
```

Commits follow Conventional Commits with an issue reference
(`type(scope): subject (#issue)`).

## Documentation

- [Architecture Guide](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/ARCHITECTURE.md): System components, pipeline stages, and threading model.
- [Socket API Specification](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/API.md): Binary frame format, JSON schemas, and client examples.
- [Configuration Reference](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/CONFIGURATION.md): Command-line options, environment variables, and client integration settings.
- [Installation & Deployment Guide](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/INSTALL.md): Prerequisites, build steps, systemd setup, and host deployment.
- [Memory Optimization Report](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/MEMORY_OPTIMIZATION.md): Root-cause analysis, TDD verification, and production benchmarks for RSS bounding (#19).
- [Development Standards](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/STANDARDS.md): Rust code quality, error handling, and testing policies.
- [Mermaid Diagrams](https://github.com/alfonsodg/rapidocr-worker/blob/main/docs/DIAGRAMS.md): High-level sequence and flow diagrams.

## License

Licensed under the [Apache License, Version 2.0](https://github.com/alfonsodg/rapidocr-worker/blob/main/LICENSE).
Copyright 2026 Alfonso de la Guarda.
