# Installation and Deployment Guide

This guide describes how to build, install, and run `rapidocr-worker` on Linux systems.

## Prerequisites

1. **Rust Toolchain**: Rust 1.84+ (`cargo`, `rustc`).
2. **ONNX Runtime Dynamic Library (`libonnxruntime.so`)**:
   The worker dynamically loads ONNX Runtime via `ORT_DYLIB_PATH`. Point it at
   the library on your system (the `onnxruntime` Python wheel ships one under
   `onnxruntime/capi/`), for example:

   ```text
   /usr/local/lib/libonnxruntime.so
   ```

3. **ONNX Model Weights**:
   - Detection model: `ch_PP-OCRv4_det_infer.onnx` (4.7 MB)
   - Direction classification model: `ch_ppocr_mobile_v2.0_cls_infer.onnx` (1.5 MB)
   - Recognition model (Latin/Spanish): `ppocrv5_latin_rec_mobile.onnx` (7.7 MB; character dictionary embedded in ONNX metadata)

## Build from Source

Clone the repository and build the release binary:

```bash
git clone git@github.com:alfonsodg/rapidocr-worker.git
cd rapidocr-worker

# Build optimized release binary
cargo build --release

# Binary generated at target/release/rapidocr-worker
ls -lh target/release/rapidocr-worker
```

Run test suite to verify binary integrity:

```bash
cargo test
cargo clippy -- -D warnings
```

## Production Installation

### 1. Install Worker Binary

Copy the compiled release binary to `/usr/local/bin`:

```bash
sudo install -m 0755 target/release/rapidocr-worker /usr/local/bin/rapidocr-worker
```

### 2. Configure Systemd Template Unit

Copy the template unit file to `/etc/systemd/system/`:

```bash
sudo cp configs/systemd/rapidocr-worker@.service /etc/systemd/system/rapidocr-rust-worker@.service
sudo systemctl daemon-reload
```

Inspect the unit configuration in `/etc/systemd/system/rapidocr-rust-worker@.service`:

```ini
[Unit]
Description=Rust RapidOCR worker %i
After=network.target

[Service]
Type=simple
User=rapidocr
Group=rapidocr
WorkingDirectory=/opt/rapidocr-worker
Environment=ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ExecStart=/usr/local/bin/rapidocr-worker \
  --socket /run/rapidocr-rust-%i/worker.sock \
  --det-model /opt/models/ch_PP-OCRv4_det_infer.onnx \
  --cls-model /opt/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --rec-model /opt/models/ppocrv5_latin_rec_mobile.onnx \
  --limit-side-len 736 \
  --threads 4
Restart=on-failure
RestartSec=2
TimeoutStopSec=20
RuntimeDirectory=rapidocr-rust-%i
RuntimeDirectoryMode=0770
NoNewPrivileges=true
ReadWritePaths=/run/rapidocr-rust-%i
RestrictAddressFamilies=AF_UNIX

[Install]
WantedBy=multi-user.target
```

### 3. Enable and Start Worker Instances

For a balanced 16-thread CPU allocation with 4 threads per worker, spawn 4 worker instances:

```bash
sudo systemctl enable --now rapidocr-rust-worker@{1..4}.service
```

Verify service status:

```bash
systemctl status rapidocr-rust-worker@{1..4}.service
ls -la /run/rapidocr-rust-*/worker.sock
```

### 4. Point Your Client at the Socket

Register the worker socket(s) in your client, one per worker instance:

```text
/run/rapidocr-rust-1/worker.sock
/run/rapidocr-rust-2/worker.sock
/run/rapidocr-rust-3/worker.sock
/run/rapidocr-rust-4/worker.sock
```

The framing contract and a reference Python client live in
[API.md](API.md).

## Verification & Smoke Test

Test socket framing directly with a test script:

```bash
python3 tests/test_socket_contract.py
```
