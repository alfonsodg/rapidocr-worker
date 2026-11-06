# Configuration Guide

`rapidocr-worker` is configured via command-line arguments and runtime environment variables.

## Command-Line Arguments

The binary accepts both canonical long flags and convenient short flags:

| Flag | Argument | Default | Required | Description |
|---|---|---|---|---|
| `--socket` | `<path>` | None | Yes | Absolute filesystem path for the UNIX domain stream socket (e.g., `/run/rapidocr-rust-1/worker.sock`). |
| `--det-model`, `--det` | `<path>` | None | Yes | Path to detection ONNX model (`ch_PP-OCRv4_det_infer.onnx`). |
| `--cls-model`, `--cls` | `<path>` | None | Yes | Path to direction classification ONNX model (`ch_ppocr_mobile_v2.0_cls_infer.onnx`). |
| `--rec-model`, `--rec` | `<path>` | None | Yes | Path to text recognition ONNX model (`ppocrv5_latin_rec_mobile.onnx`). |
| `--threads` | `<integer>` | `4` | No | Total intra-op thread budget allocated across worker sessions. |
| `--limit-side-len` | `<integer>` | `736` | No | Minimum side length: the shorter image side is scaled to this value before detection (`limit_type=min`, Python parity). |
| `--max-frame-bytes` | `<integer>` | `67108864` | No | Maximum accepted request frame size in bytes (metadata + image). Larger declared frames are rejected before allocation with `frame too large`. |
| `--io-timeout-s` | `<integer>` | `60` | No | Per-read/write timeout in seconds for client connections, bounding stalled clients. |
| `--rec-batch`, `--batch-size` | `<integer>` | `6` | No | Base recognition batch size. Compact crops dynamically scale up to 2x this value. |
| `--rec-workers`, `--workers` | `<integer>` | `1` | No | Number of parallel recognition session workers utilizing scoped worker threads. |
| `--no-cls` | None | `false` | No | Disable direction classification stage entirely for guaranteed upright documents. |
| `--cls-full` | None | `false` | No | Force exhaustive per-box direction classification, disabling sample-based fast path. |
| `--cls-sampled` | None | `false` | No | Opt-in page-level orientation sampling to bypass per-box CLS on confidently upright pages; any non-upright sample falls back to per-box classification. |

| `--keys` | `<path>` | None | No | External character dictionary, one symbol per line. Required for models that carry no ONNX `character` metadata, such as PP-OCRv6. When the model does embed the dictionary, the embedded one wins. |
| `--det-normalization` | `scalar` \| `imagenet` | `scalar` | No | Detection input normalization. `scalar` is `(v - 0.5) / 0.5` (PP-OCRv4). `imagenet` uses per-channel mean/std (PP-OCRv6). The wrong profile does not fail loudly, it silently lowers detection confidence. |
| `--det-profile` | `ppocrv4` \| `ppocrv6` | `ppocrv4` | No | Detection post-processing thresholds. `ppocrv4` = 0.3 / 0.5 / 1000 / 1.6; `ppocrv6` = 0.2 / 0.45 / 3000 / 1.4, as specified in each model's `inference.yml`. |

### Example CLI Invocation

```bash
/usr/local/bin/rapidocr-worker \
  --socket /run/rapidocr-rust-1/worker.sock \
  --det-model /opt/models/ch_PP-OCRv4_det_infer.onnx \
  --cls-model /opt/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --rec-model /opt/models/ppocrv5_latin_rec_mobile.onnx \
  --threads 4 \
  --limit-side-len 736
```

### PP-OCRv6 Invocation

PP-OCRv6 needs all three of the additions above: it ships no embedded
dictionary, and its detector expects different normalization and thresholds than
PP-OCRv4.

```bash
/usr/local/bin/rapidocr-worker \
  --socket /run/rapidocr-rust-1/worker.sock \
  --det-model /opt/models/PP-OCRv6_small_det.onnx \
  --cls-model /opt/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --rec-model /opt/models/PP-OCRv6_small_rec.onnx \
  --keys /opt/models/PP-OCRv6_small_rec_keys.txt \
  --det-normalization imagenet \
  --det-profile ppocrv6 \
  --threads 4 \
  --limit-side-len 960
```

Generate the keys file from the model's `inference.yml`: take the
`PostProcess.character_dict` entries in order, one symbol per line, with no blank
line and no added space. The decoder inserts `blank` at index 0 and the space at
the end itself, matching the embedded-metadata layout.

A published Spanish pair uses the same mechanism: the checkpoint has no embedded
metadata, so it must be loaded with its own dictionary.

```bash
/usr/local/bin/rapidocr-worker \
  --socket /run/rapidocr-rust-1/worker.sock \
  --det-model /opt/models/ch_PP-OCRv4_det_infer.onnx \
  --cls-model /opt/models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
  --rec-model /opt/models/spanish_rec_v18.onnx \
  --keys /opt/models/ppocr_keys_es.txt \
  --threads 4
```

Both files come from the RapidOCR release `spanish-multidoc-v1`
(<https://github.com/alfonsodg/RapidOCR/releases/tag/spanish-multidoc-v1>) and are
mirrored on Hugging Face
(<https://huggingface.co/alfonsodlg/rapidocr-spanish>). Mixing the checkpoint with
a different vocabulary does not fail loudly: it decodes valid-looking text from
shifted indices.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ORT_DYLIB_PATH` | Yes | None | Absolute path to the ONNX Runtime dynamic library (`libonnxruntime.so`). Required by the `ort` dynamic-load feature. |
| `RUST_LOG` | No | `info` | Logging filter level (`error`, `warn`, `info`, `debug`, `trace`). |
| `RUST_BACKTRACE` | No | `0` | Set to `1` or `full` to enable backtraces on panic. |

## Client Integration

Any client can drive the worker over its UNIX domain socket; see
[API.md](API.md) for the framing contract and a reference Python client.

When several worker instances are active, distribute requests across their
sockets (one socket per worker instance):

```text
/run/rapidocr-rust-1/worker.sock
/run/rapidocr-rust-2/worker.sock
...
```

A client that also has an in-process engine can switch between both paths by
re-pointing its socket list; an empty socket list means "no socket workers",
which is the natural rollback lever.

## Resource Sizing Guidelines

| Server CPU Budget | Recommended Workers | Threads per Worker (`--threads`) | Total ONNX Threads |
|---|---|---|---|
| 8 vCPUs | 2 | 4 | 8 |
| 16 vCPUs | 4 | 4 | 16 |
| 32 vCPUs | 6 | 4 | 24 |
| 48+ vCPUs | 4 to 8 | 4 | 16 to 32 |

* Inter-op threads are hard-bounded to `1` inside each ONNX session to prevent CPU oversubscription and thread pool contention.
* Keeping `--threads 4` per worker ensures balanced SIMD/AVX2 utilization for both detection convolution layers and recognition GEMM operations.
* `--rec-workers`: On hosts with high core counts, setting `--rec-workers 2` or `4` allows the worker to evaluate multiple recognition batches concurrently across scoped worker threads while dividing the `--threads` budget evenly among sessions, avoiding intra-op thread barriers.
