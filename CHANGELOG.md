# Changelog

All notable changes to this project will be documented in this file.
Format based on [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- PP-OCRv6 support: `--det-profile {ppocrv4,ppocrv6}` selects the detection
  post-processing, `--det-normalization {scalar,imagenet}` selects the input
  normalization, and `--keys` loads an external character dictionary for models
  whose ONNX metadata carries none.
- Aspect-ratio bucketing (`plan_rec_batches`) categorizing text crops into compact, normal, and wide buckets, scaling compact batches up to 2x base size to eliminate zero-padding overhead (#9).
- Optional sampled page-level direction classification (`evaluate_sample_orientation`, opt-in via `--cls-sampled`) that bypasses per-box CLS only on confidently upright pages and falls back to per-box classification otherwise (#9).
- CLI configuration options `--rec-batch`, `--rec-workers`, `--no-cls`, `--cls-full`, and `--cls-sampled` for tuning OCR inference (#9).
- Scoped thread recognition session pool allocating independent ONNX sessions across worker threads to maximize multicore throughput without intra-op thread barriers (#9).
- Reading order sorting (`sort_boxes` and `sort_boxes_and_scores`) matching Python RapidOCR/PaddleOCR line grouping logic (#3).
- Configurable `--limit-side-len` CLI flag for high-density document preservation; default 736 after the #8 parity alignment (#3, #8).
- CLI argument aliases `--det-model`, `--cls-model`, and `--rec-model` matching production systemd service templates (#4).
- Production systemd unit template `configs/systemd/rapidocr-worker@.service` and `deploy/systemd/rapidocr-worker@.service` (#4).
- Reading order reference contract test `tests/test_reading_order_3.py` (#3).

### Changed

- Repository moved to public GitHub (`github.com/alfonsodg/rapidocr-worker`) under the Apache-2.0 license with a fresh single-commit history (#21).
- Bounded ONNX Runtime inter-op execution threads to 1 (`with_inter_threads(1)`) to eliminate multi-worker CPU thread contention (#4).
- Bounding boxes returned from `boxes_from_bitmap` are now strictly sorted top-to-bottom and left-to-right in a single pass (#3, #15).
- OpenCV-compatible bilinear interpolation in `rec::resize_norm_into` and `cls::resize_norm_img_cls` eliminating orientation flip regression on dense text lines (#1).
- Unconstrained dynamic recognition batch tensor width matching aspect ratio without 320px truncation (#1).
- Vertical text crops are now rotated 90° counterclockwise before text direction classification (#1).
- Character metadata parsing now strips trailing newlines with `lines()` preserving exact space token index in Latin models (#1).
- Aligned README, ARCHITECTURE, CONFIGURATION, INSTALL and API with the measured parity evidence and current defaults (#11).

### Fixed

- Direction classification reads a centred 4:1 window instead of squashing the whole text line into the 192px canvas, which made dense-page lines illegible and flipped them by 180 degrees; `CLS_THRESH` moved to 0.97, the measured point where no page regresses while a 180-degree page is still recovered.
- Resynced `deploy/systemd/rapidocr-worker@.service` with `configs/` after the 736px detection parity change (#12).
- Socket input bounds: frames larger than `--max-frame-bytes` are rejected before allocation, connections get configurable read/write timeouts, and inference failures no longer abort the worker process (#13).

### Removed

- Dropped container deployment artifacts (Dockerfile, Compose, Helm) in favor of the binary + systemd delivery model (#21).
- Removed the non-functional `--keys` CLI option; the usage text now reports the real 736 default (#15).
