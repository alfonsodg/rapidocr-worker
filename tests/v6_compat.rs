//! PP-OCRv6 compatibility: external dictionary, per-channel det
//! normalization, and configurable det post-processing thresholds.
//!
//! These tests encode the three measured blockers. Before the change:
//!
//! 1. `CtcDecoder` could only read the dictionary from ONNX `character`
//!    metadata. PP-OCRv6 ships no such metadata, so `OcrEngine::new` panicked on
//!    `.expect("decoder")`.
//! 2. `normalize_chw` applied one scalar mean/std. PP-OCRv6 specifies per-channel
//!    ImageNet normalization; measured detection confidence (p99) drops from
//!    0.972 to 0.825 when the scalar path is used.
//! 3. `DbPostProcess` thresholds were hardcoded to the PP-OCRv4 values
//!    (0.3 / 0.5 / 1000 / 1.6). PP-OCRv6 specifies 0.2 / 0.45 / 3000 / 1.4.

use rapidocr_worker::config::{DetNormalization, DetProfile, WorkerConfig};
use rapidocr_worker::det::{normalize_chw, DbPostProcess};
use rapidocr_worker::rec::CtcDecoder;

fn labels(n: usize) -> String {
    (0..n)
        .map(|i| format!("c{i}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn decoder_accepts_an_external_dictionary_file() {
    // PP-OCRv6 carries 18,708 symbols and no ONNX metadata; the dictionary has
    // to come from a file. The layout must match the metadata path exactly:
    // blank at 0, the symbols, then the trailing space.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("v6_keys.txt");
    std::fs::write(&path, labels(5)).expect("write keys");

    let decoder = CtcDecoder::from_keys_file(&path).expect("decoder from keys file");

    assert_eq!(decoder.characters.len(), 7, "blank + 5 symbols + space");
    assert_eq!(decoder.characters[0], "blank");
    assert_eq!(decoder.characters[1], "c0");
    assert_eq!(decoder.characters[5], "c4");
    assert_eq!(decoder.characters[6], " ");
}

#[test]
fn decoder_rejects_a_missing_or_empty_keys_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("nope.txt");
    assert!(CtcDecoder::from_keys_file(&missing).is_err());

    let empty = dir.path().join("empty.txt");
    std::fs::write(&empty, "\n\n").expect("write empty");
    assert!(CtcDecoder::from_keys_file(&empty).is_err());
}

#[test]
fn decoder_from_keys_file_matches_metadata_layout() {
    // The two sources must be interchangeable so a model swap does not shift
    // every label by one.
    let meta = labels(6);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("keys.txt");
    std::fs::write(&path, &meta).expect("write keys");

    let from_meta = CtcDecoder::from_metadata(&meta).expect("metadata decoder");
    let from_file = CtcDecoder::from_keys_file(&path).expect("keys decoder");

    assert_eq!(from_meta.characters, from_file.characters);
}

#[test]
fn normalize_chw_honours_per_channel_mean_and_std() {
    // A pure red pixel: (1, 0, 0) after scaling.
    let img = image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]));

    let imagenet = normalize_chw(&img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    // (1 - 0.485) / 0.229, (0 - 0.456) / 0.224, (0 - 0.406) / 0.225
    assert!((imagenet[[0, 0, 0, 0]] - 2.2489).abs() < 1e-3);
    assert!((imagenet[[0, 1, 0, 0]] - (-2.0357)).abs() < 1e-3);
    assert!((imagenet[[0, 2, 0, 0]] - (-1.8044)).abs() < 1e-3);

    // The deployed PP-OCRv4 scalar path must stay identical to the old output.
    let scalar = normalize_chw(&img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    assert!((scalar[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
    assert!((scalar[[0, 1, 0, 0]] - (-1.0)).abs() < 1e-6);
    assert!((scalar[[0, 2, 0, 0]] - (-1.0)).abs() < 1e-6);
}

#[test]
fn det_post_process_is_configurable_for_ppocrv6() {
    let v4 = DbPostProcess::default();
    assert_eq!(v4.thresh, 0.3);
    assert_eq!(v4.box_thresh, 0.5);
    assert_eq!(v4.max_candidates, 1000);
    assert_eq!(v4.unclip_ratio, 1.6);

    let v6 = DbPostProcess::ppocrv6();
    assert_eq!(v6.thresh, 0.2);
    assert_eq!(v6.box_thresh, 0.45);
    assert_eq!(v6.max_candidates, 3000);
    assert_eq!(v6.unclip_ratio, 1.4);
}

#[test]
fn worker_config_accepts_v6_model_and_keys_paths() {
    let cfg = WorkerConfig::parse(vec![
        "rapidocr-worker",
        "--socket",
        "/tmp/x.sock",
        "--det",
        "/m/v6_det.onnx",
        "--rec",
        "/m/v6_rec.onnx",
        "--keys",
        "/m/v6_keys.txt",
        "--det-normalization",
        "imagenet",
        "--det-profile",
        "ppocrv6",
    ])
    .expect("parse v6 config");

    assert_eq!(cfg.det_path, "/m/v6_det.onnx");
    assert_eq!(cfg.rec_path, "/m/v6_rec.onnx");
    assert_eq!(cfg.keys_path.as_deref(), Some("/m/v6_keys.txt"));
    assert_eq!(cfg.det_normalization, DetNormalization::Imagenet);
    assert_eq!(cfg.det_profile, DetProfile::Ppocrv6);
}
