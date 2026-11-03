//! Integration test for aspect-ratio bucketing, sampled direction classification,
//! and worker thread configuration (#8, #9).

use rapidocr_worker::cls::{evaluate_sample_orientation, PageOrientation};
use rapidocr_worker::config::WorkerConfig;
use rapidocr_worker::rec::plan_rec_batches;

#[test]
fn test_worker_config_perf_defaults_and_overrides() {
    let args_default = vec![
        "rapidocr-worker",
        "--socket",
        "/run/worker.sock",
        "--det",
        "/models/det.onnx",
        "--cls",
        "/models/cls.onnx",
        "--rec",
        "/models/rec.onnx",
    ];
    let cfg = WorkerConfig::parse(&args_default).expect("default args");
    assert_eq!(cfg.rec_batch, 6);
    assert_eq!(cfg.rec_workers, 1);
    assert!(!cfg.no_cls);
    assert!(!cfg.cls_sampled);

    let args_custom = vec![
        "rapidocr-worker",
        "--socket",
        "/run/worker.sock",
        "--det",
        "/models/det.onnx",
        "--cls",
        "/models/cls.onnx",
        "--rec",
        "/models/rec.onnx",
        "--rec-batch",
        "16",
        "--rec-workers",
        "4",
        "--no-cls",
        "--cls-full",
    ];
    let cfg_custom = WorkerConfig::parse(&args_custom).expect("custom args");
    assert_eq!(cfg_custom.rec_batch, 16);
    assert_eq!(cfg_custom.rec_workers, 4);
    assert!(cfg_custom.no_cls);
    assert!(!cfg_custom.cls_sampled);
}

#[test]
fn test_aspect_ratio_bucketing_distribution() {
    // Construct realistic scanned page distribution:
    // 30 compact words (ratio 1.5 - 4.0), 10 medium lines (ratio 7.0 - 9.0), 5 wide lines (ratio 15.0 - 20.0)
    let mut crops = Vec::new();
    for i in 0..30 {
        crops.push((i, 2.5));
    }
    for i in 30..40 {
        crops.push((i, 8.0));
    }
    for i in 40..45 {
        crops.push((i, 16.0));
    }

    let batches = plan_rec_batches(&crops, 6);

    // Verify all 45 crops are accounted for
    let mut indices: Vec<usize> = batches.iter().flatten().copied().collect();
    indices.sort_unstable();
    assert_eq!(indices, (0..45).collect::<Vec<usize>>());

    // Verify compact batches hold up to 12 items
    assert!(batches[0].len() <= 12);
    assert!(batches[0].len() > 6);

    // Verify wide line batches do not exceed wide cap (4)
    let last_batch = batches.last().expect("non-empty batches");
    assert!(last_batch.len() <= 4);
}

#[test]
fn test_sample_orientation_edge_cases() {
    // 100% upright
    // 0.97 is the confidence threshold, so these must clear it.
    let upright = vec![("0", 0.99), ("0", 0.985), ("0", 0.98)];
    assert_eq!(
        evaluate_sample_orientation(&upright),
        PageOrientation::Upright
    );

    // 100% inverted
    // 0.97 is the confidence threshold, so these must clear it.
    let inverted = vec![("180", 0.985), ("180", 0.99), ("180", 0.98)];
    assert_eq!(
        evaluate_sample_orientation(&inverted),
        PageOrientation::Inverted
    );

    // Low confidence upright should fall back to mixed for safety
    let low_conf = vec![("0", 0.70), ("0", 0.90)];
    assert_eq!(
        evaluate_sample_orientation(&low_conf),
        PageOrientation::Mixed
    );

    // Contradictory orientations
    let mixed = vec![("0", 0.95), ("180", 0.95)];
    assert_eq!(evaluate_sample_orientation(&mixed), PageOrientation::Mixed);
}
