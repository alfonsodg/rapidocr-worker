//! The worker response carries per-stage timings for operators (#8).

use rapidocr_worker::config::WorkerConfig;

#[test]
fn worker_config_parses_the_documented_invocation() {
    // The production unit invocation must keep parsing so the stage timings
    // reach the response payload on every deployed worker.
    let args = vec![
        "rapidocr-worker",
        "--socket",
        "/run/rapidocr-rust-1/worker.sock",
        "--det-model",
        "/models/det.onnx",
        "--cls-model",
        "/models/cls.onnx",
        "--rec-model",
        "/models/rec.onnx",
        "--limit-side-len",
        "736",
        "--threads",
        "8",
    ];
    let cfg = WorkerConfig::parse(&args).expect("valid production args");
    assert_eq!(cfg.limit_side_len, 736);
    assert_eq!(cfg.threads, 8);
}
