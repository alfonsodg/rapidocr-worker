//! Direction-classification configuration contract for #16.

use rapidocr_worker::config::WorkerConfig;

fn base_args() -> Vec<&'static str> {
    vec![
        "rapidocr-worker",
        "--socket",
        "/run/worker.sock",
        "--det",
        "/models/det.onnx",
        "--rec",
        "/models/rec.onnx",
    ]
}

#[test]
fn missing_cls_model_without_no_cls_is_flagged() {
    let cfg = WorkerConfig::parse(base_args()).expect("valid args");
    assert!(cfg.cls_path.is_empty());
    assert!(
        cfg.cls_disabled_without_flag(),
        "missing --cls-model without --no-cls must be flagged"
    );
}

#[test]
fn explicit_no_cls_and_provided_cls_model_are_not_flagged() {
    let mut args = base_args();
    args.push("--cls");
    args.push("/models/cls.onnx");
    let cfg = WorkerConfig::parse(args).expect("valid args");
    assert!(!cfg.cls_disabled_without_flag());

    let mut args = base_args();
    args.push("--no-cls");
    let cfg = WorkerConfig::parse(args).expect("valid args");
    assert!(!cfg.cls_disabled_without_flag());
}
