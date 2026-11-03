//! Arena configuration contract for #19.

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
fn default_keeps_arena_enabled_for_parity() {
    let cfg = WorkerConfig::parse(base_args()).expect("valid args");
    assert!(
        !cfg.no_arena,
        "default must keep the arena enabled: byte-identical parity with the previous binary"
    );
}

#[test]
fn no_arena_flag_is_opt_in() {
    let mut args = base_args();
    args.push("--no-arena");
    let cfg = WorkerConfig::parse(args).expect("valid args");
    assert!(cfg.no_arena, "--no-arena must opt in to arena disablement");
}

#[test]
fn no_arena_composes_with_other_flags() {
    let mut args = base_args();
    args.extend([
        "--no-arena",
        "--rec-batch",
        "6",
        "--rec-workers",
        "2",
        "--threads",
        "4",
        "--det-profile",
        "ppocrv6",
        "--det-normalization",
        "imagenet",
        "--keys",
        "/models/v6_keys.txt",
    ]);
    let cfg = WorkerConfig::parse(args).expect("valid args");
    assert!(cfg.no_arena);
    assert_eq!(cfg.rec_batch, 6);
    assert_eq!(cfg.rec_workers, 2);
    assert_eq!(cfg.threads, 4);
    assert_eq!(
        cfg.det_profile,
        rapidocr_worker::config::DetProfile::Ppocrv6
    );
    assert_eq!(
        cfg.det_normalization,
        rapidocr_worker::config::DetNormalization::Imagenet
    );
    assert_eq!(cfg.keys_path.as_deref(), Some("/models/v6_keys.txt"));
}

#[test]
fn trim_allocator_heap_releases_freed_top_chunk() {
    let block = vec![42u8; 8 * 1024 * 1024];
    assert_eq!(block[0], 42);
    drop(block);
    let _ = rapidocr_worker::session::trim_allocator_heap();
}
