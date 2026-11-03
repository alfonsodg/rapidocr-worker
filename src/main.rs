//! Full RapidOCR worker in Rust: det + DB post + cls + rec + CTC + socket (#9).

use std::io::Write;
use std::os::unix::net::{UnixListener, UnixStream};
use std::time::{Duration, Instant};

use rapidocr_worker::config::WorkerConfig;
use rapidocr_worker::engine::OcrEngine;
use rapidocr_worker::protocol::{read_request, validate_meta, write_response};

/// Handle one request frame and return the JSON response payload.
///
/// Framing and input bounds live in [`rapidocr_worker::protocol`]; inference
/// failures are returned as errors instead of aborting the process (#13).
fn handle_request(
    engine: &mut OcrEngine,
    stream: &mut UnixStream,
    max_frame_bytes: usize,
) -> Result<Vec<u8>, String> {
    let (meta, image) = read_request(stream, max_frame_bytes)?;
    validate_meta(&meta)?;

    let started = Instant::now();
    let result = engine.process(&image);
    let elapsed = started.elapsed().as_secs_f32();

    let payload = match result {
        Ok((text, confidence, timings)) => serde_json::json!({
            "ok": true, "text": text, "confidence": confidence, "elapsed_s": elapsed,
            "det_ms": timings.det_ms, "cls_ms": timings.cls_ms, "rec_ms": timings.rec_ms
        }),
        Err(error) => serde_json::json!({ "ok": false, "error": error }),
    };
    serde_json::to_vec(&payload).map_err(|e| format!("json: {e}"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = match WorkerConfig::parse(&args) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("Error: {err}");
            eprintln!("usage: rapidocr-worker --socket <path> --det|--det-model <det.onnx> [--cls|--cls-model <cls.onnx>] --rec|--rec-model <rec.onnx> [--threads N] [--limit-side-len 736] [--rec-batch N] [--rec-workers N] [--no-cls] [--cls-full] [--cls-sampled] [--no-arena] [--max-frame-bytes N] [--io-timeout-s N]");
            std::process::exit(2);
        }
    };

    let mut engine = OcrEngine::new(&config);
    let _ = std::fs::remove_file(&config.socket_path);
    let listener = UnixListener::bind(&config.socket_path).expect("bind socket");
    eprintln!("worker listening on {}", config.socket_path);
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let timeout = Some(Duration::from_secs(config.io_timeout_seconds));
                if let Err(e) = stream.set_read_timeout(timeout) {
                    eprintln!("read timeout setup: {e}");
                }
                if let Err(e) = stream.set_write_timeout(timeout) {
                    eprintln!("write timeout setup: {e}");
                }
                let payload = match handle_request(&mut engine, &mut stream, config.max_frame_bytes)
                {
                    Ok(payload) => payload,
                    Err(e) => serde_json::to_vec(&serde_json::json!({"ok": false, "error": e}))
                        .unwrap_or_default(),
                };
                if let Err(e) = write_response(&mut stream, &payload) {
                    eprintln!("write error: {e}");
                }
                let _ = stream.flush();
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}
