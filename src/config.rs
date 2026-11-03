//! Command-line argument parsing and worker configuration (#4, #3).

pub const DEFAULT_LIMIT_SIDE_LEN: u32 = 736;
pub const DEFAULT_REC_BATCH: usize = 6;
/// Maximum accepted request frame size (metadata + image bytes) in bytes.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
/// Read/write timeout for client connections in seconds.
pub const DEFAULT_IO_TIMEOUT_SECONDS: u64 = 60;

/// Detection input normalization profile.
///
/// PP-OCRv4 normalizes with a single scalar `0.5`; PP-OCRv6 specifies per-channel
/// ImageNet statistics. Applying the wrong profile does not fail loudly, it just
/// silently degrades box confidence, so the choice is explicit.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum DetNormalization {
    /// `(v - 0.5) / 0.5` for every channel — PP-OCRv4 and earlier.
    #[default]
    Scalar05,
    /// ImageNet per-channel mean/std — PP-OCRv6.
    Imagenet,
}

impl DetNormalization {
    /// Mean and std triples this profile applies to detection input.
    pub fn mean_std(self) -> ([f32; 3], [f32; 3]) {
        match self {
            Self::Scalar05 => ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Self::Imagenet => ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        }
    }

    /// Parse the `--det-normalization` value.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "scalar" | "scalar0.5" => Ok(Self::Scalar05),
            "imagenet" => Ok(Self::Imagenet),
            other => Err(format!(
                "invalid --det-normalization {other:?}: expected 'scalar' or 'imagenet'"
            )),
        }
    }
}

/// Detection post-processing threshold profile.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum DetProfile {
    /// PP-OCRv4 thresholds (0.3 / 0.5 / 1000 / 1.6).
    #[default]
    Ppocrv4,
    /// PP-OCRv6 thresholds (0.2 / 0.45 / 3000 / 1.4).
    Ppocrv6,
}

impl DetProfile {
    /// Parse the `--det-profile` value.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "ppocrv4" | "v4" => Ok(Self::Ppocrv4),
            "ppocrv6" | "v6" => Ok(Self::Ppocrv6),
            other => Err(format!(
                "invalid --det-profile {other:?}: expected 'ppocrv4' or 'ppocrv6'"
            )),
        }
    }
}

/// Configuration options parsed from command line arguments.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct WorkerConfig {
    pub socket_path: String,
    pub det_path: String,
    pub cls_path: String,
    pub rec_path: String,
    pub threads: usize,
    pub limit_side_len: u32,
    pub keys_path: Option<String>,
    pub rec_batch: usize,
    pub rec_workers: usize,
    pub no_cls: bool,
    pub cls_sampled: bool,
    /// Disable the ONNX Runtime CPU memory arena and memory patterns (#19).
    pub no_arena: bool,
    pub max_frame_bytes: usize,
    pub io_timeout_seconds: u64,
    pub det_normalization: DetNormalization,
    pub det_profile: DetProfile,
}

impl WorkerConfig {
    /// Parse command-line arguments into [`WorkerConfig`].
    ///
    /// Accepts both short flags (`--det`, `--cls`, `--rec`) and long flags (`--det-model`,
    /// `--cls-model`, `--rec-model`) for parity with systemd unit configurations.
    /// Supports `--threads`, `--limit-side-len` (default 736), `--rec-batch`,
    /// `--rec-workers`, `--no-cls`, `--cls-full`, `--cls-sampled` (opt-in),
    /// `--no-arena` (opt-in), `--max-frame-bytes` (default 64 MiB) and
    /// `--io-timeout-s` (default 60s).
    pub fn parse<I, T>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let args_vec: Vec<String> = args.into_iter().map(|s| s.as_ref().to_string()).collect();
        let mut socket_path = String::new();
        let mut det_path = String::new();
        let mut cls_path = String::new();
        let mut rec_path = String::new();
        let mut threads = 4usize;
        let mut limit_side_len = DEFAULT_LIMIT_SIDE_LEN;
        let mut rec_batch = DEFAULT_REC_BATCH;
        let mut rec_workers = 1usize;
        let mut no_cls = false;
        let mut cls_sampled = false;
        let mut no_arena = false;
        let mut det_normalization = DetNormalization::default();
        let mut det_profile = DetProfile::default();
        let mut keys_path: Option<String> = None;
        let mut max_frame_bytes = DEFAULT_MAX_FRAME_BYTES;
        let mut io_timeout_seconds = DEFAULT_IO_TIMEOUT_SECONDS;

        let mut i = 1;
        while i < args_vec.len() {
            let flag = args_vec[i].as_str();
            let next_val = args_vec.get(i + 1).map(|s| s.as_str());
            match flag {
                "--socket" => {
                    socket_path = next_val.unwrap_or_default().to_string();
                    i += 2;
                }
                "--det" | "--det-model" => {
                    det_path = next_val.unwrap_or_default().to_string();
                    i += 2;
                }
                "--cls" | "--cls-model" => {
                    cls_path = next_val.unwrap_or_default().to_string();
                    i += 2;
                }
                "--rec" | "--rec-model" => {
                    rec_path = next_val.unwrap_or_default().to_string();
                    i += 2;
                }
                "--threads" => {
                    if let Some(val) = next_val.and_then(|v| v.parse::<usize>().ok()) {
                        if val > 0 {
                            threads = val;
                        }
                    }
                    i += 2;
                }
                "--limit-side-len" | "--det-limit-side-len" => {
                    if let Some(val) = next_val.and_then(|v| v.parse::<u32>().ok()) {
                        if val > 0 {
                            limit_side_len = val;
                        }
                    }
                    i += 2;
                }
                "--max-frame-bytes" => {
                    if let Some(val) = next_val.and_then(|v| v.parse::<usize>().ok()) {
                        if val > 0 {
                            max_frame_bytes = val;
                        }
                    }
                    i += 2;
                }
                "--io-timeout-s" | "--io-timeout" => {
                    if let Some(val) = next_val.and_then(|v| v.parse::<u64>().ok()) {
                        if val > 0 {
                            io_timeout_seconds = val;
                        }
                    }
                    i += 2;
                }
                "--rec-batch" | "--batch-size" => {
                    if let Some(val) = next_val.and_then(|v| v.parse::<usize>().ok()) {
                        if val > 0 {
                            rec_batch = val;
                        }
                    }
                    i += 2;
                }
                "--rec-workers" | "--workers" => {
                    if let Some(val) = next_val.and_then(|v| v.parse::<usize>().ok()) {
                        if val > 0 {
                            rec_workers = val;
                        }
                    }
                    i += 2;
                }
                "--no-cls" => {
                    no_cls = true;
                    i += 1;
                }
                "--cls-full" => {
                    cls_sampled = false;
                    i += 1;
                }
                "--cls-sampled" => {
                    cls_sampled = true;
                    i += 1;
                }
                "--no-arena" => {
                    no_arena = true;
                    i += 1;
                }
                "--keys" => {
                    keys_path = next_val.map(|s| s.to_string());
                    i += 2;
                }
                "--det-normalization" => {
                    if let Some(val) = next_val {
                        det_normalization = DetNormalization::parse(val)?;
                    }
                    i += 2;
                }
                "--det-profile" => {
                    if let Some(val) = next_val {
                        det_profile = DetProfile::parse(val)?;
                    }
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }

        if socket_path.is_empty() || det_path.is_empty() || rec_path.is_empty() {
            return Err("missing required arguments: --socket, --det/--det-model, and --rec/--rec-model are required".into());
        }

        Ok(Self {
            socket_path,
            det_path,
            cls_path,
            rec_path,
            threads,
            limit_side_len,
            rec_batch,
            rec_workers,
            no_cls,
            cls_sampled,
            no_arena,
            max_frame_bytes,
            io_timeout_seconds,
            det_normalization,
            det_profile,
            keys_path,
        })
    }

    /// True when direction classification is disabled because `--cls-model`
    /// was not provided and `--no-cls` was not passed explicitly (#16).
    pub fn cls_disabled_without_flag(&self) -> bool {
        !self.no_cls && self.cls_path.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_short_arguments() {
        let args = vec![
            "rapidocr-worker",
            "--socket",
            "/tmp/worker.sock",
            "--det",
            "/path/det.onnx",
            "--cls",
            "/path/cls.onnx",
            "--rec",
            "/path/rec.onnx",
            "--threads",
            "8",
        ];
        let cfg = WorkerConfig::parse(&args).expect("valid short args");
        assert_eq!(cfg.socket_path, "/tmp/worker.sock");
        assert_eq!(cfg.det_path, "/path/det.onnx");
        assert_eq!(cfg.cls_path, "/path/cls.onnx");
        assert_eq!(cfg.rec_path, "/path/rec.onnx");
        assert_eq!(cfg.threads, 8);
        assert_eq!(cfg.limit_side_len, DEFAULT_LIMIT_SIDE_LEN);
    }

    #[test]
    fn parse_long_model_aliases_and_limit_side_len() {
        let args = vec![
            "rapidocr-worker",
            "--socket",
            "/run/worker.sock",
            "--det-model",
            "/models/det.onnx",
            "--cls-model",
            "/models/cls.onnx",
            "--rec-model",
            "/models/rec.onnx",
            "--limit-side-len",
            "1280",
        ];
        let cfg = WorkerConfig::parse(&args).expect("valid long args");
        assert_eq!(cfg.det_path, "/models/det.onnx");
        assert_eq!(cfg.cls_path, "/models/cls.onnx");
        assert_eq!(cfg.rec_path, "/models/rec.onnx");
        assert_eq!(cfg.limit_side_len, 1280);
    }

    #[test]
    fn parse_missing_required_arguments_errors() {
        let args = vec!["rapidocr-worker", "--socket", "/run/worker.sock"];
        assert!(WorkerConfig::parse(&args).is_err());
    }

    #[test]
    fn parse_no_arena_flag_and_default() {
        // Default keeps parity with the current binary: arena enabled (#19).
        let args_default = vec![
            "rapidocr-worker",
            "--socket",
            "/run/worker.sock",
            "--det",
            "/models/det.onnx",
            "--rec",
            "/models/rec.onnx",
        ];
        let cfg = WorkerConfig::parse(&args_default).expect("valid args");
        assert!(!cfg.no_arena);

        let mut args_flag = args_default.clone();
        args_flag.push("--no-arena");
        let cfg = WorkerConfig::parse(&args_flag).expect("valid args");
        assert!(cfg.no_arena);
    }
    #[test]
    fn parse_performance_tuning_flags() {
        let args = vec![
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
            "12",
            "--rec-workers",
            "4",
            "--no-cls",
            "--cls-full",
        ];
        let cfg = WorkerConfig::parse(&args).expect("valid perf args");
        assert_eq!(cfg.rec_batch, 12);
        assert_eq!(cfg.rec_workers, 4);
        assert!(cfg.no_cls);
        assert!(!cfg.cls_sampled);
    }

    #[test]
    fn parse_socket_limit_flags_and_defaults() {
        let args = vec![
            "rapidocr-worker",
            "--socket",
            "/run/worker.sock",
            "--det",
            "/models/det.onnx",
            "--rec",
            "/models/rec.onnx",
        ];
        let cfg = WorkerConfig::parse(&args).expect("default args");
        assert_eq!(cfg.max_frame_bytes, DEFAULT_MAX_FRAME_BYTES);
        assert_eq!(cfg.io_timeout_seconds, DEFAULT_IO_TIMEOUT_SECONDS);

        let args_custom = vec![
            "rapidocr-worker",
            "--socket",
            "/run/worker.sock",
            "--det",
            "/models/det.onnx",
            "--rec",
            "/models/rec.onnx",
            "--max-frame-bytes",
            "1048576",
            "--io-timeout-s",
            "30",
        ];
        let cfg_custom = WorkerConfig::parse(&args_custom).expect("custom args");
        assert_eq!(cfg_custom.max_frame_bytes, 1048576);
        assert_eq!(cfg_custom.io_timeout_seconds, 30);
    }
}
