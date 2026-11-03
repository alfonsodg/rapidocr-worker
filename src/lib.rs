//! Library target for the RapidOCR worker so integration tests can exercise
//! the engine modules (#8). The binary in main.rs wires them into the socket
//! service.

pub mod batch_runner;
pub mod cls;
pub mod config;
pub mod contours;
pub mod det;
pub mod engine;
pub mod geometry;
pub mod order;
pub mod perspective;
pub mod protocol;
pub mod rec;
pub mod session;

/// Recognition input height in pixels (PaddleOCR rec shape 3x48xW).
pub const REC_HEIGHT: u32 = 48;
/// Minimum recognition canvas width in pixels (PaddleOCR rec shape 3xHx320).
pub const REC_WIDTH: u32 = 320;
/// Crops per recognition batch (PaddleOCR rec_batch_num).
pub const REC_BATCH: usize = 6;
/// Classification input height in pixels.
pub const CLS_HEIGHT: u32 = 48;
/// Classification input width in pixels.
pub const CLS_WIDTH: u32 = 192;
