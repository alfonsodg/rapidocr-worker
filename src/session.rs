//! ONNX Runtime session builder and execution timing contracts (#9, #19).

use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

#[cfg(all(target_os = "linux", target_env = "gnu"))]
unsafe extern "C" {
    fn malloc_trim(pad: usize) -> i32;
}

/// Release free pages at the top of the glibc heap back to the OS.
///
/// Once per-stage transient buffers (`resized`, `det_tensor`, `det_out`, `boxes`,
/// `img`, `crops`) are dropped in strict reverse-lifetime order, the detection
/// high-water region is no longer pinned by downstream allocations and
/// `malloc_trim(0)` reclaims it immediately.
pub fn trim_allocator_heap() -> bool {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        malloc_trim(0) != 0
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        false
    }
}

/// Per-stage wall-clock times for one page in milliseconds, exposed in the worker response.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct StageTimings {
    pub det_ms: f32,
    pub cls_ms: f32,
    pub rec_ms: f32,
}

/// Helper to build an ONNX Runtime session with intra- and inter-thread configurations.
///
/// # Arguments
/// * `path` - Filesystem path to the ONNX model file.
/// * `threads` - Number of intra-operator parallel threads to assign to this session.
/// * `arena_enabled` - When `false`, disables the ONNX Runtime CPU memory arena
///   (`DisableCpuMemArena`) and memory patterns (`DisableMemPattern`) so the runtime
///   releases inference scratch buffers back to the allocator instead of retaining the
///   high-water working set per session (#19). Memory patterns are pointless with
///   the dynamic input shapes this worker feeds (variable crop widths and page sizes),
///   so disabling both together is coherent; the flag is opt-in and defaults to arena
///   enabled for byte-identical parity with the previous binary.
pub fn build_session(path: &str, threads: usize, arena_enabled: bool) -> Session {
    let builder = Session::builder()
        .expect("builder")
        .with_intra_threads(threads)
        .expect("intra_threads")
        .with_inter_threads(1)
        .expect("inter_threads")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("opt");
    let mut builder = if arena_enabled {
        builder
    } else {
        builder
            .with_memory_pattern(false)
            .expect("memory_pattern")
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])
            .expect("execution_providers")
    };
    builder.commit_from_file(path).expect("session")
}
