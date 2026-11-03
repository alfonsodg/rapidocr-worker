"""Guard ONNX memory-pattern disable and per-stage heap release for RSS bounding."""

from pathlib import Path


def test_session_disables_memory_pattern() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    session_rs = repo_root / "src" / "session.rs"
    assert session_rs.exists()
    text = session_rs.read_text(encoding="utf-8")
    assert "with_memory_pattern(false)" in text, "ONNX memory pattern must be disabled to bound RSS"
    assert "pub fn trim_allocator_heap()" in text, "session.rs must expose trim_allocator_heap()"
    assert "malloc_trim(0)" in text, "trim_allocator_heap() must invoke malloc_trim(0) on glibc"


def test_engine_drops_stage_transients_before_downstream_allocations() -> None:
    """Ensure det_out/det_tensor are dropped before boxes/crops and heap is trimmed after rec.

    Keeping det_out, det_tensor, resized, or img alive across boxes/crops/cls/rec
    forces downstream allocations above the ~200 MB detection high-water mark in
    glibc's sbrk [heap], pinning RSS at 156-258 MB instead of 74-75 MB.
    """
    repo_root = Path(__file__).resolve().parent.parent
    engine_rs = repo_root / "src" / "engine.rs"
    text = engine_rs.read_text(encoding="utf-8")

    assert "drop(bitmap);" in text, "det bitmap copy must be dropped after boxes_from_bitmap"
    assert "drop(boxes);" in text, "boxes must be dropped right after crop extraction"
    assert "drop(img);" in text, "full decoded image must be dropped right after crop extraction"
    assert "drop(crops);" in text, "crops must be dropped right after recognition stage"
    assert "trim_allocator_heap();" in text, "process() must trim the allocator heap after dropping transients"

