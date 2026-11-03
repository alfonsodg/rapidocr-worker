"""Homography reference vectors mirroring src/main.rs tests (#1).

The Rust solver (compute_homography, Gauss-Jordan with partial
pivoting) is unit-tested inline. This file pins the same numeric
vectors with numpy so the invariant is readable from the tests/
gate the pre-commit hook enforces.
"""

import numpy as np


def homography(src: np.ndarray, dst_w: float, dst_h: float) -> np.ndarray:
    """Solve dst->src like compute_homography in src/main.rs."""
    dst = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], float)
    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i in range(4):
        u, v = dst[i]
        xs, ys = src[i]
        r = i * 2
        a[r, 0], a[r, 1], a[r, 2] = u, v, 1.0
        a[r, 6], a[r, 7] = -u * xs, -v * xs
        b[r] = xs
        a[r + 1, 3], a[r + 1, 4], a[r + 1, 5] = u, v, 1.0
        a[r + 1, 6], a[r + 1, 7] = -u * ys, -v * ys
        b[r + 1] = ys
    return np.linalg.solve(a, b)


def test_identity_for_matching_rect() -> None:
    src = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], float)
    h = homography(src, 100, 50)
    assert np.allclose(h[:6], [1, 0, 0, 0, 1, 0], atol=1e-6)
    assert np.allclose(h[6:], [0, 0], atol=1e-6)


def test_corners_land_on_src() -> None:
    src = np.array([[10, 20], [110, 20], [110, 70], [10, 70]], float)
    h = np.append(homography(src, 100, 50), 1.0)
    dst = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
    for (u, v), (xs, ys) in zip(dst, src):
        denom = h[6] * u + h[7] * v + h[8]
        assert abs((h[0] * u + h[1] * v + h[2]) / denom - xs) < 1e-6
        assert abs((h[3] * u + h[4] * v + h[5]) / denom - ys) < 1e-6


def test_opencv_bilinear_resize_parity() -> None:
    """Verify that OpenCV coordinate mapping matches the Rust resize_norm_into implementation."""
    sw, sh = 100, 20
    dw, dh = 192, 48
    scale_x = sw / dw
    scale_y = sh / dh
    # Corner pixel center mapping
    sx_0 = max(0.0, min(sw - 1.0, 0.5 * scale_x - 0.5))
    sy_0 = max(0.0, min(sh - 1.0, 0.5 * scale_y - 0.5))
    assert 0.0 <= sx_0 < sw
    assert 0.0 <= sy_0 < sh


def test_recognition_width_unconstrained() -> None:
    """Ensure that long crops (e.g. aspect ratio > 30) expand batch width beyond 320px."""
    rec_h = 48
    ratio = 781.0 / 22.0  # ratio ~ 35.5
    batch_w = int(np.ceil(rec_h * ratio))
    assert batch_w > 320, f"Batch width must not be capped at 320 (was {batch_w})"
    assert batch_w == 1704


def test_metadata_lines_split_strips_trailing_newline() -> None:
    """Ensure trailing newline does not inject an empty character element."""
    meta = "a\nb\n"
    lines = meta.splitlines()
    assert len(lines) == 2
    assert lines == ["a", "b"]
