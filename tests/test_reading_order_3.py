"""Reading order reference test mirroring PaddleOCR / RapidOCR sorted_boxes (#3).

Pins the reading order logic (sorting top-to-bottom, left-to-right within
a vertical row tolerance dy < 10px) to verify exact parity between Python and
Rust implementations.
"""

from typing import List
import numpy as np


def python_sorted_boxes(dt_boxes: np.ndarray) -> List[np.ndarray]:
    """RapidOCR python reference implementation (ch_ppocr_det/main.py:221)."""
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if (
                abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10
                and boxes[j + 1][0][0] < boxes[j][0][0]
            ):
                tmp = boxes[j]
                boxes[j] = boxes[j + 1]
                boxes[j + 1] = tmp
            else:
                break
    return boxes


def test_sorted_boxes_reading_order_parity() -> None:
    # 2 lines with 2 words each, scrambled in detection output
    # Line 1 (y ~ 20): Word A (x=30, y=22), Word B (x=150, y=20)
    # Line 2 (y ~ 80): Word C (x=30, y=79), Word D (x=180, y=80)
    box_b = np.array([[150, 20], [200, 20], [200, 35], [150, 35]], dtype=float)
    box_d = np.array([[180, 80], [230, 80], [230, 95], [180, 95]], dtype=float)
    box_a = np.array([[30, 22], [80, 22], [80, 37], [30, 37]], dtype=float)
    box_c = np.array([[30, 79], [80, 79], [80, 94], [30, 94]], dtype=float)

    raw_boxes = np.array([box_b, box_d, box_a, box_c])
    ordered = python_sorted_boxes(raw_boxes)

    # Word A (x=30, y=22) should be first
    assert ordered[0][0][0] == 30.0 and ordered[0][0][1] == 22.0
    # Word B (x=150, y=20) second
    assert ordered[1][0][0] == 150.0 and ordered[1][0][1] == 20.0
    # Word C (x=30, y=79) third
    assert ordered[2][0][0] == 30.0 and ordered[2][0][1] == 79.0
    # Word D (x=180, y=80) fourth
    assert ordered[3][0][0] == 180.0 and ordered[3][0][1] == 80.0
