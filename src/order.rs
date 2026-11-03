//! Box ordering and natural reading order sorting algorithms (#3).

use crate::geometry::Point;

/// Order rect corners like Python get_mini_boxes (sort by x, then
/// top-left first): [top-left, top-right, bottom-right, bottom-left].
/// perspective_crop relies on this winding.
pub fn order_mini_box(rect: &[Point]) -> Vec<Point> {
    if rect.len() != 4 {
        return rect.to_vec();
    }
    let mut pts = rect.to_vec();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    let (i1, i4) = if pts[1].y > pts[0].y { (0, 1) } else { (1, 0) };
    let (i2, i3) = if pts[3].y > pts[2].y { (2, 3) } else { (3, 2) };
    vec![pts[i1], pts[i2], pts[i3], pts[i4]]
}

/// Sort bounding boxes and their associated confidence scores in natural reading order.
///
/// Parity implementation of Python RapidOCR / PaddleOCR `sorted_boxes`:
/// 1. Primary order sorts by top-left corner `(y, x)`.
/// 2. Secondary order swaps adjacent boxes within vertical line grouping tolerance
///    (`abs(y2 - y1) < 10.0` px) if the later box appears further to the left (`x2 < x1`),
///    ensuring natural left-to-right reading order within lines.
///
/// # Arguments
/// * `boxes` - Mutable slice of bounding boxes, each represented by a vector of points.
/// * `scores` - Mutable slice of confidence scores matching each box.
#[allow(dead_code)]
pub fn sort_boxes_and_scores(boxes: &mut [Vec<Point>], scores: &mut [f32]) {
    assert_eq!(
        boxes.len(),
        scores.len(),
        "boxes and scores count must match"
    );
    let n = boxes.len();
    if n <= 1 {
        return;
    }

    // Step 1: initial sort indices by top-left (y, x).
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let ay = boxes[a].first().map(|p| p.y).unwrap_or(0.0);
        let by = boxes[b].first().map(|p| p.y).unwrap_or(0.0);
        let ax = boxes[a].first().map(|p| p.x).unwrap_or(0.0);
        let bx = boxes[b].first().map(|p| p.x).unwrap_or(0.0);
        ay.partial_cmp(&by)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Step 2: insertion-sort pass to group text boxes on the same horizontal text line (dy < 10.0 px).
    // In PaddleOCR, boxes within 10px vertical difference are treated as belonging to the same
    // line, and therefore ordered strictly left-to-right by x coordinate.
    for i in 0..(n - 1) {
        for j in (0..=i).rev() {
            let idx_next = indices[j + 1];
            let idx_curr = indices[j];
            if boxes[idx_next].is_empty() || boxes[idx_curr].is_empty() {
                break;
            }
            let dy = (boxes[idx_next][0].y - boxes[idx_curr][0].y).abs();
            if dy < 10.0 && boxes[idx_next][0].x < boxes[idx_curr][0].x {
                indices.swap(j, j + 1);
            } else {
                break;
            }
        }
    }

    // Apply sorted permutation to both boxes and scores in-place.
    let orig_boxes = boxes.to_vec();
    let orig_scores = scores.to_vec();
    for (new_pos, &old_idx) in indices.iter().enumerate() {
        boxes[new_pos] = orig_boxes[old_idx].clone();
        scores[new_pos] = orig_scores[old_idx];
    }
}

/// Sort bounding boxes in natural reading order (top-to-bottom, left-to-right within lines).
///
/// Convenience wrapper over [`sort_boxes_and_scores`].
///
/// # Arguments
/// * `boxes` - Mutable slice of bounding boxes to reorder in-place.
#[allow(dead_code)]
pub fn sort_boxes(boxes: &mut [Vec<Point>]) {
    let mut dummy_scores = vec![0.0f32; boxes.len()];
    sort_boxes_and_scores(boxes, &mut dummy_scores);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_box_orders_top_left_first() {
        // Python get_mini_boxes parity: sorted by x, TL first.
        let rect = vec![
            Point { x: 10.0, y: 0.0 },
            Point { x: 10.0, y: 5.0 },
            Point { x: 0.0, y: 0.0 },
            Point { x: 0.0, y: 5.0 },
        ];
        let out = order_mini_box(&rect);
        assert_eq!((out[0].x, out[0].y), (0.0, 0.0));
        assert_eq!((out[1].x, out[1].y), (10.0, 0.0));
        assert_eq!((out[2].x, out[2].y), (10.0, 5.0));
        assert_eq!((out[3].x, out[3].y), (0.0, 5.0));
    }

    #[test]
    fn sort_boxes_orders_multiline_and_same_line_left_to_right() {
        // Line 1: Word B (x=150, y=20) and Word A (x=30, y=22) -> dy=2 < 10, so A then B.
        // Line 2: Word D (x=180, y=80) and Word C (x=30, y=79) -> dy=1 < 10, so C then D.
        let mut boxes = vec![
            vec![
                Point { x: 150.0, y: 20.0 },
                Point { x: 200.0, y: 20.0 },
                Point { x: 200.0, y: 35.0 },
                Point { x: 150.0, y: 35.0 },
            ],
            vec![
                Point { x: 180.0, y: 80.0 },
                Point { x: 230.0, y: 80.0 },
                Point { x: 230.0, y: 95.0 },
                Point { x: 180.0, y: 95.0 },
            ],
            vec![
                Point { x: 30.0, y: 22.0 },
                Point { x: 80.0, y: 22.0 },
                Point { x: 80.0, y: 37.0 },
                Point { x: 30.0, y: 37.0 },
            ],
            vec![
                Point { x: 30.0, y: 79.0 },
                Point { x: 80.0, y: 79.0 },
                Point { x: 80.0, y: 94.0 },
                Point { x: 30.0, y: 94.0 },
            ],
        ];
        let mut scores = vec![0.90, 0.95, 0.85, 0.88];
        sort_boxes_and_scores(&mut boxes, &mut scores);

        // Word A (x=30, y=22)
        assert_eq!(boxes[0][0].x, 30.0);
        assert_eq!(scores[0], 0.85);

        // Word B (x=150, y=20)
        assert_eq!(boxes[1][0].x, 150.0);
        assert_eq!(scores[1], 0.90);

        // Word C (x=30, y=79)
        assert_eq!(boxes[2][0].x, 30.0);
        assert_eq!(scores[2], 0.88);

        // Word D (x=180, y=80)
        assert_eq!(boxes[3][0].x, 180.0);
        assert_eq!(scores[3], 0.95);
    }

    #[test]
    fn sort_boxes_handles_distinct_vertical_lines() {
        // Multi-line in separate blocks: Box 1 (y=10), Box 2 (y=50), Box 3 (y=100)
        let mut boxes = vec![
            vec![Point { x: 10.0, y: 100.0 }],
            vec![Point { x: 10.0, y: 10.0 }],
            vec![Point { x: 10.0, y: 50.0 }],
        ];
        sort_boxes(&mut boxes);
        assert_eq!(boxes[0][0].y, 10.0);
        assert_eq!(boxes[1][0].y, 50.0);
        assert_eq!(boxes[2][0].y, 100.0);
    }

    #[test]
    fn sort_boxes_empty_and_single() {
        let mut empty: Vec<Vec<Point>> = Vec::new();
        sort_boxes(&mut empty);
        assert!(empty.is_empty());

        let mut single = vec![vec![Point { x: 5.0, y: 5.0 }]];
        sort_boxes(&mut single);
        assert_eq!(single.len(), 1);
    }
}
