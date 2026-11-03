//! PP-OCR det postprocess: DB binarization, contour tracing,
//! box scoring, and geometric unclip.

use crate::contours::find_contours;
use crate::geometry::{min_area_rect, offset_polygon, point_in_polygon, Point, Polygon};
use crate::order::{order_mini_box, sort_boxes_and_scores};
use ndarray::Array4;

/// Normalize an RGB image to (1, 3, H, W) with per-channel mean/std for det inference.
///
/// PP-OCRv4 uses a scalar `[0.5, 0.5, 0.5]`; PP-OCRv6 specifies per-channel
/// ImageNet statistics (`inference.yml`). Applying the scalar path to v6 was
/// measured to drop detection confidence (p99) from 0.972 to 0.825, so the
/// channel values are explicit rather than a single scalar.
pub fn normalize_chw(img: &image::RgbImage, mean: [f32; 3], std: [f32; 3]) -> Array4<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut arr = Array4::<f32>::zeros((1, 3, h, w));
    for (x, y, px) in img.enumerate_pixels() {
        for c in 0..3usize {
            let v = px[c] as f32 / 255.0;
            *arr.get_mut([0, c, y as usize, x as usize]).unwrap() = (v - mean[c]) / std[c];
        }
    }
    arr
}

/// Resize with OpenCV INTER_LINEAR semantics (half-pixel centers, clamped edges).
pub fn cv_resize_bilinear(img: &image::RgbImage, nw: u32, nh: u32) -> image::RgbImage {
    let (sw, sh) = (img.width(), img.height());
    if sw == 0 || sh == 0 || nw == 0 || nh == 0 {
        return image::RgbImage::new(nw.max(1), nh.max(1));
    }
    let scale_x = sw as f32 / nw as f32;
    let scale_y = sh as f32 / nh as f32;
    let mut out = image::RgbImage::new(nw, nh);
    for dy in 0..nh {
        let sy = ((dy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = sy - y0 as f32;
        for dx in 0..nw {
            let sx = ((dx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = sx.floor() as u32;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = sx - x0 as f32;
            let p00 = img.get_pixel(x0, y0).0;
            let p10 = img.get_pixel(x1, y0).0;
            let p01 = img.get_pixel(x0, y1).0;
            let p11 = img.get_pixel(x1, y1).0;
            let mut px = [0u8; 3];
            for c in 0..3 {
                let v = p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
                    + p10[c] as f32 * fx * (1.0 - fy)
                    + p01[c] as f32 * (1.0 - fx) * fy
                    + p11[c] as f32 * fx * fy;
                px[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(dx, dy, image::Rgb(px));
        }
    }
    out
}

/// DetResizeForTest parity with Python: limit_type=min, int truncation, then
/// round to a multiple of 32; no resize when the min side already exceeds the
/// limit. Returns None when the truncated size is empty (Python returns None).
pub fn det_resize_target(width: u32, height: u32, limit_side_len: u32) -> Option<(u32, u32)> {
    let (wf, hf) = (width as f32, height as f32);
    let min_side = width.min(height) as f32;
    let ratio = if min_side < limit_side_len as f32 {
        limit_side_len as f32 / min_side
    } else {
        1.0
    };
    let rw = (wf * ratio) as u32;
    let rh = (hf * ratio) as u32;
    if rw == 0 || rh == 0 {
        return None;
    }
    let nw = ((rw as f32 / 32.0).round() as u32).max(1) * 32;
    let nh = ((rh as f32 / 32.0).round() as u32).max(1) * 32;
    Some((nw, nh))
}

pub struct DbPostProcess {
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
    pub min_size: f32,
}

impl Default for DbPostProcess {
    fn default() -> Self {
        Self {
            thresh: 0.3,
            box_thresh: 0.5,
            max_candidates: 1000,
            unclip_ratio: 1.6,
            min_size: 3.0,
        }
    }
}

impl DbPostProcess {
    /// PP-OCRv6 detection thresholds, taken from its `inference.yml`.
    ///
    /// The v4 defaults are noticeably stricter (0.3 / 0.5 / 1000 / 1.6); running
    /// a v6 detector with them changes which boxes survive, so the profile is
    /// selected explicitly rather than inferred from the model file.
    pub fn ppocrv6() -> Self {
        Self {
            thresh: 0.2,
            box_thresh: 0.45,
            max_candidates: 3000,
            unclip_ratio: 1.4,
            min_size: 3.0,
        }
    }
}

impl DbPostProcess {
    /// Extract text boxes from the det probability map.
    pub fn boxes_from_bitmap(
        &self,
        pred: &[f32],
        width: usize,
        height: usize,
        dest_width: usize,
        dest_height: usize,
    ) -> (Vec<Vec<Point>>, Vec<f32>) {
        let bitmap = self.binarize(pred, width * height);
        let bitmap = Self::dilate_2x2(&bitmap, width, height);
        let contours = find_contours(&bitmap, width, height);
        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        for contour in contours.iter().take(self.max_candidates) {
            if contour.len() < 3 {
                continue;
            }
            let (rect, sside) = min_area_rect(contour);
            if sside < self.min_size {
                continue;
            }
            let rect = order_mini_box(&rect);
            let score = box_score_fast(pred, width, height, &rect);
            if self.box_thresh > score {
                continue;
            }
            let expanded = self.unclip(&rect);
            let (final_rect, final_side) = min_area_rect(&expanded);
            if final_side < self.min_size + 2.0 {
                continue;
            }
            let final_rect = order_mini_box(&final_rect);
            let scaled: Vec<Point> = final_rect
                .iter()
                .map(|p| Point {
                    x: (p.x / width as f32 * dest_width as f32)
                        .round()
                        .clamp(0.0, dest_width as f32),
                    y: (p.y / height as f32 * dest_height as f32)
                        .round()
                        .clamp(0.0, dest_height as f32),
                })
                .collect();
            boxes.push(scaled);
            scores.push(score);
        }
        sort_boxes_and_scores(&mut boxes, &mut scores);
        (boxes, scores)
    }

    fn binarize(&self, pred: &[f32], len: usize) -> Vec<u8> {
        pred.iter()
            .take(len)
            .map(|&v| if v > self.thresh { 1 } else { 0 })
            .collect()
    }

    /// Dilate the binarized map with a 2x2 kernel (Python use_dilation parity):
    /// a pixel stays set when it or any of its top/left neighbors is set.
    pub fn dilate_2x2(bitmap: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut out = vec![0u8; bitmap.len()];
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if bitmap[idx] == 1
                    || (x > 0 && bitmap[idx - 1] == 1)
                    || (y > 0 && bitmap[idx - width] == 1)
                    || (x > 0 && y > 0 && bitmap[idx - width - 1] == 1)
                {
                    out[idx] = 1;
                }
            }
        }
        out
    }

    /// Offset a polygon outward by the unclip distance (parity with
    /// pyclipper JT_ROUND + ET_CLOSEDPOLYGON).
    fn unclip(&self, pts: &[Point]) -> Vec<Point> {
        let poly = Polygon::from_points(pts);
        let distance = poly.area() * self.unclip_ratio / poly.perimeter();
        offset_polygon(pts, distance)
    }
}

/// Score a box by averaging prediction values inside its polygon (not just the bounding rect).
fn box_score_fast(pred: &[f32], width: usize, _height: usize, rect: &[Point]) -> f32 {
    let xs: Vec<f32> = rect.iter().map(|p| p.x).collect();
    let ys: Vec<f32> = rect.iter().map(|p| p.y).collect();
    let xmin = xs.iter().cloned().fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let xmax = xs
        .iter()
        .cloned()
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(width as f32 - 1.0) as usize;
    let ymin = ys.iter().cloned().fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let ymax = ys
        .iter()
        .cloned()
        .fold(f32::MIN, f32::max)
        .ceil()
        .min((pred.len() / width) as f32 - 1.0) as usize;
    if xmax <= xmin || ymax <= ymin || rect.len() < 3 {
        return 0.0;
    }
    // Polygon rasterization: for each pixel in the bounding rect, test if it's
    // inside the polygon using the even-odd rule, then average the prediction.
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            if point_in_polygon(x as f32 + 0.5, y as f32 + 0.5, rect) {
                sum += pred[y * width + x];
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_resize_target_keeps_full_resolution_above_limit() {
        // Python limit_type=min: a 1240x1754 page already exceeds 736, so the
        // ratio is 1.0 and only the round-to-32 applies.
        let (nw, nh) = det_resize_target(1240, 1754, 736).expect("target");
        assert_eq!(nw, 1248);
        assert_eq!(nh, 1760);
    }

    #[test]
    fn det_resize_target_downscales_small_sides_to_the_limit() {
        // A 600x800 page sits below the limit: ratio = 736/600, then round-32.
        // int(800*736/600)=981 -> round(981/32)*32 = 992 (verified in Python).
        let (nw, nh) = det_resize_target(600, 800, 736).expect("target");
        assert_eq!(nw, 736);
        assert_eq!(nh, 992);
    }

    #[test]
    fn det_resize_target_rejects_empty_sizes() {
        assert!(det_resize_target(0, 0, 736).is_none());
    }

    #[test]
    fn cv_resize_bilinear_is_identity_for_same_size() {
        let mut img = image::RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([10, 20, 30]));
        img.put_pixel(1, 0, image::Rgb([40, 50, 60]));
        img.put_pixel(0, 1, image::Rgb([70, 80, 90]));
        img.put_pixel(1, 1, image::Rgb([100, 110, 120]));
        let out = cv_resize_bilinear(&img, 2, 2);
        assert_eq!(out.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(out.get_pixel(1, 1).0, [100, 110, 120]);
    }

    #[test]
    fn cv_resize_bilinear_interpolates_the_center() {
        let mut img = image::RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([0, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([255, 255, 255]));
        let out = cv_resize_bilinear(&img, 4, 1);
        // Half-pixel centers: dst 1 and 2 sample the midpoint between columns.
        let mid = out.get_pixel(1, 0).0[0];
        assert!(mid > 60 && mid < 195, "mid {mid}");
    }

    #[test]
    fn boxes_from_bitmap_returns_boxes_in_reading_order() {
        // Two boxes on the same row; contour discovery order is not stable
        // (HashMap iteration), so the det postprocess must sort them itself.
        let (w, h) = (64usize, 16usize);
        let mut pred = vec![0.0f32; w * h];
        for y in 2..12 {
            for x in 4..20 {
                pred[y * w + x] = 0.9;
            }
            for x in 40..56 {
                pred[y * w + x] = 0.9;
            }
        }
        let post = DbPostProcess::default();
        let (boxes, scores) = post.boxes_from_bitmap(&pred, w, h, w, h);
        assert_eq!(boxes.len(), 2, "boxes {boxes:?}");
        assert_eq!(scores.len(), 2);
        assert!(
            boxes[0][0].x < boxes[1][0].x,
            "left-to-right reading order required: {boxes:?}"
        );
    }
}
