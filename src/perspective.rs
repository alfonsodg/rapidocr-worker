//! Perspective transformation and planar homography solver for text crops (#1).

use crate::geometry::Point;

/// Order 4 points as top-left, top-right, bottom-right, bottom-left.
pub fn order_points(pts: &[Point]) -> [Point; 4] {
    let mut p = [pts[0], pts[1], pts[2], pts[3]];
    // Sort by y, then x: topmost first.
    p.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));
    // Top two: left is TL, right is TR. Bottom two: left is BL, right is BR.
    if p[0].x > p[1].x {
        p.swap(0, 1);
    }
    if p[2].x > p[3].x {
        p.swap(2, 3);
    }
    [p[0], p[1], p[3], p[2]] // TL, TR, BR, BL
}

/// Compute the 3x3 homography mapping dst rectangle corners to src points.
/// Returns [h11..h33] with h33=1, solving dst→src (for inverse warping).
pub fn compute_homography(src: &[Point; 4], dst_w: u32, dst_h: u32) -> [f32; 9] {
    let dst = [
        (0.0f32, 0.0f32),
        (dst_w as f32, 0.0),
        (dst_w as f32, dst_h as f32),
        (0.0, dst_h as f32),
    ];
    let mut a = [[0.0f32; 8]; 8];
    let mut b = [0.0f32; 8];
    for i in 0..4 {
        let (u, v) = dst[i];
        let (xs, ys) = (src[i].x, src[i].y);
        let row = i * 2;
        a[row][0] = u;
        a[row][1] = v;
        a[row][2] = 1.0;
        a[row][6] = -u * xs;
        a[row][7] = -v * xs;
        b[row] = xs;
        a[row + 1][3] = u;
        a[row + 1][4] = v;
        a[row + 1][5] = 1.0;
        a[row + 1][6] = -u * ys;
        a[row + 1][7] = -v * ys;
        b[row + 1] = ys;
    }
    for col in 0..8 {
        let mut pivot = col;
        for r in col + 1..8 {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let d = a[col][col];
        if d.abs() < 1e-10 {
            continue;
        }
        for cell in a[col].iter_mut().skip(col) {
            *cell /= d;
        }
        b[col] /= d;
        let pivot_row = a[col];
        for (r, row_vals) in a.iter_mut().enumerate() {
            if r == col || row_vals[col].abs() <= 1e-10 {
                continue;
            }
            let f = row_vals[col];
            for (dst, src) in row_vals.iter_mut().skip(col).zip(pivot_row[col..].iter()) {
                *dst -= f * src;
            }
            b[r] -= f * b[col];
        }
    }
    [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1.0]
}

/// Perspective crop: map the source quadrilateral to a rectangle with bilinear sampling.
pub fn perspective_crop(img: &image::RgbImage, pts: &[Point]) -> image::RgbImage {
    if pts.len() < 4 {
        let xs: Vec<u32> = pts.iter().map(|p| p.x.max(0.0) as u32).collect();
        let ys: Vec<u32> = pts.iter().map(|p| p.y.max(0.0) as u32).collect();
        let xmin = xs.iter().copied().min().unwrap_or(0);
        let ymin = ys.iter().copied().min().unwrap_or(0);
        let xmax = xs
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .min(img.width().saturating_sub(1))
            .max(xmin);
        let ymax = ys
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .min(img.height().saturating_sub(1))
            .max(ymin);
        let w = xmax - xmin + 1;
        let h = ymax - ymin + 1;
        if w == 0 || h == 0 {
            return image::RgbImage::new(1, 1);
        }
        return image::imageops::crop_imm(img, xmin, ymin, w, h).to_image();
    }

    let ordered = order_points(pts);
    let w_top =
        ((ordered[1].x - ordered[0].x).powi(2) + (ordered[1].y - ordered[0].y).powi(2)).sqrt();
    let w_bot =
        ((ordered[2].x - ordered[3].x).powi(2) + (ordered[2].y - ordered[3].y).powi(2)).sqrt();
    let h_left =
        ((ordered[3].x - ordered[0].x).powi(2) + (ordered[3].y - ordered[0].y).powi(2)).sqrt();
    let h_right =
        ((ordered[2].x - ordered[1].x).powi(2) + (ordered[2].y - ordered[1].y).powi(2)).sqrt();
    let dst_w = (w_top.max(w_bot)).max(1.0) as u32;
    let dst_h = (h_left.max(h_right)).max(1.0) as u32;
    if dst_w < 2 || dst_h < 2 || dst_w > 5000 || dst_h > 5000 {
        return image::RgbImage::new(1, 1);
    }

    let h = compute_homography(&ordered, dst_w, dst_h);
    let (sw, sh) = (img.width() as i32, img.height() as i32);
    let mut out = image::RgbImage::new(dst_w, dst_h);
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let (u, v) = (dx as f32, dy as f32);
            let denom = h[6] * u + h[7] * v + h[8];
            if denom.abs() < 1e-10 {
                continue;
            }
            let sx = (h[0] * u + h[1] * v + h[2]) / denom;
            let sy = (h[3] * u + h[4] * v + h[5]) / denom;
            let mut px = [0u8; 3];
            for (c, slot) in px.iter_mut().enumerate() {
                let value = cubic_sample(img, sw, sh, sx, sy, c);
                *slot = value.round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(dx, dy, image::Rgb(px));
        }
    }
    out
}

/// OpenCV INTER_CUBIC weight (A = -0.75) for the distance from the sample point.
fn cubic_weight(t: f32) -> f32 {
    const A: f32 = -0.75;
    let a = t.abs();
    if a <= 1.0 {
        (A + 2.0) * a * a * a - (A + 3.0) * a * a + 1.0
    } else if a < 2.0 {
        A * a * a * a - 5.0 * A * a * a + 8.0 * A * a - 4.0 * A
    } else {
        0.0
    }
}

/// Sample one channel with cv2 INTER_CUBIC semantics (4x4, BORDER_REPLICATE).
fn cubic_sample(img: &image::RgbImage, sw: i32, sh: i32, sx: f32, sy: f32, c: usize) -> f32 {
    let x0 = sx.floor() as i32;
    let y0 = sy.floor() as i32;
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;
    let sample = |xi: i32, yi: i32| -> f32 {
        let xc = xi.clamp(0, sw - 1);
        let yc = yi.clamp(0, sh - 1);
        img.get_pixel(xc as u32, yc as u32).0[c] as f32
    };
    let mut acc = 0.0f32;
    for m in -1..=2i32 {
        let wy = cubic_weight(fy - m as f32);
        for n in -1..=2i32 {
            let wx = cubic_weight(fx - n as f32);
            acc += wx * wy * sample(x0 + n, y0 + m);
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_identity_for_matching_rect() {
        let src = [
            Point { x: 0.0, y: 0.0 },
            Point { x: 100.0, y: 0.0 },
            Point { x: 100.0, y: 50.0 },
            Point { x: 0.0, y: 50.0 },
        ];
        let h = compute_homography(&src, 100, 50);
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (got, want) in h.iter().zip(identity.iter()) {
            assert!((got - want).abs() < 1e-3, "h {h:?}");
        }
    }

    #[test]
    fn homography_maps_corners_to_src() {
        let src = [
            Point { x: 10.0, y: 20.0 },
            Point { x: 110.0, y: 20.0 },
            Point { x: 110.0, y: 70.0 },
            Point { x: 10.0, y: 70.0 },
        ];
        let h = compute_homography(&src, 100, 50);
        let dst = [(0.0f32, 0.0f32), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)];
        for (i, (u, v)) in dst.iter().enumerate() {
            let denom = h[6] * u + h[7] * v + h[8];
            let xs = (h[0] * u + h[1] * v + h[2]) / denom;
            let ys = (h[3] * u + h[4] * v + h[5]) / denom;
            assert!((xs - src[i].x).abs() < 1e-2, "x {xs} vs {}", src[i].x);
            assert!((ys - src[i].y).abs() < 1e-2, "y {ys} vs {}", src[i].y);
        }
    }
}
