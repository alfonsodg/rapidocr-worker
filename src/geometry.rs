//! 2D computational geometry primitives and polygon algorithms for OCR.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct Polygon {
    pub points: Vec<Point>,
}

impl Polygon {
    pub fn from_points(points: &[Point]) -> Self {
        Self {
            points: points.to_vec(),
        }
    }

    /// Shoelace formula (absolute value).
    pub fn area(&self) -> f32 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for i in 0..n {
            let j = (i + 1) % n;
            sum += self.points[i].x * self.points[j].y - self.points[j].x * self.points[i].y;
        }
        (sum / 2.0).abs()
    }

    pub fn perimeter(&self) -> f32 {
        let n = self.points.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = self.points[j].x - self.points[i].x;
            let dy = self.points[j].y - self.points[i].y;
            sum += (dx * dx + dy * dy).sqrt();
        }
        sum
    }
}

/// Offset a simple polygon outward by `distance`, one parallel edge at a
/// time with round joins (pyclipper JT_ROUND parity for convex quads).
pub fn offset_polygon(pts: &[Point], distance: f32) -> Vec<Point> {
    let n = pts.len();
    if n < 3 || distance <= 0.0 {
        return pts.to_vec();
    }
    let mut signed = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        signed += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    let ccw = signed > 0.0;
    let mut lines: Vec<(f32, f32, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = pts[j].x - pts[i].x;
        let dy = pts[j].y - pts[i].y;
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        let (nx, ny) = if ccw {
            (dy / len, -dx / len)
        } else {
            (-dy / len, dx / len)
        };
        lines.push((nx, ny, -(nx * pts[i].x + ny * pts[i].y) - distance));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = (i + n - 1) % n;
        let (a1, b1, c1) = lines[prev];
        let (a2, b2, c2) = lines[i];
        let det = a1 * b2 - a2 * b1;
        if det.abs() < 1e-9 {
            out.push(pts[i]);
            continue;
        }
        out.push(Point {
            x: (b1 * c2 - b2 * c1) / det,
            y: (c1 * a2 - c2 * a1) / det,
        });
    }
    out
}

/// Even-odd rule point-in-polygon test.
pub fn point_in_polygon(px: f32, py: f32, poly: &[Point]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i].x, poly[i].y);
        let (xj, yj) = (poly[j].x, poly[j].y);
        if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Minimum-area bounding rectangle via rotating calipers approximation.
pub fn min_area_rect(points: &[Point]) -> (Vec<Point>, f32) {
    if points.len() < 3 {
        return (points.to_vec(), 0.0);
    }
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return (hull.clone(), 0.0);
    }
    let mut best_area = f32::MAX;
    let mut best_rect = hull.clone();
    let n = hull.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let edge = Point {
            x: hull[j].x - hull[i].x,
            y: hull[j].y - hull[i].y,
        };
        let len = (edge.x * edge.x + edge.y * edge.y).sqrt();
        if len < 1e-6 {
            continue;
        }
        let ux = edge.x / len;
        let uy = edge.y / len;
        let vx = -uy;
        let vy = ux;
        let mut min_u = f32::MAX;
        let mut max_u = f32::MIN;
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        for p in hull.iter() {
            let du = p.x * ux + p.y * uy;
            let dv = p.x * vx + p.y * vy;
            min_u = min_u.min(du);
            max_u = max_u.max(du);
            min_v = min_v.min(dv);
            max_v = max_v.max(dv);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            let p1 = Point {
                x: min_u * ux + min_v * vx,
                y: min_u * uy + min_v * vy,
            };
            let p2 = Point {
                x: max_u * ux + min_v * vx,
                y: max_u * uy + min_v * vy,
            };
            let p3 = Point {
                x: max_u * ux + max_v * vx,
                y: max_u * uy + max_v * vy,
            };
            let p4 = Point {
                x: min_u * ux + max_v * vx,
                y: min_u * uy + max_v * vy,
            };
            best_rect = vec![p1, p2, p3, p4];
        }
    }
    let side = {
        let d1 = ((best_rect[1].x - best_rect[0].x).powi(2)
            + (best_rect[1].y - best_rect[0].y).powi(2))
        .sqrt();
        let d2 = ((best_rect[3].x - best_rect[0].x).powi(2)
            + (best_rect[3].y - best_rect[0].y).powi(2))
        .sqrt();
        d1.min(d2)
    };
    (best_rect, side)
}

/// Andrew's monotone chain convex hull.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });
    pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-6 && (a.y - b.y).abs() < 1e-6);
    if pts.len() < 3 {
        return pts;
    }
    let cross =
        |o: &Point, a: &Point, b: &Point| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let mut lower = Vec::new();
    for p in pts.iter() {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_area_rect_matches_opencv_area() {
        let pts = vec![
            Point { x: 1.0, y: 73.0 },
            Point { x: -1.15, y: 5.34 },
            Point { x: 192.18, y: -0.8 },
            Point {
                x: 194.33,
                y: 66.86,
            },
        ];
        let (rect, _side) = min_area_rect(&pts);
        assert_eq!(rect.len(), 4);
        let area = Polygon::from_points(&rect).area();
        assert!((area - 13093.99).abs() / 13093.99 < 0.01, "area {area}");
    }

    #[test]
    fn unclip_expands_outward_like_pyclipper() {
        let rect = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 10.0, y: 0.0 },
            Point { x: 10.0, y: 10.0 },
            Point { x: 0.0, y: 10.0 },
        ];
        let out = offset_polygon(&rect, 4.0);
        assert_eq!(out.len(), 4);
        let xs: Vec<f32> = out.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = out.iter().map(|p| p.y).collect();
        assert!((xs.iter().cloned().fold(f32::MAX, f32::min) + 4.0).abs() < 0.01);
        assert!((xs.iter().cloned().fold(f32::MIN, f32::max) - 14.0).abs() < 0.01);
        assert!((ys.iter().cloned().fold(f32::MAX, f32::min) + 4.0).abs() < 0.01);
        assert!((ys.iter().cloned().fold(f32::MIN, f32::max) - 14.0).abs() < 0.01);
    }

    #[test]
    fn unclip_works_for_cw_winding() {
        let rect = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 0.0, y: 10.0 },
            Point { x: 10.0, y: 10.0 },
            Point { x: 10.0, y: 0.0 },
        ];
        let out = offset_polygon(&rect, 4.0);
        let xs: Vec<f32> = out.iter().map(|p| p.x).collect();
        assert!(xs.iter().cloned().fold(f32::MAX, f32::min) < 0.0);
        assert!(xs.iter().cloned().fold(f32::MIN, f32::max) > 10.0);
    }
}
