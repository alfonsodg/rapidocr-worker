//! Contour tracing parity with cv2.findContours RETR_LIST plus
//! CHAIN_APPROX_SIMPLE.
//!
//! Union-find labeling (8-connectivity) plus Moore boundary walk with
//! collinear compression. The old flood fill returned all interior
//! pixels, which collapsed into one hull in min_area_rect and lost
//! text boxes (quality regression 4/11 vs 11/11).

use crate::geometry::Point;

pub fn find_contours(bitmap: &[u8], width: usize, height: usize) -> Vec<Vec<Point>> {
    let mut labels = vec![0u32; width * height];
    let mut current: u32 = 0;
    let mut parent: Vec<u32> = vec![0];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if bitmap[idx] == 0 {
                continue;
            }
            let mut roots = Vec::new();
            // 4 already-scanned 8-neighbors: W, NW, N, NE.
            let neighbors = [
                (x > 0, x - 1, y),
                (x > 0 && y > 0, x - 1, y - 1),
                (y > 0, x, y - 1),
                (x + 1 < width && y > 0, x + 1, y - 1),
            ];
            for (ok, nx, ny) in neighbors {
                if ok {
                    let l = labels[ny * width + nx];
                    if l != 0 {
                        roots.push(find_root(&mut parent, l));
                    }
                }
            }
            if roots.is_empty() {
                current += 1;
                parent.push(current);
                labels[idx] = current;
            } else {
                let root = *roots.iter().min().unwrap();
                labels[idx] = root;
                for other in roots {
                    let a = find_root(&mut parent, root);
                    let b = find_root(&mut parent, other);
                    if a != b {
                        if a < b {
                            parent[b as usize] = a;
                        } else {
                            parent[a as usize] = b;
                        }
                    }
                }
            }
        }
    }
    let mut groups: std::collections::HashMap<u32, Vec<(i32, i32)>> =
        std::collections::HashMap::new();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if labels[idx] != 0 {
                let r = find_root(&mut parent, labels[idx]);
                groups.entry(r).or_default().push((x as i32, y as i32));
            }
        }
    }
    let mut contours = Vec::new();
    for pixels in groups.values() {
        if pixels.len() < 3 {
            continue;
        }
        let set: std::collections::HashSet<(i32, i32)> = pixels.iter().copied().collect();
        let start = *pixels
            .iter()
            .min_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)))
            .unwrap();
        let boundary = moore_boundary(&set, start, width as i32, height as i32);
        if boundary.len() >= 3 {
            contours.push(approx_poly(&boundary));
        }
    }
    contours
}

fn find_root(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        x = parent[x as usize];
    }
    x
}

/// Moore-neighbor boundary tracing with Jacob's stopping criterion.
fn moore_boundary(
    set: &std::collections::HashSet<(i32, i32)>,
    start: (i32, i32),
    width: i32,
    height: i32,
) -> Vec<Point> {
    const DIRS: [(i32, i32); 8] = [
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    ];
    let mut boundary = vec![Point {
        x: start.0 as f32,
        y: start.1 as f32,
    }];
    let mut current = start;
    let mut from_dir = 0usize;
    loop {
        let mut found: Option<((i32, i32), usize)> = None;
        for k in 1..=8 {
            let di = (from_dir + k) % 8;
            let nb = (current.0 + DIRS[di].0, current.1 + DIRS[di].1);
            if nb.0 >= 0 && nb.1 >= 0 && nb.0 < width && nb.1 < height && set.contains(&nb) {
                found = Some((nb, di));
                break;
            }
        }
        let (next, dir_idx) = match found {
            Some(v) => v,
            None => break,
        };
        from_dir = (dir_idx + 4) % 8;
        current = next;
        if current == start && boundary.len() > 2 {
            break;
        }
        boundary.push(Point {
            x: current.0 as f32,
            y: current.1 as f32,
        });
        if boundary.len() > set.len() + 4 {
            break;
        }
    }
    boundary
}

/// Compress collinear runs (CHAIN_APPROX_SIMPLE parity).
fn approx_poly(pts: &[Point]) -> Vec<Point> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut out = Vec::new();
    let n = pts.len();
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let cur = pts[i];
        let next = pts[(i + 1) % n];
        if ((cur.x - prev.x) * (next.y - cur.y) - (cur.y - prev.y) * (next.x - cur.x)).abs()
            > f32::EPSILON
        {
            out.push(cur);
        }
    }
    if out.len() < 3 {
        return pts.to_vec();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_rect(
        width: usize,
        height: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
    ) -> Vec<u8> {
        let mut m = vec![0u8; width * height];
        for y in y0..=y1 {
            for x in x0..=x1 {
                m[y * width + x] = 1;
            }
        }
        m
    }

    #[test]
    fn contours_trace_boundary_not_interior() {
        let contours = find_contours(&mask_rect(6, 5, 1, 1, 4, 3), 6, 5);
        assert_eq!(contours.len(), 1);
        assert!(contours[0].len() <= 8, "walk: {:?}", contours[0].len());
        assert!(contours[0].len() >= 4);
    }

    #[test]
    fn contours_find_each_component() {
        let mut m = mask_rect(12, 6, 1, 1, 3, 3);
        for y in 1..=3 {
            for x in 7..=9 {
                m[y * 12 + x] = 1;
            }
        }
        assert_eq!(find_contours(&m, 12, 6).len(), 2);
    }

    #[test]
    fn contours_find_nested_text_inside_hole() {
        let mut m = mask_rect(12, 12, 1, 1, 10, 10);
        for y in 4..=7 {
            for x in 4..=7 {
                m[y * 12 + x] = 0;
            }
        }
        m[5 * 12 + 5] = 1;
        m[5 * 12 + 6] = 1;
        m[6 * 12 + 5] = 1;
        assert_eq!(find_contours(&m, 12, 12).len(), 2);
    }
}
