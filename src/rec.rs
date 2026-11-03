//! CTC label decode for the rec model.

pub struct CtcDecoder {
    /// Character list with "blank" at index 0 and " " appended at the end.
    pub characters: Vec<String>,
}

impl CtcDecoder {
    /// Read the character dictionary from the ONNX model metadata (newline-separated).
    pub fn from_metadata(meta: &str) -> Result<Self, String> {
        if meta.trim().is_empty() {
            return Err("empty character metadata".into());
        }
        let parsed: Vec<String> = meta.lines().map(|s| s.to_string()).collect();
        Self::from_symbols(parsed)
    }

    /// Read the character dictionary from an external keys file.
    ///
    /// PP-OCRv6 ships no `character` ONNX metadata, so the dictionary has to come
    /// from a file. The layout matches the metadata path exactly — one symbol per
    /// line, no blank entry, no trailing space — so the two sources stay
    /// interchangeable and a model swap cannot shift every label by one.
    pub fn from_keys_file(path: &std::path::Path) -> Result<Self, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|error| format!("read keys file {}: {error}", path.display()))?;
        if contents.trim().is_empty() {
            return Err(format!("empty keys file {}", path.display()));
        }
        let parsed: Vec<String> = contents.lines().map(|s| s.to_string()).collect();
        Self::from_symbols(parsed)
    }

    /// Build the decoder from an ordered symbol list (blank at 0, space last).
    fn from_symbols(parsed: Vec<String>) -> Result<Self, String> {
        if parsed.iter().all(|symbol| symbol.trim().is_empty()) {
            return Err("empty character dictionary".into());
        }
        let mut characters = vec!["blank".to_string()];
        characters.extend(parsed);
        characters.push(" ".to_string());
        Ok(Self { characters })
    }

    /// Greedy CTC decode: argmax per timestep, remove consecutive duplicates, drop blanks.
    pub fn decode(&self, probs: &[f32], num_classes: usize, num_timesteps: usize) -> (String, f32) {
        let mut text = String::new();
        let mut confidences = Vec::new();
        let mut prev_id = 0usize;
        for t in 0..num_timesteps {
            let offset = t * num_classes;
            let mut best_id = 0usize;
            let mut best_prob = 0.0f32;
            for c in 0..num_classes {
                let p = probs[offset + c];
                if p > best_prob {
                    best_prob = p;
                    best_id = c;
                }
            }
            if best_id != 0 && best_id != prev_id {
                if let Some(ch) = self.characters.get(best_id) {
                    text.push_str(ch);
                }
                confidences.push(best_prob);
            }
            prev_id = best_id;
        }
        let mean = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };
        (text, mean)
    }
}

/// Aspect ratio boundary where crops clamp to the minimum REC_WIDTH (320px).
pub const COMPACT_ASPECT_RATIO: f32 = 6.666_667;
/// Aspect ratio boundary above which crops are considered wide lines (>576px canvas).
pub const WIDE_ASPECT_RATIO: f32 = 12.0;

/// Plan recognition batches using aspect-ratio bucketing to minimize zero padding (#9).
///
/// Groups sorted crops into homogeneous aspect-ratio batches:
/// - Compact crops (ratio <= 6.67): batched up to `base_batch * 2` since they all clamp to
///   the minimum canvas width (320px), maximizing SIMD compute density without padding waste.
/// - Wide lines (ratio > 12.0): batched in smaller sizes to prevent oversized tensor allocations.
/// - Medium lines: batched with `base_batch`.
///
/// # Arguments
/// * `sorted_crops` - Slice of `(original_crop_index, aspect_ratio)` sorted by aspect ratio.
/// * `base_batch` - Base batch size configured for recognition (e.g. 6).
///
/// # Returns
/// A vector of batches, each containing the original crop indices to be processed together.
pub fn plan_rec_batches(sorted_crops: &[(usize, f32)], base_batch: usize) -> Vec<Vec<usize>> {
    if sorted_crops.is_empty() {
        return Vec::new();
    }
    let mut batches = Vec::new();
    let compact_cap = (base_batch * 2).max(base_batch);
    let wide_cap = (base_batch.saturating_sub(2)).max(2).min(base_batch);

    let mut current_batch = Vec::new();
    let mut current_cap = base_batch;

    for &(idx, ratio) in sorted_crops {
        let target_cap = if ratio <= COMPACT_ASPECT_RATIO {
            compact_cap
        } else if ratio > WIDE_ASPECT_RATIO {
            wide_cap
        } else {
            base_batch
        };

        if current_batch.is_empty() {
            current_cap = target_cap;
            current_batch.push(idx);
        } else if current_batch.len() < current_cap && current_cap == target_cap {
            current_batch.push(idx);
        } else {
            batches.push(current_batch);
            current_batch = vec![idx];
            current_cap = target_cap;
        }
    }
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }
    batches
}

/// Resize image to (target_w, target_h) using OpenCV-compatible bilinear interpolation,
/// normalize with (v - 0.5) / 0.5 and write into an ndarray view of shape [3, target_h, target_w].
/// Vec-backed wrapper for integration tests: renders the resize+normalize
/// output of one crop without exposing ndarray views to external crates.
pub fn rec_resize_norm_into_vec(
    img: &image::RgbImage,
    target_w: usize,
    target_h: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut arr = ndarray::Array3::<f32>::zeros((3, target_h, target_w));
    resize_norm_into(img, target_w, target_h, arr.view_mut());
    (vec![3, target_h, target_w], arr.into_raw_vec_and_offset().0)
}

pub fn resize_norm_into(
    img: &image::RgbImage,
    target_w: usize,
    target_h: usize,
    mut out_slice: ndarray::ArrayViewMut3<f32>,
) {
    let (sw, sh) = (img.width(), img.height());
    if sw == 0 || sh == 0 || target_w == 0 || target_h == 0 {
        return;
    }
    let scale_x = sw as f32 / target_w as f32;
    let scale_y = sh as f32 / target_h as f32;

    let x_coords: Vec<(u32, u32, f32)> = (0..target_w)
        .map(|dx| {
            let sx = ((dx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = sx.floor() as u32;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = sx - x0 as f32;
            (x0, x1, fx)
        })
        .collect();

    let y_coords: Vec<(u32, u32, f32)> = (0..target_h)
        .map(|dy| {
            let sy = ((dy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (sh - 1) as f32);
            let y0 = sy.floor() as u32;
            let y1 = (y0 + 1).min(sh - 1);
            let fy = sy - y0 as f32;
            (y0, y1, fy)
        })
        .collect();

    for (dy, &(y0, y1, fy)) in y_coords.iter().enumerate() {
        for (dx, &(x0, x1, fx)) in x_coords.iter().enumerate() {
            let p00 = img.get_pixel(x0, y0).0;
            let p10 = img.get_pixel(x1, y0).0;
            let p01 = img.get_pixel(x0, y1).0;
            let p11 = img.get_pixel(x1, y1).0;
            for c in 0..3usize {
                let v = (p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
                    + p10[c] as f32 * fx * (1.0 - fy)
                    + p01[c] as f32 * (1.0 - fx) * fy
                    + p11[c] as f32 * fx * fy)
                    / 255.0;
                out_slice[[c, dy, dx]] = (v - 0.5) / 0.5;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_decode_filters_blanks_and_duplicates() {
        let decoder = CtcDecoder {
            characters: vec![
                "blank".into(),
                "A".into(),
                "B".into(),
                "C".into(),
                " ".into(),
            ],
        };
        // 4 timesteps, 5 classes
        // t0: class 1 (A), t1: class 1 (A, duplicate), t2: class 0 (blank), t3: class 2 (B)
        let probs = vec![
            0.1, 0.9, 0.0, 0.0, 0.0, // t0 -> A
            0.1, 0.8, 0.0, 0.0, 0.0, // t1 -> A (dup)
            0.9, 0.0, 0.0, 0.0, 0.0, // t2 -> blank
            0.1, 0.0, 0.8, 0.0, 0.0, // t3 -> B
        ];
        let (text, conf) = decoder.decode(&probs, 5, 4);
        assert_eq!(text, "AB");
        assert!((conf - 0.85).abs() < 1e-5);
    }

    #[test]
    fn resize_norm_into_normalizes_to_unit_range() {
        let img = image::RgbImage::from_pixel(2, 2, image::Rgb([255, 0, 128]));
        let mut arr = ndarray::Array3::<f32>::zeros((3, 4, 4));
        resize_norm_into(&img, 4, 4, arr.view_mut());
        // Red channel: 255 -> (1.0 - 0.5) / 0.5 = 1.0
        assert!((arr[[0, 0, 0]] - 1.0).abs() < 1e-4);
        // Green channel: 0 -> (0.0 - 0.5) / 0.5 = -1.0
        assert!((arr[[1, 0, 0]] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn from_metadata_handles_trailing_newline_without_extra_element() {
        let meta = "a\nb\n";
        let decoder = CtcDecoder::from_metadata(meta).unwrap();
        // blank at 0, "a" at 1, "b" at 2, " " at 3 => len 4
        assert_eq!(decoder.characters.len(), 4);
        assert_eq!(decoder.characters[0], "blank");
        assert_eq!(decoder.characters[1], "a");
        assert_eq!(decoder.characters[2], "b");
        assert_eq!(decoder.characters[3], " ");
    }

    #[test]
    fn plan_rec_batches_empty_returns_empty() {
        assert!(plan_rec_batches(&[], 6).is_empty());
    }

    #[test]
    fn plan_rec_batches_compact_uses_double_capacity() {
        // 14 compact items (ratio 3.0), base_batch 6 -> compact_cap 12
        // Should produce 2 batches: one of 12 and one of 2
        let crops: Vec<(usize, f32)> = (0..14).map(|i| (i, 3.0)).collect();
        let batches = plan_rec_batches(&crops, 6);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 12);
        assert_eq!(batches[1].len(), 2);
    }

    #[test]
    fn plan_rec_batches_wide_uses_reduced_capacity() {
        // 5 wide items (ratio 15.0), base_batch 6 -> wide_cap 4
        // Should produce 2 batches: one of 4 and one of 1
        let crops: Vec<(usize, f32)> = (0..5).map(|i| (i, 15.0)).collect();
        let batches = plan_rec_batches(&crops, 6);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 4);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn plan_rec_batches_preserves_all_indices_exactly_once() {
        let mut crops = Vec::new();
        for i in 0..10 {
            crops.push((i, 2.0)); // compact
        }
        for i in 10..18 {
            crops.push((i, 8.0)); // medium
        }
        for i in 18..25 {
            crops.push((i, 16.0)); // wide
        }
        let batches = plan_rec_batches(&crops, 6);
        let mut seen = std::collections::HashSet::new();
        for batch in batches {
            for idx in batch {
                assert!(seen.insert(idx), "duplicate index {idx}");
            }
        }
        assert_eq!(seen.len(), 25);
    }
}
