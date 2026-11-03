//! Text direction classifier port (ch_ppocr_cls).
//!
//! Parity with rapidocr_onnxruntime TextClassifier: batch 6, shape
//! 3x48x192, thresh 0.9 over labels ['0', '180'].

use ndarray::Array4;
use ort::session::Session;

pub use crate::{CLS_HEIGHT, CLS_WIDTH};

pub const CLS_THRESH: f32 = 0.97;
pub const CLS_BATCH: usize = 6;
pub const CLS_LABELS: [&str; 2] = ["0", "180"];

/// Aspect ratio of the window fed to the classifier for long lines.
///
/// The classifier canvas is 48x192, i.e. 4:1. A text line wider than that gets
/// **truncated**, which squashes the glyphs and makes the model vote confidently
/// but wrongly. Feeding a centred window of roughly canvas aspect keeps the
/// characters legible, which is all the classifier needs: direction is a global
/// property of the line, not of its full width.
pub const CLS_CROP_ASPECT: f32 = 4.0;

/// Reduce a wide crop to a window the classifier can actually read.
///
/// Verified on a dense horizontal page: classifying the squashed full line gave
/// 12 false "rotate 180" votes out of 38 long lines; classifying a window gave 2.
/// Each false vote destroyed a line, since the recogniser then rejected the
/// flipped crop and the `c >= 0.5` filter dropped it.
///
/// The window is chosen by **information content**, not by position. A centred
/// window is only safe when the line's characters happen to be centred: an MRZ
/// line is ~22:1 and its left third carries the document number and dates while
/// the rest is `<` filler, so a centred 4:1 window covers pure filler, carries no
/// direction evidence, and made the classifier answer "180" confidently. That
/// rotated the issuing MRZ line and lost its check digit.
///
/// The window with the highest column-gradient energy is the one containing real
/// glyph transitions, which is what the classifier needs on both line shapes.
/// Crops that already fit the canvas are returned untouched, so behaviour for
/// ordinary lines is unchanged.
pub fn cls_crop_for_orientation(img: &image::RgbImage) -> image::RgbImage {
    let (w, h) = (img.width(), img.height());
    if h == 0 || w == 0 {
        return img.clone();
    }
    let target_w = (h as f32 * CLS_CROP_ASPECT).round() as u32;
    if w <= CLS_WIDTH || w <= target_w {
        return img.clone();
    }
    let width = target_w.min(w);
    let x0 = best_window_start(img, width);
    image::imageops::crop_imm(img, x0, 0, width, h).to_image()
}

/// Start offset of the `width`-wide window with the most horizontal transitions.
///
/// The score is the sum of absolute column-to-column ink differences, which is
/// high over characters and near zero over filler and margin. Prefix sums keep
/// the scan linear in the crop width.
fn best_window_start(img: &image::RgbImage, width: u32) -> u32 {
    let (w, h) = (img.width(), img.height());
    if w <= width {
        return 0;
    }
    // Mean ink per column, 0.0 = white, 1.0 = black.
    let mut ink = vec![0.0f32; w as usize];
    for (x, column) in ink.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for y in 0..h {
            let p = img.get_pixel(x as u32, y);
            let luma = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
            sum += 1.0 - luma / 255.0;
        }
        *column = sum / h as f32;
    }
    // prefix[i] = sum of |ink[i]-ink[i-1]| for columns 1..=i.
    let mut prefix = vec![0.0f32; w as usize];
    for i in 1..w as usize {
        prefix[i] = prefix[i - 1] + (ink[i] - ink[i - 1]).abs();
    }
    let width = width as usize;
    let mut best_start = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    // `last` is the final valid start; the window is never empty because w > width.
    let last = w as usize - width;
    for x0 in 0..=last {
        // Transitions strictly inside [x0, x0+width).
        let end = x0 + width - 1;
        let score = prefix[end] - prefix[x0];
        if score > best_score {
            best_score = score;
            best_start = x0;
        }
    }
    best_start as u32
}

/// Argmax over the two cls logits (no softmax, like ClsPostProcess).
pub fn cls_postprocess(logits: &[f32]) -> (&'static str, f32) {
    let mut best_id = 0usize;
    let mut best_score = logits.first().copied().unwrap_or(0.0);
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_score {
            best_score = v;
            best_id = i;
        }
    }
    (CLS_LABELS.get(best_id).copied().unwrap_or("0"), best_score)
}

/// Rotate the crop 180 degrees when label is 180 over threshold.
pub fn needs_rotation(label: &str, score: f32) -> bool {
    label.contains("180") && score > CLS_THRESH
}

/// Inferred page orientation from a sampled subset of text bounding boxes (#9).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PageOrientation {
    /// All sampled crops are upright (0 degrees) with high confidence.
    Upright,
    /// All sampled crops are inverted (180 degrees) with high confidence.
    Inverted,
    /// Mixed or indeterminate orientation across samples requiring full per-box evaluation.
    Mixed,
}

/// Evaluate page-level text orientation from a representative sample of classification decisions.
///
/// # Arguments
/// * `samples` - Slice of `(label, score)` pairs returned by the direction classifier for sample crops.
///
/// # Returns
/// * [`PageOrientation::Upright`] if all samples are confidently upright ("0" > `CLS_THRESH`).
/// * [`PageOrientation::Inverted`] if all samples are confidently inverted ("180" > `CLS_THRESH`).
/// * [`PageOrientation::Mixed`] if orientations disagree or any score falls below confidence threshold.
pub fn evaluate_sample_orientation(samples: &[(&str, f32)]) -> PageOrientation {
    if samples.is_empty() {
        return PageOrientation::Upright;
    }
    let mut upright_count = 0;
    let mut inverted_count = 0;
    for &(label, score) in samples {
        if label.contains("180") && score > CLS_THRESH {
            inverted_count += 1;
        } else if label == "0" && score > CLS_THRESH {
            upright_count += 1;
        }
    }
    // High-confidence consensus allows skipping classification on the remaining crops
    if upright_count == samples.len() {
        PageOrientation::Upright
    } else if inverted_count == samples.len() {
        PageOrientation::Inverted
    } else {
        PageOrientation::Mixed
    }
}

/// Whether the remaining crops can skip per-box classification after sampling.
///
/// Only a confident upright sample set allows the fast path; inverted or mixed
/// evidence falls back to full per-box classification (Python parity, #9).
pub fn can_skip_remaining(orientation: PageOrientation) -> bool {
    orientation == PageOrientation::Upright
}

/// Resize to height 48 keeping aspect ratio, pad right to width 192 (OpenCV bilinear parity).
pub fn resize_norm_img_cls(img: &image::RgbImage) -> ndarray::Array4<f32> {
    let (sw, sh) = (img.width(), img.height());
    let ch = crate::CLS_HEIGHT as usize;
    let cw = crate::CLS_WIDTH as usize;
    let mut arr = ndarray::Array4::<f32>::zeros((1, 3, ch, cw));
    if sw == 0 || sh == 0 {
        return arr;
    }
    let ratio = sw as f32 / sh as f32;
    let resized_w = ((ch as f32 * ratio).ceil() as usize).min(cw).max(1);
    let view = arr.slice_mut(ndarray::s![0, .., .., 0..resized_w]);
    crate::rec::resize_norm_into(img, resized_w, ch, view);
    arr
}

/// Rotate an RGB image 180 degrees.
pub fn rotate180(img: &image::RgbImage) -> image::RgbImage {
    image::imageops::rotate180(img)
}

/// Run classification on a single chunk of crops, optionally capturing labels and scores.
///
/// Prepares a batch tensor of shape `[N, 3, 48, 192]`, executes the ONNX direction
/// classification forward pass, and rotates inverted crops by 180 degrees in place.
pub fn run_cls_chunk(
    session: &mut Session,
    chunk: &[usize],
    crops: &mut [(usize, image::RgbImage)],
    mut decisions_out: Option<&mut Vec<(String, f32)>>,
) -> Result<(), String> {
    let n = chunk.len();
    let mut input = Array4::<f32>::zeros((n, 3, CLS_HEIGHT as usize, CLS_WIDTH as usize));
    for (bi, &ci) in chunk.iter().enumerate() {
        // Judge direction from a readable window, not from a squashed full line.
        // Truncation at CLS_WIDTH turned 63:1 lines into an illegible 4:1
        // smear and produced confident, wrong rotations.
        let window = cls_crop_for_orientation(&crops[ci].1);
        let norm = resize_norm_img_cls(&window);
        input
            .slice_mut(ndarray::s![bi, .., .., ..])
            .assign(&norm.slice(ndarray::s![0, .., .., ..]));
    }
    let tensor = ort::value::Tensor::from_array(input).expect("cls tensor");
    let out = session
        .run(ort::inputs![tensor])
        .map_err(|e| format!("cls run: {e}"))?;
    let (shape, data) = out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("cls extract: {e}"))?;
    let nl = shape[1] as usize;
    for (bi, &ci) in chunk.iter().enumerate() {
        let end = (bi * nl + nl).min(data.len());
        let (label, score) = cls_postprocess(&data[bi * nl..end]);
        if let Some(ref mut decs) = decisions_out {
            decs.push((label.to_string(), score));
        }
        if needs_rotation(label, score) {
            crops[ci].1 = rotate180(&crops[ci].1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postprocess_picks_180_over_thresh() {
        let (label, score) = cls_postprocess(&[0.05, 0.95]);
        assert_eq!(label, "180");
        assert!((score - 0.95).abs() < 1e-6);
    }

    #[test]
    fn postprocess_picks_zero_class() {
        let (label, score) = cls_postprocess(&[0.8, 0.2]);
        assert_eq!(label, "0");
        assert!((score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rotation_only_for_180_over_thresh() {
        // 0.97 is the measured optimum: it is the highest threshold that
        // still recovers a page flipped by 180 degrees, and the lowest that
        // leaves no page worse off than running without classification.
        assert!(needs_rotation("180", 0.99));
        assert!(!needs_rotation("180", 0.95));
        assert!(!needs_rotation("180", 0.5));
        assert!(!needs_rotation("0", 0.99));
    }

    #[test]
    fn resize_pads_right_with_zeros() {
        let img = image::RgbImage::from_pixel(48, 48, image::Rgb([255, 255, 255]));
        let arr = resize_norm_img_cls(&img);
        assert_eq!(arr.shape(), &[1, 3, 48, 192]);
        // white pixel -> (1.0 - 0.5) / 0.5 = 1.0 in normalized space
        assert!((arr[[0, 0, 0, 0]] - 1.0).abs() < 1e-5);
        // padding columns stay zero
        assert_eq!(arr[[0, 0, 0, 191]], 0.0);
    }

    #[test]
    fn rotate180_swaps_corners() {
        let mut img = image::RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([10, 20, 30]));
        img.put_pixel(1, 0, image::Rgb([40, 50, 60]));
        let out = rotate180(&img);
        assert_eq!(out.get_pixel(0, 0).0, [40, 50, 60]);
        assert_eq!(out.get_pixel(1, 0).0, [10, 20, 30]);
    }

    #[test]
    fn sample_orientation_all_zero_is_upright() {
        let samples = vec![("0", 0.99), ("0", 0.98), ("0", 0.975)];
        assert_eq!(
            evaluate_sample_orientation(&samples),
            PageOrientation::Upright
        );
    }

    #[test]
    fn sample_orientation_all_180_is_inverted() {
        let samples = vec![("180", 0.99), ("180", 0.98), ("180", 0.975)];
        assert_eq!(
            evaluate_sample_orientation(&samples),
            PageOrientation::Inverted
        );
    }

    #[test]
    fn sample_orientation_mixed_returns_mixed() {
        let samples = vec![("0", 0.95), ("180", 0.92)];
        assert_eq!(
            evaluate_sample_orientation(&samples),
            PageOrientation::Mixed
        );
    }

    #[test]
    fn sample_orientation_empty_is_upright() {
        assert_eq!(evaluate_sample_orientation(&[]), PageOrientation::Upright);
    }

    #[test]
    fn only_confident_upright_samples_allow_skipping_remaining_crops() {
        assert!(can_skip_remaining(PageOrientation::Upright));
        assert!(!can_skip_remaining(PageOrientation::Inverted));
        assert!(!can_skip_remaining(PageOrientation::Mixed));
    }
}
