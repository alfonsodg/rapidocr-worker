//! Direction-classification fixes.
//!
//! The deployed stage fed the classifier an entire text line scaled to 48px high
//! and **truncated at 192px wide**. A dense-page line has an aspect ratio of
//! 25:1 to 63:1, so truncation squashes a 63:1 line into ~4:1 — the glyphs become
//! illegible and the classifier answers confidently and wrongly.
//!
//! Measured on `05x. CERTIADULTO.pdf` page 3 (a perfectly horizontal page, 60
//! detected lines): classifying the squashed full line produced **12** false
//! "rotate 180" votes; classifying a centred 4:1 window of the same line
//! produced **2**. Each false vote destroyed a line, because the recogniser then
//! refused the flipped crop as low-confidence and the final `c >= 0.5` filter
//! dropped it.

use rapidocr_worker::cls::{
    cls_crop_for_orientation, needs_rotation, CLS_CROP_ASPECT, CLS_HEIGHT, CLS_WIDTH,
};

/// Build a synthetic text line of a given aspect ratio.
fn line(width: u32, height: u32) -> image::RgbImage {
    image::RgbImage::from_pixel(width, height, image::Rgb([200, 200, 200]))
}

/// Paint a text-like band: alternating narrow ink and paper columns.
///
/// A solid block is not text: real glyphs have ink/paper transitions across the
/// line, which is exactly the signal the window selection relies on.
fn glyph_band(crop: &mut image::RgbImage, x0: u32, x1: u32) {
    let h = crop.height();
    for x in x0..x1 {
        let shade = if (x / 3) % 2 == 0 { 15 } else { 245 };
        for y in 0..h {
            crop.put_pixel(x, y, image::Rgb([shade, shade, shade]));
        }
    }
}

/// Fraction of dark pixels in an image, used to tell text from margin.
fn ink_ratio(img: &image::RgbImage) -> f32 {
    img.pixels().filter(|p| p[0] < 60).count() as f32 / img.pixels().count() as f32
}

#[test]
fn short_crops_are_passed_through_unchanged() {
    // A normal line that already fits the 192px canvas needs no cropping, so the
    // deployed behaviour is preserved for the common case.
    let crop = line(120, 40);
    let out = cls_crop_for_orientation(&crop);
    assert_eq!((out.width(), out.height()), (120, 40));
}

#[test]
fn a_very_wide_line_is_reduced_to_a_readable_window() {
    // 63:1 is the worst case measured on a dense page.
    let crop = line(2520, 40);
    let out = cls_crop_for_orientation(&crop);
    let ratio = out.width() as f32 / out.height() as f32;
    assert!(
        (ratio - CLS_CROP_ASPECT).abs() < 0.35,
        "expected roughly {CLS_CROP_ASPECT}:1, got {ratio:.2}:1"
    );
    // It must still fit the classifier canvas without truncation.
    let scaled_w = (CLS_HEIGHT as f32 * ratio).ceil() as u32;
    assert!(
        scaled_w <= CLS_WIDTH,
        "window scales to {scaled_w}px, wider than the {CLS_WIDTH}px canvas"
    );
    assert_eq!(out.height(), crop.height(), "the window keeps full height");
}

#[test]
fn the_window_lands_on_text_wherever_it_is() {
    // Direction is judged from real glyphs, so the window must contain text. The
    // text may be anywhere on the line: an MRZ line carries its document number
    // and dates on the left and `<` filler on the right, so a fixed
    // centred window would capture filler only.
    let mut crop = line(2000, 40);
    glyph_band(&mut crop, 100, 500);
    let out = cls_crop_for_orientation(&crop);
    assert!(
        ink_ratio(&out) > 0.15,
        "window looks like margin, not text: {:.2}",
        ink_ratio(&out)
    );
}

#[test]
fn a_centred_window_is_still_chosen_for_centred_text() {
    // The wide-crop gain must survive: when the text is centred, the window is centred.
    let mut crop = line(2000, 40);
    glyph_band(&mut crop, 800, 1200);
    let out = cls_crop_for_orientation(&crop);
    assert!(
        ink_ratio(&out) > 0.15,
        "window looks like margin, not text: {:.2}",
        ink_ratio(&out)
    );
}

#[test]
fn mrz_filler_never_wins_over_real_glyphs() {
    // Worst case from an MRZ line: informative glyphs on the left, long `<` filler on
    // the right. A centred window covers filler only, carries no direction
    // evidence, and made the classifier answer "180" confidently — which rotated
    // the issuing MRZ line and lost its check digit.
    let mut crop = line(1800, 40);
    glyph_band(&mut crop, 60, 420);
    // Filler: a solid run with no column transitions.
    for x in 500..1800 {
        for y in 0..40 {
            crop.put_pixel(x, y, image::Rgb([30, 30, 30]));
        }
    }
    let out = cls_crop_for_orientation(&crop);
    // A window inside the filler is uniformly dark (mean ~30); a window on the
    // glyph band mixes ink and paper (mean ~130). Brightness separates them
    // without depending on antialiasing, which synthetic strokes do not have.
    let mean = out.pixels().map(|p| p[0] as f32).sum::<f32>() / out.pixels().count() as f32;
    assert!(mean > 80.0, "window mean {mean:.0} is filler, not glyphs");
    // And it must actually carry ink, not just background paper.
    assert!(
        ink_ratio(&out) > 0.2,
        "window has no ink: {:.2}",
        ink_ratio(&out)
    );
}

#[test]
fn a_wide_window_never_exceeds_the_canvas() {
    // Property check across the aspect ratios actually observed (25:1 .. 63:1).
    for ratio in [25.0f32, 34.0, 47.0, 55.0, 63.0] {
        let crop = line((40.0 * ratio) as u32, 40);
        let out = cls_crop_for_orientation(&crop);
        let scaled_w = (CLS_HEIGHT as f32 * out.width() as f32 / out.height() as f32).ceil() as u32;
        assert!(
            scaled_w <= CLS_WIDTH,
            "ratio {ratio}: scaled to {scaled_w}px"
        );
    }
}

#[test]
fn rotation_still_requires_confidence() {
    // 0.97 is the measured optimum: high enough that no page regresses,
    // low enough to still recover a page flipped by 180 degrees.
    assert!(needs_rotation("180", 0.99));
    assert!(!needs_rotation("180", 0.95));
}
