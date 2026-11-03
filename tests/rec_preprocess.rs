//! Integration coverage for the rec preprocessing contract.

use rapidocr_worker::rec::rec_resize_norm_into_vec;

#[test]
fn parallel_rec_preprocess_matches_expected_normalization() {
    // 3x2 gradient image resized to 4x4: per-pixel bilinear + normalize to [-1, 1].
    let mut img = image::RgbImage::new(3, 2);
    for (x, _, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x * 80) as u8, 40, 200]);
    }
    let (shape, data) = rec_resize_norm_into_vec(&img, 4, 4);
    assert_eq!(shape, vec![3usize, 4, 4]);
    assert_eq!(data.len(), 48);
    // Red channel at (dx=0): input x=0 -> value 0 -> normalized (0/255 - 0.5)/0.5 = -1.
    assert!((data[0] + 1.0).abs() < 1e-4);
    // Red channel at (dx=3): input x=2 -> 160 -> (160/255 - 0.5)/0.5 ~ 0.2549.
    assert!((data[3 * 4 + 3] - 0.2549).abs() < 1e-3);
    // Every value stays within the unit range after normalization.
    assert!(data.iter().all(|v| (-1.0..=1.0).contains(v)));
}

#[test]
fn parallel_rec_preprocess_is_deterministic_across_runs() {
    // Determinism guards the bit-exact parity contract with the Python engine.
    let img = image::RgbImage::from_pixel(37, 23, image::Rgb([91, 143, 7]));
    let first = rec_resize_norm_into_vec(&img, 96, 32).1;
    for _ in 0..8 {
        assert_eq!(first, rec_resize_norm_into_vec(&img, 96, 32).1);
    }
}
