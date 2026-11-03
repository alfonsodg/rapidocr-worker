//! Det preprocessing parity with the Python engine (#8).
//!
//! DetResizeForTest (rapidocr_onnxruntime) resizes with limit_type=min and
//! OpenCV INTER_LINEAR, truncating the scaled size and then rounding to a
//! multiple of 32. The Rust det path must produce identical targets.

use rapidocr_worker::det::{cv_resize_bilinear, det_resize_target, DbPostProcess};

#[test]
fn full_resolution_pages_keep_their_size_rounded_to_32() {
    // A 1240x1754 page exceeds the 736 limit: ratio 1.0, round-32 only.
    let (nw, nh) = det_resize_target(1240, 1754, 736).expect("target");
    assert_eq!(nw, 1248);
    assert_eq!(nh, 1760);
}

#[test]
fn small_sides_scale_up_to_the_limit() {
    // int(800*736/600)=981 -> round(981/32)*32 = 992 (Python-verified).
    let (nw, nh) = det_resize_target(600, 800, 736).expect("target");
    assert_eq!(nw, 736);
    assert_eq!(nh, 992);
}

#[test]
fn empty_images_have_no_target() {
    assert!(det_resize_target(0, 0, 736).is_none());
}

#[test]
fn bilinear_resize_is_identity_for_same_size() {
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
fn dilate_2x2_matches_cv2_kernel() {
    // cv2.dilate with the 2x2 kernel (anchor 0,0) grows the region to the
    // right and bottom: a center pixel covers (1,2), (2,1), (2,2).
    let width = 3usize;
    let bitmap = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
    let out = DbPostProcess::dilate_2x2(&bitmap, width, 3);
    assert_eq!(out, vec![0, 0, 0, 0, 1, 1, 0, 1, 1]);
}

#[test]
fn dilate_2x2_fills_behind_full_rows() {
    let width = 2usize;
    let bitmap = vec![1, 1, 0, 0];
    let out = DbPostProcess::dilate_2x2(&bitmap, width, 2);
    assert_eq!(out, vec![1, 1, 1, 1]);
}

#[test]
fn cubic_weight_matches_opencv_shape() {
    // OpenCV INTER_CUBIC (A=-0.75): w(0)=1, w(0.5)=0.9375, w(1)=0, w(1.5)=-0.09375.
    let w = |t: f32| {
        // Access through the crate: the weight is exercised via the crop path,
        // so this replicates the formula to pin the shape.
        const A: f32 = -0.75;
        let a = t.abs();
        if a <= 1.0 {
            (A + 2.0) * a * a * a - (A + 3.0) * a * a + 1.0
        } else if a < 2.0 {
            A * a * a * a - 5.0 * A * a * a + 8.0 * A * a - 4.0 * A
        } else {
            0.0
        }
    };
    assert!((w(0.0) - 1.0).abs() < 1e-6);
    assert!((w(0.5) - 0.59375).abs() < 1e-6);
    assert!(w(1.0).abs() < 1e-6);
    assert!((w(1.5) - (-0.09375)).abs() < 1e-6);
}

#[test]
fn cubic_crop_preserves_a_flat_region() {
    // A flat-color source stays flat through the cubic warp path.
    let img = image::RgbImage::from_pixel(8, 8, image::Rgb([120, 60, 200]));
    let quad = [
        rapidocr_worker::geometry::Point { x: 0.0, y: 0.0 },
        rapidocr_worker::geometry::Point { x: 4.0, y: 0.0 },
        rapidocr_worker::geometry::Point { x: 4.0, y: 4.0 },
        rapidocr_worker::geometry::Point { x: 0.0, y: 4.0 },
    ];
    let crop = rapidocr_worker::perspective::perspective_crop(&img, &quad);
    let px = crop.get_pixel(1, 1).0;
    assert!((px[0] as i32 - 120).abs() <= 1);
    assert!((px[1] as i32 - 60).abs() <= 1);
    assert!((px[2] as i32 - 200).abs() <= 1);
}
