extern crate glycocr_rs as glycocr;

use glycocr::error::GlycOCRError;
use glycocr::pipeline::crop::{crop_and_pad_bbox, preprocess_image_to_tensor};
use glycocr::types::BoundingBox;
use image::{DynamicImage, Rgb, RgbImage};

#[test]
fn test_extreme_boundary_0_to_1000_full_image() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(1000, 1000));
    let bbox = BoundingBox::new(0.0, 0.0, 1000.0, 1000.0);
    let crop = crop_and_pad_bbox(&img, &bbox, 0.0).expect("Full image crop should succeed");
    assert_eq!(crop.width(), 1000);
    assert_eq!(crop.height(), 1000);
}

#[test]
fn test_extreme_boundary_0_to_1_full_image() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(800, 600));
    let bbox = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
    let crop =
        crop_and_pad_bbox(&img, &bbox, 0.0).expect("Full image crop 0..1 scale should succeed");
    assert_eq!(crop.width(), 800);
    assert_eq!(crop.height(), 600);
}

#[test]
fn test_extreme_boundary_degenerate_height_ymin_equals_ymax() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(1000, 1000));
    let bbox = BoundingBox::new(1000.0, 0.0, 1000.0, 1000.0);
    let crop = crop_and_pad_bbox(&img, &bbox, 0.0).expect("Degenerate height should not panic");
    assert!(crop.width() >= 1);
    assert!(crop.height() >= 1);
    assert!(crop.width() <= 1000);
    assert!(crop.height() <= 1000);
}

#[test]
fn test_extreme_boundary_degenerate_width_xmin_equals_xmax() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(1000, 1000));
    let bbox = BoundingBox::new(0.0, 500.0, 1000.0, 500.0);
    let crop = crop_and_pad_bbox(&img, &bbox, 0.0).expect("Degenerate width should not panic");
    assert!(crop.width() >= 1);
    assert!(crop.height() >= 1);
}

#[test]
fn test_extreme_boundary_negative_coordinates() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(500, 500));
    let bbox = BoundingBox::new(-200.0, -100.0, 300.0, 400.0);
    let crop = crop_and_pad_bbox(&img, &bbox, 0.1).expect("Negative coordinates should not panic");
    assert!(crop.width() <= 500);
    assert!(crop.height() <= 500);
    assert!(crop.width() >= 1);
    assert!(crop.height() >= 1);
}

#[test]
fn test_extreme_boundary_out_of_bounds_inputs() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(500, 500));
    let bbox = BoundingBox::new(1500.0, 1500.0, 2000.0, 2000.0);
    let crop = crop_and_pad_bbox(&img, &bbox, 0.0).expect("Out of bounds coords should not panic");
    assert!(crop.width() >= 1);
    assert!(crop.height() >= 1);
    assert!(crop.width() <= 500);
    assert!(crop.height() <= 500);
}

#[test]
fn test_extreme_boundary_nan_coordinates() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(500, 500));
    let bbox_nan_ymin = BoundingBox::new(f32::NAN, 0.0, 100.0, 100.0);
    let res = crop_and_pad_bbox(&img, &bbox_nan_ymin, 0.0);
    assert!(res.is_err());
    match res.unwrap_err() {
        GlycOCRError::ImageError(msg) => assert!(msg.contains("Invalid bounding box")),
        err => panic!("Expected ImageError, got {:?}", err),
    }

    let bbox_nan_all = BoundingBox::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    assert!(crop_and_pad_bbox(&img, &bbox_nan_all, 0.0).is_err());
}

#[test]
fn test_extreme_boundary_zero_dimension_images() {
    let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);

    let img_0x0 = DynamicImage::ImageRgb8(RgbImage::new(0, 0));
    assert!(crop_and_pad_bbox(&img_0x0, &bbox, 0.0).is_err());
    assert!(preprocess_image_to_tensor(&img_0x0).is_err());

    let img_0x500 = DynamicImage::ImageRgb8(RgbImage::new(0, 500));
    assert!(crop_and_pad_bbox(&img_0x500, &bbox, 0.0).is_err());
    assert!(preprocess_image_to_tensor(&img_0x500).is_err());

    let img_500x0 = DynamicImage::ImageRgb8(RgbImage::new(500, 0));
    assert!(crop_and_pad_bbox(&img_500x0, &bbox, 0.0).is_err());
    assert!(preprocess_image_to_tensor(&img_500x0).is_err());
}

#[test]
fn test_extreme_boundary_minimal_1x1_image() {
    let img = DynamicImage::ImageRgb8(RgbImage::new(1, 1));
    let bbox = BoundingBox::new(0.0, 0.0, 1000.0, 1000.0);
    let crop = crop_and_pad_bbox(&img, &bbox, 0.1).expect("1x1 image crop should succeed");
    assert_eq!(crop.width(), 1);
    assert_eq!(crop.height(), 1);

    let tensor = preprocess_image_to_tensor(&img).expect("1x1 image preprocessing should succeed");
    assert_eq!(tensor.dims(), &[1, 3, 448, 448]);
}

#[test]
fn test_candle_tensor_shape_and_float_bounds() {
    let images = vec![
        DynamicImage::ImageRgb8(RgbImage::new(10, 10)),
        DynamicImage::ImageRgb8(RgbImage::new(100, 1000)),
        DynamicImage::ImageRgb8(RgbImage::new(1920, 1080)),
        DynamicImage::ImageRgb8(RgbImage::new(1, 1)),
    ];

    for img in images {
        let tensor = preprocess_image_to_tensor(&img).expect("Preprocessing tensor failed");
        assert_eq!(
            tensor.dims(),
            &[1, 3, 448, 448],
            "Tensor shape must be [1, 3, 448, 448]"
        );

        let vec_data = tensor
            .flatten_all()
            .expect("Flatten failed")
            .to_vec1::<f32>()
            .expect("to_vec1 failed");
        assert_eq!(vec_data.len(), 3 * 448 * 448);

        for &val in &vec_data {
            assert!(
                (-1.0..=1.0).contains(&val),
                "Tensor float value {} outside [-1.0, 1.0] bounds",
                val
            );
        }
    }
}

#[test]
fn test_candle_tensor_exact_color_normalization() {
    // Pure White Image
    let mut white_img = RgbImage::new(50, 50);
    for p in white_img.pixels_mut() {
        *p = Rgb([255, 255, 255]);
    }
    let white_tensor = preprocess_image_to_tensor(&DynamicImage::ImageRgb8(white_img)).unwrap();
    let white_vals = white_tensor
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for &val in &white_vals {
        assert!(
            (val - 1.0).abs() < 1e-5,
            "White pixel should normalize to +1.0, got {}",
            val
        );
    }

    // Pure Black Image
    let mut black_img = RgbImage::new(50, 50);
    for p in black_img.pixels_mut() {
        *p = Rgb([0, 0, 0]);
    }
    let black_tensor = preprocess_image_to_tensor(&DynamicImage::ImageRgb8(black_img)).unwrap();
    let black_vals = black_tensor
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for &val in &black_vals {
        assert!(
            (val - (-1.0)).abs() < 1e-5,
            "Black pixel should normalize to -1.0, got {}",
            val
        );
    }
}
