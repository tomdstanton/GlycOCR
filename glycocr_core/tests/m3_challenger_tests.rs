extern crate glycocr_rs as glycocr;

use candle_core::{DType, Device, IndexOp};
use glycocr::model::dummy::DummyVlmEngine;
use glycocr::model::engine::VlmEngine;
use glycocr::model::paligemma::CandlePaliGemmaEngine;
use glycocr::utils::{image_to_tensor, select_device, select_device_auto};
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

// Compile-time trait bound assertions for Send + Sync
fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn test_vlm_engine_trait_bounds_send_sync() {
    // Assert at compile-time that engines and trait objects satisfy Send + Sync
    assert_send::<DummyVlmEngine>();
    assert_sync::<DummyVlmEngine>();
    assert_send::<CandlePaliGemmaEngine>();
    assert_sync::<CandlePaliGemmaEngine>();
    assert_send::<Arc<dyn VlmEngine>>();
    assert_sync::<Arc<dyn VlmEngine>>();
}

#[test]
fn test_high_concurrency_thread_safety_dummy() {
    let engine: Arc<dyn VlmEngine> = Arc::new(DummyVlmEngine::new());
    let mut handles = vec![];
    let thread_count = 64;

    for i in 0..thread_count {
        let engine_clone = Arc::clone(&engine);
        let handle = thread::spawn(move || {
            let img = DynamicImage::new_rgb8((50 + i * 2) as u32, (50 + i * 2) as u32);
            let bboxes = engine_clone.detect_diagrams(&img).expect("detect failed");
            assert_eq!(bboxes.len(), 1);
            let ocr = engine_clone.ocr_diagram(&img).expect("ocr failed");
            assert_eq!(ocr, "α-D-Glcp-(1->4)-D-Glcp");
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("thread panicked");
    }
}

#[test]
fn test_high_concurrency_thread_safety_candlepali() {
    let engine: Arc<dyn VlmEngine> = Arc::new(CandlePaliGemmaEngine::new("cpu", None).unwrap());
    let mut handles = vec![];
    let thread_count = 64;

    for i in 0..thread_count {
        let engine_clone = Arc::clone(&engine);
        let handle = thread::spawn(move || {
            let img = DynamicImage::new_rgb8((100 + i) as u32, (100 + i) as u32);
            let bboxes = engine_clone.detect_diagrams(&img).expect("detect failed");
            assert_eq!(bboxes.len(), 1);
            let ocr = engine_clone.ocr_diagram(&img).expect("ocr failed");
            assert_eq!(ocr, "α-D-Galp-(1->3)-β-D-GlcpNAc");
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("thread panicked");
    }
}

#[test]
fn test_tensor_shape_1_3_448_448_various_inputs() {
    let dev = Device::Cpu;

    // Test with various input aspect ratios and resolutions
    let test_cases = vec![
        (1, 1),
        (100, 200),
        (448, 448),
        (1920, 1080),
        (4000, 3000),
        (7, 13),
    ];

    for (w, h) in test_cases {
        let img = DynamicImage::new_rgb8(w, h);
        let tensor = image_to_tensor(&img, 448, 448, &dev).expect("tensor conversion failed");

        assert_eq!(
            tensor.dims(),
            &[1, 3, 448, 448],
            "Shape mismatch for input {}x{}",
            w,
            h
        );
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.elem_count(), 3 * 448 * 448);
    }
}

#[test]
fn test_tensor_shape_custom_target_dimensions() {
    let dev = Device::Cpu;
    let img = DynamicImage::new_rgb8(800, 600);

    let custom_targets = vec![(224, 224), (448, 448), (512, 512), (128, 256)];
    for (tw, th) in custom_targets {
        let tensor = image_to_tensor(&img, tw, th, &dev).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, th as usize, tw as usize]);
        assert_eq!(tensor.elem_count(), (3 * th * tw) as usize);
    }
}

#[test]
fn test_tensor_color_formats() {
    let dev = Device::Cpu;

    // RGB8
    let rgb_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([128, 64, 32])));
    let t_rgb = image_to_tensor(&rgb_img, 448, 448, &dev).unwrap();
    assert_eq!(t_rgb.dims(), &[1, 3, 448, 448]);

    // RGBA8
    let rgba_img =
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(100, 100, Rgba([128, 64, 32, 255])));
    let t_rgba = image_to_tensor(&rgba_img, 448, 448, &dev).unwrap();
    assert_eq!(t_rgba.dims(), &[1, 3, 448, 448]);

    // Luma8 (Grayscale)
    let luma_img = DynamicImage::ImageLuma8(GrayImage::from_pixel(100, 100, Luma([200])));
    let t_luma = image_to_tensor(&luma_img, 448, 448, &dev).unwrap();
    assert_eq!(t_luma.dims(), &[1, 3, 448, 448]);
}

#[test]
fn test_image_to_tensor_pixel_normalization_bounds() {
    let dev = Device::Cpu;

    // All Black Image (0, 0, 0)
    let black_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, Rgb([0, 0, 0])));
    let black_tensor = image_to_tensor(&black_img, 10, 10, &dev).unwrap();
    let min_val: f32 = black_tensor
        .flatten_all()
        .unwrap()
        .min(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    let max_val: f32 = black_tensor
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert_eq!(min_val, 0.0);
    assert_eq!(max_val, 0.0);

    // All White Image (255, 255, 255)
    let white_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, Rgb([255, 255, 255])));
    let white_tensor = image_to_tensor(&white_img, 10, 10, &dev).unwrap();
    let min_val_w: f32 = white_tensor
        .flatten_all()
        .unwrap()
        .min(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    let max_val_w: f32 = white_tensor
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!((min_val_w - 1.0).abs() < 1e-5);
    assert!((max_val_w - 1.0).abs() < 1e-5);

    // Mid Gray Image (128, 128, 128)
    let gray_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(10, 10, Rgb([128, 128, 128])));
    let gray_tensor = image_to_tensor(&gray_img, 10, 10, &dev).unwrap();
    let val_g: f32 = gray_tensor.i((0, 0, 5, 5)).unwrap().to_scalar().unwrap();
    assert!((val_g - (128.0 / 255.0)).abs() < 1e-4);
}

#[test]
fn test_image_to_tensor_rgb_channel_isolation() {
    let dev = Device::Cpu;

    // Pure Red Image (255, 0, 0)
    let red_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(5, 5, Rgb([255, 0, 0])));
    let tensor_red = image_to_tensor(&red_img, 5, 5, &dev).unwrap();
    let r = tensor_red
        .i((0, 0, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let g = tensor_red
        .i((0, 1, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let b = tensor_red
        .i((0, 2, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        (r - 1.0).abs() < 1e-5,
        "Red channel should be 1.0, got {}",
        r
    );
    assert_eq!(g, 0.0, "Green channel should be 0.0, got {}", g);
    assert_eq!(b, 0.0, "Blue channel should be 0.0, got {}", b);

    // Pure Green Image (0, 255, 0)
    let green_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(5, 5, Rgb([0, 255, 0])));
    let tensor_green = image_to_tensor(&green_img, 5, 5, &dev).unwrap();
    let r_g = tensor_green
        .i((0, 0, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let g_g = tensor_green
        .i((0, 1, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let b_g = tensor_green
        .i((0, 2, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(r_g, 0.0);
    assert!((g_g - 1.0).abs() < 1e-5);
    assert_eq!(b_g, 0.0);

    // Pure Blue Image (0, 0, 255)
    let blue_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(5, 5, Rgb([0, 0, 255])));
    let tensor_blue = image_to_tensor(&blue_img, 5, 5, &dev).unwrap();
    let r_b = tensor_blue
        .i((0, 0, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let g_b = tensor_blue
        .i((0, 1, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let b_b = tensor_blue
        .i((0, 2, 0, 0))
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(r_b, 0.0);
    assert_eq!(g_b, 0.0);
    assert!((b_b - 1.0).abs() < 1e-5);
}

#[test]
fn test_device_selection_cases() {
    assert!(matches!(select_device("cpu").unwrap(), Device::Cpu));
    assert!(matches!(select_device("CPU").unwrap(), Device::Cpu));
    assert!(matches!(
        select_device("  metal  ").unwrap(),
        Device::Cpu | Device::Metal(_)
    ));
    assert!(matches!(
        select_device("auto").unwrap(),
        Device::Cpu | Device::Metal(_) | Device::Cuda(_)
    ));
    assert!(select_device_auto().is_ok());
    assert!(select_device("invalid").is_err());
    assert!(select_device("").is_err());
    assert!(select_device("   ").is_err());
    assert!(select_device("tpu").is_err());
}

#[test]
fn test_performance_tensor_conversion_stress() {
    let dev = Device::Cpu;
    // Standard pipeline image dimensions (448x448)
    let img = DynamicImage::new_rgb8(448, 448);

    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        let tensor = image_to_tensor(&img, 448, 448, &dev).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 448, 448]);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!(
        "Average 448x448 -> (1,3,448,448) tensor conversion time: {:.2} ms over {} iterations",
        avg_ms, iterations
    );

    let threshold_ms = if cfg!(debug_assertions) { 2000.0 } else { 50.0 };
    assert!(
        avg_ms < threshold_ms,
        "Tensor conversion averaged {:.2} ms, exceeding threshold of {} ms!",
        avg_ms,
        threshold_ms
    );
}
