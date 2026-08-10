use candle_core::{Device, IndexOp};
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use std::sync::Arc;
use std::thread;
use tempfile::NamedTempFile;

extern crate glycocr_rs as glycocr;

use glycocr::error::GlycOCRError;
use glycocr::model::dummy::DummyVlmEngine;
use glycocr::model::engine::VlmEngine;
use glycocr::model::paligemma::CandlePaliGemmaEngine;
use glycocr::utils::{image_to_tensor, select_device, select_device_auto};

#[test]
fn test_dummy_vlm_engine_extended_images() {
    let engine = DummyVlmEngine::new();

    // Test 1x1 RGB
    let img_1x1 = DynamicImage::new_rgb8(1, 1);
    let boxes = engine.detect_diagrams(&img_1x1).unwrap();
    assert_eq!(boxes.len(), 1);
    let ocr = engine.ocr_diagram(&img_1x1).unwrap();
    assert_eq!(ocr, "α-D-Glcp-(1->4)-D-Glcp");

    // Test 0x0 GrayImage
    let img_0x0 = DynamicImage::ImageLuma8(GrayImage::new(0, 0));
    let boxes_0 = engine.detect_diagrams(&img_0x0).unwrap();
    assert_eq!(boxes_0.len(), 1);

    // Test large image
    let img_large = DynamicImage::new_rgb8(4000, 4000);
    let boxes_large = engine.detect_diagrams(&img_large).unwrap();
    assert_eq!(boxes_large.len(), 1);
}

#[test]
fn test_candlepali_engine_device_resolution() {
    // Standard CPU
    let cpu_engine = CandlePaliGemmaEngine::new("cpu", None).unwrap();
    assert!(matches!(cpu_engine.device, Device::Cpu));
    assert_eq!(cpu_engine.device_name, "cpu");

    // Metal (macOS fallback if unavailable)
    let metal_engine = CandlePaliGemmaEngine::new("metal", None).unwrap();
    assert!(matches!(
        metal_engine.device,
        Device::Cpu | Device::Metal(_)
    ));

    // Auto resolution
    let auto_engine = CandlePaliGemmaEngine::new("auto", None).unwrap();
    assert!(matches!(
        auto_engine.device,
        Device::Cpu | Device::Metal(_) | Device::Cuda(_)
    ));

    // Direct helper select_device_auto
    let dev_auto = select_device_auto().unwrap();
    assert!(matches!(
        dev_auto,
        Device::Cpu | Device::Metal(_) | Device::Cuda(_)
    ));

    // Direct helper select_device
    let dev_cpu = select_device("cpu").unwrap();
    assert!(matches!(dev_cpu, Device::Cpu));

    // Case-insensitive device string should succeed
    let cpu_upper = CandlePaliGemmaEngine::new("CPU", None).unwrap();
    assert!(matches!(cpu_upper.device, Device::Cpu));

    let err = CandlePaliGemmaEngine::new("tpu_v4", None).unwrap_err();
    assert!(matches!(err, GlycOCRError::ModelError(_)));
    assert!(err.to_string().contains("Unsupported device 'tpu_v4'"));

    // Empty device string
    let err_empty = CandlePaliGemmaEngine::new("   ", None).unwrap_err();
    assert!(matches!(err_empty, GlycOCRError::ModelError(_)));
}

#[test]
fn test_candlepali_engine_weights_path_validation() {
    // Non-existent weights path
    let err = CandlePaliGemmaEngine::new(
        "cpu",
        Some(std::path::Path::new(
            "/tmp/non_existent_weights_12345.safetensors",
        )),
    )
    .unwrap_err();
    assert!(matches!(err, GlycOCRError::ModelError(_)));
    assert!(err.to_string().contains("Model weights file not found"));

    // Existing temp file as weights path
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_path_buf();
    let engine = CandlePaliGemmaEngine::new("cpu", Some(temp_path.as_path())).unwrap();
    assert_eq!(engine.weights_path(), Some(temp_path.as_path()));
    assert!(!engine.is_loaded());
}

#[test]
fn test_candlepali_engine_inference_various_image_types() {
    let engine = CandlePaliGemmaEngine::new("cpu", None).unwrap();

    // 1. RGBA Image
    let rgba_img =
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(100, 100, Rgba([128, 64, 32, 255])));
    let boxes = engine
        .detect_diagrams(&rgba_img)
        .expect("RGBA detect failed");
    assert_eq!(boxes.len(), 1);
    let ocr = engine.ocr_diagram(&rgba_img).expect("RGBA ocr failed");
    assert!(!ocr.is_empty());

    // 2. Grayscale (Luma8) Image
    let gray_img = DynamicImage::ImageLuma8(GrayImage::from_pixel(200, 150, Luma([200])));
    let boxes_gray = engine
        .detect_diagrams(&gray_img)
        .expect("Gray detect failed");
    assert_eq!(boxes_gray.len(), 1);
    let ocr_gray = engine.ocr_diagram(&gray_img).expect("Gray ocr failed");
    assert_eq!(ocr_gray, "α-D-Galp-(1->3)-β-D-GlcpNAc");

    // 3. Extreme Aspect Ratios: Very wide (1000x10) and Very tall (10x1000)
    let wide_img = DynamicImage::new_rgb8(1000, 10);
    assert!(engine.detect_diagrams(&wide_img).is_ok());

    let tall_img = DynamicImage::new_rgb8(10, 1000);
    assert!(engine.detect_diagrams(&tall_img).is_ok());
}

#[test]
fn test_image_to_tensor_exact_channels_and_values() {
    let dev = Device::Cpu;

    // Test specific pixel pattern: Red (255, 0, 0), Green (0, 255, 0), Blue (0, 0, 255)
    // 2x2 image:
    // Top-Left: Red, Top-Right: Green
    // Bottom-Left: Blue, Bottom-Right: White (255, 255, 255)
    let mut imgbuf = RgbImage::new(2, 2);
    imgbuf.put_pixel(0, 0, Rgb([255, 0, 0]));
    imgbuf.put_pixel(1, 0, Rgb([0, 255, 0]));
    imgbuf.put_pixel(0, 1, Rgb([0, 0, 255]));
    imgbuf.put_pixel(1, 1, Rgb([255, 255, 255]));
    let dyn_img = DynamicImage::ImageRgb8(imgbuf);

    let tensor = image_to_tensor(&dyn_img, 2, 2, &dev).expect("Tensor creation failed");

    // Check Tensor shape: (1, 3, 2, 2) -> (Batch, Channel, Height, Width)
    assert_eq!(tensor.dims(), &[1, 3, 2, 2]);

    // Top-Left (0, 0): R=1.0, G=0.0, B=0.0
    let r_00: f32 = tensor.i((0, 0, 0, 0)).unwrap().to_scalar().unwrap();
    let g_00: f32 = tensor.i((0, 1, 0, 0)).unwrap().to_scalar().unwrap();
    let b_00: f32 = tensor.i((0, 2, 0, 0)).unwrap().to_scalar().unwrap();
    assert!((r_00 - 1.0).abs() < 1e-5);
    assert_eq!(g_00, 0.0);
    assert_eq!(b_00, 0.0);

    // Top-Right (0, 1): R=0.0, G=1.0, B=0.0
    let r_01: f32 = tensor.i((0, 0, 0, 1)).unwrap().to_scalar().unwrap();
    let g_01: f32 = tensor.i((0, 1, 0, 1)).unwrap().to_scalar().unwrap();
    let b_01: f32 = tensor.i((0, 2, 0, 1)).unwrap().to_scalar().unwrap();
    assert_eq!(r_01, 0.0);
    assert!((g_01 - 1.0).abs() < 1e-5);
    assert_eq!(b_01, 0.0);

    // Bottom-Left (1, 0): R=0.0, G=0.0, B=1.0
    let r_10: f32 = tensor.i((0, 0, 1, 0)).unwrap().to_scalar().unwrap();
    let g_10: f32 = tensor.i((0, 1, 1, 0)).unwrap().to_scalar().unwrap();
    let b_10: f32 = tensor.i((0, 2, 1, 0)).unwrap().to_scalar().unwrap();
    assert_eq!(r_10, 0.0);
    assert_eq!(g_10, 0.0);
    assert!((b_10 - 1.0).abs() < 1e-5);

    // Bottom-Right (1, 1): R=1.0, G=1.0, B=1.0
    let r_11: f32 = tensor.i((0, 0, 1, 1)).unwrap().to_scalar().unwrap();
    let g_11: f32 = tensor.i((0, 1, 1, 1)).unwrap().to_scalar().unwrap();
    let b_11: f32 = tensor.i((0, 2, 1, 1)).unwrap().to_scalar().unwrap();
    assert!((r_11 - 1.0).abs() < 1e-5);
    assert!((g_11 - 1.0).abs() < 1e-5);
    assert!((b_11 - 1.0).abs() < 1e-5);
}

#[test]
fn test_trait_object_concurrency() {
    let dummy_engine: Arc<dyn VlmEngine> = Arc::new(DummyVlmEngine::new());
    let pali_engine: Arc<dyn VlmEngine> =
        Arc::new(CandlePaliGemmaEngine::new("cpu", None).unwrap());

    let mut handles = vec![];

    for i in 0..10 {
        let engine_ref = if i % 2 == 0 {
            Arc::clone(&dummy_engine)
        } else {
            Arc::clone(&pali_engine)
        };

        let handle = thread::spawn(move || {
            let img = DynamicImage::new_rgb8(120, 120);
            let bboxes = engine_ref.detect_diagrams(&img).unwrap();
            assert!(!bboxes.is_empty());
            let text = engine_ref.ocr_diagram(&img).unwrap();
            assert!(!text.is_empty());
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("Thread execution failed");
    }
}
