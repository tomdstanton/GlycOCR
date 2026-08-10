extern crate glycocr_rs as glycocr;

use glycocr::error::GlycOCRError;
use glycocr::model::engine::VlmEngine;
use glycocr::model::paligemma::CandlePaliGemmaEngine;
use glycocr::utils::select_device;
use image::{DynamicImage, GrayImage, RgbImage};
use std::path::Path;
use tempfile::NamedTempFile;

#[test]
fn test_select_device_case_insensitivity_and_whitespace() {
    let test_cases = vec![
        ("cPu", true),
        ("MEtaL", true),
        ("CuDa", true),
        ("AUTO", true),
        ("  cPu  ", true),
        ("\tMEtaL\n", true),
        ("  CuDa  ", true),
        ("  AUTO  ", true),
        ("gpu", true),
        ("GPU", true),
    ];

    for (input, should_succeed) in test_cases {
        let res = select_device(input);
        if should_succeed {
            assert!(
                res.is_ok(),
                "Device string '{}' should have succeeded, but failed with: {:?}",
                input,
                res.err()
            );
        }
    }
}

#[test]
fn test_select_device_invalid_strings() {
    let invalid_inputs = vec![
        "invalid_device",
        "TPU_V4",
        "opencl",
        "vulkan",
        "",
        "   ",
        "cpu_cuda",
        "12345",
    ];

    for input in invalid_inputs {
        let res = select_device(input);
        assert!(
            res.is_err(),
            "Invalid device string '{}' should have returned error",
            input
        );
        match res.unwrap_err() {
            GlycOCRError::ModelError(msg) => {
                assert!(msg.contains("Unsupported device"));
            }
            err => panic!("Unexpected error type for '{}': {:?}", input, err),
        }
    }
}

#[test]
fn test_candlepali_engine_constructor_device_variations() {
    let valid_devices = vec!["cPu", "MEtaL", "CuDa", "AUTO", "  cPu \n"];
    for dev_str in valid_devices {
        let engine = CandlePaliGemmaEngine::new(dev_str, None);
        assert!(
            engine.is_ok(),
            "CandlePaliGemmaEngine::new failed for device '{}': {:?}",
            dev_str,
            engine.err()
        );
        let engine = engine.unwrap();
        assert_eq!(engine.weights_path(), None);
        assert!(!engine.is_loaded());
    }

    let invalid_devices = vec!["cPu_invalid", "TPU", ""];
    for dev_str in invalid_devices {
        let engine = CandlePaliGemmaEngine::new(dev_str, None);
        assert!(
            engine.is_err(),
            "CandlePaliGemmaEngine::new should fail for invalid device '{}'",
            dev_str
        );
    }
}

#[test]
fn test_candlepali_engine_weights_path_handling() {
    // 1. None path
    let engine_none = CandlePaliGemmaEngine::new("cPu", None).unwrap();
    assert_eq!(engine_none.weights_path(), None);

    // 2. Non-existent file path
    let missing_path = Path::new("/tmp/non_existent_paligemma_weights_99999.safetensors");
    let err_missing = CandlePaliGemmaEngine::new("cPu", Some(missing_path)).unwrap_err();
    match err_missing {
        GlycOCRError::ModelError(msg) => {
            assert!(msg.contains("Model weights file not found"));
            assert!(msg.contains("non_existent_paligemma_weights_99999"));
        }
        err => panic!(
            "Expected ModelError for missing weights file, got {:?}",
            err
        ),
    }

    // 3. Existing valid file path
    let temp_file = NamedTempFile::new().unwrap();
    let real_path = temp_file.path();
    let engine_valid = CandlePaliGemmaEngine::new("cPu", Some(real_path)).unwrap();
    assert_eq!(engine_valid.weights_path(), Some(real_path));
    assert!(!engine_valid.is_loaded());
}

#[test]
fn test_candlepali_engine_methods_without_crashing() {
    let engine = CandlePaliGemmaEngine::new("cPu", None).unwrap();

    // Normal RGB image
    let rgb_img = DynamicImage::ImageRgb8(RgbImage::new(448, 448));
    let boxes = engine
        .detect_diagrams(&rgb_img)
        .expect("detect_diagrams failed");
    assert_eq!(boxes.len(), 1);
    assert_eq!(boxes[0].ymin, 50.0);
    assert_eq!(boxes[0].xmin, 50.0);
    assert_eq!(boxes[0].ymax, 450.0);
    assert_eq!(boxes[0].xmax, 450.0);

    let ocr = engine.ocr_diagram(&rgb_img).expect("ocr_diagram failed");
    assert_eq!(ocr, "α-D-Galp-(1->3)-β-D-GlcpNAc");

    // Extreme/edge case images
    let tiny_img = DynamicImage::new_rgb8(1, 1);
    assert!(engine.detect_diagrams(&tiny_img).is_ok());
    assert!(engine.ocr_diagram(&tiny_img).is_ok());

    let zero_img = DynamicImage::ImageLuma8(GrayImage::new(0, 0));
    assert!(engine.detect_diagrams(&zero_img).is_ok());
    assert!(engine.ocr_diagram(&zero_img).is_ok());

    let huge_img = DynamicImage::new_rgb8(2000, 2000);
    assert!(engine.detect_diagrams(&huge_img).is_ok());
    assert!(engine.ocr_diagram(&huge_img).is_ok());
}
