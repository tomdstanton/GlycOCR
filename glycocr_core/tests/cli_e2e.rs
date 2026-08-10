//! GlycOCR End-to-End (E2E) Opaque-Box Integration Test Harness
//!
//! Tiers 1-4 requirement-driven test suite as specified in TEST_INFRA.md and SCOPE.md.
//! Uses in-process execution via `glycocr::run_cli_from`.

extern crate glycocr_rs as glycocr;

use glycocr::model::dummy::DummyVlmEngine;
use glycocr::model::engine::VlmEngine;
use glycocr::pipeline::crop::crop_and_pad_bbox;
use glycocr::run_cli_from;
use glycocr::types::{BoundingBox, DetectedDiagram, DocumentScanResult, PageResult};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use lopdf::{Document, Object, Stream, dictionary};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{NamedTempFile, TempDir};

// =========================================================================
// HELPER UTILITIES & SYNTHETIC FIXTURE GENERATORS
// =========================================================================

pub fn setup_temp_dir() -> TempDir {
    TempDir::new().expect("Failed to create temporary directory")
}

pub fn create_synthetic_pdf(pages: usize) -> NamedTempFile {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();

    for _ in 0..pages {
        let content_id = doc.add_object(Stream::new(
            dictionary! {},
            b"q 0 0 200 200 re f Q".to_vec(),
        ));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        });
        page_ids.push(Object::Reference(page_id));
    }

    let pages_obj = dictionary! {
        "Type" => "Pages",
        "Count" => pages as i64,
        "Kids" => page_ids,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_obj));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", Object::Reference(catalog_id));

    let temp_file = tempfile::Builder::new()
        .suffix(".pdf")
        .tempfile()
        .expect("Failed to create temp pdf file");
    doc.save(temp_file.path())
        .expect("Failed to save synthetic pdf");
    temp_file
}

pub fn create_synthetic_image(width: u32, height: u32) -> NamedTempFile {
    let mut img = RgbImage::new(width, height);
    for pixel in img.pixels_mut() {
        *pixel = Rgb([200, 200, 200]);
    }
    let temp_file = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("Failed to create temp image file");
    img.save_with_format(temp_file.path(), image::ImageFormat::Png)
        .expect("Failed to save synthetic image");
    temp_file
}

pub fn create_corrupt_pdf() -> NamedTempFile {
    let temp_file = tempfile::Builder::new()
        .suffix(".pdf")
        .tempfile()
        .expect("Failed to create temp corrupt pdf file");
    fs::write(temp_file.path(), b"NOT_A_VALID_PDF_STREAM_DATA_BYTES").unwrap();
    temp_file
}

pub fn create_corrupt_image() -> NamedTempFile {
    let temp_file = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("Failed to create temp corrupt image file");
    fs::write(temp_file.path(), b"NOT_A_VALID_PNG_IMAGE_BYTES").unwrap();
    temp_file
}

pub fn create_empty_file() -> NamedTempFile {
    NamedTempFile::new().expect("Failed to create empty temp file")
}

pub fn get_real_pdf_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/data/whitfield-et-al-2025-o-antigen-polysaccharides-in-klebsiella-pneumoniae-structures-and-molecular-basis-for-antigenic.pdf")
}

pub fn get_real_image_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/data/kpsc_K1_cps_SNFG.jpg")
}

// =========================================================================
// TIER 1: FEATURE COVERAGE (60 TESTS - 5 TESTS PER FEATURE 1..12)
// =========================================================================
mod tier1_feature_coverage {
    use super::*;

    // --- Feature 1: Cargo Package Setup & CLI Binary Entry ---
    #[test]
    fn test_f1_01_cli_help_flag() {
        let res = run_cli_from(["glycocr", "--help"]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("glycocr"));
    }

    #[test]
    fn test_f1_02_cli_version_flag() {
        let res = run_cli_from(["glycocr", "--version"]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("0.1.0"));
    }

    #[test]
    fn test_f1_03_infer_subcommand_help() {
        let res = run_cli_from(["glycocr", "infer", "--help"]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("--pdf"));
    }

    #[test]
    fn test_f1_04_dummy_image_inference() {
        let img_file = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img_file.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_f1_05_dummy_pdf_inference() {
        let pdf_file = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf_file.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    // --- Feature 2: Error Handling Framework ---
    #[test]
    fn test_f2_01_missing_file_error() {
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--pdf",
            "/nonexistent/path/doc.pdf",
            "--dummy",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("PdfError") || err.contains("not found"));
    }

    #[test]
    fn test_f2_02_invalid_device_name() {
        let pdf_file = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--pdf",
            pdf_file.path().to_str().unwrap(),
            "--device",
            "quantum_gpu",
            "--dummy",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("Unsupported device"));
    }

    #[test]
    fn test_f2_06_device_auto_and_case_normalization() {
        let pdf_file = create_synthetic_pdf(1);
        let pdf_path = pdf_file.path().to_str().unwrap();

        let res_auto = run_cli_from([
            "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "auto",
        ]);
        assert!(res_auto.is_ok(), "Failed on --device auto: {:?}", res_auto);

        let res_upper_auto = run_cli_from([
            "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "AUTO",
        ]);
        assert!(
            res_upper_auto.is_ok(),
            "Failed on --device AUTO: {:?}",
            res_upper_auto
        );

        let res_spaced_auto = run_cli_from([
            "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "  auto  ",
        ]);
        assert!(
            res_spaced_auto.is_ok(),
            "Failed on --device '  auto  ': {:?}",
            res_spaced_auto
        );
    }

    #[test]
    fn test_f2_03_missing_required_input_flag() {
        let res = run_cli_from(["glycocr", "infer", "--dummy"]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("must be provided"));
    }

    #[test]
    fn test_f2_04_conflicting_input_flags() {
        let pdf_file = create_synthetic_pdf(1);
        let img_file = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--pdf",
            pdf_file.path().to_str().unwrap(),
            "--image",
            img_file.path().to_str().unwrap(),
            "--dummy",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("Cannot specify both"));
    }

    #[test]
    fn test_f2_05_corrupt_image_error() {
        let corrupt_img = create_corrupt_image();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            corrupt_img.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("ImageError") || err.contains("Failed to open image"));
    }

    // --- Feature 3: Core Data Structures ---
    #[test]
    fn test_f3_01_bounding_box_instantiation() {
        let bbox = BoundingBox {
            ymin: 100.0,
            xmin: 100.0,
            ymax: 400.0,
            xmax: 400.0,
            label: Some("SNFG".into()),
        };
        assert_eq!(bbox.ymin, 100.0);
        assert_eq!(bbox.label.as_deref(), Some("SNFG"));
    }

    #[test]
    fn test_f3_02_detected_diagram_fields() {
        let diagram = DetectedDiagram {
            bbox: BoundingBox {
                ymin: 0.0,
                xmin: 0.0,
                ymax: 500.0,
                xmax: 500.0,
                label: None,
            },
            cropped_path: Some("crop.png".into()),
            iupac: "α-D-Glcp-(1->4)-D-Glcp".into(),
            confidence: 0.98,
        };
        assert_eq!(diagram.iupac, "α-D-Glcp-(1->4)-D-Glcp");
        assert_eq!(diagram.confidence, 0.98);
    }

    #[test]
    fn test_f3_03_page_result_structure() {
        let page = PageResult {
            page_number: 1,
            diagrams: vec![],
        };
        assert_eq!(page.page_number, 1);
        assert!(page.diagrams.is_empty());
    }

    #[test]
    fn test_f3_04_document_scan_result_serde() {
        let scan = DocumentScanResult {
            pdf_path: "sample.pdf".into(),
            total_pages: 2,
            pages: vec![
                PageResult {
                    page_number: 1,
                    diagrams: vec![],
                },
                PageResult {
                    page_number: 2,
                    diagrams: vec![],
                },
            ],
        };
        let json = serde_json::to_string(&scan).unwrap();
        let decoded: DocumentScanResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_pages, 2);
    }

    #[test]
    fn test_f3_05_iupac_string_non_empty() {
        let img_file = create_synthetic_image(100, 100);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("res.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img_file.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ]);
        assert!(res.is_ok());
        let content = fs::read_to_string(out_json).unwrap();
        let scan: DocumentScanResult = serde_json::from_str(&content).unwrap();
        assert!(!scan.pages[0].diagrams[0].iupac.is_empty());
    }

    // --- Feature 4: PDF Ingestion & Page Rendering ---
    #[test]
    fn test_f4_01_single_page_pdf_ingestion() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_f4_02_multi_page_pdf_ingestion() {
        let pdf = create_synthetic_pdf(3);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("out.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ]);
        assert!(res.is_ok());
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        assert_eq!(scan.total_pages, 3);
    }

    #[test]
    fn test_f4_03_pdf_extract_function_direct() {
        let pdf = create_synthetic_pdf(2);
        let pages = glycocr::pipeline::pdf::extract_pdf_pages(pdf.path()).unwrap();
        assert_eq!(pages.len(), 2);
    }

    #[test]
    fn test_f4_04_pdf_rendering_dimensions() {
        let pdf = create_synthetic_pdf(1);
        let pages = glycocr::pipeline::pdf::extract_pdf_pages(pdf.path()).unwrap();
        use image::GenericImageView;
        assert_eq!(pages[0].dimensions(), (448, 448));
    }

    #[test]
    fn test_f4_05_pdf_page_indexing_accuracy() {
        let pdf = create_synthetic_pdf(4);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("out.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ])
        .unwrap();
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        assert_eq!(scan.pages.len(), 4);
        for (i, page) in scan.pages.iter().enumerate() {
            assert_eq!(page.page_number, i + 1);
        }
    }

    // --- Feature 5: Bounding Box Cropping & Preprocessing ---
    #[test]
    fn test_f5_01_crop_and_pad_bbox_standard() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let bbox = BoundingBox {
            ymin: 100.0,
            xmin: 100.0,
            ymax: 400.0,
            xmax: 400.0,
            label: None,
        };
        let crop = crop_and_pad_bbox(&img, &bbox, 0.1).unwrap();
        assert!(crop.width() > 0 && crop.height() > 0);
    }

    #[test]
    fn test_f5_02_crop_full_image_boundary() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let bbox = BoundingBox {
            ymin: 0.0,
            xmin: 0.0,
            ymax: 1000.0,
            xmax: 1000.0,
            label: None,
        };
        let crop = crop_and_pad_bbox(&img, &bbox, 0.0).unwrap();
        assert_eq!(crop.dimensions(), (100, 100));
    }

    #[test]
    fn test_f5_03_crop_padding_expansion() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let bbox = BoundingBox {
            ymin: 200.0,
            xmin: 200.0,
            ymax: 800.0,
            xmax: 800.0,
            label: None,
        };
        let crop_no_pad = crop_and_pad_bbox(&img, &bbox, 0.0).unwrap();
        let crop_pad = crop_and_pad_bbox(&img, &bbox, 0.1).unwrap();
        assert!(crop_pad.width() >= crop_no_pad.width());
    }

    #[test]
    fn test_f5_04_crop_coordinate_clamping() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let bbox = BoundingBox {
            ymin: -50.0,
            xmin: -50.0,
            ymax: 1200.0,
            xmax: 1200.0,
            label: None,
        };
        let crop = crop_and_pad_bbox(&img, &bbox, 0.1);
        assert!(crop.is_ok());
    }

    #[test]
    fn test_f5_05_crop_small_subregion() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(500, 500));
        let bbox = BoundingBox {
            ymin: 100.0,
            xmin: 100.0,
            ymax: 200.0,
            xmax: 200.0,
            label: None,
        };
        let crop = crop_and_pad_bbox(&img, &bbox, 0.1).unwrap();
        assert!(crop.width() > 10);
    }

    // --- Feature 6: Decoupled Model Trait (VlmEngine) ---
    #[test]
    fn test_f6_01_dummy_engine_trait_implementation() {
        let engine = DummyVlmEngine::new();
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let bboxes = engine.detect_diagrams(&img).unwrap();
        assert!(!bboxes.is_empty());
    }

    #[test]
    fn test_f6_02_dummy_engine_ocr_trait_call() {
        let engine = DummyVlmEngine::new();
        let crop = DynamicImage::ImageRgb8(RgbImage::new(50, 50));
        let iupac = engine.ocr_diagram(&crop).unwrap();
        assert!(iupac.contains("Glcp"));
    }

    #[test]
    fn test_f6_03_trait_object_dispatch() {
        let engine: Box<dyn VlmEngine> = Box::new(DummyVlmEngine::new());
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        assert!(engine.detect_diagrams(&img).is_ok());
    }

    #[test]
    fn test_f6_04_engine_returns_valid_bbox_coordinates() {
        let engine = DummyVlmEngine::new();
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let bboxes = engine.detect_diagrams(&img).unwrap();
        let box0 = &bboxes[0];
        assert!(box0.ymin < box0.ymax && box0.xmin < box0.xmax);
    }

    #[test]
    fn test_f6_05_engine_returns_non_empty_ocr_string() {
        let engine = DummyVlmEngine::new();
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let text = engine.ocr_diagram(&img).unwrap();
        assert!(!text.trim().is_empty());
    }

    // --- Feature 7: Fallback Dummy Stub (DummyVlmEngine) ---
    #[test]
    fn test_f7_01_dummy_flag_activation() {
        let img = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_f7_02_dummy_engine_returns_mock_boxes() {
        let engine = DummyVlmEngine::new();
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let boxes = engine.detect_diagrams(&img).unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].label.as_deref(), Some("SNFG"));
    }

    #[test]
    fn test_f7_03_dummy_engine_returns_mock_iupac() {
        let engine = DummyVlmEngine::new();
        let crop = DynamicImage::ImageRgb8(RgbImage::new(50, 50));
        let iupac = engine.ocr_diagram(&crop).unwrap();
        assert_eq!(iupac, "α-D-Glcp-(1->4)-D-Glcp");
    }

    #[test]
    fn test_f7_04_dummy_pipeline_execution_without_weights() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_f7_05_dummy_engine_thread_safety() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DummyVlmEngine>();
    }

    // --- Feature 8: Candle PaliGemma Model Engine ---
    #[test]
    fn test_f8_01_candle_engine_cpu_initialization() {
        let engine = glycocr::model::paligemma::CandlePaliGemmaEngine::new("cpu", None);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_f8_02_candle_engine_metal_initialization() {
        let engine = glycocr::model::paligemma::CandlePaliGemmaEngine::new("metal", None);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_f8_03_candle_engine_cuda_initialization() {
        let engine = glycocr::model::paligemma::CandlePaliGemmaEngine::new("cuda", None);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_f8_04_candle_engine_detect_diagrams() {
        let engine = glycocr::model::paligemma::CandlePaliGemmaEngine::new("cpu", None).unwrap();
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let boxes = engine.detect_diagrams(&img).unwrap();
        assert!(!boxes.is_empty());
    }

    #[test]
    fn test_f8_05_candle_engine_ocr_diagram() {
        let engine = glycocr::model::paligemma::CandlePaliGemmaEngine::new("cpu", None).unwrap();
        let crop = DynamicImage::ImageRgb8(RgbImage::new(50, 50));
        let text = engine.ocr_diagram(&crop).unwrap();
        assert!(text.contains("Galp"));
    }

    // --- Feature 9: Two-Pass Pipeline Runner ---
    #[test]
    fn test_f9_01_pipeline_runner_instantiation() {
        let engine = DummyVlmEngine::new();
        let _runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
    }

    #[test]
    fn test_f9_02_pipeline_runner_image_execution() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let img = create_synthetic_image(100, 100);
        let res = runner.run_image(img.path()).unwrap();
        assert_eq!(res.total_pages, 1);
    }

    #[test]
    fn test_f9_03_pipeline_runner_pdf_execution() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let pdf = create_synthetic_pdf(2);
        let res = runner.run_pdf(pdf.path()).unwrap();
        assert_eq!(res.total_pages, 2);
    }

    #[test]
    fn test_f9_04_two_pass_detection_and_crop() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let img = create_synthetic_image(200, 200);
        let res = runner.run_image(img.path()).unwrap();
        assert!(!res.pages[0].diagrams.is_empty());
        assert_eq!(res.pages[0].diagrams[0].iupac, "α-D-Glcp-(1->4)-D-Glcp");
    }

    #[test]
    fn test_f9_05_pipeline_aggregates_all_pages() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let pdf = create_synthetic_pdf(5);
        let res = runner.run_pdf(pdf.path()).unwrap();
        assert_eq!(res.pages.len(), 5);
    }

    // --- Feature 10: CLI Commands & Test Entry Point ---
    #[test]
    fn test_f10_01_run_cli_from_valid_pdf_args() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_f10_02_run_cli_from_valid_image_args() {
        let img = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_f10_03_run_cli_from_custom_output_flag() {
        let img = create_synthetic_image(100, 100);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("out.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "-o",
            out_json.to_str().unwrap(),
        ]);
        assert!(res.is_ok());
        assert!(out_json.exists());
    }

    #[test]
    fn test_f10_04_run_cli_from_json_stdout_flag() {
        let img = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_f10_05_run_cli_from_verbose_flag() {
        let img = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "-v",
        ]);
        assert!(res.is_ok());
    }

    // --- Feature 11: JSON Serialization & Output ---
    #[test]
    fn test_f11_01_json_output_file_created() {
        let img = create_synthetic_image(100, 100);
        let temp = setup_temp_dir();
        let out_file = temp.path().join("scan.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_file.to_str().unwrap(),
        ]);
        assert!(res.is_ok());
        assert!(out_file.exists());
    }

    #[test]
    fn test_f11_02_json_file_deserialization_validity() {
        let img = create_synthetic_image(100, 100);
        let temp = setup_temp_dir();
        let out_file = temp.path().join("scan.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_file.to_str().unwrap(),
        ])
        .unwrap();
        let json_str = fs::read_to_string(out_file).unwrap();
        let scan: DocumentScanResult = serde_json::from_str(&json_str).unwrap();
        assert_eq!(scan.total_pages, 1);
    }

    #[test]
    fn test_f11_03_json_schema_keys_present() {
        let img = create_synthetic_image(100, 100);
        let temp = setup_temp_dir();
        let out_file = temp.path().join("scan.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_file.to_str().unwrap(),
        ])
        .unwrap();
        let json_str = fs::read_to_string(out_file).unwrap();
        assert!(json_str.contains("\"pdf_path\""));
        assert!(json_str.contains("\"total_pages\""));
        assert!(json_str.contains("\"pages\""));
    }

    #[test]
    fn test_f11_04_json_pretty_printed() {
        let img = create_synthetic_image(100, 100);
        let temp = setup_temp_dir();
        let out_file = temp.path().join("scan.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_file.to_str().unwrap(),
        ])
        .unwrap();
        let json_str = fs::read_to_string(out_file).unwrap();
        assert!(json_str.contains('\n'));
    }

    #[test]
    fn test_f11_05_json_output_diagrams_array() {
        let img = create_synthetic_image(100, 100);
        let temp = setup_temp_dir();
        let out_file = temp.path().join("scan.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_file.to_str().unwrap(),
        ])
        .unwrap();
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_file).unwrap()).unwrap();
        assert!(!scan.pages[0].diagrams.is_empty());
    }

    // --- Feature 12: End-to-End E2E Test Suite ---
    #[test]
    fn test_f12_01_full_e2e_pipeline_image_to_json() {
        let img = create_synthetic_image(200, 200);
        let temp = setup_temp_dir();
        let out_file = temp.path().join("e2e_img.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_file.to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
        assert!(out_file.exists());
    }

    #[test]
    fn test_f12_02_full_e2e_pipeline_pdf_to_json() {
        let pdf = create_synthetic_pdf(2);
        let temp = setup_temp_dir();
        let out_file = temp.path().join("e2e_pdf.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_file.to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_file).unwrap()).unwrap();
        assert_eq!(scan.total_pages, 2);
    }

    #[test]
    fn test_f12_03_e2e_test_harness_isolation() {
        let temp1 = setup_temp_dir();
        let temp2 = setup_temp_dir();
        assert_ne!(temp1.path(), temp2.path());
    }

    #[test]
    fn test_f12_04_e2e_real_fixture_existence_check() {
        let pdf_path = get_real_pdf_path();
        let img_path = get_real_image_path();
        if pdf_path.exists() {
            let res = run_cli_from([
                "glycocr",
                "infer",
                "--dummy",
                "--pdf",
                pdf_path.to_str().unwrap(),
            ]);
            assert!(res.is_ok());
        }
        if img_path.exists() {
            let res = run_cli_from([
                "glycocr",
                "infer",
                "--dummy",
                "--image",
                img_path.to_str().unwrap(),
            ]);
            assert!(res.is_ok());
        }
    }

    #[test]
    fn test_f12_05_e2e_document_scan_result_schema_verification() {
        let pdf = create_synthetic_pdf(1);
        let temp = setup_temp_dir();
        let out_file = temp.path().join("schema.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_file.to_str().unwrap(),
        ])
        .unwrap();
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_file).unwrap()).unwrap();
        assert_eq!(scan.total_pages, 1);
        assert_eq!(scan.pages[0].page_number, 1);
        assert_eq!(scan.pages[0].diagrams[0].confidence, 0.95);
    }
}

// =========================================================================
// TIER 2: BOUNDARY & CORNER CASES (60 TESTS - 5 TESTS PER FEATURE 1..12)
// =========================================================================
mod tier2_boundary_corner {
    use super::*;

    // --- Feature 1 Boundary Tests ---
    #[test]
    fn test_t2_f1_01_unknown_cli_flag() {
        let res = run_cli_from(["glycocr", "--unknown-flag-xyz"]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f1_02_unknown_subcommand() {
        let res = run_cli_from(["glycocr", "nonexistent_subcommand"]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f1_03_no_subcommand_provided() {
        let res = run_cli_from(["glycocr"]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f1_04_empty_arg_vector() {
        let empty_args: Vec<String> = vec![];
        let res = run_cli_from(empty_args);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f1_05_extra_positional_arg() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "extra_pos_arg",
        ]);
        assert!(res.is_err());
    }

    // --- Feature 2 Boundary Tests ---
    #[test]
    fn test_t2_f2_01_non_existent_pdf_path() {
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            "/nonexistent/path/doc.pdf",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("PdfError") || err.contains("not found"));
    }

    #[test]
    fn test_t2_f2_02_non_existent_image_path() {
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            "/nonexistent/path/img.png",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("ImageError") || err.contains("not found"));
    }

    #[test]
    fn test_t2_f2_03_non_existent_output_directory() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            "/invalid_dir_9999/out.json",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("Output directory does not exist") || err.contains("No such file"));
    }

    #[test]
    fn test_t2_f2_04_unsupported_device_name() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--device",
            "super_gpu_123",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("Unsupported device"));
    }

    #[test]
    fn test_t2_f2_05_invalid_model_weights_path() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--model-path",
            "/missing/weights.safetensors",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("Model weights file not found") || err.contains("ModelError"));
    }

    // --- Feature 3 Boundary Tests ---
    #[test]
    fn test_t2_f3_01_zero_coordinate_bounding_box() {
        let bbox = BoundingBox {
            ymin: 0.0,
            xmin: 0.0,
            ymax: 0.0,
            xmax: 0.0,
            label: None,
        };
        assert_eq!(bbox.ymin, 0.0);
        assert_eq!(bbox.ymax, 0.0);
    }

    #[test]
    fn test_t2_f3_02_extreme_bounding_box_coordinates() {
        let bbox = BoundingBox {
            ymin: 0.0,
            xmin: 0.0,
            ymax: 1000.0,
            xmax: 1000.0,
            label: Some("MAX".into()),
        };
        assert_eq!(bbox.ymax, 1000.0);
    }

    #[test]
    fn test_t2_f3_03_zero_confidence_diagram() {
        let diagram = DetectedDiagram {
            bbox: BoundingBox {
                ymin: 10.0,
                xmin: 10.0,
                ymax: 50.0,
                xmax: 50.0,
                label: None,
            },
            cropped_path: None,
            iupac: "Unknown".into(),
            confidence: 0.0,
        };
        assert_eq!(diagram.confidence, 0.0);
    }

    #[test]
    fn test_t2_f3_04_empty_pages_array_document_scan() {
        let scan = DocumentScanResult {
            pdf_path: "empty_doc.pdf".into(),
            total_pages: 0,
            pages: vec![],
        };
        assert_eq!(scan.pages.len(), 0);
    }

    #[test]
    fn test_t2_f3_05_malformed_json_deserialization_failure() {
        let malformed_json = r#"{"pdf_path": "test.pdf", "total_pages": "NOT_AN_INT"}"#;
        let res: Result<DocumentScanResult, _> = serde_json::from_str(malformed_json);
        assert!(res.is_err());
    }

    // --- Feature 4 Boundary Tests ---
    #[test]
    fn test_t2_f4_01_zero_byte_pdf_file() {
        let empty_pdf = create_empty_file();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            empty_pdf.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("empty") || err.contains("PdfError"));
    }

    #[test]
    fn test_t2_f4_02_corrupted_pdf_file_bytes() {
        let corrupt_pdf = create_corrupt_pdf();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            corrupt_pdf.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("Invalid PDF format") || err.contains("PdfError"));
    }

    #[test]
    fn test_t2_f4_03_non_pdf_extension_file() {
        let temp = setup_temp_dir();
        let txt_file = temp.path().join("doc.txt");
        fs::write(&txt_file, b"Hello World").unwrap();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            txt_file.to_str().unwrap(),
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f4_04_direct_pdf_extractor_zero_byte() {
        let empty_pdf = create_empty_file();
        let res = glycocr::pipeline::pdf::extract_pdf_pages(empty_pdf.path());
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f4_05_direct_pdf_extractor_corrupt_stream() {
        let corrupt_pdf = create_corrupt_pdf();
        let res = glycocr::pipeline::pdf::extract_pdf_pages(corrupt_pdf.path());
        assert!(res.is_err());
    }

    // --- Feature 5 Boundary Tests ---
    #[test]
    fn test_t2_f5_01_zero_byte_image_file() {
        let empty_img = create_empty_file();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            empty_img.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f5_02_1x1_pixel_image() {
        let tiny_img = create_synthetic_image(1, 1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            tiny_img.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f5_03_out_of_bounds_negative_bbox() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let bbox = BoundingBox {
            ymin: -500.0,
            xmin: -500.0,
            ymax: 500.0,
            xmax: 500.0,
            label: None,
        };
        let res = crop_and_pad_bbox(&img, &bbox, 0.1);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f5_04_out_of_bounds_exceeding_1000_bbox() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let bbox = BoundingBox {
            ymin: 500.0,
            xmin: 500.0,
            ymax: 2500.0,
            xmax: 2500.0,
            label: None,
        };
        let res = crop_and_pad_bbox(&img, &bbox, 0.1);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f5_05_crop_on_zero_dimension_image() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(0, 0));
        let bbox = BoundingBox {
            ymin: 0.0,
            xmin: 0.0,
            ymax: 100.0,
            xmax: 100.0,
            label: None,
        };
        let res = crop_and_pad_bbox(&img, &bbox, 0.1);
        assert!(res.is_err());
    }

    // --- Feature 6 Boundary Tests ---
    #[test]
    fn test_t2_f6_01_detect_diagrams_1x1_image() {
        let engine = DummyVlmEngine::new();
        let img = DynamicImage::ImageRgb8(RgbImage::new(1, 1));
        let res = engine.detect_diagrams(&img);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f6_02_ocr_diagram_1x1_crop() {
        let engine = DummyVlmEngine::new();
        let crop = DynamicImage::ImageRgb8(RgbImage::new(1, 1));
        let res = engine.ocr_diagram(&crop);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f6_03_detect_diagrams_large_image() {
        let engine = DummyVlmEngine::new();
        let img = DynamicImage::ImageRgb8(RgbImage::new(2000, 2000));
        let res = engine.detect_diagrams(&img);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f6_04_candle_engine_unsupported_device() {
        let engine = glycocr::model::paligemma::CandlePaliGemmaEngine::new("tpu", None);
        assert!(engine.is_err());
    }

    #[test]
    fn test_t2_f6_05_candle_engine_missing_weights_file() {
        let engine = glycocr::model::paligemma::CandlePaliGemmaEngine::new(
            "cpu",
            Some(Path::new("/missing/weights.bin")),
        );
        assert!(engine.is_err());
    }

    // --- Feature 7 Boundary Tests ---
    #[test]
    fn test_t2_f7_01_dummy_engine_run_on_corrupt_file() {
        let corrupt = create_corrupt_pdf();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            corrupt.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f7_02_dummy_engine_run_on_empty_file() {
        let empty = create_empty_file();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            empty.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f7_03_dummy_engine_default_trait_behavior() {
        let engine = DummyVlmEngine::default();
        let img = DynamicImage::ImageRgb8(RgbImage::new(50, 50));
        assert_eq!(engine.detect_diagrams(&img).unwrap().len(), 1);
    }

    #[test]
    fn test_t2_f7_04_dummy_engine_constant_ocr_output() {
        let engine = DummyVlmEngine::new();
        let crop1 = DynamicImage::ImageRgb8(RgbImage::new(10, 10));
        let crop2 = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        assert_eq!(
            engine.ocr_diagram(&crop1).unwrap(),
            engine.ocr_diagram(&crop2).unwrap()
        );
    }

    #[test]
    fn test_t2_f7_05_dummy_engine_with_json_and_output() {
        let img = create_synthetic_image(50, 50);
        let temp = setup_temp_dir();
        let out = temp.path().join("dummy_out.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
    }

    // --- Feature 8 Boundary Tests ---
    #[test]
    fn test_t2_f8_01_candle_engine_empty_device_str() {
        let res = glycocr::model::paligemma::CandlePaliGemmaEngine::new("", None);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f8_02_candle_engine_uppercase_device_str() {
        let res = glycocr::model::paligemma::CandlePaliGemmaEngine::new("CPU", None);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f8_03_candle_engine_existing_dummy_model_path() {
        let temp = setup_temp_dir();
        let dummy_weights = temp.path().join("model.safetensors");
        fs::write(&dummy_weights, b"weights").unwrap();
        let res = glycocr::model::paligemma::CandlePaliGemmaEngine::new(
            "cpu",
            Some(dummy_weights.as_path()),
        );
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f8_04_candle_engine_non_existent_model_path() {
        let res = glycocr::model::paligemma::CandlePaliGemmaEngine::new(
            "cpu",
            Some(Path::new("/nonexistent/model.bin")),
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f8_05_candle_engine_run_with_no_weights() {
        let engine = glycocr::model::paligemma::CandlePaliGemmaEngine::new("cpu", None).unwrap();
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        assert!(engine.detect_diagrams(&img).is_ok());
        assert!(engine.ocr_diagram(&img).is_ok());
    }

    // --- Feature 9 Boundary Tests ---
    #[test]
    fn test_t2_f9_01_runner_with_missing_pdf() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let res = runner.run_pdf(Path::new("/missing/pdf.pdf"));
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f9_02_runner_with_missing_image() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let res = runner.run_image(Path::new("/missing/image.png"));
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f9_03_runner_with_empty_pdf() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let empty = create_empty_file();
        let res = runner.run_pdf(empty.path());
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f9_04_runner_with_corrupt_pdf() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let corrupt = create_corrupt_pdf();
        let res = runner.run_pdf(corrupt.path());
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f9_05_runner_with_corrupt_image() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let corrupt = create_corrupt_image();
        let res = runner.run_image(corrupt.path());
        assert!(res.is_err());
    }

    // --- Feature 10 Boundary Tests ---
    #[test]
    fn test_t2_f10_01_cli_missing_input_flags() {
        let res = run_cli_from(["glycocr", "infer", "--dummy"]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f10_02_cli_conflicting_pdf_and_image_flags() {
        let pdf = create_synthetic_pdf(1);
        let img = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--image",
            img.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f10_03_cli_output_flag_without_path() {
        let res = run_cli_from(["glycocr", "infer", "--dummy", "-o"]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f10_04_cli_device_flag_without_value() {
        let res = run_cli_from(["glycocr", "infer", "--dummy", "--device"]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f10_05_cli_pdf_flag_without_value() {
        let res = run_cli_from(["glycocr", "infer", "--dummy", "--pdf"]);
        assert!(res.is_err());
    }

    // --- Feature 11 Boundary Tests ---
    #[test]
    fn test_t2_f11_01_json_write_to_invalid_directory() {
        let img = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "-o",
            "/non_existent_folder_abc/scan.json",
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t2_f11_02_json_deserialization_extra_fields_ignored() {
        let json_with_extra = r#"{
            "pdf_path": "test.pdf",
            "total_pages": 1,
            "pages": [],
            "extra_unrecognized_field": true
        }"#;
        let res: Result<DocumentScanResult, _> = serde_json::from_str(json_with_extra);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f11_03_json_empty_pages_array_validity() {
        let scan = DocumentScanResult {
            pdf_path: "empty.pdf".into(),
            total_pages: 0,
            pages: vec![],
        };
        let json = serde_json::to_string(&scan).unwrap();
        assert!(json.contains("\"pages\":[]") || json.contains("\"pages\": []"));
    }

    #[test]
    fn test_t2_f11_04_json_roundtrip_equality() {
        let scan = DocumentScanResult {
            pdf_path: "test.pdf".into(),
            total_pages: 1,
            pages: vec![PageResult {
                page_number: 1,
                diagrams: vec![DetectedDiagram {
                    bbox: BoundingBox {
                        ymin: 1.0,
                        xmin: 2.0,
                        ymax: 3.0,
                        xmax: 4.0,
                        label: Some("A".into()),
                    },
                    cropped_path: None,
                    iupac: "IUPAC".into(),
                    confidence: 0.9,
                }],
            }],
        };
        let json = serde_json::to_string(&scan).unwrap();
        let scan2: DocumentScanResult = serde_json::from_str(&json).unwrap();
        assert_eq!(scan, scan2);
    }

    #[test]
    fn test_t2_f11_05_json_special_characters_escaping() {
        let diagram = DetectedDiagram {
            bbox: BoundingBox {
                ymin: 0.0,
                xmin: 0.0,
                ymax: 100.0,
                xmax: 100.0,
                label: None,
            },
            cropped_path: None,
            iupac: "α-D-Glcp-(1->4)-[β-D-Glcp-(1->3)]-D-GlcpNAc".into(),
            confidence: 0.99,
        };
        let json = serde_json::to_string(&diagram).unwrap();
        assert!(json.contains("α-D-Glcp"));
    }

    // --- Feature 12 Boundary Tests ---
    #[test]
    fn test_t2_f12_01_filepath_with_spaces() {
        let temp = setup_temp_dir();
        let space_img = temp.path().join("sample image with space.png");
        let img = create_synthetic_image(50, 50);
        fs::copy(img.path(), &space_img).unwrap();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            space_img.to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f12_02_filepath_with_unicode_characters() {
        let temp = setup_temp_dir();
        let unicode_img = temp.path().join("糖锁_glycan_sample.png");
        let img = create_synthetic_image(50, 50);
        fs::copy(img.path(), &unicode_img).unwrap();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            unicode_img.to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t2_f12_03_concurrent_tempdir_isolation() {
        let t1 = setup_temp_dir();
        let t2 = setup_temp_dir();
        let f1 = t1.path().join("out.json");
        let f2 = t2.path().join("out.json");
        fs::write(&f1, b"1").unwrap();
        fs::write(&f2, b"2").unwrap();
        assert_eq!(fs::read_to_string(f1).unwrap(), "1");
        assert_eq!(fs::read_to_string(f2).unwrap(), "2");
    }

    #[test]
    fn test_t2_f12_04_repeat_cli_invocations() {
        let img = create_synthetic_image(50, 50);
        for _ in 0..5 {
            let res = run_cli_from([
                "glycocr",
                "infer",
                "--dummy",
                "--image",
                img.path().to_str().unwrap(),
            ]);
            assert!(res.is_ok());
        }
    }

    #[test]
    fn test_t2_f12_05_large_page_count_synthetic_pdf() {
        let large_pdf = create_synthetic_pdf(10);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            large_pdf.path().to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }
}

// =========================================================================
// TIER 3: CROSS-FEATURE PAIRWISE INTERACTIONS (15 TESTS)
// =========================================================================
mod tier3_cross_feature_pairwise {
    use super::*;

    #[test]
    fn test_t3_01_pdf_dummy_json_output_interaction() {
        let pdf = create_synthetic_pdf(2);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("pairwise_1.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--dummy",
            "--output",
            out_json.to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
        assert!(out_json.exists());
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        assert_eq!(scan.total_pages, 2);
    }

    #[test]
    fn test_t3_02_image_dummy_custom_output_interaction() {
        let img = create_synthetic_image(150, 150);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("pairwise_2.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--image",
            img.path().to_str().unwrap(),
            "--dummy",
            "--output",
            out_json.to_str().unwrap(),
            "-v",
        ]);
        assert!(res.is_ok());
        assert!(out_json.exists());
    }

    #[test]
    fn test_t3_03_device_cpu_candle_engine_pdf_interaction() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--device",
            "cpu",
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t3_04_device_metal_candle_engine_image_interaction() {
        let img = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--image",
            img.path().to_str().unwrap(),
            "--device",
            "metal",
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t3_05_conflicting_flags_error_framework_interaction() {
        let pdf = create_synthetic_pdf(1);
        let img = create_synthetic_image(100, 100);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--image",
            img.path().to_str().unwrap(),
            "--device",
            "cpu",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("Cannot specify both"));
    }

    #[test]
    fn test_t3_06_image_crop_preprocessing_dummy_ocr_interaction() {
        let engine = DummyVlmEngine::new();
        let img = create_synthetic_image(300, 300);
        let dynamic_img = image::open(img.path()).unwrap();
        let bboxes = engine.detect_diagrams(&dynamic_img).unwrap();
        let crop = crop_and_pad_bbox(&dynamic_img, &bboxes[0], 0.1).unwrap();
        let iupac = engine.ocr_diagram(&crop).unwrap();
        assert!(!iupac.is_empty());
    }

    #[test]
    fn test_t3_07_pdf_ingestion_error_framework_missing_dir_interaction() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            "/non_existent_parent/file.json",
        ]);
        assert!(res.is_err());
    }

    #[test]
    fn test_t3_08_dummy_engine_verbose_logging_output_file_interaction() {
        let img = create_synthetic_image(100, 100);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("verbose_out.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
            "-v",
        ]);
        assert!(res.is_ok());
        assert!(out_json.exists());
    }

    #[test]
    fn test_t3_09_multipage_pdf_crop_dummy_engine_serde_interaction() {
        let pdf = create_synthetic_pdf(3);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("multipage_serde.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ])
        .unwrap();
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        assert_eq!(scan.total_pages, 3);
        assert_eq!(scan.pages.len(), 3);
    }

    #[test]
    fn test_t3_10_zero_byte_input_error_framework_exit_interaction() {
        let empty_pdf = create_empty_file();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            empty_pdf.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("empty") || err.contains("PdfError"));
    }

    #[test]
    fn test_t3_11_corrupt_pdf_error_framework_exit_interaction() {
        let corrupt_pdf = create_corrupt_pdf();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            corrupt_pdf.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("Invalid PDF format") || err.contains("PdfError"));
    }

    #[test]
    fn test_t3_12_synthetic_1x1_image_dummy_json_export_interaction() {
        let tiny = create_synthetic_image(1, 1);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("tiny_out.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            tiny.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
        assert!(out_json.exists());
    }

    #[test]
    fn test_t3_13_pdf_custom_output_device_cuda_fallback_interaction() {
        let pdf = create_synthetic_pdf(1);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("cuda_out.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--device",
            "cuda",
            "--output",
            out_json.to_str().unwrap(),
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t3_14_image_bbox_normalization_json_export_interaction() {
        let img = create_synthetic_image(448, 448);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("norm_out.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ])
        .unwrap();
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        let bbox = &scan.pages[0].diagrams[0].bbox;
        assert!(bbox.ymin >= 0.0 && bbox.ymax <= 1000.0);
    }

    #[test]
    fn test_t3_15_document_scan_result_file_roundtrip_interaction() {
        let temp = setup_temp_dir();
        let out_json = temp.path().join("roundtrip.json");
        let pdf = create_synthetic_pdf(1);
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ])
        .unwrap();
        let content1 = fs::read_to_string(&out_json).unwrap();
        let scan1: DocumentScanResult = serde_json::from_str(&content1).unwrap();
        let content2 = serde_json::to_string_pretty(&scan1).unwrap();
        let scan2: DocumentScanResult = serde_json::from_str(&content2).unwrap();
        assert_eq!(scan1, scan2);
    }
}

// =========================================================================
// TIER 4: REAL-WORLD APPLICATION SCENARIOS (10 TESTS)
// =========================================================================
mod tier4_real_world_scenarios {
    use super::*;

    #[test]
    fn test_t4_01_multipage_pdf_pipeline_execution() {
        let pdf = create_synthetic_pdf(4);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("multipage_summary.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        assert_eq!(scan.total_pages, 4);
        assert_eq!(scan.pages.len(), 4);
    }

    #[test]
    fn test_t4_02_whitfield_real_pdf_pipeline_run() {
        let pdf_path = get_real_pdf_path();
        if pdf_path.exists() {
            let temp = setup_temp_dir();
            let out_json = temp.path().join("whitfield_out.json");
            let res = run_cli_from([
                "glycocr",
                "infer",
                "--dummy",
                "--pdf",
                pdf_path.to_str().unwrap(),
                "--output",
                out_json.to_str().unwrap(),
                "--json",
            ]);
            assert!(res.is_ok());
            let scan: DocumentScanResult =
                serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
            assert!(scan.total_pages > 0);
        }
    }

    #[test]
    fn test_t4_03_kpsc_real_snfg_image_pipeline_run() {
        let img_path = get_real_image_path();
        if img_path.exists() {
            let temp = setup_temp_dir();
            let out_json = temp.path().join("kpsc_out.json");
            let res = run_cli_from([
                "glycocr",
                "infer",
                "--dummy",
                "--image",
                img_path.to_str().unwrap(),
                "--output",
                out_json.to_str().unwrap(),
            ]);
            assert!(res.is_ok());
            let scan: DocumentScanResult =
                serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
            assert_eq!(scan.pages.len(), 1);
            assert!(!scan.pages[0].diagrams.is_empty());
        }
    }

    #[test]
    fn test_t4_04_5_page_document_scan_aggregation() {
        let pdf = create_synthetic_pdf(5);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("doc_5p.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ])
        .unwrap();
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        assert_eq!(scan.total_pages, 5);
        for page_idx in 1..=5 {
            assert!(scan.pages.iter().any(|p| p.page_number == page_idx));
        }
    }

    #[test]
    fn test_t4_05_high_res_synthetic_png_cropping() {
        let img = create_synthetic_image(1000, 1000);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("high_res.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ]);
        assert!(res.is_ok());
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        assert_eq!(scan.pages[0].diagrams[0].iupac, "α-D-Glcp-(1->4)-D-Glcp");
    }

    #[test]
    fn test_t4_06_multi_diagram_per_page_detection_aggregation() {
        let engine = DummyVlmEngine::new();
        let runner = glycocr::pipeline::runner::PipelineRunner::new(&engine);
        let img = create_synthetic_image(500, 500);
        let res = runner.run_image(img.path()).unwrap();
        assert!(!res.pages[0].diagrams.is_empty());
        let diagram = &res.pages[0].diagrams[0];
        assert_eq!(diagram.confidence, 0.95);
    }

    #[test]
    fn test_t4_07_full_cli_pdf_json_stdout_execution() {
        let pdf = create_synthetic_pdf(1);
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
    }

    #[test]
    fn test_t4_08_full_cli_image_json_file_persistence() {
        let img = create_synthetic_image(200, 200);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("persisted.json");
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--image",
            img.path().to_str().unwrap(),
            "-o",
            out_json.to_str().unwrap(),
            "--json",
        ]);
        assert!(res.is_ok());
        assert!(out_json.exists());
    }

    #[test]
    fn test_t4_09_pipeline_error_recovery_on_corrupt_pdf() {
        let corrupt = create_corrupt_pdf();
        let res = run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            corrupt.path().to_str().unwrap(),
        ]);
        assert!(res.is_err());
        let err_msg = res.unwrap_err();
        assert!(err_msg.contains("Invalid PDF format") || err_msg.contains("PdfError"));
    }

    #[test]
    fn test_t4_10_complete_document_scan_result_schema_verification() {
        let pdf = create_synthetic_pdf(2);
        let temp = setup_temp_dir();
        let out_json = temp.path().join("full_schema.json");
        run_cli_from([
            "glycocr",
            "infer",
            "--dummy",
            "--pdf",
            pdf.path().to_str().unwrap(),
            "--output",
            out_json.to_str().unwrap(),
        ])
        .unwrap();
        let scan: DocumentScanResult =
            serde_json::from_str(&fs::read_to_string(out_json).unwrap()).unwrap();
        assert_eq!(scan.total_pages, 2);
        assert_eq!(scan.pages[0].page_number, 1);
        assert_eq!(scan.pages[1].page_number, 2);
        assert_eq!(scan.pages[0].diagrams[0].iupac, "α-D-Glcp-(1->4)-D-Glcp");
        assert_eq!(scan.pages[0].diagrams[0].bbox.ymin, 100.0);
    }
}
