//! M4 Empirical Challenger Integration & Boundary Test Suite
//!
//! Comprehensive empirical stress-testing for Milestone 4 (Pipeline Runner, CLI & JSON Output).
//! Verifies invalid flag combinations, missing/corrupt files, unsupported devices, output exports,
//! and JSON schema compliance.

extern crate glycocr_rs as glycocr;

use glycocr::cli::run_cli_from;
use glycocr::types::DocumentScanResult;
use lopdf::{Document, Object, Stream, dictionary};
use std::fs;
use std::path::PathBuf;
use tempfile::{NamedTempFile, TempDir};

fn create_synthetic_pdf(pages: usize) -> NamedTempFile {
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
        .expect("Failed to create temp pdf");
    doc.save(temp_file.path())
        .expect("Failed to save synthetic pdf");
    temp_file
}

fn create_synthetic_image(width: u32, height: u32) -> NamedTempFile {
    let img = image::RgbImage::from_pixel(width, height, image::Rgb([180, 180, 180]));
    let temp_file = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("Failed to create temp image");
    img.save_with_format(temp_file.path(), image::ImageFormat::Png)
        .expect("Failed to save synthetic png");
    temp_file
}

fn create_corrupt_pdf() -> NamedTempFile {
    let temp_file = tempfile::Builder::new()
        .suffix(".pdf")
        .tempfile()
        .expect("Failed to create corrupt pdf tempfile");
    fs::write(temp_file.path(), b"NOT_A_VALID_PDF_HEADER_OR_STREAM").unwrap();
    temp_file
}

fn create_corrupt_image() -> NamedTempFile {
    let temp_file = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("Failed to create corrupt png tempfile");
    fs::write(temp_file.path(), b"CORRUPT_PNG_BYTES_HEADER_INVALID").unwrap();
    temp_file
}

// ---------------------------------------------------------------------------
// 1. INVALID FLAG COMBINATIONS
// ---------------------------------------------------------------------------
#[test]
fn test_m4_challenger_conflicting_pdf_and_image_flags() {
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
    assert!(
        res.is_err(),
        "Expected error when specifying both --pdf and --image"
    );
    let err = res.unwrap_err();
    assert!(
        err.contains("Cannot specify both"),
        "Error message was: {}",
        err
    );
}

#[test]
fn test_m4_challenger_missing_both_pdf_and_image_flags() {
    let res = run_cli_from(["glycocr", "infer", "--dummy"]);
    assert!(
        res.is_err(),
        "Expected error when omitting both --pdf and --image"
    );
    let err = res.unwrap_err();
    assert!(
        err.contains("must be provided"),
        "Error message was: {}",
        err
    );
}

#[test]
fn test_m4_challenger_missing_subcommand() {
    let res = run_cli_from(["glycocr"]);
    assert!(
        res.is_err(),
        "Expected error when no subcommand is provided"
    );
    let err = res.unwrap_err();
    assert!(
        err.contains("No subcommand provided"),
        "Error message was: {}",
        err
    );
}

#[test]
fn test_m4_challenger_invalid_subcommand() {
    let res = run_cli_from(["glycocr", "extract"]);
    assert!(res.is_err(), "Expected error for invalid subcommand");
}

#[test]
fn test_m4_challenger_unrecognized_infer_arg() {
    let pdf = create_synthetic_pdf(1);
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--pdf",
        pdf.path().to_str().unwrap(),
        "--bogus-flag",
    ]);
    assert!(res.is_err(), "Expected error for unrecognized CLI argument");
}

#[test]
fn test_m4_challenger_output_directory_does_not_exist() {
    let pdf = create_synthetic_pdf(1);
    let non_existent = PathBuf::from("/tmp/non_existent_dir_m4_challenge_999/output.json");
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--pdf",
        pdf.path().to_str().unwrap(),
        "--output",
        non_existent.to_str().unwrap(),
    ]);
    assert!(
        res.is_err(),
        "Expected error when output directory does not exist"
    );
    let err = res.unwrap_err();
    assert!(
        err.contains("Output directory does not exist") || err.contains("No such file"),
        "Error: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// 2. MISSING & CORRUPT INPUT FILES
// ---------------------------------------------------------------------------
#[test]
fn test_m4_challenger_missing_pdf_file() {
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--pdf",
        "/nonexistent_path_to_pdf_12345.pdf",
    ]);
    assert!(res.is_err(), "Expected error for non-existent PDF file");
    let err = res.unwrap_err();
    assert!(
        err.contains("PdfError") || err.contains("not found"),
        "Error: {}",
        err
    );
}

#[test]
fn test_m4_challenger_missing_image_file() {
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--image",
        "/nonexistent_path_to_image_12345.png",
    ]);
    assert!(res.is_err(), "Expected error for non-existent image file");
    let err = res.unwrap_err();
    assert!(
        err.contains("ImageError") || err.contains("not found"),
        "Error: {}",
        err
    );
}

#[test]
fn test_m4_challenger_zero_byte_pdf_file() {
    let empty_file = NamedTempFile::new().unwrap();
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--pdf",
        empty_file.path().to_str().unwrap(),
    ]);
    assert!(res.is_err(), "Expected error for zero-byte PDF file");
    let err = res.unwrap_err();
    assert!(
        err.contains("empty") || err.contains("PdfError"),
        "Error: {}",
        err
    );
}

#[test]
fn test_m4_challenger_zero_byte_image_file() {
    let empty_file = NamedTempFile::new().unwrap();
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--image",
        empty_file.path().to_str().unwrap(),
    ]);
    assert!(res.is_err(), "Expected error for zero-byte image file");
    let err = res.unwrap_err();
    assert!(
        err.contains("ImageError") || err.contains("open image"),
        "Error: {}",
        err
    );
}

#[test]
fn test_m4_challenger_corrupt_pdf_file() {
    let corrupt_pdf = create_corrupt_pdf();
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--pdf",
        corrupt_pdf.path().to_str().unwrap(),
    ]);
    assert!(res.is_err(), "Expected error for corrupt PDF file");
    let err = res.unwrap_err();
    assert!(
        err.contains("Invalid PDF format") || err.contains("PdfError"),
        "Error: {}",
        err
    );
}

#[test]
fn test_m4_challenger_corrupt_image_file() {
    let corrupt_img = create_corrupt_image();
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--image",
        corrupt_img.path().to_str().unwrap(),
    ]);
    assert!(res.is_err(), "Expected error for corrupt image file");
    let err = res.unwrap_err();
    assert!(
        err.contains("ImageError") || err.contains("Failed to open image"),
        "Error: {}",
        err
    );
}

#[test]
fn test_m4_challenger_directory_passed_as_file_input() {
    let temp_dir = TempDir::new().unwrap();
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--image",
        temp_dir.path().to_str().unwrap(),
    ]);
    assert!(
        res.is_err(),
        "Expected error when passing a directory path as image input"
    );
}

// ---------------------------------------------------------------------------
// 3. UNSUPPORTED DEVICES & DEVICE NORMALIZATION
// ---------------------------------------------------------------------------
#[test]
fn test_m4_challenger_unsupported_devices() {
    let pdf = create_synthetic_pdf(1);
    let pdf_str = pdf.path().to_str().unwrap();
    let invalid_devices = ["tpu", "vulkan", "directx", "opencl", "super_gpu_123", ""];

    for dev in invalid_devices {
        let res = run_cli_from([
            "glycocr", "infer", "--dummy", "--pdf", pdf_str, "--device", dev,
        ]);
        assert!(res.is_err(), "Expected device '{}' to be rejected", dev);
        let err = res.unwrap_err();
        assert!(
            err.contains("Unsupported device") || err.contains("Invalid"),
            "Device '{}' error: {}",
            dev,
            err
        );
    }
}

#[test]
fn test_m4_challenger_valid_device_normalization() {
    let pdf = create_synthetic_pdf(1);
    let pdf_str = pdf.path().to_str().unwrap();
    let valid_variations = [
        "auto", "AUTO", " Auto ", "cpu", "CPU", " Cpu ", "metal", "METAL", " Metal ", "cuda",
        "CUDA", " Cuda ", "gpu", "GPU", " Gpu ",
    ];

    for dev in valid_variations {
        let res = run_cli_from([
            "glycocr", "infer", "--dummy", "--pdf", pdf_str, "--device", dev,
        ]);
        assert!(
            res.is_ok(),
            "Expected device '{}' to be accepted and normalized",
            dev
        );
    }
}

// ---------------------------------------------------------------------------
// 4. OUTPUT FILE EXPORT & JSON SCHEMA COMPLIANCE
// ---------------------------------------------------------------------------
#[test]
fn test_m4_challenger_output_file_export_and_schema_validation() {
    let pdf = create_synthetic_pdf(2);
    let temp_dir = TempDir::new().unwrap();
    let output_json_path = temp_dir.path().join("scan_result.json");

    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--pdf",
        pdf.path().to_str().unwrap(),
        "--output",
        output_json_path.to_str().unwrap(),
        "--json",
    ]);

    assert!(res.is_ok(), "CLI run failed with --output and --json flags");
    assert!(
        output_json_path.exists(),
        "Output JSON file was not created"
    );

    let json_content = fs::read_to_string(&output_json_path).expect("Failed to read output JSON");
    assert!(!json_content.is_empty(), "JSON output file is empty");

    // Deserialize into DocumentScanResult
    let scan: DocumentScanResult = serde_json::from_str(&json_content)
        .expect("Failed to deserialize output JSON into DocumentScanResult schema");

    assert_eq!(scan.total_pages, 2, "Expected total_pages == 2");
    assert_eq!(scan.pages.len(), 2, "Expected 2 PageResults");

    for (idx, page) in scan.pages.iter().enumerate() {
        assert_eq!(
            page.page_number,
            idx + 1,
            "Page numbers should be 1-indexed"
        );
        assert!(
            !page.diagrams.is_empty(),
            "Expected at least one detected diagram per page in dummy mode"
        );

        for diagram in &page.diagrams {
            assert!(
                diagram.confidence >= 0.0 && diagram.confidence <= 1.0,
                "Confidence out of range [0, 1]"
            );
            assert!(
                !diagram.iupac.is_empty(),
                "IUPAC string should not be empty"
            );
            assert!(
                diagram.bbox.ymin <= diagram.bbox.ymax,
                "ymin must be <= ymax"
            );
            assert!(
                diagram.bbox.xmin <= diagram.bbox.xmax,
                "xmin must be <= xmax"
            );
        }
    }
}

#[test]
fn test_m4_challenger_image_output_file_export_schema() {
    let img = create_synthetic_image(200, 200);
    let temp_dir = TempDir::new().unwrap();
    let output_json_path = temp_dir.path().join("image_scan_result.json");

    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--image",
        img.path().to_str().unwrap(),
        "--output",
        output_json_path.to_str().unwrap(),
    ]);

    assert!(res.is_ok(), "CLI image run failed");
    assert!(
        output_json_path.exists(),
        "Image scan JSON file not created"
    );

    let json_content = fs::read_to_string(&output_json_path).unwrap();
    let scan: DocumentScanResult = serde_json::from_str(&json_content).unwrap();

    assert_eq!(scan.total_pages, 1);
    assert_eq!(scan.pages.len(), 1);
    assert_eq!(scan.pages[0].page_number, 1);
    assert_eq!(
        scan.pages[0].diagrams[0].cropped_path.as_deref(),
        Some(img.path().to_str().unwrap())
    );
}

// ---------------------------------------------------------------------------
// 5. STRESS & EDGE CASES
// ---------------------------------------------------------------------------
#[test]
fn test_m4_challenger_unicode_and_space_paths() {
    let temp_dir = TempDir::new().unwrap();
    let space_unicode_path = temp_dir
        .path()
        .join("GlycOCR Test Path 糖锁 2026/sample_image.png");
    fs::create_dir_all(space_unicode_path.parent().unwrap()).unwrap();

    let img = create_synthetic_image(128, 128);
    fs::copy(img.path(), &space_unicode_path).unwrap();

    let output_json = temp_dir.path().join("GlycOCR Test Path 糖锁 2026/out.json");

    let res = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--image",
        space_unicode_path.to_str().unwrap(),
        "--output",
        output_json.to_str().unwrap(),
    ]);

    assert!(
        res.is_ok(),
        "Execution failed for path with spaces and unicode"
    );
    assert!(output_json.exists());
}

#[test]
fn test_m4_challenger_large_page_count_pdf_stress() {
    let pdf = create_synthetic_pdf(15);
    let temp_dir = TempDir::new().unwrap();
    let out_json = temp_dir.path().join("large_pdf.json");

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
    assert_eq!(scan.total_pages, 15);
    assert_eq!(scan.pages.len(), 15);
}

#[test]
fn test_m4_challenger_real_candlepali_engine_cpu_execution() {
    let img = create_synthetic_image(64, 64);
    let res = run_cli_from([
        "glycocr",
        "infer",
        "--image",
        img.path().to_str().unwrap(),
        "--device",
        "cpu",
    ]);

    assert!(
        res.is_ok(),
        "Default CandlePaliGemmaEngine run failed on CPU"
    );
}
