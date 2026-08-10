#![cfg(not(target_os = "macos"))]

// Milestone 2 PyO3 stress tests harness
extern crate glycocr_rs as glycocr;

use glycocr::error::GlycOCRError;
use glycocr::pyo3_bindings::{PyPipelineRunner, scan_image, scan_pdf, scan_pdf_dict};
use image::{DynamicImage, Rgb, RgbImage};
use lopdf::{Document, Object, Stream, dictionary};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs;
use tempfile::{NamedTempFile, tempdir};

fn create_synthetic_pdf(pages: usize) -> NamedTempFile {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();

    for _ in 0..pages {
        let content_id = doc.add_object(Stream::new(
            dictionary! {},
            b"q 1 0 0 1 0 0 cm /Im1 Do Q".to_vec(),
        ));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
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

    let temp_file = NamedTempFile::new().expect("Failed to create temp pdf");
    doc.save(temp_file.path()).expect("Failed to save temp pdf");
    temp_file
}

fn create_synthetic_image() -> NamedTempFile {
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([200, 200, 200])));
    let temp_file = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("Failed to create temp png");
    img.save(temp_file.path()).expect("Failed to save temp png");
    temp_file
}

#[test]
fn test_pyo3_error_conversion_mapping() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let pdf_err = GlycOCRError::PdfError("PDF decode error".into());
        let py_err1: PyErr = pdf_err.into();
        assert!(py_err1.is_instance_of::<PyValueError>(py));
        assert!(py_err1.to_string().contains("PDF Error: PDF decode error"));

        let img_err = GlycOCRError::ImageError("Image decode error".into());
        let py_err2: PyErr = img_err.into();
        assert!(py_err2.is_instance_of::<PyValueError>(py));
        assert!(
            py_err2
                .to_string()
                .contains("Image Error: Image decode error")
        );

        let model_err = GlycOCRError::ModelError("Model weight missing".into());
        let py_err3: PyErr = model_err.into();
        assert!(py_err3.is_instance_of::<PyRuntimeError>(py));
        assert!(
            py_err3
                .to_string()
                .contains("Model Error: Model weight missing")
        );

        let cli_err = GlycOCRError::CliError("CLI invalid flag".into());
        let py_err4: PyErr = cli_err.into();
        assert!(py_err4.is_instance_of::<PyValueError>(py));
        assert!(py_err4.to_string().contains("CLI Error: CLI invalid flag"));

        let io_err = GlycOCRError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "File not found on disk",
        ));
        let py_err5: PyErr = io_err.into();
        assert!(py_err5.is_instance_of::<PyFileNotFoundError>(py));
        assert!(py_err5.to_string().contains("File not found on disk"));
    });
}

#[test]
fn test_scan_pdf_nonexistent_path() {
    pyo3::prepare_freethreaded_python();
    let res = scan_pdf("/nonexistent_path/fake_doc.pdf", "cpu", true, None);
    assert!(res.is_err(), "Expected error for non-existent PDF path");
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyValueError>(py));
        assert!(err.to_string().contains("PDF Error: PDF file not found"));
    });
}

#[test]
fn test_scan_pdf_zero_byte_file() {
    pyo3::prepare_freethreaded_python();
    let temp_file = NamedTempFile::new().unwrap();
    let path_str = temp_file.path().to_str().unwrap();
    let res = scan_pdf(path_str, "cpu", true, None);
    assert!(res.is_err(), "Expected error for 0-byte PDF file");
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyValueError>(py));
        assert!(err.to_string().contains("0 bytes"));
    });
}

#[test]
fn test_scan_pdf_invalid_header() {
    pyo3::prepare_freethreaded_python();
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), b"INVALID_NOT_A_PDF_FILE").unwrap();
    let path_str = temp_file.path().to_str().unwrap();

    let res = scan_pdf(path_str, "cpu", true, None);
    assert!(res.is_err(), "Expected error for invalid PDF header");
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyValueError>(py));
        assert!(err.to_string().contains("missing %PDF- header"));
    });
}

#[test]
fn test_scan_pdf_directory_path() {
    pyo3::prepare_freethreaded_python();
    let dir = tempdir().unwrap();
    let dir_str = dir.path().to_str().unwrap();

    let res = scan_pdf(dir_str, "cpu", true, None);
    assert!(
        res.is_err(),
        "Expected error for directory path passed to scan_pdf"
    );
}

#[test]
fn test_scan_pdf_invalid_device() {
    pyo3::prepare_freethreaded_python();
    let pdf = create_synthetic_pdf(1);
    let pdf_str = pdf.path().to_str().unwrap();

    let res = scan_pdf(pdf_str, "invalid_device_xyz", false, None);
    assert!(res.is_err(), "Expected error for unsupported device");
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyRuntimeError>(py));
        assert!(
            err.to_string()
                .contains("Unsupported device 'invalid_device_xyz'")
        );
    });
}

#[test]
fn test_scan_pdf_missing_model_path() {
    pyo3::prepare_freethreaded_python();
    let pdf = create_synthetic_pdf(1);
    let pdf_str = pdf.path().to_str().unwrap();
    let missing_model = "/path/to/missing_weights.safetensors";

    let res = scan_pdf(pdf_str, "cpu", false, Some(missing_model));
    assert!(
        res.is_err(),
        "Expected error for missing model weights file"
    );
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyRuntimeError>(py));
        assert!(err.to_string().contains("Model weights file not found"));
    });
}

#[test]
fn test_scan_pdf_valid_dummy_execution() {
    pyo3::prepare_freethreaded_python();
    let pdf = create_synthetic_pdf(2);
    let pdf_str = pdf.path().to_str().unwrap();

    let res = scan_pdf(pdf_str, "cpu", true, None);
    assert!(
        res.is_ok(),
        "scan_pdf should succeed in dummy mode: {:?}",
        res.err()
    );
    let json_str = res.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(parsed["total_pages"], 2);
    assert_eq!(parsed["pages"].as_array().unwrap().len(), 2);
    assert!(
        parsed["pdf_path"].as_str().unwrap().ends_with(".tmp")
            || parsed["pdf_path"].as_str().unwrap().contains("tmp")
    );
}

#[test]
fn test_scan_pdf_dict_valid_and_invalid() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let res_err = scan_pdf_dict(py, "/nonexistent/doc.pdf", "cpu", true, None);
        assert!(res_err.is_err());

        let pdf = create_synthetic_pdf(1);
        let pdf_str = pdf.path().to_str().unwrap();
        let res_ok = scan_pdf_dict(py, pdf_str, "cpu", true, None);
        assert!(res_ok.is_ok());

        let py_obj = res_ok.unwrap();
        let py_dict = py_obj.bind(py).downcast::<PyDict>().unwrap();

        let total_pages: usize = py_dict
            .get_item("total_pages")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(total_pages, 1);

        let pdf_path: String = py_dict
            .get_item("pdf_path")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(pdf_path, pdf_str);
    });
}

#[test]
fn test_scan_image_nonexistent_file() {
    pyo3::prepare_freethreaded_python();
    let res = scan_image("/nonexistent/img.png", "cpu", true, None);
    assert!(res.is_err(), "Expected error for non-existent image path");
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyValueError>(py));
        assert!(err.to_string().contains("Image file not found"));
    });
}

#[test]
fn test_scan_image_invalid_image_format() {
    pyo3::prepare_freethreaded_python();
    let temp_file = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
    fs::write(temp_file.path(), b"NOT_A_PNG_IMAGE_BINARY").unwrap();
    let img_str = temp_file.path().to_str().unwrap();

    let res = scan_image(img_str, "cpu", true, None);
    assert!(res.is_err(), "Expected error for invalid image data");
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyValueError>(py));
        assert!(err.to_string().contains("Failed to open image"));
    });
}

#[test]
fn test_scan_image_invalid_device() {
    pyo3::prepare_freethreaded_python();
    let img = create_synthetic_image();
    let img_str = img.path().to_str().unwrap();

    let res = scan_image(img_str, "bad_device_name", false, None);
    assert!(res.is_err(), "Expected error for unsupported device");
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyRuntimeError>(py));
        assert!(err.to_string().contains("Unsupported device"));
    });
}

#[test]
fn test_scan_image_missing_model_path() {
    pyo3::prepare_freethreaded_python();
    let img = create_synthetic_image();
    let img_str = img.path().to_str().unwrap();

    let res = scan_image(img_str, "cpu", false, Some("/missing/weights.safetensors"));
    assert!(
        res.is_err(),
        "Expected error for missing model weights path"
    );
    let err = res.unwrap_err();
    Python::with_gil(|py| {
        assert!(err.is_instance_of::<PyRuntimeError>(py));
        assert!(err.to_string().contains("Model weights file not found"));
    });
}

#[test]
fn test_scan_image_valid_dummy_execution() {
    pyo3::prepare_freethreaded_python();
    let img = create_synthetic_image();
    let img_str = img.path().to_str().unwrap();

    let res = scan_image(img_str, "cpu", true, None);
    assert!(
        res.is_ok(),
        "scan_image should succeed in dummy mode: {:?}",
        res.err()
    );
    let json_str = res.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(parsed["total_pages"], 1);
    assert_eq!(parsed["pages"].as_array().unwrap().len(), 1);
    let diagrams = &parsed["pages"][0]["diagrams"];
    assert_eq!(diagrams.as_array().unwrap().len(), 1);
    assert_eq!(diagrams[0]["iupac"], "α-D-Glcp-(1->4)-D-Glcp");
}

#[test]
fn test_py_pipeline_runner_pdf_dummy_flow() {
    pyo3::prepare_freethreaded_python();
    let runner = PyPipelineRunner::new("cpu", true, None);

    let err_res = runner.run_pdf("/nonexistent/file.pdf");
    assert!(err_res.is_err());

    let pdf = create_synthetic_pdf(1);
    let pdf_str = pdf.path().to_str().unwrap();
    let ok_res = runner.run_pdf(pdf_str);
    assert!(ok_res.is_ok());

    let parsed: serde_json::Value = serde_json::from_str(&ok_res.unwrap()).unwrap();
    assert_eq!(parsed["total_pages"], 1);
}

#[test]
fn test_py_pipeline_runner_pdf_dict_dummy_flow() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let runner = PyPipelineRunner::new("cpu", true, None);

        let err_res = runner.run_pdf_dict(py, "/nonexistent/file.pdf");
        assert!(err_res.is_err());

        let pdf = create_synthetic_pdf(1);
        let pdf_str = pdf.path().to_str().unwrap();
        let ok_res = runner.run_pdf_dict(py, pdf_str);
        assert!(ok_res.is_ok());

        let dict_obj = ok_res.unwrap();
        let bound_dict = dict_obj.bind(py).downcast::<PyDict>().unwrap();
        let total_pages: usize = bound_dict
            .get_item("total_pages")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(total_pages, 1);
    });
}

#[test]
fn test_py_pipeline_runner_image_dummy_flow() {
    pyo3::prepare_freethreaded_python();
    let runner = PyPipelineRunner::new("cpu", true, None);

    let err_res = runner.run_image("/nonexistent/img.png");
    assert!(err_res.is_err());

    let img = create_synthetic_image();
    let img_str = img.path().to_str().unwrap();
    let ok_res = runner.run_image(img_str);
    assert!(ok_res.is_ok());

    let parsed: serde_json::Value = serde_json::from_str(&ok_res.unwrap()).unwrap();
    assert_eq!(parsed["total_pages"], 1);
}

#[test]
fn test_py_pipeline_runner_candle_invalid_device_and_missing_path() {
    pyo3::prepare_freethreaded_python();
    let pdf = create_synthetic_pdf(1);
    let pdf_str = pdf.path().to_str().unwrap();

    let runner_invalid_dev = PyPipelineRunner::new("tpu_invalid", false, None);
    let res1 = runner_invalid_dev.run_pdf(pdf_str);
    assert!(res1.is_err());

    let runner_missing_weights =
        PyPipelineRunner::new("cpu", false, Some("/missing/weights.safetensors"));
    let res2 = runner_missing_weights.run_pdf(pdf_str);
    assert!(res2.is_err());
}

#[test]
fn test_unicode_path_handling_pyo3() {
    pyo3::prepare_freethreaded_python();
    let dir = tempdir().unwrap();
    let unicode_pdf_path = dir.path().join("glycan_🧪_test_document_αβ.pdf");
    let synthetic_pdf = create_synthetic_pdf(1);
    fs::copy(synthetic_pdf.path(), &unicode_pdf_path).unwrap();

    let pdf_str = unicode_pdf_path.to_str().unwrap();
    let res = scan_pdf(pdf_str, "cpu", true, None);
    assert!(res.is_ok(), "Unicode PDF path should be handled cleanly");

    let unicode_img_path = dir.path().join("glycan_🔬_image_αβ.png");
    let synthetic_img = create_synthetic_image();
    fs::copy(synthetic_img.path(), &unicode_img_path).unwrap();

    let img_str = unicode_img_path.to_str().unwrap();
    let res_img = scan_image(img_str, "cpu", true, None);
    assert!(
        res_img.is_ok(),
        "Unicode Image path should be handled cleanly"
    );
}
