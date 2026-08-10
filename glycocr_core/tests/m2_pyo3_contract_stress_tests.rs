#![cfg(not(target_os = "macos"))]

// Milestone 2 PyO3 contract stress tests
use glycocr_rs::DocumentScanResult;
use glycocr_rs::pyo3_bindings::{
    PyPipelineRunner, glycocr_rs as register_glycocr_rs, scan_image, scan_pdf, scan_pdf_dict,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::path::PathBuf;

fn get_sample_pdf_path() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pdf_path = manifest_dir.join("../tests/data/whitfield-et-al-2025-o-antigen-polysaccharides-in-klebsiella-pneumoniae-structures-and-molecular-basis-for-antigenic.pdf");
    pdf_path.to_string_lossy().to_string()
}

fn get_sample_image_path() -> String {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let image_path = manifest_dir.join("../tests/data/kpsc_K1_cps_SNFG.jpg");
    image_path.to_string_lossy().to_string()
}

#[test]
fn test_pyo3_pymodule_registration() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let module = PyModule::new(py, "glycocr_rs").expect("Failed to create PyModule");
        register_glycocr_rs(py, &module).expect("Failed to register glycocr_rs PyModule");

        assert!(
            module.hasattr("scan_pdf").unwrap(),
            "Module missing scan_pdf"
        );
        assert!(
            module.hasattr("scan_pdf_dict").unwrap(),
            "Module missing scan_pdf_dict"
        );
        assert!(
            module.hasattr("scan_image").unwrap(),
            "Module missing scan_image"
        );
        assert!(
            module.hasattr("PyPipelineRunner").unwrap(),
            "Module missing PyPipelineRunner"
        );
    });
}

#[test]
fn test_scan_pdf_vs_scan_pdf_dict_contracts() {
    pyo3::prepare_freethreaded_python();
    let pdf_path = get_sample_pdf_path();

    Python::with_gil(|py| {
        // 1. scan_pdf returns String (JSON)
        let json_str =
            scan_pdf(&pdf_path, "cpu", true, None).expect("scan_pdf failed with dummy=true");
        assert!(!json_str.is_empty(), "JSON output should not be empty");

        let scan_res: DocumentScanResult = serde_json::from_str(&json_str)
            .expect("scan_pdf JSON string failed serde_json deserialization");
        assert!(
            scan_res.total_pages > 0,
            "Document scan result should have total_pages > 0"
        );

        // 2. scan_pdf_dict returns Python dict (PyObject)
        let dict_pyobj = scan_pdf_dict(py, &pdf_path, "cpu", true, None)
            .expect("scan_pdf_dict failed with dummy=true");

        let bound_dict = dict_pyobj.bind(py);
        assert!(
            bound_dict.is_instance_of::<PyDict>(),
            "scan_pdf_dict result is not PyDict"
        );

        let dict = bound_dict.downcast::<PyDict>().unwrap();
        assert!(
            dict.contains("pdf_path").unwrap(),
            "Dict missing key pdf_path"
        );
        assert!(
            dict.contains("total_pages").unwrap(),
            "Dict missing key total_pages"
        );
        assert!(dict.contains("pages").unwrap(), "Dict missing key pages");

        let dict_total_pages: usize = dict
            .get_item("total_pages")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(dict_total_pages, scan_res.total_pages);

        let pages_obj = dict.get_item("pages").unwrap().unwrap();
        assert!(
            pages_obj.is_instance_of::<PyList>(),
            "pages field in dict is not PyList"
        );

        let pages_list = pages_obj.downcast::<PyList>().unwrap();
        assert_eq!(pages_list.len(), scan_res.total_pages);

        // Check deep nesting conversion: pages[0] -> diagrams[0] -> bbox, iupac, confidence
        let first_page = pages_list
            .get_item(0)
            .unwrap()
            .downcast::<PyDict>()
            .unwrap()
            .clone();
        assert!(first_page.contains("page_number").unwrap());
        assert!(first_page.contains("diagrams").unwrap());

        let diagrams_list = first_page
            .get_item("diagrams")
            .unwrap()
            .unwrap()
            .downcast::<PyList>()
            .unwrap()
            .clone();
        if !diagrams_list.is_empty() {
            let first_diagram = diagrams_list
                .get_item(0)
                .unwrap()
                .downcast::<PyDict>()
                .unwrap()
                .clone();
            assert!(first_diagram.contains("bbox").unwrap());
            assert!(first_diagram.contains("iupac").unwrap());
            assert!(first_diagram.contains("confidence").unwrap());

            let bbox_dict = first_diagram
                .get_item("bbox")
                .unwrap()
                .unwrap()
                .downcast::<PyDict>()
                .unwrap()
                .clone();
            assert!(bbox_dict.contains("ymin").unwrap());
            assert!(bbox_dict.contains("xmin").unwrap());
            assert!(bbox_dict.contains("ymax").unwrap());
            assert!(bbox_dict.contains("xmax").unwrap());
        }
    });
}

#[test]
fn test_py_pipeline_runner_class_and_methods() {
    pyo3::prepare_freethreaded_python();
    let pdf_path = get_sample_pdf_path();
    let image_path = get_sample_image_path();

    Python::with_gil(|py| {
        let runner = PyPipelineRunner::new("cpu", true, Some("/mock/path"));

        // run_pdf returns JSON string
        let json_str = runner.run_pdf(&pdf_path).expect("runner.run_pdf failed");
        let parsed: DocumentScanResult = serde_json::from_str(&json_str).expect("Valid JSON");
        assert!(parsed.total_pages > 0);

        // run_pdf_dict returns PyDict
        let dict_obj = runner
            .run_pdf_dict(py, &pdf_path)
            .expect("runner.run_pdf_dict failed");
        let dict = dict_obj.bind(py).downcast::<PyDict>().unwrap();
        assert!(dict.contains("pdf_path").unwrap());
        assert!(dict.contains("pages").unwrap());

        // run_image returns JSON string
        let img_json = runner
            .run_image(&image_path)
            .expect("runner.run_image failed");
        let img_parsed: DocumentScanResult =
            serde_json::from_str(&img_json).expect("Valid image JSON");
        assert_eq!(img_parsed.total_pages, 1);
    });
}

#[test]
fn test_scan_image_contract() {
    pyo3::prepare_freethreaded_python();
    let image_path = get_sample_image_path();

    let json_str =
        scan_image(&image_path, "cpu", true, None).expect("scan_image dummy=true failed");
    let scan_res: DocumentScanResult = serde_json::from_str(&json_str).expect("Valid JSON");
    assert_eq!(scan_res.total_pages, 1);
    assert_eq!(scan_res.pages.len(), 1);
    assert!(scan_res.pages[0].diagram_count() > 0);
}

#[test]
fn test_optional_parameters_and_defaults() {
    pyo3::prepare_freethreaded_python();
    let pdf_path = get_sample_pdf_path();

    Python::with_gil(|py| {
        // dummy=true with model_path specified
        let dict_obj = scan_pdf_dict(
            py,
            &pdf_path,
            "cpu",
            true,
            Some("/tmp/dummy_model.safetensors"),
        )
        .expect("scan_pdf_dict with optional model_path failed");
        assert!(dict_obj.bind(py).is_instance_of::<PyDict>());

        // invalid device when dummy=false should produce a model error (PyRuntimeError or PyValueError)
        let err = scan_pdf(&pdf_path, "unsupported_device_xyz", false, None);
        assert!(
            err.is_err(),
            "Invalid device with dummy=false should return PyResult::Err"
        );
    });
}

#[test]
fn test_error_mapping_for_missing_file() {
    pyo3::prepare_freethreaded_python();
    let nonexistent_pdf = "/nonexistent/path/to/missing_file.pdf";

    let err = scan_pdf(nonexistent_pdf, "cpu", true, None);
    assert!(err.is_err(), "scan_pdf on missing file should error");

    Python::with_gil(|py| {
        let py_err = err.unwrap_err();
        assert!(
            py_err.is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py)
                || py_err.is_instance_of::<pyo3::exceptions::PyValueError>(py),
            "Expected PyFileNotFoundError or PyValueError, got {:?}",
            py_err
        );
    });
}

#[test]
fn test_repeated_invocations_and_stress() {
    pyo3::prepare_freethreaded_python();
    let pdf_path = get_sample_pdf_path();

    Python::with_gil(|py| {
        for i in 0..50 {
            let json_str = scan_pdf(&pdf_path, "cpu", true, None)
                .unwrap_or_else(|_| panic!("Failed on iteration {}", i));
            assert!(!json_str.is_empty());

            let dict_pyobj = scan_pdf_dict(py, &pdf_path, "cpu", true, None)
                .unwrap_or_else(|_| panic!("Failed dict conversion on iteration {}", i));
            assert!(dict_pyobj.bind(py).is_instance_of::<PyDict>());
        }
    });
}
