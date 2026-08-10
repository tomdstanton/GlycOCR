use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use std::path::Path;

use crate::error::GlycOCRError;
use crate::model::dummy::DummyVlmEngine;
use crate::model::paligemma::CandlePaliGemmaEngine;
use crate::pipeline::runner::PipelineRunner;

/// Maps pipeline `GlycOCRError` variants to standard Python exception types.
impl From<GlycOCRError> for PyErr {
    fn from(err: GlycOCRError) -> Self {
        match err {
            GlycOCRError::PdfError(msg) => PyValueError::new_err(format!("PDF Error: {}", msg)),
            GlycOCRError::ImageError(msg) => PyValueError::new_err(format!("Image Error: {}", msg)),
            GlycOCRError::ModelError(msg) => {
                PyRuntimeError::new_err(format!("Model Error: {}", msg))
            }
            GlycOCRError::CliError(msg) => PyValueError::new_err(format!("CLI Error: {}", msg)),
            GlycOCRError::IoError(e) => PyFileNotFoundError::new_err(e.to_string()),
        }
    }
}

/// Scans a PDF file and returns a JSON string representation of `DocumentScanResult`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (pdf_path, device="cpu", dummy=false, model_path=None))]
pub fn scan_pdf(
    pdf_path: &str,
    device: &str,
    dummy: bool,
    model_path: Option<&str>,
) -> PyResult<String> {
    let path = Path::new(pdf_path);
    let scan_result = if dummy {
        let engine = DummyVlmEngine::new();
        let runner = PipelineRunner::new(&engine);
        runner.run_pdf(path)?
    } else {
        let engine = CandlePaliGemmaEngine::new(device, model_path.map(Path::new))?;
        let runner = PipelineRunner::new(&engine);
        runner.run_pdf(path)?
    };

    scan_result
        .to_json()
        .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization error: {}", e)))
}

/// Scans a PDF file and returns a native Python dictionary (`PyObject`).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (pdf_path, device="cpu", dummy=false, model_path=None))]
pub fn scan_pdf_dict(
    py: Python<'_>,
    pdf_path: &str,
    device: &str,
    dummy: bool,
    model_path: Option<&str>,
) -> PyResult<PyObject> {
    let json_str = scan_pdf(pdf_path, device, dummy, model_path)?;
    let json_mod = py.import("json")?;
    let dict_obj = json_mod.call_method1("loads", (json_str,))?;
    Ok(dict_obj.into())
}

/// Scans an image file and returns a JSON string representation of `DocumentScanResult`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (image_path, device="cpu", dummy=false, model_path=None))]
pub fn scan_image(
    image_path: &str,
    device: &str,
    dummy: bool,
    model_path: Option<&str>,
) -> PyResult<String> {
    let path = Path::new(image_path);
    let scan_result = if dummy {
        let engine = DummyVlmEngine::new();
        let runner = PipelineRunner::new(&engine);
        runner.run_image(path)?
    } else {
        let engine = CandlePaliGemmaEngine::new(device, model_path.map(Path::new))?;
        let runner = PipelineRunner::new(&engine);
        runner.run_image(path)?
    };

    scan_result
        .to_json()
        .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization error: {}", e)))
}

/// Python class wrapper for configuring and executing pipeline runs.
#[gen_stub_pyclass]
#[pyclass]
pub struct PyPipelineRunner {
    device: String,
    dummy: bool,
    model_path: Option<String>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipelineRunner {
    #[new]
    #[pyo3(signature = (device="cpu", dummy=false, model_path=None))]
    pub fn new(device: &str, dummy: bool, model_path: Option<&str>) -> Self {
        Self {
            device: device.to_string(),
            dummy,
            model_path: model_path.map(|s| s.to_string()),
        }
    }

    /// Executes PDF inference using configured runner parameters, returning a JSON string.
    pub fn run_pdf(&self, pdf_path: &str) -> PyResult<String> {
        scan_pdf(
            pdf_path,
            &self.device,
            self.dummy,
            self.model_path.as_deref(),
        )
    }

    /// Executes PDF inference using configured runner parameters, returning a native Python dict.
    pub fn run_pdf_dict(&self, py: Python<'_>, pdf_path: &str) -> PyResult<PyObject> {
        scan_pdf_dict(
            py,
            pdf_path,
            &self.device,
            self.dummy,
            self.model_path.as_deref(),
        )
    }

    /// Executes single-image inference using configured runner parameters, returning a JSON string.
    pub fn run_image(&self, image_path: &str) -> PyResult<String> {
        scan_image(
            image_path,
            &self.device,
            self.dummy,
            self.model_path.as_deref(),
        )
    }
}

/// Entrypoint for the `glycocr_rs` PyO3 C-extension module.
#[pymodule]
pub fn glycocr_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(scan_pdf_dict, m)?)?;
    m.add_function(wrap_pyfunction!(scan_image, m)?)?;
    m.add_class::<PyPipelineRunner>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
