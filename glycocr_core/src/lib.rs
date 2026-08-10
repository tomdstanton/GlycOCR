pub mod cli;
pub mod config;
pub mod error;
pub mod model;
pub mod pipeline;
pub mod pyo3_bindings;
pub mod types;
pub mod utils;

pub use cli::run_cli_from;
pub use config::Config;
pub use error::GlycOCRError;
pub use model::{CandlePaliGemmaEngine, DummyVlmEngine, VlmEngine};
pub use pipeline::{
    PipelineRunner, crop_and_pad_bbox, extract_pdf_pages, preprocess_image_to_tensor,
};
pub use pyo3_bindings::{PyPipelineRunner, scan_image, scan_pdf, scan_pdf_dict, stub_info};
pub use types::{BoundingBox, DetectedDiagram, DocumentScanResult, PageResult};
pub use utils::{image_to_tensor, select_device, select_device_auto};
