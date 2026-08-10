// Milestone 2 PyO3 error variant test harness
extern crate glycocr_rs as glycocr;

use glycocr::error::GlycOCRError;

#[test]
fn test_glycocr_error_variants() {
    let pdf_err = GlycOCRError::PdfError("PDF decode error".into());
    assert_eq!(pdf_err.to_string(), "PDF error: PDF decode error");

    let img_err = GlycOCRError::ImageError("Image decode error".into());
    assert_eq!(img_err.to_string(), "Image error: Image decode error");

    let model_err = GlycOCRError::ModelError("Model weight missing".into());
    assert_eq!(model_err.to_string(), "Model error: Model weight missing");

    let cli_err = GlycOCRError::CliError("CLI invalid flag".into());
    assert_eq!(cli_err.to_string(), "CLI error: CLI invalid flag");

    let io_err = GlycOCRError::IoError(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "File not found on disk",
    ));
    assert_eq!(io_err.to_string(), "IO error: File not found on disk");
}
