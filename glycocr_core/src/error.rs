use thiserror::Error;

/// Custom error type for the GlycOCR pipeline using `thiserror`.
#[derive(Error, Debug)]
pub enum GlycOCRError {
    /// Errors originating from PDF loading, parsing, or page rendering
    #[error("PDF error: {0}")]
    PdfError(String),

    /// Errors originating from image decoding, cropping, or preprocessing
    #[error("Image error: {0}")]
    ImageError(String),

    /// Errors originating from Model inference, weight loading, or Candle tensor operations
    #[error("Model error: {0}")]
    ModelError(String),

    /// Errors originating from CLI argument parsing or command execution
    #[error("CLI error: {0}")]
    CliError(String),

    /// Errors originating from underlying filesystem or I/O operations
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Convenience alias for `std::result::Result<T, GlycOCRError>`
pub type Result<T> = std::result::Result<T, GlycOCRError>;

impl From<image::ImageError> for GlycOCRError {
    fn from(err: image::ImageError) -> Self {
        GlycOCRError::ImageError(err.to_string())
    }
}

impl From<serde_json::Error> for GlycOCRError {
    fn from(err: serde_json::Error) -> Self {
        GlycOCRError::ModelError(format!("JSON error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_error_display() {
        let err1 = GlycOCRError::PdfError("Failed to open PDF".into());
        assert_eq!(err1.to_string(), "PDF error: Failed to open PDF");

        let err2 = GlycOCRError::ImageError("Invalid crop bounds".into());
        assert_eq!(err2.to_string(), "Image error: Invalid crop bounds");

        let err3 = GlycOCRError::ModelError("Tensor shape mismatch".into());
        assert_eq!(err3.to_string(), "Model error: Tensor shape mismatch");

        let err4 = GlycOCRError::CliError("Missing argument --pdf".into());
        assert_eq!(err4.to_string(), "CLI error: Missing argument --pdf");

        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err5: GlycOCRError = io_err.into();
        assert_eq!(err5.to_string(), "IO error: file not found");
    }

    #[test]
    fn test_image_error_conversion() {
        let img_err = image::ImageError::Limits(image::error::LimitError::from_kind(
            image::error::LimitErrorKind::DimensionError,
        ));
        let expected_msg = format!("Image error: {}", img_err);
        let glyc_err: GlycOCRError = img_err.into();
        assert_eq!(glyc_err.to_string(), expected_msg);
        assert!(matches!(glyc_err, GlycOCRError::ImageError(_)));
    }

    #[test]
    fn test_serde_json_error_conversion() {
        let serde_err: std::result::Result<serde_json::Value, serde_json::Error> =
            serde_json::from_str("invalid json");
        let json_err = serde_err.unwrap_err();
        let glyc_err: GlycOCRError = json_err.into();
        assert!(matches!(glyc_err, GlycOCRError::ModelError(_)));
        assert!(glyc_err.to_string().starts_with("Model error: JSON error:"));
    }
}
