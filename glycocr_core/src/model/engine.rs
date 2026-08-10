use crate::error::GlycOCRError;
use crate::types::BoundingBox;
use image::DynamicImage;

/// Vision-Language Model Engine trait for diagram detection and OCR inference.
pub trait VlmEngine: Send + Sync {
    /// Detects SNFG diagram bounding boxes within an image page.
    fn detect_diagrams(&self, img: &DynamicImage) -> Result<Vec<BoundingBox>, GlycOCRError>;

    /// Performs OCR on a cropped diagram image and returns an IUPAC glycan sequence string.
    fn ocr_diagram(&self, crop: &DynamicImage) -> Result<String, GlycOCRError>;
}
