use crate::error::GlycOCRError;
use crate::model::engine::VlmEngine;
use crate::types::BoundingBox;
use image::DynamicImage;

/// A mock implementation of `VlmEngine` for offline execution, unit testing, and CI testing.
///
/// Returns synthetic bounding boxes and valid IUPAC glycan strings without requiring GPU or model weights.
#[derive(Debug, Clone)]
pub struct DummyVlmEngine {
    /// Mock bounding boxes returned by `detect_diagrams`.
    mock_boxes: Vec<BoundingBox>,
    /// Default IUPAC glycan string returned by `ocr_diagram`.
    mock_iupac: String,
}

impl Default for DummyVlmEngine {
    fn default() -> Self {
        Self {
            mock_boxes: vec![BoundingBox {
                ymin: 100.0,
                xmin: 100.0,
                ymax: 400.0,
                xmax: 400.0,
                label: Some("SNFG".to_string()),
            }],
            mock_iupac: "α-D-Glcp-(1->4)-D-Glcp".to_string(),
        }
    }
}

impl DummyVlmEngine {
    /// Creates a new `DummyVlmEngine` with default mock data.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a custom `DummyVlmEngine` with user-specified mock bounding boxes and IUPAC string.
    pub fn with_mock_data(mock_boxes: Vec<BoundingBox>, mock_iupac: impl Into<String>) -> Self {
        Self {
            mock_boxes,
            mock_iupac: mock_iupac.into(),
        }
    }
}

impl VlmEngine for DummyVlmEngine {
    fn detect_diagrams(&self, _img: &DynamicImage) -> Result<Vec<BoundingBox>, GlycOCRError> {
        Ok(self.mock_boxes.clone())
    }

    fn ocr_diagram(&self, _crop: &DynamicImage) -> Result<String, GlycOCRError> {
        Ok(self.mock_iupac.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn test_dummy_vlm_engine_default() {
        let engine = DummyVlmEngine::new();
        let dummy_img = DynamicImage::new_rgb8(100, 100);

        let boxes = engine.detect_diagrams(&dummy_img).unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].ymin, 100.0);
        assert_eq!(boxes[0].xmin, 100.0);
        assert_eq!(boxes[0].ymax, 400.0);
        assert_eq!(boxes[0].xmax, 400.0);

        let ocr = engine.ocr_diagram(&dummy_img).unwrap();
        assert_eq!(ocr, "α-D-Glcp-(1->4)-D-Glcp");
    }

    #[test]
    fn test_dummy_vlm_engine_detect() {
        let engine = DummyVlmEngine::new();
        let dummy_img = DynamicImage::new_rgb8(100, 100);
        let boxes = engine
            .detect_diagrams(&dummy_img)
            .expect("detect_diagrams should succeed");
        assert!(!boxes.is_empty());
        assert_eq!(boxes[0].label, Some("SNFG".to_string()));
    }

    #[test]
    fn test_dummy_vlm_engine_ocr() {
        let engine = DummyVlmEngine::new();
        let dummy_img = DynamicImage::new_rgb8(100, 100);
        let ocr = engine
            .ocr_diagram(&dummy_img)
            .expect("ocr_diagram should succeed");
        assert_eq!(ocr, "α-D-Glcp-(1->4)-D-Glcp");
    }

    #[test]
    fn test_dummy_vlm_engine_custom() {
        let custom_box = BoundingBox::with_label(50.0, 50.0, 200.0, 200.0, "SNFG_Custom");
        let engine = DummyVlmEngine::with_mock_data(vec![custom_box.clone()], "Man5");
        let dummy_img = DynamicImage::new_rgb8(50, 50);

        let boxes = engine.detect_diagrams(&dummy_img).unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0], custom_box);

        let ocr = engine.ocr_diagram(&dummy_img).unwrap();
        assert_eq!(ocr, "Man5");
    }
}
