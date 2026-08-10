use serde::{Deserialize, Serialize};

/// Bounding box representation in normalized coordinate space (0..1000 or 0.0..1.0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Minimum y coordinate (top boundary)
    pub ymin: f32,
    /// Minimum x coordinate (left boundary)
    pub xmin: f32,
    /// Maximum y coordinate (bottom boundary)
    pub ymax: f32,
    /// Maximum x coordinate (right boundary)
    pub xmax: f32,
    /// Optional label or category for the bounding box region
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl BoundingBox {
    /// Creates a new BoundingBox without a label.
    pub fn new(ymin: f32, xmin: f32, ymax: f32, xmax: f32) -> Self {
        Self {
            ymin,
            xmin,
            ymax,
            xmax,
            label: None,
        }
    }

    /// Creates a new BoundingBox with a label.
    pub fn with_label(
        ymin: f32,
        xmin: f32,
        ymax: f32,
        xmax: f32,
        label: impl Into<String>,
    ) -> Self {
        Self {
            ymin,
            xmin,
            ymax,
            xmax,
            label: Some(label.into()),
        }
    }

    /// Calculates the width of the bounding box.
    pub fn width(&self) -> f32 {
        (self.xmax - self.xmin).max(0.0)
    }

    /// Calculates the height of the bounding box.
    pub fn height(&self) -> f32 {
        (self.ymax - self.ymin).max(0.0)
    }

    /// Calculates the surface area of the bounding box.
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Checks if the bounding box coordinates are geometrically valid (ymin <= ymax and xmin <= xmax).
    pub fn is_valid(&self) -> bool {
        self.ymin <= self.ymax && self.xmin <= self.xmax
    }

    /// Expands the bounding box outward by a given padding ratio relative to its width and height.
    /// Clamps normalized values to [0.0, 1000.0] or [0.0, 1.0].
    pub fn expand(&self, padding_ratio: f32) -> Self {
        let w = self.width();
        let h = self.height();
        let pad_x = w * padding_ratio;
        let pad_y = h * padding_ratio;
        let max_val = if self.ymax > 1.0 || self.xmax > 1.0 {
            1000.0
        } else {
            1.0
        };

        Self {
            ymin: (self.ymin - pad_y).max(0.0),
            xmin: (self.xmin - pad_x).max(0.0),
            ymax: (self.ymax + pad_y).min(max_val),
            xmax: (self.xmax + pad_x).min(max_val),
            label: self.label.clone(),
        }
    }

    /// Converts normalized coordinates to absolute image pixel coordinates `(x, y, crop_width, crop_height)`.
    pub fn to_pixel_coords(&self, img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
        let img_w_f = img_w as f32;
        let img_h_f = img_h as f32;

        let (norm_ymin, norm_xmin, norm_ymax, norm_xmax) = if self.ymax > 1.0 || self.xmax > 1.0 {
            (
                self.ymin / 1000.0,
                self.xmin / 1000.0,
                self.ymax / 1000.0,
                self.xmax / 1000.0,
            )
        } else {
            (self.ymin, self.xmin, self.ymax, self.xmax)
        };

        let px_xmin = (norm_xmin * img_w_f).clamp(0.0, img_w_f) as u32;
        let px_ymin = (norm_ymin * img_h_f).clamp(0.0, img_h_f) as u32;
        let px_xmax = (norm_xmax * img_w_f).clamp(0.0, img_w_f) as u32;
        let px_ymax = (norm_ymax * img_h_f).clamp(0.0, img_h_f) as u32;

        let crop_w = px_xmax.saturating_sub(px_xmin).max(1);
        let crop_h = px_ymax.saturating_sub(px_ymin).max(1);

        (px_xmin, px_ymin, crop_w, crop_h)
    }
}

/// Represents a detected diagram with its bounding box, cropped file path, predicted IUPAC sequence, and confidence score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedDiagram {
    /// Bounding box location of the diagram
    pub bbox: BoundingBox,
    /// Optional file path where the cropped image is saved
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cropped_path: Option<String>,
    /// Recognized IUPAC glycan string
    pub iupac: String,
    /// Model detection and OCR confidence score
    pub confidence: f32,
}

impl DetectedDiagram {
    /// Creates a new DetectedDiagram instance.
    pub fn new(bbox: BoundingBox, iupac: impl Into<String>, confidence: f32) -> Self {
        Self {
            bbox,
            cropped_path: None,
            iupac: iupac.into(),
            confidence,
        }
    }

    /// Sets the cropped image file path for the diagram.
    pub fn with_cropped_path(mut self, path: impl Into<String>) -> Self {
        self.cropped_path = Some(path.into());
        self
    }
}

/// Results aggregated for a single page of a PDF document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageResult {
    /// 1-based page number within the PDF document
    pub page_number: usize,
    /// Vector of detected diagrams found on this page
    pub diagrams: Vec<DetectedDiagram>,
}

impl PageResult {
    /// Creates a new PageResult instance.
    pub fn new(page_number: usize, diagrams: Vec<DetectedDiagram>) -> Self {
        Self {
            page_number,
            diagrams,
        }
    }

    /// Returns the number of diagrams detected on this page.
    pub fn diagram_count(&self) -> usize {
        self.diagrams.len()
    }

    /// Returns true if no diagrams were detected on this page.
    pub fn is_empty(&self) -> bool {
        self.diagrams.is_empty()
    }
}

/// Document-wide aggregation of all scan results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentScanResult {
    /// Path to the analyzed PDF document
    pub pdf_path: String,
    /// Total number of pages processed in the document
    pub total_pages: usize,
    /// Vector of page-level results
    pub pages: Vec<PageResult>,
}

impl DocumentScanResult {
    /// Creates a new DocumentScanResult instance.
    pub fn new(pdf_path: impl Into<String>, total_pages: usize, pages: Vec<PageResult>) -> Self {
        Self {
            pdf_path: pdf_path.into(),
            total_pages,
            pages,
        }
    }

    /// Calculates total count of detected diagrams across all pages.
    pub fn total_diagrams(&self) -> usize {
        self.pages.iter().map(|p| p.diagram_count()).sum()
    }

    /// Serializes the scan result to a pretty-formatted JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a DocumentScanResult from a JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_methods() {
        let bbox = BoundingBox::new(100.0, 50.0, 300.0, 250.0);
        assert_eq!(bbox.width(), 200.0);
        assert_eq!(bbox.height(), 200.0);
        assert_eq!(bbox.area(), 40000.0);
        assert!(bbox.is_valid());

        let invalid_bbox = BoundingBox::new(300.0, 250.0, 100.0, 50.0);
        assert!(!invalid_bbox.is_valid());

        // Test expand (10% padding on 200 width and height = 20px padding)
        let expanded = bbox.expand(0.10);
        assert_eq!(expanded.ymin, 80.0);
        assert_eq!(expanded.xmin, 30.0);
        assert_eq!(expanded.ymax, 320.0);
        assert_eq!(expanded.xmax, 270.0);

        // Test to_pixel_coords with 1000x1000 image dimensions
        let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
        assert_eq!((x, y, w, h), (50, 100, 200, 200));

        // Test to_pixel_coords with relative [0.0, 1.0] scale
        let rel_bbox = BoundingBox::new(0.10, 0.05, 0.30, 0.25);
        let (rx, ry, rw, rh) = rel_bbox.to_pixel_coords(1000, 1000);
        assert_eq!((rx, ry, rw, rh), (50, 100, 200, 200));
    }

    #[test]
    fn test_bounding_box_label() {
        let bbox = BoundingBox::with_label(10.0, 20.0, 30.0, 40.0, "SNFG_diagram");
        assert_eq!(bbox.label, Some("SNFG_diagram".to_string()));
    }

    #[test]
    fn test_detected_diagram_and_page_result() {
        let bbox = BoundingBox::new(100.0, 50.0, 300.0, 250.0);
        let diagram = DetectedDiagram::new(bbox.clone(), "α-D-Glcp-(1->4)-D-Glcp", 0.95)
            .with_cropped_path("/tmp/crop1.png");

        assert_eq!(diagram.cropped_path, Some("/tmp/crop1.png".to_string()));
        assert_eq!(diagram.iupac, "α-D-Glcp-(1->4)-D-Glcp");
        assert_eq!(diagram.confidence, 0.95);

        let page = PageResult::new(1, vec![diagram.clone()]);
        assert_eq!(page.page_number, 1);
        assert_eq!(page.diagram_count(), 1);
        assert!(!page.is_empty());

        let empty_page = PageResult::new(2, vec![]);
        assert_eq!(empty_page.diagram_count(), 0);
        assert!(empty_page.is_empty());
    }

    #[test]
    fn test_document_scan_result_json_roundtrip() {
        let bbox = BoundingBox::with_label(100.0, 50.0, 300.0, 250.0, "SNFG");
        let diagram = DetectedDiagram::new(bbox, "α-D-Glcp-(1->4)-D-Glcp", 0.98)
            .with_cropped_path("crops/page1_diag1.png");
        let page1 = PageResult::new(1, vec![diagram]);
        let page2 = PageResult::new(2, vec![]);
        let scan_result = DocumentScanResult::new("test_document.pdf", 2, vec![page1, page2]);

        assert_eq!(scan_result.total_diagrams(), 1);

        let json_str = scan_result.to_json().expect("Serialization failed");
        assert!(json_str.contains("test_document.pdf"));
        assert!(json_str.contains("SNFG"));

        let restored = DocumentScanResult::from_json(&json_str).expect("Deserialization failed");
        assert_eq!(scan_result, restored);
    }

    #[test]
    fn test_option_none_serde_omission() {
        let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
        let diagram = DetectedDiagram::new(bbox, "Man5", 0.90);
        let json = serde_json::to_string(&diagram).expect("Failed serde");

        // label and cropped_path should be omitted from JSON when None
        assert!(!json.contains("\"label\""));
        assert!(!json.contains("\"cropped_path\""));

        let deserialized: DetectedDiagram = serde_json::from_str(&json).expect("Deserialization");
        assert_eq!(diagram, deserialized);
    }
}
