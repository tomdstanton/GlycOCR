use crate::error::GlycOCRError;
use crate::model::engine::VlmEngine;
use crate::pipeline::crop::crop_and_pad_bbox;
use crate::pipeline::pdf::extract_pdf_pages;
use crate::types::{DetectedDiagram, DocumentScanResult, PageResult};
use std::path::Path;

pub struct PipelineRunner<'a, E: VlmEngine> {
    engine: &'a E,
}

impl<'a, E: VlmEngine> PipelineRunner<'a, E> {
    pub fn new(engine: &'a E) -> Self {
        Self { engine }
    }

    pub fn run_pdf(&self, pdf_path: &Path) -> Result<DocumentScanResult, GlycOCRError> {
        let pages = extract_pdf_pages(pdf_path)?;
        let mut page_results = Vec::new();

        for (idx, page_img) in pages.iter().enumerate() {
            let bboxes = self.engine.detect_diagrams(page_img)?;
            let mut detected_diagrams = Vec::new();

            for bbox in bboxes {
                let crop = crop_and_pad_bbox(page_img, &bbox, 0.1)?;
                let iupac = self.engine.ocr_diagram(&crop)?;
                detected_diagrams.push(DetectedDiagram {
                    bbox,
                    cropped_path: None,
                    iupac,
                    confidence: 0.95,
                });
            }

            page_results.push(PageResult {
                page_number: idx + 1,
                diagrams: detected_diagrams,
            });
        }

        Ok(DocumentScanResult {
            pdf_path: pdf_path.to_string_lossy().to_string(),
            total_pages: pages.len(),
            pages: page_results,
        })
    }

    pub fn run_image(&self, image_path: &Path) -> Result<DocumentScanResult, GlycOCRError> {
        if !image_path.exists() {
            return Err(GlycOCRError::ImageError(format!(
                "Image file not found: {}",
                image_path.display()
            )));
        }

        let img = image::open(image_path)
            .map_err(|e| GlycOCRError::ImageError(format!("Failed to open image: {}", e)))?;

        let bboxes = self.engine.detect_diagrams(&img)?;
        let mut detected_diagrams = Vec::new();

        for bbox in bboxes {
            let crop = crop_and_pad_bbox(&img, &bbox, 0.1)?;
            let iupac = self.engine.ocr_diagram(&crop)?;
            detected_diagrams.push(DetectedDiagram {
                bbox,
                cropped_path: Some(image_path.to_string_lossy().to_string()),
                iupac,
                confidence: 0.95,
            });
        }

        Ok(DocumentScanResult {
            pdf_path: image_path.to_string_lossy().to_string(),
            total_pages: 1,
            pages: vec![PageResult {
                page_number: 1,
                diagrams: detected_diagrams,
            }],
        })
    }
}
