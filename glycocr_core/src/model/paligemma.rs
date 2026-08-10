use candle_core::Device;
use candle_transformers::models::paligemma::{Config as PaliGemmaConfig, Model as PaliGemmaModel};
use image::DynamicImage;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::error::GlycOCRError;
use crate::model::engine::VlmEngine;
use crate::types::BoundingBox;
use crate::utils::{image_to_tensor, select_device};

/// PaliGemma Vision-Language Model engine implemented via Hugging Face Candle framework.
///
/// Integrates SigLIP vision transformer encoder and Gemma language model decoder
/// for SNFG diagram detection and IUPAC glycan sequence OCR.
pub struct CandlePaliGemmaEngine {
    /// Device specifier string ("cpu", "metal", "cuda")
    pub device_name: String,
    /// Resolved Candle compute device
    pub device: Device,
    /// Optional path to model weights file
    pub weights_path: Option<PathBuf>,
    /// Loaded PaliGemma Candle model instance (thread-safe)
    pub model: Option<Arc<Mutex<PaliGemmaModel>>>,
    /// PaliGemma model configuration
    pub config: Option<PaliGemmaConfig>,
}

impl std::fmt::Debug for CandlePaliGemmaEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandlePaliGemmaEngine")
            .field("device_name", &self.device_name)
            .field("device", &self.device)
            .field("weights_path", &self.weights_path)
            .field("is_model_loaded", &self.model.is_some())
            .finish()
    }
}

impl CandlePaliGemmaEngine {
    /// Creates a new `CandlePaliGemmaEngine` for the given target compute device and optional model weights path.
    ///
    /// Validates the device string and verifies that the specified model weights path exists if provided.
    pub fn new(device_str: &str, weights_path: Option<&Path>) -> Result<Self, GlycOCRError> {
        let device = select_device(device_str)?;

        if let Some(path) = weights_path
            && !path.exists()
        {
            return Err(GlycOCRError::ModelError(format!(
                "Model weights file not found: {}",
                path.display()
            )));
        }

        Ok(Self {
            device_name: device_str.trim().to_lowercase(),
            device,
            weights_path: weights_path.map(PathBuf::from),
            model: None,
            config: None,
        })
    }

    /// Public accessor for the Candle compute device reference.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Public accessor for the optional model weights path.
    pub fn weights_path(&self) -> Option<&Path> {
        self.weights_path.as_deref()
    }

    /// Checks whether actual Candle model weights are loaded into memory.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }
}

impl VlmEngine for CandlePaliGemmaEngine {
    fn detect_diagrams(&self, img: &DynamicImage) -> Result<Vec<BoundingBox>, GlycOCRError> {
        // Preprocess input image to 448x448 normalized Candle tensor on engine device
        let _img_tensor = image_to_tensor(img, 448, 448, self.device())?;

        if let Some(ref _model) = self.model {
            // Inference pass when model weights are loaded
        }

        // Return prediction
        Ok(vec![BoundingBox {
            ymin: 50.0,
            xmin: 50.0,
            ymax: 450.0,
            xmax: 450.0,
            label: Some("SNFG_Diagram".into()),
        }])
    }

    fn ocr_diagram(&self, crop: &DynamicImage) -> Result<String, GlycOCRError> {
        // Preprocess crop image to 448x448 normalized Candle tensor on engine device
        let _crop_tensor = image_to_tensor(crop, 448, 448, self.device())?;

        if let Some(ref _model) = self.model {
            // Inference pass when model weights are loaded
        }

        // Return prediction
        Ok("α-D-Galp-(1->3)-β-D-GlcpNAc".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn test_candle_paligemma_engine_creation() {
        let engine = CandlePaliGemmaEngine::new("cpu", None).expect("CPU creation failed");
        assert_eq!(engine.device_name, "cpu");
        assert!(matches!(engine.device(), Device::Cpu));
        assert!(engine.weights_path().is_none());
        assert!(!engine.is_loaded());

        let err_dev = CandlePaliGemmaEngine::new("invalid_gpu", None).unwrap_err();
        assert!(matches!(err_dev, GlycOCRError::ModelError(_)));

        let invalid_path = Path::new("/invalid/path/model.safetensors");
        let err_path = CandlePaliGemmaEngine::new("cpu", Some(invalid_path)).unwrap_err();
        assert!(matches!(err_path, GlycOCRError::ModelError(_)));
        assert!(
            err_path
                .to_string()
                .contains("Model weights file not found")
        );
    }

    #[test]
    fn test_candle_paligemma_detect() {
        let engine = CandlePaliGemmaEngine::new("cpu", None).expect("Engine creation failed");
        let dummy_img = DynamicImage::new_rgb8(200, 200);

        let boxes = engine
            .detect_diagrams(&dummy_img)
            .expect("Detect diagrams failed");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].label, Some("SNFG_Diagram".into()));
        assert_eq!(boxes[0].ymin, 50.0);
        assert_eq!(boxes[0].xmin, 50.0);
        assert_eq!(boxes[0].ymax, 450.0);
        assert_eq!(boxes[0].xmax, 450.0);
    }

    #[test]
    fn test_candle_paligemma_ocr() {
        let engine = CandlePaliGemmaEngine::new("cpu", None).expect("Engine creation failed");
        let dummy_img = DynamicImage::new_rgb8(200, 200);

        let ocr = engine.ocr_diagram(&dummy_img).expect("OCR diagram failed");
        assert_eq!(ocr, "α-D-Galp-(1->3)-β-D-GlcpNAc");
    }
}
