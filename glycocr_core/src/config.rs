use serde::{Deserialize, Serialize};

/// Pipeline configuration options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Compute device target (e.g., "cpu", "metal", "cuda")
    pub device: String,
    /// HuggingFace model repository or local ID for diagram detection
    pub detection_model_id: String,
    /// HuggingFace model repository or local ID for diagram OCR
    pub ocr_model_id: String,
    /// Bounding box expansion padding ratio (e.g. 0.10 for 10% padding)
    pub padding_ratio: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: "cpu".to_string(),
            detection_model_id: "google/paligemma-3b-pt-224".to_string(),
            ocr_model_id: "google/paligemma-3b-pt-224".to_string(),
            padding_ratio: 0.10,
        }
    }
}

impl Config {
    /// Creates a new `Config` instance with default values.
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.device, "cpu");
        assert_eq!(config.padding_ratio, 0.10);
        assert_eq!(config.detection_model_id, "google/paligemma-3b-pt-224");
        assert_eq!(config.ocr_model_id, "google/paligemma-3b-pt-224");

        let config_new = Config::new();
        assert_eq!(config, config_new);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = Config::default();
        let json = serde_json::to_string(&config).expect("Serialization failed");
        let restored: Config = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(config, restored);
    }
}
