use candle_core::{Device, Tensor};
use image::DynamicImage;

use crate::error::GlycOCRError;

/// Logs a verbose status message to stdout if `verbose` is true.
pub fn log_verbose(msg: &str, verbose: bool) {
    if verbose {
        println!("[GlycOCR] {}", msg);
    }
}

/// Automatically selects the best available compute device (CUDA -> Metal -> CPU).
pub fn select_device_auto() -> Result<Device, GlycOCRError> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = Device::new_cuda(0) {
            return Ok(dev);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(dev) = Device::new_metal(0) {
            return Ok(dev);
        }
    }

    Ok(Device::Cpu)
}

/// Selects and initializes a Candle compute device based on the provided string ("cpu", "metal", "cuda", "gpu", "auto").
/// Case-insensitive. Unknown device strings return error.
pub fn select_device(device_str: &str) -> Result<Device, GlycOCRError> {
    let normalized = device_str.trim().to_lowercase();
    match normalized.as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(target_os = "macos")]
            {
                if let Ok(dev) = Device::new_metal(0) {
                    return Ok(dev);
                }
            }
            Ok(Device::Cpu)
        }
        "cuda" | "gpu" => {
            #[cfg(feature = "cuda")]
            {
                if let Ok(dev) = Device::new_cuda(0) {
                    return Ok(dev);
                }
            }
            Ok(Device::Cpu)
        }
        "auto" => select_device_auto(),
        other => Err(GlycOCRError::ModelError(format!(
            "Unsupported device '{}'. Expected one of: cpu, metal, cuda",
            other
        ))),
    }
}

/// Converts a `DynamicImage` into an NCHW float tensor normalized to [0.0, 1.0].
/// Target shape is `(1, 3, target_height, target_width)`.
pub fn image_to_tensor(
    img: &DynamicImage,
    target_width: u32,
    target_height: u32,
    device: &Device,
) -> Result<Tensor, GlycOCRError> {
    let resized = img.resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();
    let (width, height) = rgb.dimensions();
    let raw = rgb.as_raw();
    let total_pixels = (width * height) as usize;

    let mut data = Vec::with_capacity(total_pixels * 3);
    // Red channel
    for i in 0..total_pixels {
        data.push(raw[i * 3] as f32 / 255.0);
    }
    // Green channel
    for i in 0..total_pixels {
        data.push(raw[i * 3 + 1] as f32 / 255.0);
    }
    // Blue channel
    for i in 0..total_pixels {
        data.push(raw[i * 3 + 2] as f32 / 255.0);
    }

    Tensor::from_vec(data, (1, 3, height as usize, width as usize), device)
        .map_err(|e| GlycOCRError::ModelError(format!("Failed to create tensor: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_device_cpu() {
        let dev = select_device("cpu").expect("CPU selection should succeed");
        assert!(matches!(dev, Device::Cpu));
    }

    #[test]
    fn test_select_device_auto() {
        let dev = select_device_auto().expect("Auto selection should succeed");
        let dev_str = select_device("auto").expect("Auto string selection should succeed");
        assert_eq!(format!("{:?}", dev), format!("{:?}", dev_str));
    }

    #[test]
    fn test_select_device_unsupported() {
        let err = select_device("tpu").unwrap_err();
        assert!(matches!(err, GlycOCRError::ModelError(_)));
        assert!(err.to_string().contains("Unsupported device 'tpu'"));

        let dev_upper =
            select_device("CPU").expect("Case-insensitive CPU selection should succeed");
        assert!(matches!(dev_upper, Device::Cpu));
    }

    #[test]
    fn test_image_to_tensor_shape_and_range() {
        let img = DynamicImage::new_rgb8(100, 100);
        let dev = Device::Cpu;
        let tensor = image_to_tensor(&img, 448, 448, &dev).expect("Tensor conversion failed");

        assert_eq!(tensor.dims(), &[1, 3, 448, 448]);
    }
}
