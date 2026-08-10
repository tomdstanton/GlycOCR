use crate::error::GlycOCRError;
use crate::types::BoundingBox;
use candle_core::{Device, Tensor};
use image::{DynamicImage, GenericImageView};

/// Crops a region of interest from an image based on a `BoundingBox` with padding expansion.
///
/// Coordinates are automatically denormalized from 0..1000 or 0..1 relative scale to pixel dimensions.
/// Expands width and height outward by `padding_ratio` (e.g. 0.10 for 10% expansion).
/// Enforces strict bounds clamping so `crop_imm` never panics on edge or out-of-bounds coordinates.
pub fn crop_and_pad_bbox(
    img: &DynamicImage,
    bbox: &BoundingBox,
    padding_ratio: f32,
) -> Result<DynamicImage, GlycOCRError> {
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return Err(GlycOCRError::ImageError(
            "Cannot crop an image with 0 width or height".into(),
        ));
    }

    if !bbox.is_valid()
        || bbox.ymin.is_nan()
        || bbox.xmin.is_nan()
        || bbox.ymax.is_nan()
        || bbox.xmax.is_nan()
    {
        return Err(GlycOCRError::ImageError(
            "Invalid bounding box geometry".into(),
        ));
    }

    let expanded = bbox.expand(padding_ratio);
    let (px_x, px_y, px_w, px_h) = expanded.to_pixel_coords(width, height);

    let crop_x = px_x.min(width.saturating_sub(1));
    let crop_y = px_y.min(height.saturating_sub(1));
    let max_avail_w = width - crop_x;
    let max_avail_h = height - crop_y;
    let crop_w = px_w.clamp(1, max_avail_w);
    let crop_h = px_h.clamp(1, max_avail_h);

    let cropped = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
    Ok(cropped)
}

/// Preprocesses a cropped diagram image into a normalized 448x448 Candle tensor on CPU.
pub fn preprocess_image_to_tensor(img: &DynamicImage) -> Result<Tensor, GlycOCRError> {
    preprocess_image_to_tensor_with_device(img, &Device::Cpu)
}

/// Preprocesses a cropped diagram image into a normalized 448x448 Candle tensor `[1, 3, 448, 448]` on the specified device.
/// Normalizes pixel values to SigLIP standard range `[-1.0, 1.0]` via `(pixel/255.0 - 0.5) / 0.5`.
pub fn preprocess_image_to_tensor_with_device(
    img: &DynamicImage,
    device: &Device,
) -> Result<Tensor, GlycOCRError> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(GlycOCRError::ImageError(
            "Cannot preprocess an empty image with 0 width or height".into(),
        ));
    }

    let resized = img.resize_exact(448, 448, image::imageops::FilterType::Triangle);
    let rgb_img = resized.to_rgb8();
    let raw_pixels = rgb_img.into_raw();

    let tensor_data: Vec<f32> = raw_pixels
        .iter()
        .map(|&p| (p as f32 / 255.0 - 0.5) / 0.5)
        .collect();

    let tensor = Tensor::from_vec(tensor_data, (448, 448, 3), device)
        .map_err(|e| GlycOCRError::ModelError(format!("Failed to create tensor: {e}")))?
        .permute((2, 0, 1))
        .map_err(|e| GlycOCRError::ModelError(format!("Failed to permute tensor layout: {e}")))?
        .unsqueeze(0)
        .map_err(|e| {
            GlycOCRError::ModelError(format!("Failed to unsqueeze tensor batch dimension: {e}"))
        })?;

    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_crop_and_pad_normal_bbox() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(1000, 1000));
        let bbox = BoundingBox {
            ymin: 100.0,
            xmin: 100.0,
            ymax: 400.0,
            xmax: 400.0,
            label: None,
        };
        let crop = crop_and_pad_bbox(&img, &bbox, 0.0).unwrap();
        assert_eq!(crop.dimensions(), (300, 300));
    }

    #[test]
    fn test_crop_and_pad_10_percent_padding() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(1000, 1000));
        let bbox = BoundingBox {
            ymin: 200.0,
            xmin: 200.0,
            ymax: 800.0,
            xmax: 800.0,
            label: None,
        };
        let crop = crop_and_pad_bbox(&img, &bbox, 0.1).unwrap();
        assert_eq!(crop.dimensions(), (720, 720));
    }

    #[test]
    fn test_crop_and_pad_clamping_at_edges() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(500, 500));
        let bbox = BoundingBox {
            ymin: -200.0,
            xmin: -200.0,
            ymax: 1200.0,
            xmax: 1200.0,
            label: None,
        };
        let crop = crop_and_pad_bbox(&img, &bbox, 0.1).unwrap();
        assert_eq!(crop.dimensions(), (500, 500));
    }

    #[test]
    fn test_crop_and_pad_zero_width_height_bbox() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(500, 500));
        let bbox = BoundingBox {
            ymin: 250.0,
            xmin: 250.0,
            ymax: 250.0,
            xmax: 250.0,
            label: None,
        };
        let crop = crop_and_pad_bbox(&img, &bbox, 0.0).unwrap();
        assert!(crop.width() >= 1 && crop.height() >= 1);
    }

    #[test]
    fn test_crop_and_pad_zero_size_image() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(0, 500));
        let bbox = BoundingBox {
            ymin: 0.0,
            xmin: 0.0,
            ymax: 100.0,
            xmax: 100.0,
            label: None,
        };
        let res = crop_and_pad_bbox(&img, &bbox, 0.0);
        assert!(res.is_err());
        match res.unwrap_err() {
            GlycOCRError::ImageError(msg) => assert!(msg.contains("0 width or height")),
            err => panic!("Unexpected error variant: {:?}", err),
        }
    }

    #[test]
    fn test_crop_and_pad_invalid_bbox() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(500, 500));
        let bbox = BoundingBox {
            ymin: 400.0,
            xmin: 100.0,
            ymax: 200.0,
            xmax: 300.0,
            label: None,
        };
        let res = crop_and_pad_bbox(&img, &bbox, 0.0);
        assert!(res.is_err());
        match res.unwrap_err() {
            GlycOCRError::ImageError(msg) => assert!(msg.contains("Invalid bounding box")),
            err => panic!("Unexpected error variant: {:?}", err),
        }
    }

    #[test]
    fn test_preprocess_image_to_tensor_shape() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(200, 300));
        let tensor = preprocess_image_to_tensor(&img).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 448, 448]);
    }

    #[test]
    fn test_preprocess_image_to_tensor_channel_bounds() {
        let mut img = RgbImage::new(100, 100);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([255, 255, 255]);
        }
        let dynamic_img = DynamicImage::ImageRgb8(img);
        let tensor = preprocess_image_to_tensor(&dynamic_img).unwrap();
        let vec_data = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(vec_data.len(), 3 * 448 * 448);
        for &val in vec_data.iter() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "Expected normalized white pixel 1.0, got {}",
                val
            );
        }

        let mut black_img = RgbImage::new(100, 100);
        for pixel in black_img.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }
        let dynamic_black_img = DynamicImage::ImageRgb8(black_img);
        let black_tensor = preprocess_image_to_tensor(&dynamic_black_img).unwrap();
        let vec_black = black_tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for &val in vec_black.iter() {
            assert!(
                (val - (-1.0)).abs() < 1e-5,
                "Expected normalized black pixel -1.0, got {}",
                val
            );
        }
    }

    #[test]
    fn test_preprocess_image_to_tensor_device() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(50, 50));
        let tensor = preprocess_image_to_tensor_with_device(&img, &Device::Cpu).unwrap();
        assert!(matches!(tensor.device(), Device::Cpu));
    }

    #[test]
    fn test_preprocess_image_to_tensor_zero_size() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(0, 100));
        let res = preprocess_image_to_tensor(&img);
        assert!(res.is_err());
        match res.unwrap_err() {
            GlycOCRError::ImageError(msg) => assert!(msg.contains("0 width or height")),
            err => panic!("Unexpected error variant: {:?}", err),
        }
    }
}
