extern crate glycocr_rs as glycocr;

use glycocr::{BoundingBox, DocumentScanResult};

#[test]
fn test_degenerate_bbox_ymin_greater_than_ymax() {
    let bbox = BoundingBox::new(300.0, 50.0, 100.0, 200.0);
    assert!(
        !bbox.is_valid(),
        "BoundingBox with ymin > ymax must be invalid"
    );
    assert_eq!(
        bbox.height(),
        0.0,
        "height() must clamp to 0.0 when ymin > ymax"
    );
    assert_eq!(bbox.width(), 150.0);
    assert_eq!(bbox.area(), 0.0);

    let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
    assert_eq!((x, y, w, h), (50, 300, 150, 1));
}

#[test]
fn test_degenerate_bbox_xmin_greater_than_xmax() {
    let bbox = BoundingBox::new(100.0, 500.0, 300.0, 200.0);
    assert!(
        !bbox.is_valid(),
        "BoundingBox with xmin > xmax must be invalid"
    );
    assert_eq!(
        bbox.width(),
        0.0,
        "width() must clamp to 0.0 when xmin > xmax"
    );
    assert_eq!(bbox.height(), 200.0);
    assert_eq!(bbox.area(), 0.0);

    let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
    assert_eq!((x, y, w, h), (500, 100, 1, 200));
}

#[test]
fn test_negative_coordinates_in_bbox() {
    let bbox = BoundingBox::new(-100.0, -50.0, 200.0, 300.0);
    assert!(bbox.is_valid(), "ymin <= ymax is true for -100 <= 200");
    assert_eq!(bbox.height(), 300.0);
    assert_eq!(bbox.width(), 350.0);
    assert_eq!(bbox.area(), 105000.0);

    let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
    assert_eq!(x, 0, "px_xmin should clamp to 0");
    assert_eq!(y, 0, "px_ymin should clamp to 0");
    assert_eq!(w, 300, "crop width should be px_xmax(300) - px_xmin(0)");
    assert_eq!(h, 200, "crop height should be px_ymax(200) - px_ymin(0)");
}

#[test]
fn test_all_negative_bbox_coordinates() {
    let bbox = BoundingBox::new(-200.0, -100.0, -50.0, -10.0);
    assert!(bbox.is_valid());
    let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
    assert_eq!((x, y, w, h), (0, 0, 1, 1));
}

#[test]
fn test_extreme_overflow_bbox_coordinates() {
    let bbox = BoundingBox::new(0.0, 0.0, f32::MAX, f32::MAX);
    assert!(bbox.is_valid());
    assert_eq!(bbox.width(), f32::MAX);
    assert_eq!(bbox.height(), f32::MAX);
    assert!(
        bbox.area().is_infinite(),
        "Area should overflow to infinity"
    );

    let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
    assert_eq!(
        (x, y, w, h),
        (0, 0, 1000, 1000),
        "Pixel conversion should clamp to max image dimensions"
    );
}

#[test]
fn test_zero_width_and_height_bbox() {
    let bbox = BoundingBox::new(100.0, 50.0, 100.0, 50.0);
    assert!(bbox.is_valid());
    assert_eq!(bbox.width(), 0.0);
    assert_eq!(bbox.height(), 0.0);
    assert_eq!(bbox.area(), 0.0);

    let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
    assert_eq!(
        (x, y, w, h),
        (50, 100, 1, 1),
        "Zero size box should default to min 1x1 crop size"
    );
}

#[test]
fn test_nan_bbox_coordinates() {
    let bbox = BoundingBox::new(f32::NAN, 0.0, 100.0, 100.0);
    assert!(!bbox.is_valid(), "NaN coordinate bbox must be invalid");
    let (_x, y, w, _h) = bbox.to_pixel_coords(1000, 1000);
    assert_eq!(y, 0);
    assert_eq!(w, 100);
}

#[test]
fn test_expand_with_negative_padding_ratio() {
    let bbox = BoundingBox::new(100.0, 50.0, 300.0, 250.0);
    let shrunk = bbox.expand(-0.10);
    assert_eq!(shrunk.ymin, 120.0);
    assert_eq!(shrunk.xmin, 70.0);
    assert_eq!(shrunk.ymax, 280.0);
    assert_eq!(shrunk.xmax, 230.0);
    assert!(shrunk.is_valid());

    let invalid_shrunk = bbox.expand(-0.60);
    assert_eq!(invalid_shrunk.ymin, 220.0);
    assert_eq!(invalid_shrunk.ymax, 180.0);
    assert!(
        !invalid_shrunk.is_valid(),
        "Expanding with -0.60 padding ratio produces inverted invalid box (ymin > ymax)"
    );
}

#[test]
fn test_expand_with_zero_padding_ratio() {
    let bbox = BoundingBox::new(100.0, 50.0, 300.0, 250.0);
    let expanded = bbox.expand(0.0);
    assert_eq!(bbox, expanded);
}

#[test]
fn test_expand_with_extreme_padding_ratio() {
    let bbox = BoundingBox::new(100.0, 50.0, 300.0, 250.0);
    let extreme = bbox.expand(100.0);
    assert_eq!(extreme.ymin, 0.0);
    assert_eq!(extreme.xmin, 0.0);
    assert_eq!(extreme.ymax, 1000.0);
    assert_eq!(extreme.xmax, 1000.0);
}

#[test]
fn test_expand_scale_ambiguity_overshoot_bug() {
    let bbox = BoundingBox::new(0.1, 0.1, 1.05, 0.9);
    let expanded = bbox.expand(0.10);
    assert_eq!(expanded.ymax, 1.145);

    let (x, y, w, h) = expanded.to_pixel_coords(1000, 1000);
    assert_eq!(
        (x, y, w, h),
        (0, 0, 1, 1),
        "0..1 box with ymax=1.05 gets misclassified as 0..1000 and collapses to (0,0,1,1)"
    );
}

#[test]
fn test_to_pixel_coords_top_left_0_to_1000_box_misclassified() {
    // Small box in 0..1000 scale located at top-left corner
    let bbox = BoundingBox::new(0.2, 0.1, 0.9, 0.8); // 0..1000 scale pixel values
    let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
    // Because ymax <= 1.0 and xmax <= 1.0, it gets treated as 0..1 scale and multiplied by 1000!
    assert_eq!(
        (x, y, w, h),
        (100, 200, 700, 700),
        "0..1000 box with ymax<=1.0 gets misclassified as 0..1 scale and multiplied by 1000"
    );
}

#[test]
fn test_to_pixel_coords_zero_image_dimensions() {
    let bbox = BoundingBox::new(100.0, 50.0, 300.0, 250.0);
    let (x, y, w, h) = bbox.to_pixel_coords(0, 0);
    assert_eq!((x, y, w, h), (0, 0, 1, 1));
}

#[test]
fn test_to_pixel_coords_out_of_bounds_input() {
    let bbox = BoundingBox::new(500.0, 500.0, 1500.0, 1500.0);
    let (x, y, w, h) = bbox.to_pixel_coords(1000, 1000);
    assert_eq!((x, y, w, h), (500, 500, 500, 500));

    let bbox_oob = BoundingBox::new(1200.0, 1200.0, 1500.0, 1500.0);
    let (_x, y, _w, h) = bbox_oob.to_pixel_coords(1000, 1000);
    assert_eq!(y, 1000);
    assert_eq!(h, 1);
    assert!(
        y + h > 1000,
        "Out of bounds crop coordinate: y + h = 1001 > 1000"
    );
}

#[test]
fn test_json_serde_missing_required_fields() {
    let json_missing_pages = r#"{"pdf_path": "doc.pdf", "pages": []}"#;
    let res1 = DocumentScanResult::from_json(json_missing_pages);
    assert!(res1.is_err());

    let json_missing_diagrams =
        r#"{"pdf_path": "doc.pdf", "total_pages": 1, "pages": [{"page_number": 1}]}"#;
    let res2 = DocumentScanResult::from_json(json_missing_diagrams);
    assert!(res2.is_err());

    let json_missing_bbox = r#"{"pdf_path": "doc.pdf", "total_pages": 1, "pages": [{"page_number": 1, "diagrams": [{"iupac": "Glc", "confidence": 0.9}]}]}"#;
    let res3 = DocumentScanResult::from_json(json_missing_bbox);
    assert!(res3.is_err());
}

#[test]
fn test_json_serde_optional_fields_omitted() {
    let json_optional_omitted = r#"{
        "pdf_path": "test.pdf",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "diagrams": [
                    {
                        "bbox": {"ymin": 10.0, "xmin": 20.0, "ymax": 30.0, "xmax": 40.0},
                        "iupac": "Man5",
                        "confidence": 0.95
                    }
                ]
            }
        ]
    }"#;
    let result = DocumentScanResult::from_json(json_optional_omitted)
        .expect("Optional fields omitted should deserialize cleanly");
    assert_eq!(result.pages[0].diagrams[0].bbox.label, None);
    assert_eq!(result.pages[0].diagrams[0].cropped_path, None);
}

#[test]
fn test_json_serde_invalid_type_errors() {
    let json_invalid_type = r#"{
        "pdf_path": "test.pdf",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "diagrams": [
                    {
                        "bbox": {"ymin": 10.0, "xmin": 20.0, "ymax": 30.0, "xmax": 40.0},
                        "iupac": "Man5",
                        "confidence": "high"
                    }
                ]
            }
        ]
    }"#;
    let result = DocumentScanResult::from_json(json_invalid_type);
    assert!(result.is_err());
}

#[test]
fn test_json_serde_empty_document_scan_result() {
    let empty_result = DocumentScanResult::new("empty.pdf", 0, vec![]);
    assert_eq!(empty_result.total_diagrams(), 0);
    let json = empty_result.to_json().unwrap();
    let restored = DocumentScanResult::from_json(&json).unwrap();
    assert_eq!(empty_result, restored);
}
