extern crate glycocr_rs as glycocr;

use glycocr::error::GlycOCRError;
use glycocr::extract_pdf_pages;
use image::GenericImageView;
use lopdf::{Document, Object, Stream, dictionary};
use std::fs;
use std::path::Path;
use tempfile::{NamedTempFile, tempdir};

fn create_synthetic_pdf(pages: usize) -> NamedTempFile {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();

    for _ in 0..pages {
        let content_id = doc.add_object(Stream::new(
            dictionary! {},
            b"q 1 0 0 1 0 0 cm /Im1 Do Q".to_vec(),
        ));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        page_ids.push(Object::Reference(page_id));
    }

    let pages_obj = dictionary! {
        "Type" => "Pages",
        "Count" => pages as i64,
        "Kids" => page_ids,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_obj));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", Object::Reference(catalog_id));

    let temp_file = NamedTempFile::new().expect("Failed to create temp pdf");
    doc.save(temp_file.path()).expect("Failed to save temp pdf");
    temp_file
}

#[test]
fn test_challenger_pdf_nonexistent_file() {
    let path = Path::new("/nonexistent_directory_abc123/nonexistent_file_xyz.pdf");
    let result = extract_pdf_pages(path);
    assert!(result.is_err(), "Expected error for nonexistent file");
    if let Err(GlycOCRError::PdfError(msg)) = result {
        assert!(
            msg.contains("PDF file not found"),
            "Error msg should contain 'PDF file not found': {}",
            msg
        );
    } else {
        panic!("Expected GlycOCRError::PdfError, got {:?}", result);
    }
}

#[test]
fn test_challenger_pdf_zero_byte_file() {
    let temp_file = NamedTempFile::new().unwrap();
    let result = extract_pdf_pages(temp_file.path());
    assert!(result.is_err(), "Expected error for zero byte file");
    if let Err(GlycOCRError::PdfError(msg)) = result {
        assert!(
            msg.contains("0 bytes"),
            "Error msg should mention 0 bytes: {}",
            msg
        );
    } else {
        panic!("Expected GlycOCRError::PdfError, got {:?}", result);
    }
}

#[test]
fn test_challenger_pdf_invalid_header() {
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), b"THIS_IS_NOT_A_VALID_PDF_HEADER_DATA").unwrap();
    let result = extract_pdf_pages(temp_file.path());
    assert!(result.is_err(), "Expected error for invalid header");
    if let Err(GlycOCRError::PdfError(msg)) = result {
        assert!(
            msg.contains("missing %PDF- header"),
            "Error msg should mention missing header: {}",
            msg
        );
    } else {
        panic!("Expected GlycOCRError::PdfError, got {:?}", result);
    }
}

#[test]
fn test_challenger_pdf_corrupt_payload_with_valid_header() {
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(
        temp_file.path(),
        b"%PDF-1.7\nCorrupted binary payload with no catalog or trailer",
    )
    .unwrap();
    let result = extract_pdf_pages(temp_file.path());
    assert!(result.is_err(), "Expected error for corrupt PDF body");
    if let Err(GlycOCRError::PdfError(msg)) = result {
        assert!(
            msg.contains("Failed to parse PDF"),
            "Error msg should contain 'Failed to parse PDF': {}",
            msg
        );
    } else {
        panic!("Expected GlycOCRError::PdfError, got {:?}", result);
    }
}

#[test]
fn test_challenger_pdf_directory_path() {
    let dir = tempdir().unwrap();
    let result = extract_pdf_pages(dir.path());
    assert!(
        result.is_err(),
        "Expected error when passing directory path"
    );
}

#[test]
fn test_challenger_pdf_valid_single_page_rendering() {
    let pdf_file = create_synthetic_pdf(1);
    let result = extract_pdf_pages(pdf_file.path());
    assert!(
        result.is_ok(),
        "Single-page PDF should parse successfully: {:?}",
        result.err()
    );
    let pages = result.unwrap();
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].dimensions(), (448, 448));
}

#[test]
fn test_challenger_pdf_valid_multipage_rendering() {
    let pdf_file = create_synthetic_pdf(5);
    let result = extract_pdf_pages(pdf_file.path());
    assert!(
        result.is_ok(),
        "Multi-page PDF (5 pages) should parse successfully"
    );
    let pages = result.unwrap();
    assert_eq!(pages.len(), 5);
    for (i, page) in pages.iter().enumerate() {
        assert_eq!(
            page.dimensions(),
            (448, 448),
            "Page {} should have dimensions 448x448",
            i + 1
        );
    }
}

#[test]
fn test_challenger_pdf_unicode_path_name() {
    let dir = tempdir().unwrap();
    let unicode_file_path = dir.path().join("glycan_αβ_123_🔬.pdf");
    let synthetic_pdf = create_synthetic_pdf(2);
    fs::copy(synthetic_pdf.path(), &unicode_file_path).unwrap();

    let result = extract_pdf_pages(&unicode_file_path);
    assert!(
        result.is_ok(),
        "Unicode path name should be handled correctly"
    );
    let pages = result.unwrap();
    assert_eq!(pages.len(), 2);
}

#[test]
fn test_challenger_pdf_large_page_count() {
    let pdf_file = create_synthetic_pdf(20);
    let result = extract_pdf_pages(pdf_file.path());
    assert!(result.is_ok(), "20-page PDF should parse successfully");
    let pages = result.unwrap();
    assert_eq!(pages.len(), 20);
}

#[test]
fn test_challenger_pdf_zero_pages_in_document() {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let pages_obj = dictionary! {
        "Type" => "Pages",
        "Count" => 0i64,
        "Kids" => Vec::<Object>::new(),
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_obj));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", Object::Reference(catalog_id));

    let temp_file = NamedTempFile::new().unwrap();
    doc.save(temp_file.path()).unwrap();

    let result = extract_pdf_pages(temp_file.path());
    assert!(result.is_err(), "PDF with 0 pages should return error");
    if let Err(GlycOCRError::PdfError(msg)) = result {
        assert!(
            msg.contains("0 pages"),
            "Error msg should mention 0 pages: {}",
            msg
        );
    } else {
        panic!("Expected GlycOCRError::PdfError, got {:?}", result);
    }
}

#[test]
fn test_challenger_pdf_real_fixture_file() {
    let real_pdf_path = Path::new(
        "tests/data/whitfield-et-al-2025-o-antigen-polysaccharides-in-klebsiella-pneumoniae-structures-and-molecular-basis-for-antigenic.pdf",
    );
    if real_pdf_path.exists() {
        let result = extract_pdf_pages(real_pdf_path);
        assert!(
            result.is_ok(),
            "Real PDF fixture should extract successfully: {:?}",
            result.err()
        );
        let pages = result.unwrap();
        assert!(
            !pages.is_empty(),
            "Real PDF fixture should contain at least 1 page"
        );
        for page in &pages {
            assert_eq!(page.dimensions(), (448, 448));
        }
    }
}
