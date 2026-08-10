//! Empirical Stress, Performance, and Concurrency Test Suite for `PipelineRunner`
//!
//! Written by challenger_m4_2 to stress-test:
//! 1. `PipelineRunner` throughput and latency performance over 1000+ repeat invocations.
//! 2. Multi-threaded concurrency and thread safety (`Send` / `Sync`).
//! 3. Repeated invocations, state isolation, and error recovery under stress.
//! 4. High bounding-box density, high-resolution image handling, and JSON serialization.

extern crate glycocr_rs as glycocr;

use glycocr::error::GlycOCRError;
use glycocr::model::dummy::DummyVlmEngine;
use glycocr::model::paligemma::CandlePaliGemmaEngine;
use glycocr::pipeline::runner::PipelineRunner;
use glycocr::run_cli_from;
use glycocr::types::{BoundingBox, DocumentScanResult};
use image::{DynamicImage, Rgb, RgbImage};
use lopdf::{Document, Object, Stream, dictionary};
use std::time::Instant;
use tempfile::NamedTempFile;

// Helper to create a synthetic valid PDF file
fn create_synthetic_pdf(pages: usize) -> NamedTempFile {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();

    for _ in 0..pages {
        let content_id = doc.add_object(Stream::new(
            dictionary! {},
            b"q 0 0 200 200 re f Q".to_vec(),
        ));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
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

    let temp_file = tempfile::Builder::new()
        .suffix(".pdf")
        .tempfile()
        .expect("Failed to create temp pdf file");
    doc.save(temp_file.path())
        .expect("Failed to save synthetic pdf");
    temp_file
}

// Helper to create a synthetic PNG image file
fn create_synthetic_image(width: u32, height: u32) -> (NamedTempFile, DynamicImage) {
    let mut imgbuf = RgbImage::new(width, height);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let r = (x % 256) as u8;
        let g = (y % 256) as u8;
        let b = ((x + y) % 256) as u8;
        *pixel = Rgb([r, g, b]);
    }
    let dyn_img = DynamicImage::ImageRgb8(imgbuf);
    let temp_file = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .expect("Failed to create temp image file");
    dyn_img
        .save(temp_file.path())
        .expect("Failed to save synthetic png");
    (temp_file, dyn_img)
}

#[test]
fn test_pipeline_runner_trait_bounds_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DummyVlmEngine>();
    assert_send_sync::<CandlePaliGemmaEngine>();
    assert_send_sync::<PipelineRunner<'static, DummyVlmEngine>>();
    assert_send_sync::<PipelineRunner<'static, CandlePaliGemmaEngine>>();
}

#[test]
fn test_pipeline_runner_performance_image_stress() {
    let (img_file, _dyn_img) = create_synthetic_image(400, 400);
    let engine = DummyVlmEngine::new();
    let runner = PipelineRunner::new(&engine);

    let iterations = 1000;
    let start = Instant::now();

    for i in 0..iterations {
        let res = runner
            .run_image(img_file.path())
            .expect("run_image should succeed");
        assert_eq!(res.total_pages, 1);
        assert_eq!(res.pages.len(), 1);
        assert_eq!(res.pages[0].diagrams.len(), 1);
        assert_eq!(res.pages[0].diagrams[0].iupac, "α-D-Glcp-(1->4)-D-Glcp");
        if i % 250 == 0 {
            assert_eq!(res.pdf_path, img_file.path().to_string_lossy());
        }
    }

    let elapsed = start.elapsed();
    let ops_per_sec = (iterations as f64) / elapsed.as_secs_f64();
    println!(
        "\n[PERFORMANCE] Completed {} image pipeline runs in {:.2?} ({:.1} ops/sec)",
        iterations, elapsed, ops_per_sec
    );
    assert!(
        ops_per_sec > 10.0,
        "Pipeline image execution is too slow (< 10 ops/sec)"
    );
}

#[test]
fn test_pipeline_runner_performance_pdf_stress() {
    let pdf_file = create_synthetic_pdf(3);
    let engine = DummyVlmEngine::new();
    let runner = PipelineRunner::new(&engine);

    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let res = runner
            .run_pdf(pdf_file.path())
            .expect("run_pdf should succeed");
        assert_eq!(res.total_pages, 3);
        assert_eq!(res.pages.len(), 3);
        for page in &res.pages {
            assert_eq!(page.diagrams.len(), 1);
            assert_eq!(page.diagrams[0].confidence, 0.95);
        }
    }

    let elapsed = start.elapsed();
    let pages_processed = (iterations * 3) as f64;
    let pages_per_sec = pages_processed / elapsed.as_secs_f64();
    println!(
        "[PERFORMANCE] Completed {} PDF runs ({} pages) in {:.2?} ({:.1} pages/sec)",
        iterations, pages_processed, elapsed, pages_per_sec
    );
    assert!(
        pages_per_sec > 5.0,
        "Pipeline PDF execution is too slow (< 5 pages/sec)"
    );
}

#[test]
fn test_pipeline_runner_thread_safety_shared_runner() {
    let (img_file, _) = create_synthetic_image(300, 300);
    let pdf_file = create_synthetic_pdf(2);

    let img_path = img_file.path();
    let pdf_path = pdf_file.path();

    let engine = DummyVlmEngine::new();
    let runner = PipelineRunner::new(&engine);

    let thread_count = 32;
    let ops_per_thread = 50;

    let start = Instant::now();

    std::thread::scope(|s| {
        for _ in 0..thread_count {
            s.spawn(|| {
                for _ in 0..ops_per_thread {
                    let img_res = runner.run_image(img_path).unwrap();
                    assert_eq!(img_res.total_pages, 1);
                    assert_eq!(img_res.pages[0].diagrams.len(), 1);

                    let pdf_res = runner.run_pdf(pdf_path).unwrap();
                    assert_eq!(pdf_res.total_pages, 2);
                    assert_eq!(pdf_res.pages.len(), 2);
                }
            });
        }
    });

    let elapsed = start.elapsed();
    let total_ops = thread_count * ops_per_thread * 2;
    println!(
        "[CONCURRENCY] Shared PipelineRunner across {} threads completed {} total ops in {:.2?}",
        thread_count, total_ops, elapsed
    );
}

#[test]
fn test_pipeline_runner_thread_safety_shared_candlepali_engine() {
    let (img_file, _) = create_synthetic_image(200, 200);
    let img_path = img_file.path();

    let engine = CandlePaliGemmaEngine::new("cpu", None).expect("Engine creation failed");

    let thread_count = 16;
    let ops_per_thread = 20;

    let start = Instant::now();

    std::thread::scope(|s| {
        for _ in 0..thread_count {
            s.spawn(|| {
                let runner = PipelineRunner::new(&engine);
                for _ in 0..ops_per_thread {
                    let res = runner.run_image(img_path).unwrap();
                    assert_eq!(res.total_pages, 1);
                    assert_eq!(res.pages[0].diagrams.len(), 1);
                    assert_eq!(
                        res.pages[0].diagrams[0].iupac,
                        "α-D-Galp-(1->3)-β-D-GlcpNAc"
                    );
                }
            });
        }
    });

    let elapsed = start.elapsed();
    let total_ops = thread_count * ops_per_thread;
    println!(
        "[CONCURRENCY] Shared CandlePaliGemmaEngine across {} threads completed {} ops in {:.2?}",
        thread_count, total_ops, elapsed
    );
}

#[test]
fn test_pipeline_runner_high_diagram_density_stress() {
    let bboxes: Vec<BoundingBox> = (0..50)
        .map(|i| BoundingBox {
            ymin: (i as f32 * 10.0) % 800.0,
            xmin: (i as f32 * 10.0) % 800.0,
            ymax: ((i as f32 * 10.0) % 800.0) + 50.0,
            xmax: ((i as f32 * 10.0) % 800.0) + 50.0,
            label: Some(format!("SNFG_{}", i)),
        })
        .collect();

    let engine = DummyVlmEngine::with_mock_data(bboxes, "Neu5Ac");
    let runner = PipelineRunner::new(&engine);

    let (img_file, _) = create_synthetic_image(1000, 1000);
    let res = runner
        .run_image(img_file.path())
        .expect("High density run_image failed");

    assert_eq!(res.total_pages, 1);
    assert_eq!(res.pages[0].diagrams.len(), 50);
    for (idx, diagram) in res.pages[0].diagrams.iter().enumerate() {
        assert_eq!(diagram.iupac, "Neu5Ac");
        assert_eq!(diagram.bbox.label, Some(format!("SNFG_{}", idx)));
    }
}

#[test]
fn test_pipeline_runner_high_resolution_image_stress() {
    let (img_file, _) = create_synthetic_image(3000, 3000);
    let engine = DummyVlmEngine::new();
    let runner = PipelineRunner::new(&engine);

    let start = Instant::now();
    let res = runner
        .run_image(img_file.path())
        .expect("High res run_image failed");
    let elapsed = start.elapsed();

    assert_eq!(res.total_pages, 1);
    assert_eq!(res.pages[0].diagrams.len(), 1);
    println!(
        "[STRESS] 3000x3000 high-res image pipeline run completed in {:.2?}",
        elapsed
    );
}

#[test]
fn test_pipeline_runner_repeated_invocations_interleaved_and_error_recovery() {
    let (img_file, _) = create_synthetic_image(250, 250);
    let pdf_file = create_synthetic_pdf(1);
    let engine = DummyVlmEngine::new();
    let runner = PipelineRunner::new(&engine);

    let non_existent_img = tempfile::Builder::new()
        .tempdir()
        .unwrap()
        .path()
        .join("missing.png");
    let non_existent_pdf = tempfile::Builder::new()
        .tempdir()
        .unwrap()
        .path()
        .join("missing.pdf");

    for _ in 0..100 {
        let img_res = runner.run_image(img_file.path()).unwrap();
        assert_eq!(img_res.total_pages, 1);

        let err_img = runner.run_image(&non_existent_img).unwrap_err();
        assert!(matches!(err_img, GlycOCRError::ImageError(_)));

        let pdf_res = runner.run_pdf(pdf_file.path()).unwrap();
        assert_eq!(pdf_res.total_pages, 1);

        let err_pdf = runner.run_pdf(&non_existent_pdf).unwrap_err();
        assert!(matches!(
            err_pdf,
            GlycOCRError::PdfError(_) | GlycOCRError::IoError(_)
        ));
    }
}

#[test]
fn test_pipeline_runner_json_serialization_stress() {
    let (img_file, _) = create_synthetic_image(300, 300);
    let engine = DummyVlmEngine::new();
    let runner = PipelineRunner::new(&engine);

    let scan_result = runner.run_image(img_file.path()).unwrap();

    let json_str = serde_json::to_string_pretty(&scan_result).expect("JSON serialization failed");
    assert!(json_str.contains("\"total_pages\": 1"));
    assert!(json_str.contains("\"iupac\": \"α-D-Glcp-(1->4)-D-Glcp\""));
    assert!(json_str.contains("\"confidence\": 0.95"));

    let deserialized: DocumentScanResult =
        serde_json::from_str(&json_str).expect("JSON deserialization failed");
    assert_eq!(deserialized, scan_result);
}

#[test]
fn test_cli_in_process_concurrency_stress() {
    let (img_file, _) = create_synthetic_image(200, 200);
    let img_path = img_file.path().to_string_lossy().to_string();

    let thread_count = 16;
    let ops_per_thread = 10;

    std::thread::scope(|s| {
        for _ in 0..thread_count {
            let path_ref = &img_path;
            s.spawn(move || {
                for _ in 0..ops_per_thread {
                    let args = vec!["glycocr", "infer", "--dummy", "--image", path_ref, "--json"];
                    let res = run_cli_from(args);
                    assert!(res.is_ok(), "run_cli_from failed in concurrent thread");
                }
            });
        }
    });
}
