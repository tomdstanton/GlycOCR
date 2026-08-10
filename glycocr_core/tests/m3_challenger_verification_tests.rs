extern crate glycocr_rs as glycocr;

use glycocr::cli::run_cli_from;
use glycocr::model::dummy::DummyVlmEngine;
use glycocr::model::engine::VlmEngine;
use glycocr::model::paligemma::CandlePaliGemmaEngine;
use image::{DynamicImage, RgbImage};
use lopdf::{Document, Object, Stream, dictionary};
use tempfile::NamedTempFile;

fn create_synthetic_pdf() -> NamedTempFile {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
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
    let pages_obj = dictionary! {
        "Type" => "Pages",
        "Count" => 1,
        "Kids" => vec![Object::Reference(page_id)],
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

#[test]
fn test_cli_devices_exact() {
    let pdf = create_synthetic_pdf();
    let pdf_path = pdf.path().to_str().unwrap();

    let res_cpu = run_cli_from([
        "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "cpu",
    ]);
    assert!(res_cpu.is_ok(), "CPU device failed: {:?}", res_cpu);

    let res_metal = run_cli_from([
        "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "metal",
    ]);
    assert!(res_metal.is_ok(), "Metal device failed: {:?}", res_metal);

    let res_cuda = run_cli_from([
        "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "cuda",
    ]);
    assert!(res_cuda.is_ok(), "CUDA device failed: {:?}", res_cuda);

    let res_auto = run_cli_from([
        "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "auto",
    ]);
    assert!(res_auto.is_ok(), "Auto device failed: {:?}", res_auto);
}

#[test]
fn test_cli_devices_case_and_aliases() {
    let pdf = create_synthetic_pdf();
    let pdf_path = pdf.path().to_str().unwrap();

    let res_upper_cpu = run_cli_from([
        "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "CPU",
    ]);
    assert!(
        res_upper_cpu.is_ok(),
        "Uppercase CPU device failed: {:?}",
        res_upper_cpu
    );

    let res_gpu = run_cli_from([
        "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", "gpu",
    ]);
    assert!(res_gpu.is_ok(), "GPU alias failed: {:?}", res_gpu);
}

#[test]
fn test_cli_candle_engine_with_auto() {
    let pdf = create_synthetic_pdf();
    let pdf_path = pdf.path().to_str().unwrap();

    // CLI with CandlePaliGemmaEngine (without --dummy) and --device auto
    let res = run_cli_from(["glycocr", "infer", "--pdf", pdf_path, "--device", "auto"]);
    assert!(
        res.is_ok(),
        "CLI without --dummy using --device auto failed: {:?}",
        res
    );
}

#[test]
fn test_vlm_engine_trait_implementations() {
    let dummy = DummyVlmEngine::new();
    let pali = CandlePaliGemmaEngine::new("cpu", None).unwrap();

    let engines: Vec<Box<dyn VlmEngine>> = vec![Box::new(dummy), Box::new(pali)];

    let img = DynamicImage::ImageRgb8(RgbImage::new(200, 200));

    for engine in engines {
        let bboxes = engine
            .detect_diagrams(&img)
            .expect("detect_diagrams failed");
        assert!(!bboxes.is_empty());
        let ocr = engine.ocr_diagram(&img).expect("ocr_diagram failed");
        assert!(!ocr.is_empty());
    }
}

#[test]
fn test_all_12_required_device_flags() {
    let pdf = create_synthetic_pdf();
    let pdf_path = pdf.path().to_str().unwrap();

    let valid_flags = [
        "auto", "AUTO", "Auto", "cpu", "CPU", "metal", "Metal", "cuda", "CUDA", "gpu", "GPU",
    ];

    for flag in valid_flags {
        // Test with --dummy
        let res_dummy = run_cli_from([
            "glycocr", "infer", "--dummy", "--pdf", pdf_path, "--device", flag,
        ]);
        assert!(
            res_dummy.is_ok(),
            "Device flag '{}' with --dummy failed: {:?}",
            flag,
            res_dummy
        );

        // Test without --dummy (CandlePaliGemmaEngine)
        let res_real = run_cli_from(["glycocr", "infer", "--pdf", pdf_path, "--device", flag]);
        assert!(
            res_real.is_ok(),
            "Device flag '{}' with CandlePaliGemmaEngine failed: {:?}",
            flag,
            res_real
        );
    }

    // Test invalid device flag
    let res_invalid = run_cli_from([
        "glycocr",
        "infer",
        "--dummy",
        "--pdf",
        pdf_path,
        "--device",
        "invalid_dev",
    ]);
    assert!(
        res_invalid.is_err(),
        "Invalid device flag 'invalid_dev' should fail"
    );
    let err_msg = res_invalid.unwrap_err();
    assert!(
        err_msg.contains("Unsupported device 'invalid_dev'"),
        "Expected error message to contain 'Unsupported device \'invalid_dev\'', got: {}",
        err_msg
    );
}
