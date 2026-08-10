use crate::error::GlycOCRError;
use crate::model::dummy::DummyVlmEngine;
use crate::model::paligemma::CandlePaliGemmaEngine;
use crate::pipeline::runner::PipelineRunner;
use clap::{Args, Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "glycocr",
    version = "0.1.0",
    about = "GlycOCR SNFG Diagram Detection & OCR Pipeline"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Predict SNFG diagrams and OCR IUPAC strings from input PDF or Image
    Infer(InferArgs),
}

#[derive(Args, Debug)]
pub struct InferArgs {
    /// Path to input PDF document
    #[arg(short, long)]
    pub pdf: Option<PathBuf>,

    /// Path to input diagram image file
    #[arg(short, long)]
    pub image: Option<PathBuf>,

    /// Use fallback offline DummyVlmEngine
    #[arg(long)]
    pub dummy: bool,

    /// Computing device selection (auto, cpu, metal, cuda, gpu)
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Path to custom PaliGemma model weights
    #[arg(long)]
    pub model_path: Option<String>,

    /// Output destination file for JSON results
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output formatted JSON to stdout or file
    #[arg(long)]
    pub json: bool,

    /// Enable verbose logging output
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run_cli() -> Result<(), String> {
    let args = std::env::args_os();
    run_cli_from(args)
}

pub fn run_cli_from<I, T>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(parsed) => parsed,
        Err(e) => return Err(e.to_string()),
    };

    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            return Err(
                "No subcommand provided. Use 'glycocr infer --help' for usage.".to_string(),
            );
        }
    };

    match command {
        Commands::Infer(infer_args) => run_infer(infer_args).map_err(|e| e.to_string()),
    }
}

fn run_infer(args: InferArgs) -> Result<(), GlycOCRError> {
    if args.pdf.is_none() && args.image.is_none() {
        return Err(GlycOCRError::CliError(
            "Either --pdf or --image input flag must be provided.".to_string(),
        ));
    }

    if args.pdf.is_some() && args.image.is_some() {
        return Err(GlycOCRError::CliError(
            "Cannot specify both --pdf and --image simultaneously.".to_string(),
        ));
    }

    let dev_lower = args.device.trim().to_lowercase();
    let valid_devices = ["auto", "cpu", "metal", "cuda", "gpu"];
    if !valid_devices.contains(&dev_lower.as_str()) {
        return Err(GlycOCRError::CliError(format!(
            "Unsupported device '{}'. Valid options: auto, cpu, metal, cuda, gpu",
            args.device
        )));
    }

    let scan_result = if args.dummy {
        let dummy_engine = DummyVlmEngine::new();
        let runner = PipelineRunner::new(&dummy_engine);
        if let Some(ref pdf_path) = args.pdf {
            runner.run_pdf(pdf_path)?
        } else if let Some(ref img_path) = args.image {
            runner.run_image(img_path)?
        } else {
            unreachable!()
        }
    } else {
        let engine = CandlePaliGemmaEngine::new(
            &dev_lower,
            args.model_path.as_deref().map(std::path::Path::new),
        )?;
        let runner = PipelineRunner::new(&engine);
        if let Some(ref pdf_path) = args.pdf {
            runner.run_pdf(pdf_path)?
        } else if let Some(ref img_path) = args.image {
            runner.run_image(img_path)?
        } else {
            unreachable!()
        }
    };

    let json_str = serde_json::to_string_pretty(&scan_result)
        .map_err(|e| GlycOCRError::CliError(format!("JSON serialization failed: {}", e)))?;

    if let Some(ref out_path) = args.output {
        if let Some(parent) = out_path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            return Err(GlycOCRError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Output directory does not exist: {}", parent.display()),
            )));
        }
        fs::write(out_path, &json_str)?;
    } else if args.json {
        println!("{}", json_str);
    } else {
        println!(
            "Scanned document '{}': Total pages={}, Total diagrams detected={}",
            scan_result.pdf_path,
            scan_result.total_pages,
            scan_result
                .pages
                .iter()
                .map(|p| p.diagrams.len())
                .sum::<usize>()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_normalization_and_validation() {
        let valid_devices = vec![
            "auto", "AUTO", "  auto  ", "cpu", "CPU", "metal", "cuda", "gpu", "GPU",
        ];
        for dev in valid_devices {
            let res = Cli::try_parse_from([
                "glycocr",
                "infer",
                "--dummy",
                "--pdf",
                "sample.pdf",
                "--device",
                dev,
            ]);
            assert!(res.is_ok(), "Device '{}' parsing should succeed", dev);
        }
    }
}
