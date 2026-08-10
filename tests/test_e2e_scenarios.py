"""End-to-end integration scenario tests for GlycOCR user workflows."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from typer.testing import CliRunner

from glycocr.cli import app
from glycocr.data.dataset import GlycOCRDataset
from glycocr.inference.predictor import GlycOCR
from glycocr.models.model import GlycOCRModel
from glycocr.models.parser import GlycanParseResult
from glycocr.training.trainer import GlycOCRTrainer

runner = CliRunner()


def test_e2e_pdf_to_json_scan_workflow(sample_pdf_path: Path, mock_glycocr_rs) -> None:
    """E2E Scenario 1: Scientific PDF ingestion -> Rust scan_pdf execution -> valid JSON output."""
    result = runner.invoke(app, ["infer", "--pdf", str(sample_pdf_path), "--device", "cpu", "--json"])
    assert result.exit_code == 0

    scan_data = json.loads(result.stdout.strip())
    assert "pages" in scan_data
    assert len(scan_data["pages"]) > 0
    diagrams = scan_data["pages"][0]["diagrams"]
    assert len(diagrams) > 0
    assert diagrams[0]["iupac"] == "Gal(b1-4)Glc"


def test_e2e_image_to_biocuration_workflow(sample_image_path: Path, mock_hf_processor, mock_hf_model) -> None:
    """E2E Scenario 2: SNFG diagram image -> Predictor prediction -> IUPACParser NetworkX graph generation."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.processor = mock_hf_processor
    model_obj.model = mock_hf_model

    mock_hf_model.generate.return_value = torch.tensor([[10, 11, 12, 13, 100, 101, 102]])
    mock_hf_processor.batch_decode.return_value = ["Gal(b1-4)Glc"]

    predictor = GlycOCR(model=model_obj)
    parse_result = predictor.predict(sample_image_path)

    assert isinstance(parse_result, GlycanParseResult)
    assert parse_result.is_valid is True
    assert parse_result.iupac == "Gal(b1-4)Glc"
    assert parse_result.graph is not None


def test_e2e_synthesis_to_training_workflow(tmp_path: Path, mock_hf_processor, mock_hf_model) -> None:
    """E2E Scenario 3: Synthesis -> binary dataset -> GlycOCRDataset -> Trainer loop -> Checkpoint."""
    iupac_file = tmp_path / "iupac_input.txt"
    iupac_file.write_text("Gal(b1-4)Glc\nMan(a1-3)Man\n", encoding="utf-8")
    dataset_dir = tmp_path / "binary_soa_dataset"
    output_dir = tmp_path / "model_checkpoint"

    mock_synth = MagicMock()
    mock_synth.synthesize.return_value = torch.zeros((3, 64, 64), dtype=torch.uint8)

    with patch("glycocr.data.synthesizer.IUPACSynthesizer", return_value=mock_synth):
        synth_res = runner.invoke(app, ["prep", "synthesize", str(iupac_file), str(dataset_dir)])
        assert synth_res.exit_code == 0

    assert (dataset_dir / "images.bin").exists()
    assert (dataset_dir / "strings.bin").exists()
    assert (dataset_dir / "index.npz").exists()

    # Load dataset from synthesized binary directory
    dataset = GlycOCRDataset(data_dir=dataset_dir, processor=mock_hf_processor, max_length=16, degrade_prob=0.0)
    assert len(dataset) == 2

    # Initialize model and trainer
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.processor = mock_hf_processor
    model_obj.model = mock_hf_model
    model_obj.save_pretrained = MagicMock()

    trainer = GlycOCRTrainer(
        model=model_obj,
        train_dataset=dataset,
        output_dir=str(output_dir),
        num_train_epochs=1,
    )

    mock_hf_trainer_instance = MagicMock()
    mock_hf_trainer_instance.train.return_value = "trained"

    with patch("glycocr.training.trainer._GlycOCRHFTrainer", return_value=mock_hf_trainer_instance):
        res = trainer.train()
        assert res == "trained"


def test_e2e_cli_error_recovery_workflow(tmp_path: Path) -> None:
    """E2E Scenario 4: Error recovery and validation checks across CLI subcommands."""
    # 1. Invalid infer invocation
    res_infer = runner.invoke(app, ["infer", "--pdf", "nonexistent.pdf", "--image", "nonexistent.png"])
    assert res_infer.exit_code != 0
    assert "Must specify exactly one of --pdf or --image" in res_infer.stdout

    # 2. Deploy missing arguments
    res_deploy = runner.invoke(app, ["deploy"])
    assert res_deploy.exit_code != 0

    # 3. Train on non-existent dataset dir
    res_train = runner.invoke(app, ["train", str(tmp_path / "non_existent_dir"), str(tmp_path / "out")])
    assert res_train.exit_code != 0 or "Error" in res_train.stdout or "FileNotFoundError" in str(res_train.exception)
