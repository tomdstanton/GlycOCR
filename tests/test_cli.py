"""Unit tests for GlycOCR Typer CLI infer, train, prep, and deploy commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from typer.testing import CliRunner

from glycocr.cli import app

runner = CliRunner()


def test_infer_help() -> None:
    """Test infer --help outputs expected options."""
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0
    assert "--pdf" in result.stdout
    assert "--image" in result.stdout
    assert "--dummy" in result.stdout
    assert "--device" in result.stdout
    assert "--model-path" in result.stdout or "-m" in result.stdout
    assert "--output" in result.stdout or "-o" in result.stdout
    assert "--json" in result.stdout


def test_infer_no_args_fails() -> None:
    """Test calling infer with neither --pdf nor --image fails with exit code 1."""
    result = runner.invoke(app, ["infer"])
    assert result.exit_code != 0
    assert "Must specify exactly one of --pdf or --image" in result.stdout


def test_infer_both_pdf_and_image_fails() -> None:
    """Test calling infer with both --pdf and --image fails with exit code 1."""
    result = runner.invoke(app, ["infer", "--pdf", "sample.pdf", "--image", "sample.png"])
    assert result.exit_code != 0
    assert "Must specify exactly one of --pdf or --image" in result.stdout


def test_infer_pdf_dummy_stdout(tmp_path: Path) -> None:
    """Test infer with --pdf and --dummy flag."""
    pdf_file = tmp_path / "test_doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    result = runner.invoke(app, ["infer", "--pdf", str(pdf_file), "--dummy"])
    assert result.exit_code == 0
    assert "pdf_path" in result.stdout
    assert "Gal(b1-4)Glc" in result.stdout


def test_infer_image_dummy_stdout(tmp_path: Path) -> None:
    """Test infer with --image and --dummy flag."""
    img_file = tmp_path / "test_img.png"
    img_file.write_bytes(b"dummy image content")

    result = runner.invoke(app, ["infer", "--image", str(img_file), "--dummy"])
    assert result.exit_code == 0
    assert "pdf_path" in result.stdout
    assert "Gal(b1-4)Glc" in result.stdout


def test_infer_json_flag(tmp_path: Path) -> None:
    """Test infer with --json outputs raw JSON string to stdout."""
    img_file = tmp_path / "test_img.png"
    img_file.write_bytes(b"dummy image content")

    result = runner.invoke(app, ["infer", "--image", str(img_file), "--dummy", "--json"])
    assert result.exit_code == 0

    data = json.loads(result.stdout.strip())
    assert isinstance(data, dict)
    assert data["dummy"] is True


def test_infer_output_file(tmp_path: Path) -> None:
    """Test infer saving JSON result to output file."""
    img_file = tmp_path / "test_img.png"
    img_file.write_bytes(b"dummy image content")
    out_file = tmp_path / "output.json"

    result = runner.invoke(app, ["infer", "--image", str(img_file), "--dummy", "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    data = json.loads(content)
    assert data["dummy"] is True
    assert "Output saved to:" in result.stdout


def test_infer_mocked_glycocr_rs_pdf(tmp_path: Path) -> None:
    """Test infer delegates to glycocr_rs.scan_pdf when available."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"pdf content")

    mock_rs = MagicMock()
    mock_rs.scan_pdf.return_value = json.dumps({"pdf_path": str(pdf_file), "pages": []})

    with patch("glycocr.cli.glycocr_rs", mock_rs):
        result = runner.invoke(
            app,
            ["infer", "--pdf", str(pdf_file), "--device", "cuda", "--model-path", "/tmp/model"],
        )
        assert result.exit_code == 0
        mock_rs.scan_pdf.assert_called_once_with(str(pdf_file), device="cuda", dummy=False, model_path="/tmp/model")


def test_infer_mocked_glycocr_rs_image(tmp_path: Path) -> None:
    """Test infer delegates to glycocr_rs.scan_image when available."""
    img_file = tmp_path / "sample.png"
    img_file.write_bytes(b"image content")

    mock_rs = MagicMock()
    mock_rs.scan_image.return_value = json.dumps({"pdf_path": str(img_file), "pages": []})

    with patch("glycocr.cli.glycocr_rs", mock_rs):
        result = runner.invoke(
            app,
            ["infer", "--image", str(img_file), "--device", "mps"],
        )
        assert result.exit_code == 0
        mock_rs.scan_image.assert_called_once_with(str(img_file), device="mps", dummy=False, model_path=None)


def test_infer_mocked_glycocr_rs_error(tmp_path: Path) -> None:
    """Test infer handles error raised by glycocr_rs gracefully."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"pdf content")

    mock_rs = MagicMock()
    mock_rs.scan_pdf.side_effect = RuntimeError("Rust engine failed")

    with patch("glycocr.cli.glycocr_rs", mock_rs):
        result = runner.invoke(app, ["infer", "--pdf", str(pdf_file)])
        assert result.exit_code != 0
        assert "Inference error:" in result.stdout


def test_cli_version() -> None:
    """Test glycocr --version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "glycocr" in result.stdout


def test_cli_train_command(mock_binary_dataset_dir, tmp_path: Path) -> None:
    """Test glycocr train command execution with mocked trainer."""
    out_dir = tmp_path / "model_out"

    with (
        patch("glycocr.models.model.GlycOCRModel"),
        patch("glycocr.data.dataset.GlycOCRDataset"),
        patch("glycocr.training.trainer.GlycOCRTrainer") as mock_trainer_cls,
    ):
        mock_trainer_inst = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_inst

        result = runner.invoke(app, ["train", str(mock_binary_dataset_dir), str(out_dir), "-e", "1", "-b", "2"])
        assert result.exit_code == 0
        mock_trainer_cls.assert_called_once()
        mock_trainer_inst.train.assert_called_once()


def test_cli_prep_synthesize_command(tmp_path: Path) -> None:
    """Test glycocr prep synthesize subcommand."""
    iupac_file = tmp_path / "iupacs.txt"
    iupac_file.write_text("Gal(b1-4)Glc\nMan(a1-3)Man\n", encoding="utf-8")
    out_dir = tmp_path / "synth_out"

    mock_synth = MagicMock()
    mock_synth.synthesize.return_value = torch.zeros((3, 64, 64), dtype=torch.uint8)

    with patch("glycocr.data.synthesizer.IUPACSynthesizer", return_value=mock_synth):
        result = runner.invoke(app, ["prep", "synthesize", str(iupac_file), str(out_dir)])
        assert result.exit_code == 0
        assert (out_dir / "images.bin").exists()
        assert (out_dir / "strings.bin").exists()
        assert (out_dir / "index.npz").exists()


def test_cli_prep_fetch_command(tmp_path: Path) -> None:
    """Test glycocr prep fetch subcommand with mocked glycowork."""
    out_file = tmp_path / "fetched.txt"

    mock_df = {"glycan": MagicMock(dropna=MagicMock(return_value=MagicMock(tolist=lambda: ["Gal(b1-4)Glc"])))}

    with (
        patch("glycowork.glycan_data.loader.df_glycan", mock_df, create=True),
        patch("glycowork.motif.processing.canonicalize_iupac", side_effect=lambda x: x, create=True),
    ):
        result = runner.invoke(app, ["prep", "fetch", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()


def test_cli_deploy_command(tmp_path: Path) -> None:
    """Test glycocr deploy subcommand."""
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    with patch("glycocr.deploy.deploy_to_huggingface") as mock_deploy:
        result = runner.invoke(app, ["deploy", "user/repo", str(model_dir), "--token", "hf_12345"])
        assert result.exit_code == 0
        mock_deploy.assert_called_once_with(repo_id="user/repo", model_path=str(model_dir), token="hf_12345")
