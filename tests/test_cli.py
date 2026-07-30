"""Unit tests for Typer CLI commands and options."""

from typer.testing import CliRunner

from glycocr.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Test CLI --help option prints app help message and subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "infer" in result.stdout
    assert "train" in result.stdout
    assert "synthesize" in result.stdout


def test_cli_infer_help() -> None:
    """Test infer subcommand --help."""
    result = runner.invoke(app, ["infer", "--help"])
    assert result.exit_code == 0
    assert "IMAGE" in result.stdout or "image" in result.stdout.lower()


def test_cli_train_help() -> None:
    """Test train subcommand --help."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "dataset_dir" in result.stdout.lower() or "dataset" in result.stdout.lower()


def test_cli_synthesize_help() -> None:
    """Test synthesize subcommand --help."""
    result = runner.invoke(app, ["synthesize", "--help"])
    assert result.exit_code == 0
    assert "iupac_list" in result.stdout.lower() or "iupac" in result.stdout.lower()
