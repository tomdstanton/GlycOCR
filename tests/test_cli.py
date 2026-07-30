"""Unit tests for Typer CLI commands and options."""

from typer.testing import CliRunner

from glycocr.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Test CLI --help option prints app help message and subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "predict" in result.stdout
    assert "train" in result.stdout
    assert "synthesize" in result.stdout


def test_cli_predict_help() -> None:
    """Test predict subcommand --help."""
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--image" in result.stdout or "-i" in result.stdout


def test_cli_train_help() -> None:
    """Test train subcommand --help."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.stdout or "-d" in result.stdout


def test_cli_synthesize_help() -> None:
    """Test synthesize subcommand --help."""
    result = runner.invoke(app, ["synthesize", "--help"])
    assert result.exit_code == 0
    assert "--iupac-list" in result.stdout
