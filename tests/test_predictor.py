"""Unit tests for GlycOCR high-level predictor API."""

from unittest.mock import MagicMock, patch

import pytest

from glycocr.inference.predictor import GlycOCR
from glycocr.models.parser import GlycanParseResult


def test_predictor_init_with_model(mock_hf_model) -> None:
    """Test GlycOCR predictor initialization with provided model instance."""
    predictor = GlycOCR(model=mock_hf_model)
    assert predictor.model == mock_hf_model
    assert predictor.parser is not None


def test_predictor_unloaded_model_raises_error() -> None:
    """Test predictor.predict() raises ValueError when model is None."""
    predictor = GlycOCR(model=None)
    with pytest.raises(ValueError, match="Model is not loaded"):
        predictor.predict("dummy.png")


def test_predictor_predict_returns_parse_result(mock_hf_model, sample_image_path) -> None:
    """Test predictor.predict() generates IUPAC string and returns GlycanParseResult."""
    mock_model = MagicMock()
    mock_model.generate.return_value = "Gal(b1-4)Glc"

    predictor = GlycOCR(model=mock_model)
    res = predictor.predict(sample_image_path)

    assert isinstance(res, GlycanParseResult)
    assert res.iupac == "Gal(b1-4)Glc"
    assert res.is_valid is True
    assert res.graph is not None
    mock_model.generate.assert_called_once_with(sample_image_path, prompt="caption en\n")


def test_predictor_load_pretrained_device_selection() -> None:
    """Test load_pretrained auto device selection logic."""
    mock_model_instance = MagicMock()
    mock_model_instance.model = MagicMock()

    with (
        patch(
            "glycocr.models.model.GlycOCRModel.from_pretrained", return_value=mock_model_instance
        ) as mock_from_pretrained,
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        predictor = GlycOCR.load_pretrained("/path/to/weights")
        assert predictor.model == mock_model_instance
        mock_from_pretrained.assert_called_once_with("/path/to/weights", device="cpu")


def test_predictor_load_pretrained_cuda_selection() -> None:
    """Test load_pretrained selects cuda when available."""
    mock_model_instance = MagicMock()
    mock_model_instance.model = MagicMock()

    with (
        patch(
            "glycocr.models.model.GlycOCRModel.from_pretrained", return_value=mock_model_instance
        ) as mock_from_pretrained,
        patch("torch.cuda.is_available", return_value=True),
    ):
        predictor = GlycOCR.load_pretrained("/path/to/weights")
        assert predictor.model == mock_model_instance
        mock_from_pretrained.assert_called_once_with("/path/to/weights", device="cuda")


def test_predictor_load_pretrained_merge_and_unload() -> None:
    """Test merge_and_unload() is called if available on model."""
    mock_model_instance = MagicMock()
    mock_peft = MagicMock()
    mock_peft.merge_and_unload = MagicMock(return_value="merged_model")
    mock_model_instance.model = mock_peft

    with patch("glycocr.models.model.GlycOCRModel.from_pretrained", return_value=mock_model_instance):
        predictor = GlycOCR.load_pretrained("/path/to/weights", device="cpu")
        mock_peft.merge_and_unload.assert_called_once()
        assert predictor.model is not None
        assert predictor.model.model == "merged_model"
