"""Unit tests for GlycOCRModel architecture, PEFT adapter configuration, forward pass, and generation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from glycocr.models.model import GlycOCRModel


def test_model_paligemma2_default_config() -> None:
    """Test default configuration parameters for GlycOCRModel."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.model_name = "google/paligemma2-3b-pt-448"
    model_obj.lora_r = 8
    model_obj.lora_alpha = 16
    model_obj.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    assert model_obj.model_name == "google/paligemma2-3b-pt-448"
    assert model_obj.lora_r == 8
    assert model_obj.lora_alpha == 16
    assert model_obj.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]


def test_model_init_with_mocks(mock_hf_processor, mock_hf_model) -> None:
    """Test initialization of GlycOCRModel using mocked HF dependencies."""
    with (
        patch("glycocr.models.model.AutoProcessor.from_pretrained", return_value=mock_hf_processor),
        patch("glycocr.models.model.PaliGemmaForConditionalGeneration.from_pretrained", return_value=mock_hf_model),
        patch("glycocr.models.model.get_peft_model", return_value=mock_hf_model),
    ):
        model = GlycOCRModel(device="cpu")
        assert model.model_name == "google/paligemma2-3b-pt-448"
        assert model.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]
        assert model.processor == mock_hf_processor
        assert model.model == mock_hf_model


def test_model_custom_lora_targets(mock_hf_processor, mock_hf_model) -> None:
    """Test initializing GlycOCRModel with custom target_modules, lora_r, lora_alpha."""
    with (
        patch("glycocr.models.model.AutoProcessor.from_pretrained", return_value=mock_hf_processor),
        patch("glycocr.models.model.PaliGemmaForConditionalGeneration.from_pretrained", return_value=mock_hf_model),
        patch("glycocr.models.model.get_peft_model", return_value=mock_hf_model),
    ):
        model = GlycOCRModel(lora_r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
        assert model.lora_r == 16
        assert model.lora_alpha == 32
        assert model.target_modules == ["q_proj", "v_proj"]


def test_model_forward_pass(mock_hf_processor, mock_hf_model) -> None:
    """Test model forward pass with pixel_values and input_ids/labels."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.processor = mock_hf_processor
    model_obj.model = mock_hf_model

    pixel_values = torch.zeros((2, 3, 448, 448), dtype=torch.float32)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    labels = torch.tensor([[-100, 2, 3], [-100, 5, 6]], dtype=torch.long)

    out = model_obj.forward(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
    assert out.loss == torch.tensor(0.42)
    mock_hf_model.assert_called_with(
        pixel_values=pixel_values,
        input_ids=input_ids,
        labels=labels,
    )


def test_model_forward_pass_default_prompt(mock_hf_processor, mock_hf_model) -> None:
    """Test forward pass automatically generating prompt input_ids if input_ids is None."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.processor = mock_hf_processor
    model_obj.model = mock_hf_model

    pixel_values = torch.zeros((1, 3, 448, 448), dtype=torch.float32)
    out = model_obj.forward(pixel_values=pixel_values)
    assert out.loss == torch.tensor(0.42)


def test_model_generate_prompt_slicing() -> None:
    """Test that GlycOCRModel.generate() correctly slices prompt tokens before decoding."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)

    mock_processor = MagicMock()
    mock_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "pixel_values": torch.zeros((1, 3, 448, 448), dtype=torch.float32),
    }
    mock_processor.return_value = mock_inputs

    mock_base_model = MagicMock()
    mock_base_model.parameters.side_effect = lambda: iter([torch.tensor([0.0])])
    mock_generated_ids = torch.tensor([[1, 2, 3, 4, 100, 101, 102]])
    mock_base_model.generate.return_value = mock_generated_ids

    def fake_batch_decode(output_ids, skip_special_tokens=True):
        assert torch.equal(output_ids, torch.tensor([[100, 101, 102]]))
        return ["  Gal(b1-4)Glc  "]

    mock_processor.batch_decode = MagicMock(side_effect=fake_batch_decode)

    model_obj.processor = mock_processor
    model_obj.model = mock_base_model

    dummy_img = torch.zeros((3, 100, 100), dtype=torch.uint8)
    result = model_obj.generate(dummy_img, prompt="caption en\n")

    assert result == "Gal(b1-4)Glc"
    mock_base_model.generate.assert_called_once()
    mock_processor.batch_decode.assert_called_once()


def test_model_generate_from_file_path(sample_image_path, mock_hf_processor, mock_hf_model) -> None:
    """Test generate() given a valid image file path."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.processor = mock_hf_processor
    model_obj.model = mock_hf_model

    res = model_obj.generate(sample_image_path)
    assert res == "Gal(b1-4)Glc"


def test_model_generate_from_float_tensor(mock_hf_processor, mock_hf_model) -> None:
    """Test generate() with 3D and 4D float image tensors."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.processor = mock_hf_processor
    model_obj.model = mock_hf_model

    img_3d = torch.zeros((3, 448, 448), dtype=torch.float32)
    res_3d = model_obj.generate(img_3d)
    assert res_3d == "Gal(b1-4)Glc"

    img_4d = torch.zeros((1, 3, 448, 448), dtype=torch.float32)
    res_4d = model_obj.generate(img_4d)
    assert res_4d == "Gal(b1-4)Glc"


def test_model_generate_invalid_type() -> None:
    """Test generate() raising TypeError for unsupported input types."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.model = MagicMock()
    model_obj.model.parameters.side_effect = lambda: iter([torch.tensor([0.0])])

    with pytest.raises(TypeError, match="Unsupported image type"):
        model_obj.generate(12345)  # type: ignore


def test_model_save_and_load_pretrained(tmp_path: Path, mock_hf_processor, mock_hf_model) -> None:
    """Test save_pretrained and from_pretrained method logic."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    model_obj.model = mock_hf_model
    model_obj.processor = mock_hf_processor

    save_dir = tmp_path / "saved_model"
    model_obj.save_pretrained(save_dir)

    mock_hf_model.save_pretrained.assert_called_once_with(str(save_dir))
    mock_hf_processor.save_pretrained.assert_called_once_with(save_dir)

    mock_peft_model_instance = MagicMock()
    mock_peft_model_instance.merge_and_unload.return_value = mock_hf_model

    with (
        patch("glycocr.models.model.AutoProcessor.from_pretrained", return_value=mock_hf_processor),
        patch("glycocr.models.model.PaliGemmaForConditionalGeneration.from_pretrained", return_value=mock_hf_model),
        patch("glycocr.models.model.PeftModel.from_pretrained", return_value=mock_peft_model_instance),
    ):
        restored = GlycOCRModel.from_pretrained(save_dir, device="cpu")
        assert restored.model == mock_hf_model
        assert restored.processor == mock_hf_processor
