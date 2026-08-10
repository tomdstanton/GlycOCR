"""Empirical stress tests for prompt token slicing in GlycOCRModel.generate()."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from glycocr.models.model import GlycOCRModel


class MockBatchEncoding(dict[str, Any]):
    """Mock BatchEncoding object that supports both dict indexing and attribute access."""

    def __init__(self, input_ids: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(input_ids=input_ids, **kwargs)
        self.input_ids = input_ids


def _create_mock_model_and_processor(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    decoded_raw_text: str,
) -> tuple[GlycOCRModel, MagicMock, MagicMock]:
    """Helper to construct a GlycOCRModel with mocked processor and model."""
    model_obj = GlycOCRModel.__new__(GlycOCRModel)

    input_ids_tensor = torch.tensor([prompt_token_ids])
    generated_ids_tensor = torch.tensor([prompt_token_ids + output_token_ids])

    mock_processor = MagicMock()
    mock_processor.tokenizer.return_tensors = MagicMock()
    mock_batch_enc = MockBatchEncoding(input_ids=input_ids_tensor)
    mock_processor.tokenizer.return_value = mock_batch_enc
    mock_processor.return_value = {
        "input_ids": input_ids_tensor,
        "pixel_values": torch.zeros((1, 3, 448, 448), dtype=torch.float32),
    }

    mock_base_model = MagicMock()
    mock_base_model.parameters.return_value = iter([torch.tensor([0.0])])
    mock_base_model.generate.return_value = generated_ids_tensor

    def fake_batch_decode(output_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        # Assert that prompt tokens were strictly sliced out
        expected_output_ids = torch.tensor([output_token_ids])
        assert torch.equal(output_ids, expected_output_ids), (
            f"Expected output_ids {expected_output_ids.tolist()}, but got {output_ids.tolist()}"
        )
        return [decoded_raw_text]

    mock_processor.batch_decode = MagicMock(side_effect=fake_batch_decode)

    model_obj.processor = mock_processor
    model_obj.model = mock_base_model
    return model_obj, mock_processor, mock_base_model


@pytest.mark.parametrize(
    "prompt_len, prompt_tokens",
    [
        (1, [10]),
        (4, [10, 11, 12, 13]),
        (10, list(range(10, 20))),
        (25, list(range(10, 35))),
        (50, list(range(10, 60))),
    ],
)
def test_prompt_slicing_varying_lengths(prompt_len: int, prompt_tokens: list[int]) -> None:
    """Stress test: Prompt token slicing with varying prompt lengths (1 to 50 tokens)."""
    output_tokens = [100, 101, 102]
    decoded_text = "  Man(a1-3)Man(b1-4)GlcNAc  "

    model_obj, _, mock_base_model = _create_mock_model_and_processor(
        prompt_token_ids=prompt_tokens,
        output_token_ids=output_tokens,
        decoded_raw_text=decoded_text,
    )

    dummy_img = torch.zeros((3, 64, 64), dtype=torch.uint8)
    result = model_obj.generate(dummy_img, prompt="test_prompt")

    assert result == "Man(a1-3)Man(b1-4)GlcNAc"
    mock_base_model.generate.assert_called_once()


def test_prompt_slicing_empty_output() -> None:
    """Stress test: When model generates 0 new tokens (e.g. immediate EOS)."""
    prompt_tokens = [1, 2, 3, 4]
    output_tokens: list[int] = []

    model_obj, _, _ = _create_mock_model_and_processor(
        prompt_token_ids=prompt_tokens,
        output_token_ids=output_tokens,
        decoded_raw_text="",
    )

    dummy_img = torch.zeros((3, 64, 64), dtype=torch.uint8)
    result = model_obj.generate(dummy_img, prompt="caption en\n")

    assert result == ""


def test_prompt_slicing_multiline_iupac() -> None:
    """Stress test: Decoding multi-line IUPAC strings with complex structure."""
    prompt_tokens = [1, 2, 3]
    output_tokens = [200, 201, 202, 203, 204]
    multiline_iupac = (
        "Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)\n[Neu5Ac(a2-6)GalNAc(b1-4)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
    )

    model_obj, _, _ = _create_mock_model_and_processor(
        prompt_token_ids=prompt_tokens,
        output_token_ids=output_tokens,
        decoded_raw_text=f"  {multiline_iupac}  \n",
    )

    dummy_img = torch.zeros((3, 64, 64), dtype=torch.uint8)
    result = model_obj.generate(dummy_img, prompt="caption en\n")

    assert result == multiline_iupac


def test_prompt_slicing_special_tokens_in_output() -> None:
    """Stress test: Special tokens inside output tokens are handled correctly."""
    prompt_tokens = [1, 2, 3, 4]
    # 100 = Gal, 1 = EOS, 0 = PAD, 2 = UNK
    output_tokens = [100, 101, 1, 0, 2]
    decoded_clean_text = "Gal(b1-4)Glc"

    model_obj, mock_processor, _ = _create_mock_model_and_processor(
        prompt_token_ids=prompt_tokens,
        output_token_ids=output_tokens,
        decoded_raw_text=decoded_clean_text,
    )

    dummy_img = torch.zeros((3, 64, 64), dtype=torch.uint8)
    result = model_obj.generate(dummy_img, prompt="caption en\n")

    assert result == "Gal(b1-4)Glc"
    # Ensure skip_special_tokens=True was passed to batch_decode
    assert mock_processor.batch_decode.call_args.kwargs.get("skip_special_tokens") is True


def test_prompt_slicing_across_image_input_types(tmp_path: Path) -> None:
    """Stress test: Prompt token slicing across all image input types (str, Path, uint8, float32)."""
    prompt_tokens = [10, 20, 30]
    output_tokens = [500, 501]
    decoded_text = "Fuc(a1-2)Gal"

    # Create dummy image file on disk
    from PIL import Image

    img_path = tmp_path / "test_snfg.png"
    pil_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    pil_img.save(img_path)

    # 1. Test str path
    model_obj, _, _ = _create_mock_model_and_processor(prompt_tokens, output_tokens, decoded_text)
    res_str = model_obj.generate(str(img_path), prompt="caption en\n")
    assert res_str == "Fuc(a1-2)Gal"

    # 2. Test Path object
    model_obj, _, _ = _create_mock_model_and_processor(prompt_tokens, output_tokens, decoded_text)
    res_path = model_obj.generate(img_path, prompt="caption en\n")
    assert res_path == "Fuc(a1-2)Gal"

    # 3. Test uint8 Tensor (3D)
    model_obj, _, _ = _create_mock_model_and_processor(prompt_tokens, output_tokens, decoded_text)
    uint8_3d = torch.zeros((3, 100, 100), dtype=torch.uint8)
    res_u3d = model_obj.generate(uint8_3d, prompt="caption en\n")
    assert res_u3d == "Fuc(a1-2)Gal"

    # 4. Test uint8 Tensor (4D)
    model_obj, _, _ = _create_mock_model_and_processor(prompt_tokens, output_tokens, decoded_text)
    uint8_4d = torch.zeros((1, 3, 100, 100), dtype=torch.uint8)
    res_u4d = model_obj.generate(uint8_4d, prompt="caption en\n")
    assert res_u4d == "Fuc(a1-2)Gal"

    # 5. Test float32 Tensor (3D)
    model_obj, _, _ = _create_mock_model_and_processor(prompt_tokens, output_tokens, decoded_text)
    float_3d = torch.zeros((3, 448, 448), dtype=torch.float32)
    res_f3d = model_obj.generate(float_3d, prompt="caption en\n")
    assert res_f3d == "Fuc(a1-2)Gal"

    # 6. Test float32 Tensor (4D)
    model_obj, _, _ = _create_mock_model_and_processor(prompt_tokens, output_tokens, decoded_text)
    float_4d = torch.zeros((1, 3, 448, 448), dtype=torch.float32)
    res_f4d = model_obj.generate(float_4d, prompt="caption en\n")
    assert res_f4d == "Fuc(a1-2)Gal"


def test_no_prompt_contamination_assurance() -> None:
    """Stress test: Contamination check to verify prompt tokens never reach batch_decode."""
    prompt_tokens = [9999, 8888, 7777]  # Prompt tokens
    output_tokens = [123, 456]  # Model output tokens

    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    input_ids_tensor = torch.tensor([prompt_tokens])
    generated_ids_tensor = torch.tensor([prompt_tokens + output_tokens])

    mock_processor = MagicMock()
    mock_processor.return_value = {
        "input_ids": input_ids_tensor,
        "pixel_values": torch.zeros((1, 3, 448, 448), dtype=torch.float32),
    }

    mock_base_model = MagicMock()
    mock_base_model.parameters.return_value = iter([torch.tensor([0.0])])
    mock_base_model.generate.return_value = generated_ids_tensor

    received_slice: list[int] = []

    def mock_batch_decode(output_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        nonlocal received_slice
        received_slice = output_ids.squeeze(0).tolist()
        return ["GalNAc"]

    mock_processor.batch_decode = MagicMock(side_effect=mock_batch_decode)
    model_obj.processor = mock_processor
    model_obj.model = mock_base_model

    dummy_img = torch.zeros((3, 64, 64), dtype=torch.uint8)
    res = model_obj.generate(dummy_img, prompt="caption en\n")

    # Assert no prompt tokens exist in received slice
    for pt in prompt_tokens:
        assert pt not in received_slice, f"Prompt token {pt} was found in sliced output_ids!"

    assert received_slice == output_tokens
    assert res == "GalNAc"


def test_plain_dict_tokenizer_return_type_repro() -> None:
    """Stress test / Bug reproduction: Tokenizer returning plain dict causes AttributeError in float32 branch."""
    prompt_tokens = [10, 20]
    output_tokens = [100, 200]
    input_ids_tensor = torch.tensor([prompt_tokens])
    generated_ids_tensor = torch.tensor([prompt_tokens + output_tokens])

    model_obj = GlycOCRModel.__new__(GlycOCRModel)
    mock_processor = MagicMock()

    # Plain dict return without attribute access .input_ids
    mock_processor.tokenizer.return_value = {"input_ids": input_ids_tensor}

    mock_base_model = MagicMock()
    mock_base_model.parameters.return_value = iter([torch.tensor([0.0])])
    mock_base_model.generate.return_value = generated_ids_tensor

    model_obj.processor = mock_processor
    model_obj.model = mock_base_model

    float_3d = torch.zeros((3, 448, 448), dtype=torch.float32)

    with pytest.raises(AttributeError, match="'dict' object has no attribute 'input_ids'"):
        model_obj.generate(float_3d, prompt="caption en\n")
