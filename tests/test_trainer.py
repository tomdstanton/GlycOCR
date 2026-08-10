"""Unit tests for GlycOCRTrainer, DataCollatorForGlycOCR, and SigLIP input preparation."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from glycocr.training.trainer import DataCollatorForGlycOCR, GlycOCRTrainer, _GlycOCRHFTrainer


def test_data_collator_empty_list() -> None:
    """Test DataCollatorForGlycOCR returns empty dict for empty features list."""
    collator = DataCollatorForGlycOCR()
    assert collator([]) == {}


def test_data_collator_batching_tensors() -> None:
    """Test DataCollatorForGlycOCR stacks pixel_values, input_ids, labels, and lists raw_images."""
    collator = DataCollatorForGlycOCR()

    f1 = {
        "raw_images": torch.zeros((3, 100, 100), dtype=torch.uint8),
        "pixel_values": torch.zeros((3, 448, 448), dtype=torch.float32),
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
    }
    f2 = {
        "raw_images": torch.ones((3, 100, 100), dtype=torch.uint8),
        "pixel_values": torch.ones((3, 448, 448), dtype=torch.float32),
        "input_ids": torch.tensor([4, 5, 6], dtype=torch.long),
        "labels": torch.tensor([-100, 5, 6], dtype=torch.long),
    }

    batch = collator([f1, f2])

    assert len(batch["raw_images"]) == 2
    assert batch["pixel_values"].shape == (2, 3, 448, 448)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)


def test_hf_trainer_prepare_inputs_siglip_normalization(mock_hf_model) -> None:
    """Test _GlycOCRHFTrainer._prepare_inputs resizes raw images to 448x448 and normalizes with SigLIP mean/std."""
    hf_trainer = _GlycOCRHFTrainer.__new__(_GlycOCRHFTrainer)
    hf_trainer.model = mock_hf_model

    # Input batch with raw_images
    raw_img = torch.full((3, 100, 100), fill_value=255, dtype=torch.uint8)
    inputs = {"raw_images": [raw_img]}

    # Mock super()._prepare_inputs to return input dict
    with patch("transformers.Trainer._prepare_inputs", side_effect=lambda x: x):
        processed = hf_trainer._prepare_inputs(inputs)

    assert "pixel_values" in processed
    assert "raw_images" not in processed
    assert processed["pixel_values"].shape == (1, 3, 448, 448)

    # 255 / 255 = 1.0; SigLIP norm: (1.0 - 0.5) / 0.5 = 1.0
    val = processed["pixel_values"][0, 0, 0, 0].item()
    assert abs(val - 1.0) < 1e-4


def test_glycocr_trainer_init(mock_hf_model) -> None:
    """Test GlycOCRTrainer initialization arguments."""
    trainer = GlycOCRTrainer(
        model=mock_hf_model,
        output_dir="./test_output",
        learning_rate=1e-4,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        fp16=True,
    )

    assert trainer.model == mock_hf_model
    assert trainer.output_dir == "./test_output"
    assert trainer.learning_rate == 1e-4
    assert trainer.num_train_epochs == 2
    assert trainer.per_device_train_batch_size == 2
    assert trainer.fp16 is True


def test_glycocr_trainer_train_missing_dataset_raises_error(mock_hf_model) -> None:
    """Test GlycOCRTrainer.train() raises ValueError if no dataset is provided."""
    trainer = GlycOCRTrainer(model=mock_hf_model)
    with pytest.raises(ValueError, match="No training dataset provided"):
        trainer.train()


def test_glycocr_trainer_train_execution(mock_hf_model) -> None:
    """Test GlycOCRTrainer.train() instantiates HF Trainer and calls train()."""
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 10

    trainer = GlycOCRTrainer(
        model=mock_hf_model,
        train_dataset=mock_dataset,
        output_dir="./test_out",
    )

    mock_hf_trainer_instance = MagicMock()
    mock_hf_trainer_instance.train.return_value = "train_result"

    with patch("glycocr.training.trainer._GlycOCRHFTrainer", return_value=mock_hf_trainer_instance):
        result = trainer.train()
        assert result == "train_result"
        mock_hf_trainer_instance.train.assert_called_once()
