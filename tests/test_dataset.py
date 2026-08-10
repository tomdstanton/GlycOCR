"""Unit tests for GlycOCRDataset loading, sequence alignment, label masking, and binary memory-mapped I/O."""

from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from glycocr.data.dataset import GlycOCRDataset


def test_dataset_sequence_alignment_and_masking(mock_hf_processor) -> None:
    """Test that GlycOCRDataset aligns input_ids and labels lengths and masks prompt & padding with -100."""
    prompt_ids = [10, 11, 12, 13]  # length 4
    target_ids = [20, 21, 22]  # length 3
    eos_id = 1
    pad_id = 0

    def fake_tokenizer(text, add_special_tokens=False):
        if text == "caption en\n":
            return {"input_ids": prompt_ids}
        else:
            return {"input_ids": target_ids}

    mock_hf_processor.tokenizer = MagicMock(side_effect=fake_tokenizer)
    mock_hf_processor.tokenizer.eos_token_id = eos_id
    mock_hf_processor.tokenizer.pad_token_id = pad_id

    max_length = 16
    dummy_img = torch.zeros((3, 64, 64), dtype=torch.uint8)
    items = [(dummy_img, "Man(a1-3)Man")]

    dataset = GlycOCRDataset(items=items, processor=mock_hf_processor, max_length=max_length, degrade_prob=0.0)
    sample = dataset[0]

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    assert input_ids.shape == (max_length,)
    assert labels.shape == (max_length,)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long

    expected_input_ids = [10, 11, 12, 13, 20, 21, 22, 1] + [0] * (max_length - 8)
    assert input_ids.tolist() == expected_input_ids

    expected_labels = [-100] * 4 + [20, 21, 22, 1] + [-100] * 8
    assert labels.tolist() == expected_labels


def test_dataset_len_and_item_shapes(sample_image_path, mock_hf_processor) -> None:
    """Test len(dataset) and keys returned by __getitem__."""
    items = [
        (sample_image_path, "Gal(b1-4)Glc"),
        (torch.zeros((3, 100, 100), dtype=torch.uint8), "Man(a1-3)Man"),
    ]
    dataset = GlycOCRDataset(items=items, processor=mock_hf_processor, max_length=32, degrade_prob=0.0)

    assert len(dataset) == 2
    item = dataset[0]
    assert "raw_images" in item
    assert "input_ids" in item
    assert "labels" in item
    assert item["raw_images"].shape[0] == 3
    assert item["input_ids"].shape == (32,)
    assert item["labels"].shape == (32,)


def test_dataset_item_with_pil_and_numpy(mock_hf_processor) -> None:
    """Test dataset item fetching when input is PIL Image or Numpy array."""
    pil_img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    np_img = np.zeros((64, 64, 3), dtype=np.uint8)

    items = [(pil_img, "Gal(b1-4)Glc"), (np_img, "Man(a1-3)Man")]
    dataset = GlycOCRDataset(items=items, processor=mock_hf_processor, max_length=16, degrade_prob=0.0)

    assert len(dataset) == 2
    sample1 = dataset[0]
    sample2 = dataset[1]

    assert sample1["raw_images"].shape == (3, 64, 64)
    assert sample2["raw_images"].shape == (3, 64, 64)


def test_dataset_binary_memmap_loading(mock_binary_dataset_dir, mock_hf_processor) -> None:
    """Test loading items lazily from binary dataset directory (images.bin, strings.bin, index.npz)."""
    dataset = GlycOCRDataset(
        data_dir=mock_binary_dataset_dir,
        processor=mock_hf_processor,
        max_length=16,
        degrade_prob=0.0,
    )

    assert len(dataset) == 1
    sample = dataset[0]

    assert "raw_images" in sample
    assert "input_ids" in sample
    assert "labels" in sample
    assert sample["raw_images"].shape == (3, 64, 64)


def test_dataset_degradation_pipeline(sample_image_path, mock_hf_processor) -> None:
    """Test that degradation pipeline is executed without error when degrade_prob > 0."""
    items = [(sample_image_path, "Gal(b1-4)Glc")]
    dataset = GlycOCRDataset(items=items, processor=mock_hf_processor, max_length=16, degrade_prob=1.0)

    sample = dataset[0]
    assert sample["raw_images"] is not None
    assert sample["raw_images"].dim() == 3
