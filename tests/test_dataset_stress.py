"""Empirical stress test suite for GlycOCRDataset sequence alignment and loss masking."""

from unittest.mock import MagicMock

import pytest
import torch

from glycocr.data.dataset import GlycOCRDataset


def create_mock_processor(
    prompt_ids: list[int] | torch.Tensor,
    target_ids: list[int] | torch.Tensor,
    eos_id: int | None = 1,
    pad_id: int | None = 0,
):
    """Helper to construct a mock processor/tokenizer for dataset stress tests."""
    mock_processor = MagicMock()

    def fake_tokenizer(text, add_special_tokens=False):
        if text == "caption en\n":
            return {"input_ids": prompt_ids}
        else:
            return {"input_ids": target_ids}

    mock_processor.tokenizer = MagicMock(side_effect=fake_tokenizer)
    mock_processor.tokenizer.eos_token_id = eos_id
    mock_processor.tokenizer.pad_token_id = pad_id
    return mock_processor


def verify_dataset_sample_invariants(
    sample: dict[str, torch.Tensor],
    max_length: int,
    expected_prompt_len: int,
    expected_combined_len: int,
):
    """Verify core structural invariants on a dataset sample."""
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    # 1. Shape and dtype assertions
    assert input_ids.shape == (max_length,), f"input_ids shape {input_ids.shape} != ({max_length},)"
    assert labels.shape == (max_length,), f"labels shape {labels.shape} != ({max_length},)"
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long

    # 2. Prompt masking assertion: labels[:prompt_len] must be strictly -100
    prompt_len = min(expected_prompt_len, expected_combined_len)
    assert (labels[:prompt_len] == -100).all().item(), f"Prompt labels not strictly -100: {labels[:prompt_len]}"

    # 3. Padding masking assertion: labels[combined_len:] must be strictly -100
    combined_len = min(expected_combined_len, max_length)
    if combined_len < max_length:
        assert (labels[combined_len:] == -100).all().item(), (
            f"Padding labels not strictly -100: {labels[combined_len:]}"
        )

    # 4. Target matching assertion: target tokens in labels[prompt_len:combined_len] match input_ids exactly
    if prompt_len < combined_len:
        target_labels = labels[prompt_len:combined_len]
        target_inputs = input_ids[prompt_len:combined_len]
        assert torch.equal(target_labels, target_inputs), (
            f"Target labels {target_labels} do not match target input_ids {target_inputs}"
        )


def test_stress_short_iupac_string():
    """Stress test with short IUPAC string (1 target token)."""
    prompt_ids = [10, 11, 12, 13]
    target_ids = [99]
    eos_id = 1
    pad_id = 0
    max_length = 16

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(items=[(dummy_img, "Neu5Ac")], processor=proc, max_length=max_length, num_image_tokens=0)
    sample = ds[0]

    # combined: prompt (4) + target (1) + eos (1) = 6 tokens
    verify_dataset_sample_invariants(sample, max_length=max_length, expected_prompt_len=4, expected_combined_len=6)
    assert sample["input_ids"].tolist() == [10, 11, 12, 13, 99, 1] + [0] * 10
    assert sample["labels"].tolist() == [-100, -100, -100, -100, 99, 1] + [-100] * 10


def test_stress_extremely_long_iupac_string_truncation():
    """Stress test with extremely long IUPAC string requiring truncation."""
    prompt_ids = [10, 11, 12, 13]  # 4 tokens
    target_ids = list(range(100, 300))  # 200 tokens
    eos_id = 1
    pad_id = 0
    max_length = 32

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(
        items=[(dummy_img, "VeryLongIUPAC...")],
        processor=proc,
        max_length=max_length,
        degrade_prob=0.0,
        num_image_tokens=0,
    )
    sample = ds[0]

    # combined untruncated = 4 + 200 + 1 = 205 tokens -> truncated to 32
    verify_dataset_sample_invariants(sample, max_length=max_length, expected_prompt_len=4, expected_combined_len=205)

    # First 4 labels should be -100
    assert sample["labels"][:4].tolist() == [-100] * 4
    # Remaining 28 labels should match target_ids[:28]
    assert sample["labels"][4:].tolist() == list(range(100, 128))
    # No padding should be present
    assert len(sample["labels"]) == 32


def test_stress_prompt_exceeds_max_length():
    """Stress test where prompt tokens alone exceed max_length."""
    prompt_ids = list(range(10, 30))  # 20 tokens
    target_ids = [50, 51, 52]
    eos_id = 1
    pad_id = 0
    max_length = 10

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(items=[(dummy_img, "Test")], processor=proc, max_length=max_length, num_image_tokens=0)
    sample = ds[0]

    verify_dataset_sample_invariants(sample, max_length=max_length, expected_prompt_len=20, expected_combined_len=24)
    # All 10 labels should be -100
    assert sample["labels"].tolist() == [-100] * 10
    # input_ids should be truncated prompt
    assert sample["input_ids"].tolist() == list(range(10, 20))


def test_stress_eos_token_id_none():
    """Stress test when tokenizer.eos_token_id is None."""
    prompt_ids = [10, 11]
    target_ids = [20, 21]
    eos_id = None
    pad_id = 0
    max_length = 8

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(items=[(dummy_img, "Gal(b1-4)Glc")], processor=proc, max_length=max_length, num_image_tokens=0)
    sample = ds[0]

    # combined: prompt (2) + target (2) = 4 tokens
    verify_dataset_sample_invariants(sample, max_length=max_length, expected_prompt_len=2, expected_combined_len=4)
    assert sample["input_ids"].tolist() == [10, 11, 20, 21, 0, 0, 0, 0]
    assert sample["labels"].tolist() == [-100, -100, 20, 21, -100, -100, -100, -100]


def test_stress_pad_token_id_none():
    """Stress test when tokenizer.pad_token_id is None (should default pad_id to 0)."""
    prompt_ids = [10, 11]
    target_ids = [20, 21]
    eos_id = 1
    pad_id = None
    max_length = 8

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(items=[(dummy_img, "Gal(b1-4)Glc")], processor=proc, max_length=max_length, num_image_tokens=0)
    sample = ds[0]

    # combined: prompt (2) + target (2) + eos (1) = 5 tokens
    verify_dataset_sample_invariants(sample, max_length=max_length, expected_prompt_len=2, expected_combined_len=5)
    # Default pad_id is 0
    assert sample["input_ids"].tolist() == [10, 11, 20, 21, 1, 0, 0, 0]
    assert sample["labels"].tolist() == [-100, -100, 20, 21, 1, -100, -100, -100]


def test_stress_exact_combined_length_equals_max_length():
    """Stress test when len(combined) == max_length exactly (no padding, no truncation)."""
    prompt_ids = [10, 11]
    target_ids = [20, 21, 22]
    eos_id = 1
    pad_id = 0
    max_length = 6  # 2 + 3 + 1 = 6

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(items=[(dummy_img, "ExactFit")], processor=proc, max_length=max_length, num_image_tokens=0)
    sample = ds[0]

    verify_dataset_sample_invariants(sample, max_length=max_length, expected_prompt_len=2, expected_combined_len=6)
    assert sample["input_ids"].tolist() == [10, 11, 20, 21, 22, 1]
    assert sample["labels"].tolist() == [-100, -100, 20, 21, 22, 1]


@pytest.mark.parametrize("max_len", [1, 2, 5, 8, 128, 512])
def test_stress_varied_max_lengths(max_len: int):
    """Stress test across varied max_lengths."""
    prompt_ids = [10, 11, 12]
    target_ids = [20, 21, 22, 23, 24]
    eos_id = 1
    pad_id = 0

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(items=[(dummy_img, "MultiLenTest")], processor=proc, max_length=max_len, num_image_tokens=0)
    sample = ds[0]

    # total untruncated combined = 3 + 5 + 1 = 9
    verify_dataset_sample_invariants(sample, max_length=max_len, expected_prompt_len=3, expected_combined_len=9)


def test_stress_empty_target_string():
    """Stress test with empty IUPAC string (0 target tokens)."""
    prompt_ids = [10, 11]
    target_ids = []
    eos_id = 1
    pad_id = 0
    max_length = 6

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(items=[(dummy_img, "")], processor=proc, max_length=max_length, num_image_tokens=0)
    sample = ds[0]

    # combined: 2 prompt + 0 target + 1 eos = 3 tokens
    verify_dataset_sample_invariants(sample, max_length=max_length, expected_prompt_len=2, expected_combined_len=3)
    assert sample["input_ids"].tolist() == [10, 11, 1, 0, 0, 0]
    assert sample["labels"].tolist() == [-100, -100, 1, -100, -100, -100]


def test_stress_tokenizer_returns_torch_tensors():
    """Stress test when tokenizer returns 1D PyTorch Tensors instead of lists."""
    prompt_ids = torch.tensor([10, 11, 12], dtype=torch.long)
    target_ids = torch.tensor([20, 21], dtype=torch.long)
    eos_id = 1
    pad_id = 0
    max_length = 8

    proc = create_mock_processor(prompt_ids, target_ids, eos_id=eos_id, pad_id=pad_id)
    dummy_img = torch.zeros((3, 32, 32), dtype=torch.uint8)
    ds = GlycOCRDataset(items=[(dummy_img, "TensorTest")], processor=proc, max_length=max_length, num_image_tokens=0)
    sample = ds[0]

    # combined: 3 prompt + 2 target + 1 eos = 6 tokens
    verify_dataset_sample_invariants(sample, max_length=max_length, expected_prompt_len=3, expected_combined_len=6)
    assert sample["input_ids"].tolist() == [10, 11, 12, 20, 21, 1, 0, 0]
    assert sample["labels"].tolist() == [-100, -100, -100, 20, 21, 1, -100, -100]
