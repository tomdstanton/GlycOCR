"""Pytest central fixture configuration for zero-network minimal model testing."""

import json
import sys
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def mock_hf_processor():
    """Mock Hugging Face AutoProcessor for PaliGemma 2."""
    processor = MagicMock()
    tokenizer = MagicMock()

    def fake_tokenizer_call(text, add_special_tokens=False, return_tensors=None, **kwargs):
        if isinstance(text, list):
            ids = [[10, 11, 12, 13] for _ in text]
        else:
            ids = [10, 11, 12, 13]

        t = torch.tensor(ids, dtype=torch.long)
        if return_tensors == "pt" and t.dim() == 1:
            t = t.unsqueeze(0)

        res = MagicMock()
        res.input_ids = t
        res.__getitem__ = lambda self, k: t if k == "input_ids" else None
        return res

    tokenizer.side_effect = fake_tokenizer_call
    tokenizer.return_value = fake_tokenizer_call("caption en\n")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.batch_decode.return_value = ["Gal(b1-4)Glc"]

    processor.tokenizer = tokenizer
    processor.batch_decode.return_value = ["Gal(b1-4)Glc"]

    def fake_processor_call(text=None, images=None, return_tensors=None, **kwargs):
        return {
            "pixel_values": torch.zeros((1, 3, 448, 448), dtype=torch.float32),
            "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        }

    processor.side_effect = fake_processor_call
    processor.return_value = fake_processor_call()

    processor.save_pretrained = MagicMock()
    return processor


@pytest.fixture
def mock_hf_model():
    """Mock Hugging Face PaliGemma 2 Model with LoRA adapters."""
    model = MagicMock()
    param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    model.parameters.side_effect = lambda: iter([param])

    model.generate.return_value = torch.tensor([[10, 11, 12, 13, 100, 101, 102]], dtype=torch.long)

    output_obj = MagicMock()
    output_obj.loss = torch.tensor(0.42, dtype=torch.float32)
    output_obj.logits = torch.zeros((1, 10, 1000), dtype=torch.float32)
    model.return_value = output_obj

    model.save_pretrained = MagicMock()
    model.merge_and_unload = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_glycocr_rs(monkeypatch):
    """Mock compiled rust PyO3 extension module glycocr_rs."""
    mock_module = MagicMock()

    default_json_pdf = json.dumps(
        {
            "pdf_path": "test.pdf",
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "diagrams": [
                        {
                            "bbox": [10.0, 10.0, 100.0, 100.0],
                            "cropped_path": None,
                            "iupac": "Gal(b1-4)Glc",
                            "confidence": 0.95,
                        }
                    ],
                }
            ],
            "dummy": True,
        }
    )

    default_json_img = json.dumps(
        {
            "pdf_path": "test.png",
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "diagrams": [
                        {
                            "bbox": [0.0, 0.0, 448.0, 448.0],
                            "cropped_path": "test.png",
                            "iupac": "Gal(b1-4)Glc",
                            "confidence": 0.95,
                        }
                    ],
                }
            ],
            "dummy": True,
        }
    )

    mock_module.scan_pdf.return_value = default_json_pdf
    mock_module.scan_pdf_dict.return_value = json.loads(default_json_pdf)
    mock_module.scan_image.return_value = default_json_img

    mock_runner = MagicMock()
    mock_runner.run_pdf.return_value = default_json_pdf
    mock_runner.run_pdf_dict.return_value = json.loads(default_json_pdf)
    mock_runner.run_image.return_value = default_json_img
    mock_module.PyPipelineRunner = MagicMock(return_value=mock_runner)

    monkeypatch.setitem(sys.modules, "glycocr_rs", mock_module)
    import glycocr.cli

    monkeypatch.setattr(glycocr.cli, "glycocr_rs", mock_module)

    return mock_module


@pytest.fixture
def sample_image_path(tmp_path):
    """Path to a temporary sample PNG image file."""
    img_path = tmp_path / "sample.png"
    from PIL import Image

    img = Image.new("RGB", (448, 448), color=(255, 255, 255))
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Path to a temporary sample PDF file."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 sample pdf content")
    return pdf_path


@pytest.fixture
def sample_iupac_string():
    """Sample valid IUPAC string."""
    return "Gal(b1-4)Glc"


@pytest.fixture
def mock_binary_dataset_dir(tmp_path):
    """Creates a temporary binary dataset directory with dummy images.bin, strings.bin, index.npz."""
    import io

    import numpy as np
    from PIL import Image

    dataset_dir = tmp_path / "binary_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    str_bytes = b"Gal(b1-4)Glc"

    img_bin = dataset_dir / "images.bin"
    str_bin = dataset_dir / "strings.bin"
    index_npz = dataset_dir / "index.npz"

    img_bin.write_bytes(img_bytes)
    str_bin.write_bytes(str_bytes)

    np.savez(
        index_npz,
        img_offsets=np.array([0], dtype=np.uint64),
        img_lengths=np.array([len(img_bytes)], dtype=np.uint32),
        str_offsets=np.array([0], dtype=np.uint64),
        str_lengths=np.array([len(str_bytes)], dtype=np.uint32),
    )

    return dataset_dir
