"""Unit tests for model architecture wrapper, dataset, and trainer modules."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from glycocr.data.dataset import GlycOCRDataset
from glycocr.data.synthesizer import IUPACSynthesizer
from glycocr.models.model import GlycOCRModel
from glycocr.training.trainer import DataCollatorForGlycOCR, GlycOCRTrainer

# --- MOCKING HF TRANSFORMERS ---


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.peft_config = {"default": MagicMock(target_modules=["q_proj", "v_proj"], r=8, lora_alpha=16)}

    def forward(self, pixel_values, input_ids=None, labels=None, **kwargs):
        loss = torch.tensor(2.0, requires_grad=True)
        return MagicMock(loss=loss)

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])

    def save_pretrained(self, path):
        import json

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text(
            json.dumps({"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]})
        )


def mock_get_peft_model(base_model, peft_config):
    return DummyModel()


def setup_mocks():
    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()
    mock_processor.tokenizer.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))
    mock_processor.return_value = {
        "pixel_values": torch.zeros((1, 3, 768, 768)),
        "input_ids": torch.tensor([[1, 2, 3]]),
    }
    mock_processor.batch_decode.return_value = ["Gal(b1-4)GlcNAc"]

    p_model = patch("glycocr.models.model.AutoModelForCausalLM")
    p_proc = patch("glycocr.models.model.AutoProcessor")
    p_peft = patch("glycocr.models.model.get_peft_model", side_effect=mock_get_peft_model)
    p_peft_cls = patch("glycocr.models.model.PeftModel")

    return p_model, p_proc, p_peft, p_peft_cls, mock_processor


# --- TESTS ---


def test_model_instantiation() -> None:
    p_model, p_proc, p_peft, p_peft_cls, _ = setup_mocks()
    with p_model, p_proc, p_peft, p_peft_cls:
        model = GlycOCRModel()
        assert model.model_name == "microsoft/Florence-2-base"
        assert model.lora_r == 8
        assert model.lora_alpha == 16
        assert "q_proj" in model.target_modules
        assert "v_proj" in model.target_modules

        peft_config = model.model.peft_config.get("default", None)
        assert peft_config is not None
        assert set(getattr(peft_config, "target_modules")) == {"q_proj", "v_proj"}


def test_dataset_indexing() -> None:
    img1 = torch.ones((3, 100, 100), dtype=torch.float32)
    img2 = torch.ones((3, 200, 200), dtype=torch.float32)
    pairs = [(img1, "Gal(b1-4)Glc"), (img2, "Man(a1-3)Man")]

    dataset = GlycOCRDataset(items=pairs, max_length=128)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert set(item0.keys()) == {"raw_images", "input_ids", "labels"}
    assert item0["raw_images"].shape == torch.Size([3, 100, 100])
    assert item0["raw_images"].dtype == torch.float32
    assert item0["input_ids"].dim() == 1
    assert item0["labels"].shape == torch.Size([128])
    assert item0["labels"].dtype == torch.int64

    assert -100 in item0["labels"]


def test_dataloader_and_forward_pass() -> None:
    img1 = torch.ones((3, 128, 128), dtype=torch.float32)
    img2 = torch.ones((3, 128, 128), dtype=torch.float32)
    pairs = [(img1, "Gal(b1-4)GlcNAc"), (img2, "Neu5Ac(a2-3)Gal")]

    p_model, p_proc, p_peft, p_peft_cls, mock_processor = setup_mocks()
    with p_model, p_proc as mock_proc, p_peft, p_peft_cls:
        mock_proc.from_pretrained.return_value = mock_processor
        model = GlycOCRModel()
        dataset = GlycOCRDataset(items=pairs, processor=model.processor, max_length=64)
        dataloader = DataLoader(dataset, batch_size=2)

        batch = next(iter(dataloader))
        assert batch["raw_images"][0].shape == torch.Size([3, 128, 128])
        assert batch["labels"].shape == torch.Size([2, 3])

        # Manually prepare inputs simulating Trainer
        from glycocr.training.trainer import _GlycOCRHFTrainer

        trainer = _GlycOCRHFTrainer(model=model)
        batch = trainer._prepare_inputs(batch)

        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )

        assert hasattr(outputs, "loss")
        assert outputs.loss is not None
        assert outputs.loss.item() > 0.0


def test_single_sample_overfitting() -> None:
    synth = IUPACSynthesizer(target_size=(384, 384))
    iupac_target = "Gal(b1-4)GlcNAc"
    image = synth.synthesize(iupac_target)

    tensor_img = image

    p_model, p_proc, p_peft, p_peft_cls, mock_processor = setup_mocks()
    with p_model, p_proc as mock_proc, p_peft, p_peft_cls:
        mock_proc.from_pretrained.return_value = mock_processor
        model = GlycOCRModel()
        dataset = GlycOCRDataset(
            items=[(image, iupac_target)],
            processor=model.processor,
            max_length=64,
        )
        sample = dataset[0]

        device = next(model.parameters()).device
        pixel_values = sample["raw_images"].unsqueeze(0).to(device)
        # Mock what trainer does
        import kornia

        pixel_values = kornia.geometry.transform.resize(pixel_values, (768, 768), interpolation="bilinear")
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        pixel_values = (pixel_values - mean) / std

        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=3e-3,
        )

        model.train()
        for step in range(2):
            optimizer.zero_grad()
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model.generate(tensor_img)

        assert prediction == iupac_target, f"Expected '{iupac_target}', got '{prediction}'"


def test_model_generate() -> None:
    img = torch.ones((3, 64, 64), dtype=torch.float32)
    p_model, p_proc, p_peft, p_peft_cls, mock_processor = setup_mocks()
    with p_model, p_proc as mock_proc, p_peft, p_peft_cls:
        mock_proc.from_pretrained.return_value = mock_processor
        model = GlycOCRModel()
        prediction = model.generate(img)
        assert isinstance(prediction, str)


def test_model_save_and_load_pretrained() -> None:
    p_model, p_proc, p_peft, p_peft_cls, mock_processor = setup_mocks()
    with p_model, p_proc as mock_proc, p_peft, p_peft_cls as mock_peft:
        mock_proc.from_pretrained.return_value = mock_processor
        mock_peft.from_pretrained.return_value = DummyModel()

        model = GlycOCRModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "florence2_lora"
            model.save_pretrained(save_path)
            assert (save_path / "adapter_config.json").exists()

            reloaded_model = GlycOCRModel.from_pretrained(save_path)
            assert isinstance(reloaded_model, GlycOCRModel)
            assert reloaded_model.lora_r == 8


def test_trainer_instantiation() -> None:
    p_model, p_proc, p_peft, p_peft_cls, _ = setup_mocks()
    with p_model, p_proc, p_peft, p_peft_cls:
        model = GlycOCRModel()
        trainer = GlycOCRTrainer(model=model, output_dir="/tmp/checkpoints")
        assert trainer.output_dir == "/tmp/checkpoints"
        assert trainer.learning_rate == 5e-4
        assert trainer.fp16 is False


def test_data_collator() -> None:
    collator = DataCollatorForGlycOCR()
    f1 = {
        "raw_images": torch.zeros((3, 128, 128)),
        "input_ids": torch.tensor([1, 2, 3]),
        "labels": torch.tensor([4, 5, 6]),
    }
    f2 = {
        "raw_images": torch.ones((3, 128, 128)),
        "input_ids": torch.tensor([1, 2, 3]),
        "labels": torch.tensor([4, 5, 6]),
    }
    batch = collator([f1, f2])
    assert batch["raw_images"][0].shape == torch.Size([3, 128, 128])
    assert batch["input_ids"].shape == torch.Size([2, 3])
    assert batch["labels"].shape == torch.Size([2, 3])


def test_hf_trainer_execution() -> None:
    img = torch.ones((3, 64, 64), dtype=torch.float32)
    p_model, p_proc, p_peft, p_peft_cls, mock_processor = setup_mocks()
    with (
        p_model,
        p_proc as mock_proc,
        p_peft,
        p_peft_cls,
        patch("glycocr.training.trainer.Trainer.train") as mock_train,
    ):
        mock_proc.from_pretrained.return_value = mock_processor
        mock_train.return_value = MagicMock()
        model = GlycOCRModel()
        dataset = GlycOCRDataset(
            items=[(img, "Gal(b1-4)Glc")],
            processor=model.processor,
            max_length=16,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = GlycOCRTrainer(
                model=model,
                train_dataset=dataset,
                output_dir=tmp_dir,
                learning_rate=1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
            )
            train_result = trainer.train()
            assert train_result is not None
