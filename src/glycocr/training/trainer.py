"""Training pipeline wrapping Hugging Face Trainer with fp16 and LoRA optimization."""

from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from glycocr.data.dataset import GlycOCRDataset
from glycocr.models.model import GlycOCRModel


class DataCollatorForGlycOCR:
    """Custom data collator to batch pixel_values, input_ids, and labels for GlycOCRModel."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate list of feature dictionaries into batched torch tensors."""
        batch: dict[str, Any] = {}

        if not features:
            return batch

        if "raw_images" in features[0]:
            batch["raw_images"] = [f["raw_images"] for f in features]

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])

        if "input_ids" in features[0]:
            batch["input_ids"] = torch.stack([f["input_ids"] for f in features])

        if "labels" in features[0]:
            batch["labels"] = torch.stack([f["labels"] for f in features])

        return batch


class _GlycOCRHFTrainer(Trainer):  # type: ignore
    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        inputs = super()._prepare_inputs(inputs)

        if "raw_images" in inputs:
            import kornia

            raw_images = inputs.pop("raw_images")

            # Target resolution for PaliGemma 2 (google/paligemma2-3b-pt-448) is 448x448
            target_size = (448, 448)

            resized_images = []
            for img in raw_images:
                # img is (C, H, W) uint8 tensor on device
                img_float = img.float() / 255.0
                img_float = img_float.unsqueeze(0)
                img_resized = kornia.geometry.transform.resize(img_float, target_size, interpolation="bilinear")
                resized_images.append(img_resized.squeeze(0))

            pixel_values = torch.stack(resized_images)

            # Normalize with SigLIP mean/std expected by PaliGemma 2
            device = pixel_values.device
            mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

            pixel_values = (pixel_values - mean) / std
            inputs["pixel_values"] = pixel_values.to(next(self.model.parameters()).dtype)

        return inputs


class GlycOCRTrainer:
    """Wrapper managing training and validation execution for GlycOCRModel."""

    def __init__(
        self,
        model: GlycOCRModel,
        train_dataset: GlycOCRDataset | None = None,
        eval_dataset: GlycOCRDataset | None = None,
        output_dir: str = "./output",
        learning_rate: float = 5e-4,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        fp16: bool = False,
        gradient_accumulation_steps: int = 1,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize trainer parameters and output configuration."""
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.extra_kwargs = kwargs

    def train(
        self,
        train_dataset: GlycOCRDataset | None = None,
        eval_dataset: GlycOCRDataset | None = None,
        resume_from_checkpoint: bool | str = False,
    ) -> Any:  # noqa: ANN401
        """Execute model training loop on provided dataset."""
        dataset_to_use = train_dataset if train_dataset is not None else self.train_dataset
        eval_to_use = eval_dataset if eval_dataset is not None else self.eval_dataset

        if dataset_to_use is None:
            raise ValueError("No training dataset provided to GlycOCRTrainer.")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            fp16=self.fp16,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            remove_unused_columns=False,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=3,
            eval_strategy="no" if eval_to_use is None else "epoch",
            report_to="none",
            **self.extra_kwargs,
        )

        data_collator = DataCollatorForGlycOCR()

        hf_trainer = _GlycOCRHFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_to_use,
            eval_dataset=eval_to_use,
            data_collator=data_collator,
        )

        return hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
