"""PyTorch Dataset implementation formatted for Florence-2 Vision-Language Model."""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from glycocr.models.model import apply_florence2_patches


class GlycOCRDataset(Dataset):
    """Dataset loading (image, IUPAC string) pairs formatted for Florence-2 processing."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        items: list[tuple[Any, str]] | None = None,
        processor: Any = None,
        task_prompt: str = "<MORE_DETAILED_CAPTION>",
        max_length: int = 128,
        target_size: tuple[int, int] | None = (768, 768),
    ) -> None:
        """Initialize dataset with image-IUPAC pairs or binary dataset directory."""
        self.data_dir = Path(data_dir) if data_dir else None
        self.items = items if items else []
        self.task_prompt = task_prompt
        self.max_length = max_length
        self.target_size = target_size

        if self.data_dir and self.data_dir.exists():
            import numpy as np
            self.index = np.load(self.data_dir / "index.npz")
            self.img_offsets = self.index['img_offsets']
            self.img_lengths = self.index['img_lengths']
            self.str_offsets = self.index['str_offsets']
            self.str_lengths = self.index['str_lengths']
            self.num_samples = len(self.img_offsets)
            
            # File handles for binary IO (opened per worker or lazily)
            self._img_f = None
            self._str_f = None
        else:
            self.num_samples = len(self.items)

        if processor is None:
            apply_florence2_patches()
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", trust_remote_code=True
            )
        else:
            self.processor = processor

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return self.num_samples

    def _get_from_binary(self, idx: int) -> tuple[torch.Tensor, str]:
        """Load image and IUPAC string from binary blob using SoA index."""
        if self._img_f is None:
            self._img_f = open(self.data_dir / "images.bin", "rb")
            self._str_f = open(self.data_dir / "strings.bin", "rb")

        # Load string
        self._str_f.seek(self.str_offsets[idx])
        iupac_bytes = self._str_f.read(self.str_lengths[idx])
        iupac = iupac_bytes.decode('utf-8')

        # Load image bytes and decode with torchvision
        self._img_f.seek(self.img_offsets[idx])
        img_bytes = self._img_f.read(self.img_lengths[idx])
        
        # Convert bytes to tensor via torchvision
        import torchvision
        byte_tensor = torch.frombuffer(img_bytes, dtype=torch.uint8)
        image_tensor = torchvision.io.decode_image(byte_tensor, mode=torchvision.io.image.ImageReadMode.RGB)
        
        return image_tensor, iupac

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve and process a sample by index."""
        if self.data_dir:
            image_tensor, iupac = self._get_from_binary(idx)
            
            # Apply Kornia augmentations / resizing
            import kornia
            # image_tensor is (C, H, W) in uint8 [0, 255]
            image_tensor = image_tensor.float() / 255.0 # (C, H, W) [0.0, 1.0]
            
            if self.target_size is not None:
                # Kornia resize expects (B, C, H, W), so we add batch dim
                image_tensor = image_tensor.unsqueeze(0)
                image_tensor = kornia.geometry.transform.resize(
                    image_tensor, self.target_size, interpolation='bilinear'
                )
                image_tensor = image_tensor.squeeze(0)
                
            # For Florence-2 processor, we need to pass a list of tensors or PIL images
            # But the processor might expect a PIL image or numpy array. Wait, if we use torchvision,
            # we can skip the processor's image transform and do it ourselves, but for simplicity
            # we'll convert to numpy for the processor or handle it directly.
            # Wait, Florence-2 processor accepts PIL Images or NumPy arrays or PyTorch Tensors!
            # It expects RGB.
            image = image_tensor
        else:
            image_item, iupac = self.items[idx]
            if isinstance(image_item, (str, Path)):
                image = Image.open(image_item).convert("RGB")
            elif isinstance(image_item, Image.Image):
                image = image_item.convert("RGB")
            else:
                image = Image.fromarray(image_item).convert("RGB")
            
            if self.target_size is not None:
                image = image.resize(self.target_size)

        inputs = self.processor(
            text=self.task_prompt, images=image, return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].squeeze(0).to(torch.float32)
        input_ids = inputs["input_ids"].squeeze(0)

        labels = self.processor.tokenizer(
            text=iupac,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids.squeeze(0)

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels = labels.clone()
            labels[labels == pad_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }
