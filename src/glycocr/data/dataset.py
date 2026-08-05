"""PyTorch Dataset implementation formatted for Florence-2 Vision-Language Model."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from glycocr.models.model import apply_florence2_patches


class GlycOCRDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset loading (image, IUPAC string) pairs formatted for Florence-2 processing."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        items: list[tuple[Any, str]] | None = None,
        processor: Any = None,  # noqa: ANN401
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
            self.img_offsets = self.index["img_offsets"]
            self.img_lengths = self.index["img_lengths"]
            self.str_offsets = self.index["str_offsets"]
            self.str_lengths = self.index["str_lengths"]
            self.num_samples = len(self.img_offsets)

            # Memory-mapped arrays for binary IO (initialized lazily per worker)
            self._img_memmap = None
            self._str_memmap = None
        else:
            self.num_samples = len(self.items)

        if processor is None:
            apply_florence2_patches()
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        else:
            self.processor = processor

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return self.num_samples

    def _get_from_binary(self, idx: int) -> tuple[torch.Tensor, str]:
        """Load image and IUPAC string from binary blob using SoA index and np.memmap."""
        if self._img_memmap is None:
            import numpy as np

            assert self.data_dir is not None
            self._img_memmap = np.memmap(self.data_dir / "images.bin", dtype="uint8", mode="r")
            self._str_memmap = np.memmap(self.data_dir / "strings.bin", dtype="uint8", mode="r")

        # Load string
        str_start = self.str_offsets[idx]
        str_end = str_start + self.str_lengths[idx]
        assert self._str_memmap is not None
        iupac_bytes = self._str_memmap[str_start:str_end].tobytes()
        iupac = iupac_bytes.decode("utf-8")

        # Load image bytes and decode with torchvision
        img_start = self.img_offsets[idx]
        img_end = img_start + self.img_lengths[idx]
        assert self._img_memmap is not None
        img_bytes = self._img_memmap[img_start:img_end].tobytes()

        # Convert bytes to tensor via torchvision
        import torch
        import torchvision

        byte_tensor = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8)
        image_tensor = torchvision.io.decode_image(byte_tensor, mode=torchvision.io.image.ImageReadMode.RGB)

        return image_tensor, iupac

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Retrieve and process a sample by index."""
        if self.data_dir:
            image_tensor, iupac = self._get_from_binary(index)
        else:
            image_item, iupac = self.items[index]
            if isinstance(image_item, (str, Path)):
                import torchvision

                img_bytes = Path(image_item).read_bytes()
                image_tensor = torchvision.io.decode_image(
                    torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8),
                    mode=torchvision.io.image.ImageReadMode.RGB,
                )
            elif isinstance(image_item, torch.Tensor):
                image_tensor = image_item
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor.squeeze(0)
            else:
                import kornia
                import numpy as np

                np_img = np.array(image_item)
                image_tensor = kornia.utils.image_to_tensor(np_img, keepdim=False).squeeze(0)

        # Process the text using processor's tokenizer
        input_ids = self.processor.tokenizer(text=self.task_prompt, return_tensors="pt").input_ids.squeeze(0)

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
            "raw_images": image_tensor,
            "input_ids": input_ids,
            "labels": labels,
        }
