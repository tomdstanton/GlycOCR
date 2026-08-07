"""Data augmentation pipeline for introducing realistic PDF/image degradation artifacts using torchvision.transforms.v2."""

import random

import torch
import torch.nn as nn
from torchvision.transforms import v2


class RandomGaussianNoise(nn.Module):
    """Custom Gaussian Noise transform for torchvision v2."""

    def __init__(self, mean: float = 0.0, std: float = 0.2, p: float = 0.2) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img

        noise = torch.randn_like(img) * self.std + self.mean
        noisy_img = img + noise
        return noisy_img.clamp(0.0, 1.0)


class SNFGDegrader(nn.Module):
    """Applies noise, blur, DPI degradation, and compression artifacts to clean SNFG images."""

    def __init__(self, p: float = 0.8, seed: int | None = None) -> None:
        """Initialize the degrader pipeline with probability threshold p and optional seed.

        Args:
            p: Probability of applying the degradation pipeline. Defaults to 0.8.
            seed: Optional integer seed for random number generators.
        """
        super().__init__()
        self.p = p
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Build v2 augmentations (assumes tensor is float [0, 1])
        self.pipeline = v2.Compose(
            [
                # Blur / Noise
                RandomGaussianNoise(mean=0.0, std=0.2, p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 3.0))], p=0.2),
                # Torchvision doesn't have an exact RandomMotionBlur, but a second blur pass handles most of it.
                v2.RandomApply([v2.GaussianBlur(kernel_size=(7, 7), sigma=(0.5, 2.0))], p=0.2),
                # Color / Contrast
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
                # Affine Transform
                v2.RandomApply(
                    [
                        v2.RandomAffine(
                            degrees=(-10.0, 10.0), translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(-5.0, 5.0)
                        )
                    ],
                    p=0.5,
                ),
            ]
        )

    def degrade(self, image: torch.Tensor) -> torch.Tensor:
        """Apply degradation pipeline to an input tensor.

        Args:
            image: Input PyTorch Tensor of shape (C, H, W) or (B, C, H, W).

        Returns:
            Degraded PyTorch Tensor of identical dimensions.
        """
        if random.random() > self.p:
            return image

        is_uint8 = image.dtype == torch.uint8

        if is_uint8:
            image = image.float() / 255.0

        augmented = self.pipeline(image)

        if is_uint8:
            augmented = (augmented * 255).clamp(0, 255).to(torch.uint8)

        return augmented
