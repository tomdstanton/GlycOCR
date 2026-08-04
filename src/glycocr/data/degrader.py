"""Data augmentation pipeline for introducing realistic PDF/image degradation artifacts using Kornia."""

import random

import kornia as K
import numpy as np
import torch
import torch.nn as nn


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
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.pipeline = K.augmentation.AugmentationSequential(
            # Blur / Noise
            K.augmentation.RandomGaussianNoise(mean=0.0, std=0.2, p=0.2),
            K.augmentation.RandomGaussianBlur((5, 5), (1.0, 3.0), p=0.2),
            K.augmentation.RandomMotionBlur(3, 35.0, 0.5, p=0.2),
            # Color / Contrast
            K.augmentation.RandomGrayscale(p=0.2),
            K.augmentation.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            # Affine Transform
            K.augmentation.RandomAffine(
                degrees=(-10.0, 10.0),
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=(-5.0, 5.0),
                p=0.5,
            ),
            random_apply=True,
        )

    def degrade(self, image: torch.Tensor) -> torch.Tensor:
        """Apply degradation pipeline to an input tensor.

        Args:
            image: Input PyTorch Tensor of shape (C, H, W) or (B, C, H, W) in range [0, 1].

        Returns:
            Degraded PyTorch Tensor of identical dimensions.
        """
        if random.random() > self.p:
            return image

        is_unbatched = image.dim() == 3
        if is_unbatched:
            image = image.unsqueeze(0)

        augmented = self.pipeline(image)

        if is_unbatched:
            augmented = augmented.squeeze(0)

        return augmented
