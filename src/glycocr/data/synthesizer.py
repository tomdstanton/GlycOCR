"""Synthetic SNFG diagram renderer for IUPAC glycan strings."""

import torch
import torchvision
from glycowork.motif.draw import GlycoDraw


class IUPACSynthesizer:
    """Renders IUPAC-condensed glycan strings into SNFG diagram images."""

    def __init__(
        self,
        target_size: tuple[int, int] = (384, 384),
        bg_color: tuple[int, int, int] = (255, 255, 255),
        output_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the synthesizer with target image dimensions and background color.

        Args:
            target_size: Desired (width, height) of the output image. Defaults to (384, 384).
            bg_color: RGB background color tuple for padding. Defaults to white (255, 255, 255).
            output_size: Alias for target_size for backward compatibility.
        """
        if output_size is not None:
            target_size = output_size
        self.target_size = target_size
        self.output_size = target_size
        self.bg_color = bg_color

    def synthesize(self, iupac_string: str) -> torch.Tensor:
        """Synthesize a single SNFG PNG image from an IUPAC string.

        Args:
            iupac_string: IUPAC-condensed glycan string representation.

        Returns:
            RGB PyTorch Tensor of shape (3, H, W) with diagram padded on bg_color.

        Raises:
            ValueError: If iupac_string is invalid, empty, or drawing fails.
        """
        if not isinstance(iupac_string, str) or not iupac_string.strip():
            raise ValueError(f"Invalid IUPAC string: '{iupac_string}'")

        try:
            drawing = GlycoDraw(iupac_string, suppress=True)
            png_bytes = drawing._repr_png_()
        except Exception as err:
            raise ValueError(f"Failed to render IUPAC string '{iupac_string}': {err}") from err

        if not png_bytes:
            raise ValueError(f"Failed to generate PNG bytes for IUPAC string: '{iupac_string}'")

        try:
            tensor_img = torchvision.io.decode_image(
                torch.frombuffer(bytearray(png_bytes), dtype=torch.uint8), mode=torchvision.io.ImageReadMode.RGB
            )
        except Exception as err:
            raise ValueError(f"Failed to load rendered image for '{iupac_string}': {err}") from err

        target_w, target_h = self.target_size
        _, h, w = tensor_img.shape

        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions for rendered IUPAC string '{iupac_string}': {w}x{h}")

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        float_img = tensor_img.float().unsqueeze(0) / 255.0

        import kornia.geometry.transform as kg

        # Kornia resize
        resized_img = kg.resize(float_img, size=(new_h, new_w), interpolation="bilinear", antialias=True)

        # Calculate padding needed to reach target_size
        pad_bottom = target_h - new_h
        pad_right = target_w - new_w

        # Kornia pad requires padding as (left, right, top, bottom)
        pad_left = pad_right // 2
        pad_right = pad_right - pad_left
        pad_top = pad_bottom // 2
        pad_bottom = pad_bottom - pad_top

        padded_img = torch.nn.functional.pad(
            resized_img,
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=1.0,  # White background in [0,1]
        )

        # Convert back to uint8 tensor in [0, 255]
        padded_img = (padded_img.squeeze(0) * 255.0).to(torch.uint8)

        return padded_img

    def synthesize_batch(self, iupac_strings: list[str]) -> list[torch.Tensor]:
        """Synthesize a batch of SNFG PNG images from IUPAC strings.

        Args:
            iupac_strings: List of IUPAC-condensed glycan strings.

        Returns:
            List of RGB PyTorch Tensors.
        """
        return [self.synthesize(s) for s in iupac_strings]
