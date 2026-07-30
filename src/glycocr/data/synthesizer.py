"""Synthetic SNFG diagram renderer for IUPAC glycan strings."""

import io

from glycowork.motif.draw import GlycoDraw
import torch
import torch.nn.functional as F
import torchvision


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
            raise ValueError(
                f"Failed to render IUPAC string '{iupac_string}': {err}"
            ) from err

        if not png_bytes:
            raise ValueError(
                f"Failed to generate PNG bytes for IUPAC string: '{iupac_string}'"
            )

        try:
            tensor_img = torchvision.io.decode_image(
                torch.frombuffer(bytearray(png_bytes), dtype=torch.uint8), 
                mode=torchvision.io.ImageReadMode.RGB
            )
        except Exception as err:
            raise ValueError(
                f"Failed to load rendered image for '{iupac_string}': {err}"
            ) from err

        target_w, target_h = self.target_size
        _, h, w = tensor_img.shape

        if w <= 0 or h <= 0:
            raise ValueError(
                f"Invalid image dimensions for rendered IUPAC string '{iupac_string}': {w}x{h}"
            )

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Convert to float for interpolation, add batch dim
        float_img = tensor_img.float().unsqueeze(0)
        
        resized_img = F.interpolate(
            float_img, size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0).to(torch.uint8)

        # Create padded background (C, H, W)
        padded_img = torch.tensor(self.bg_color, dtype=torch.uint8).view(3, 1, 1).expand(3, target_h, target_w).clone()
        
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        # Paste the resized image into the center
        padded_img[:, paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized_img

        return padded_img

    def synthesize_batch(self, iupac_strings: list[str]) -> list[torch.Tensor]:
        """Synthesize a batch of SNFG PNG images from IUPAC strings.

        Args:
            iupac_strings: List of IUPAC-condensed glycan strings.

        Returns:
            List of RGB PyTorch Tensors.
        """
        return [self.synthesize(s) for s in iupac_strings]
