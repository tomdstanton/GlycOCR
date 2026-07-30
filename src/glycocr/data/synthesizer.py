"""Synthetic SNFG diagram renderer for IUPAC glycan strings."""

import io

from glycowork.motif.draw import GlycoDraw
from PIL import Image


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

    def synthesize(self, iupac_string: str) -> Image.Image:
        """Synthesize a single SNFG PNG image from an IUPAC string.

        Args:
            iupac_string: IUPAC-condensed glycan string representation.

        Returns:
            RGB PIL Image of size target_size with diagram padded on bg_color.

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
            raw_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except Exception as err:
            raise ValueError(
                f"Failed to load rendered image for '{iupac_string}': {err}"
            ) from err

        target_w, target_h = self.target_size
        w, h = raw_img.size

        if w <= 0 or h <= 0:
            raise ValueError(
                f"Invalid image dimensions for rendered IUPAC string '{iupac_string}': {w}x{h}"
            )

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized_img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        padded_img = Image.new("RGB", self.target_size, self.bg_color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded_img.paste(resized_img, (paste_x, paste_y))

        return padded_img

    def synthesize_batch(self, iupac_strings: list[str]) -> list[Image.Image]:
        """Synthesize a batch of SNFG PNG images from IUPAC strings.

        Args:
            iupac_strings: List of IUPAC-condensed glycan strings.

        Returns:
            List of RGB PIL Images.
        """
        return [self.synthesize(s) for s in iupac_strings]
