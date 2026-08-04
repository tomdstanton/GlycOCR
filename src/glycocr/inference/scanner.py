"""Document scanner for finding and predicting SNFG images in PDFs."""

from collections.abc import Iterator
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None  # type: ignore

import torch

from glycocr.inference.predictor import GlycOCR
from glycocr.models.parser import GlycanParseResult


class DocumentScanner:
    """Scans PDFs for SNFG images and streams IUPAC string predictions."""

    def __init__(self, predictor: GlycOCR | None = None) -> None:
        """Initialize with a GlycOCR predictor instance."""
        if fitz is None:
            raise ImportError("PyMuPDF is required for document scanning. Install it with `uv pip install pymupdf`.")
        self.predictor = predictor or GlycOCR.load_pretrained()

    def scan_pdf(self, pdf_path: str | Path) -> Iterator[tuple[int, int, GlycanParseResult]]:
        """Scan a PDF and yield (page_num, image_idx, parse_result)."""
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Convert bytes to Tensor (C, H, W) via torchvision
                import torchvision

                try:
                    tensor_img = torchvision.io.decode_image(
                        torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8),
                        mode=torchvision.io.ImageReadMode.RGB,
                    )

                    # Predict IUPAC
                    result = self.predictor.predict(tensor_img)
                    yield (page_num + 1, img_idx + 1, result)
                except Exception:
                    # Ignore decoding errors or predict failures for non-SNFG items
                    continue
