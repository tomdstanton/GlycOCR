"""GlycOCR: Symbol Nomenclature for Glycans (SNFG) Vision-Language OCR Package."""

from glycocr.inference.predictor import GlycOCR

try:
    from glycocr._version import version as __version__
except ImportError:
    __version__ = "0.1.0.dev0"

__all__ = ["GlycOCR", "__version__"]
