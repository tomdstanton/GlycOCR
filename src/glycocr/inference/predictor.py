"""High-level Python inference API for predicting IUPAC strings from SNFG diagrams."""

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from glycocr.models.model import GlycOCRModel
    from glycocr.models.parser import GlycanParseResult


class GlycOCR:
    """Main user-facing API class for loading models and executing SNFG OCR predictions."""

    def __init__(self, model: "GlycOCRModel | None" = None) -> None:
        """Initialize GlycOCR predictor with optional pre-instantiated model instance."""
        self.model = model
        
        from glycocr.models.parser import GlycOCRParser
        self.parser = GlycOCRParser()

    @classmethod
    def load_pretrained(cls, model_path: str | Path | None = None, device: "str | torch.device | None" = None) -> "GlycOCR":
        """Load pretrained GlycOCR model weights and return predictor instance."""
        from glycocr.models.model import GlycOCRModel
        
        if model_path is None:
            model = GlycOCRModel(device=device)
        else:
            model = GlycOCRModel.from_pretrained(model_path, device=device)
        return cls(model=model)

    def predict(self, image: "str | Path | torch.Tensor") -> "GlycanParseResult":
        """Predict IUPAC string and validation status for a given input image."""
        if self.model is None:
            raise ValueError("Model is not loaded. Use load_pretrained() or pass a model to __init__.")
        
        iupac_string = self.model.generate(image)
        parsed = self.parser.parse(iupac_string)
        return parsed
