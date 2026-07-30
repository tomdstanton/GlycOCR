"""PyTorch model wrapper for Florence-2 VLM with PEFT/LoRA adapter tuning."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def apply_florence2_patches() -> None:
    """Apply runtime monkey-patches for Florence-2 compatibility with recent transformers versions."""
    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None

    if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
        PreTrainedTokenizerBase.additional_special_tokens = property(
            lambda self: getattr(self, "_additional_special_tokens", [])
        )

    orig_sdpa_can_dispatch = getattr(PreTrainedModel, "_sdpa_can_dispatch", None)
    if orig_sdpa_can_dispatch is not None and not getattr(
        PreTrainedModel, "_florence2_patched", False
    ):

        def patched_sdpa_can_dispatch(
            self: PreTrainedModel, is_init_check: bool = False
        ) -> bool:
            try:
                return orig_sdpa_can_dispatch(self, is_init_check=is_init_check)
            except AttributeError:
                return False

        PreTrainedModel._sdpa_can_dispatch = patched_sdpa_can_dispatch
        PreTrainedModel._florence2_patched = True


class GlycOCRModel(nn.Module):
    """Florence-2 VLM wrapper with PEFT/LoRA adapters targeting attention layers."""

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        lora_r: int = 8,
        lora_alpha: int = 16,
        target_modules: list[str] | Sequence[str] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize Florence-2 model architecture and apply LoRA configuration."""
        super().__init__()
        apply_florence2_patches()

        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.target_modules = (
            list(target_modules) if target_modules is not None else ["q_proj", "v_proj"]
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        base_model = base_model.to(torch.float32)

        if (
            not hasattr(base_model, "generation_config")
            or base_model.generation_config is None
        ):
            try:
                from transformers import GenerationConfig

                base_model.generation_config = GenerationConfig.from_model_config(
                    base_model.config
                )
            except Exception:
                pass

        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(base_model, peft_config)

        if device is not None:
            self.to(device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass for model training and loss calculation."""
        kwargs.pop("num_items_in_batch", None)
        if input_ids is None:
            batch_size = pixel_values.shape[0]
            prompt_inputs = self.processor.tokenizer(
                text=["<MORE_DETAILED_CAPTION>"] * batch_size,
                return_tensors="pt",
                padding=True,
            )
            input_ids = prompt_inputs.input_ids.to(pixel_values.device)

        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            **kwargs,
        )

    def generate(
        self,
        image: torch.Tensor | str | Path,
        prompt: str = "<MORE_DETAILED_CAPTION>",
        max_new_tokens: int = 128,
    ) -> str:
        """Generate IUPAC text prediction given an input SNFG image."""
        device = next(self.model.parameters()).device

        if isinstance(image, (str, Path)):
            import torchvision
            # Use fast binary loading to avoid Pillow
            img_bytes = Path(image).read_bytes()
            tensor = torchvision.io.decode_image(
                torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8), 
                mode=torchvision.io.ImageReadMode.RGB
            )
            # Pass numpy array to AutoProcessor (it expects HWC numpy array if not PIL)
            np_img = tensor.permute(1, 2, 0).numpy()
            inputs = self.processor(text=prompt, images=np_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float32)
            input_ids = inputs["input_ids"].to(device=device)
        elif isinstance(image, torch.Tensor):
            pixel_values = image.to(device=device, dtype=torch.float32)
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            prompt_inputs = self.processor.tokenizer(text=prompt, return_tensors="pt")
            input_ids = prompt_inputs.input_ids.to(device=device)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            use_cache=False,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return generated_text.strip()

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save PEFT adapters and processor to directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | Path,
        model_name: str = "microsoft/Florence-2-base",
        device: str | torch.device | None = None,
    ) -> "GlycOCRModel":
        """Load fine-tuned model and processor from directory."""
        apply_florence2_patches()
        load_path = Path(load_directory)

        instance = cls.__new__(cls)
        super(GlycOCRModel, instance).__init__()

        try:
            processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        base_model = base_model.to(torch.float32)

        if (
            not hasattr(base_model, "generation_config")
            or base_model.generation_config is None
        ):
            try:
                from transformers import GenerationConfig

                base_model.generation_config = GenerationConfig.from_model_config(
                    base_model.config
                )
            except Exception:
                pass

        peft_model = PeftModel.from_pretrained(base_model, load_path)

        instance.model_name = model_name
        peft_cfg = getattr(peft_model, "peft_config", {})
        default_cfg = (
            peft_cfg.get("default", None) if isinstance(peft_cfg, dict) else None
        )
        instance.lora_r = getattr(default_cfg, "r", 8)
        instance.lora_alpha = getattr(default_cfg, "lora_alpha", 16)
        instance.target_modules = list(
            getattr(default_cfg, "target_modules", ["q_proj", "v_proj"])
        )
        instance.processor = processor
        instance.model = peft_model

        if device is not None:
            instance.to(device)

        return instance
