"""Unit tests for image degradation engine module."""

import torch

from glycocr.data.degrader import SNFGDegrader
from glycocr.data.synthesizer import IUPACSynthesizer


def test_degrader_instantiation() -> None:
    """Test instantiation and attribute initialization of SNFGDegrader."""
    degrader = SNFGDegrader()
    assert degrader.p == 0.8

    custom_degrader = SNFGDegrader(p=0.5, seed=42)
    assert custom_degrader.p == 0.5
    assert custom_degrader.seed == 42


def test_degrader_degrade_output_type_and_shape() -> None:
    """Verify SNFGDegrader.degrade(clean_tensor) produces valid Tensor of equal size."""
    synth = IUPACSynthesizer(target_size=(384, 384))
    clean_img = synth.synthesize("Gal(b1-4)GlcNAc")
    clean_tensor = clean_img.float() / 255.0

    degrader = SNFGDegrader(p=1.0)
    degraded_tensor = degrader.degrade(clean_tensor)

    assert isinstance(degraded_tensor, torch.Tensor)
    assert degraded_tensor.shape == clean_tensor.shape == (3, 384, 384)


def test_degrader_p_zero() -> None:
    """Verify SNFGDegrader(p=0.0) produces identical tensor."""
    synth = IUPACSynthesizer(target_size=(384, 384))
    clean_img = synth.synthesize("Gal(b1-4)GlcNAc")
    clean_tensor = clean_img.float() / 255.0

    degrader = SNFGDegrader(p=0.0)
    degraded_tensor = degrader.degrade(clean_tensor)

    assert torch.allclose(clean_tensor, degraded_tensor)


def test_degrader_tensor_and_array_conversion() -> None:
    """Verify output tensor converts back to expected shapes."""
    synth = IUPACSynthesizer(target_size=(384, 384))
    clean_img = synth.synthesize("Gal(b1-4)GlcNAc")
    clean_tensor = clean_img.float() / 255.0

    degrader = SNFGDegrader(p=0.8)
    degraded_tensor = degrader.degrade(clean_tensor)

    assert degraded_tensor.shape == torch.Size([3, 384, 384])
    assert degraded_tensor.dtype == torch.float32
