"""Unit tests for synthetic SNFG diagram generator module."""

import pytest
from PIL import Image

from glycocr.data.synthesizer import IUPACSynthesizer


def test_synthesizer_instantiation() -> None:
    """Test instantiation and attribute initialization of IUPACSynthesizer."""
    synth = IUPACSynthesizer()
    assert synth.target_size == (384, 384)
    assert synth.output_size == (384, 384)
    assert synth.bg_color == (255, 255, 255)

    custom_synth = IUPACSynthesizer(target_size=(512, 512), bg_color=(200, 200, 200))
    assert custom_synth.target_size == (512, 512)
    assert custom_synth.bg_color == (200, 200, 200)

    legacy_synth = IUPACSynthesizer(output_size=(256, 256))
    assert legacy_synth.target_size == (256, 256)


def test_synthesizer_single_image() -> None:
    """Verify IUPACSynthesizer.synthesize("Gal(b1-4)GlcNAc") returns RGB PIL Image with size (384, 384)."""
    synth = IUPACSynthesizer(target_size=(384, 384))
    img = synth.synthesize("Gal(b1-4)GlcNAc")

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (384, 384)


def test_synthesizer_invalid_iupac_raises_value_error() -> None:
    """Verify invalid IUPAC string raises ValueError."""
    synth = IUPACSynthesizer()

    with pytest.raises(ValueError):
        synth.synthesize("")

    with pytest.raises(ValueError):
        synth.synthesize("  ")

    with pytest.raises(ValueError):
        synth.synthesize("][")

    with pytest.raises(ValueError):
        synth.synthesize("[[]]")


def test_synthesizer_batch() -> None:
    """Verify synthesize_batch returns a list of RGB PIL Images of expected size."""
    synth = IUPACSynthesizer(target_size=(384, 384))
    iupac_list = ["Gal(b1-4)GlcNAc", "Neu5Ac(a2-3)Gal(b1-4)Glc"]
    images = synth.synthesize_batch(iupac_list)

    assert isinstance(images, list)
    assert len(images) == 2
    for img in images:
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (384, 384)
