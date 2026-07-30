"""Unit tests for output validation and parsing layer."""

from glycocr.models.parser import GlycOCRParser


def test_parser_instantiation() -> None:
    """Test instantiation of GlycOCRParser."""
    parser = GlycOCRParser()
    assert isinstance(parser, GlycOCRParser)
