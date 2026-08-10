"""Unit tests for GlycOCRParser IUPAC validation, parsing, and NetworkX graph generation."""

from glycocr.models.parser import GlycanParseResult, GlycOCRParser


def test_parser_instantiation() -> None:
    """Test instantiation of GlycOCRParser."""
    parser = GlycOCRParser()
    assert isinstance(parser, GlycOCRParser)


def test_parser_validate_valid_iupac() -> None:
    """Test validate() returns True for canonical valid IUPAC strings."""
    parser = GlycOCRParser()
    assert parser.validate("Gal(b1-4)Glc") is True
    assert parser.validate("Man(a1-3)Man(b1-4)GlcNAc") is True


def test_parser_validate_invalid_iupac() -> None:
    """Test validate() returns False for malformed/invalid IUPAC strings."""
    parser = GlycOCRParser()
    assert parser.validate("Gal(b1-4[Glc") is False
    assert parser.validate("][[]") is False


def test_parser_parse_valid_returns_graph() -> None:
    """Test parse() with valid IUPAC string returns GlycanParseResult with is_valid=True and NetworkX graph."""
    parser = GlycOCRParser()
    res = parser.parse("Gal(b1-4)Glc")

    assert isinstance(res, GlycanParseResult)
    assert res.is_valid is True
    assert res.iupac == "Gal(b1-4)Glc"
    assert res.error is None
    assert res.graph is not None

    import networkx as nx

    assert isinstance(res.graph, nx.Graph) or hasattr(res.graph, "nodes")


def test_parser_parse_invalid_returns_error() -> None:
    """Test parse() with invalid IUPAC string returns GlycanParseResult with is_valid=False and error string."""
    parser = GlycOCRParser()
    res = parser.parse("Gal(b1-4[Glc")

    assert isinstance(res, GlycanParseResult)
    assert res.is_valid is False
    assert res.iupac == "Gal(b1-4[Glc"
    assert res.error is not None
    assert res.graph is None


def test_parser_parse_branched_iupac() -> None:
    """Test parse() with complex branched IUPAC string."""
    parser = GlycOCRParser()
    iupac = "Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc"
    res = parser.parse(iupac)

    assert res.is_valid is True
    assert res.graph is not None
