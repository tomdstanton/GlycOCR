"""Output validation and parsing layer leveraging glycowork."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GlycanParseResult:
    """Structured output for parsed and validated IUPAC glycan strings."""

    iupac: str
    is_valid: bool
    error: str | None = None
    graph: Any | None = None


class GlycOCRParser:
    """Validates and parses IUPAC-condensed string predictions into biological representations."""

    def validate(self, iupac_string: str) -> bool:
        """Validate chemical/topological validity of an IUPAC string prediction."""
        try:
            from glycowork.motif.graph import glycan_to_nxGraph
            from glycowork.motif.processing import canonicalize_iupac

            canonical = canonicalize_iupac(iupac_string)
            glycan_to_nxGraph(canonical)
            return True
        except Exception:
            return False

    def parse(self, iupac_string: str) -> GlycanParseResult:
        """Parse IUPAC string into graph/structural representations and metadata."""
        try:
            from glycowork.motif.graph import glycan_to_nxGraph
            from glycowork.motif.processing import canonicalize_iupac

            canonical = canonicalize_iupac(iupac_string)
            graph = glycan_to_nxGraph(canonical)
            return GlycanParseResult(
                iupac=canonical,
                is_valid=True,
                graph=graph,
            )
        except Exception as e:
            return GlycanParseResult(
                iupac=iupac_string,
                is_valid=False,
                error=str(e),
            )
