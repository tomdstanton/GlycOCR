"""Output validation and parsing layer leveraging glycowork."""

from typing import Any
import networkx as nx

from glycowork.motif.graph import glycan_to_nxGraph
from glycowork.motif.processing import canonicalize_iupac


from pydantic import BaseModel, ConfigDict, Field

class GlycanParseResult(BaseModel):
    """Structured output for parsed and validated IUPAC glycan strings."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    iupac: str = Field(description="The canonical IUPAC string, or raw string if parsing failed.")
    is_valid: bool = Field(description="Whether the IUPAC string is chemically and topologically valid.")
    error: str | None = Field(default=None, description="Error message if validation failed.")
    graph: Any | None = Field(default=None, description="NetworkX graph representation of the glycan.")


class GlycOCRParser:
    """Validates and parses IUPAC-condensed string predictions into biological representations."""

    def validate(self, iupac_string: str) -> bool:
        """Validate chemical/topological validity of an IUPAC string prediction."""
        try:
            canonical = canonicalize_iupac(iupac_string)
            glycan_to_nxGraph(canonical)
            return True
        except Exception:
            return False

    def parse(self, iupac_string: str) -> GlycanParseResult:
        """Parse IUPAC string into graph/structural representations and metadata."""
        try:
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
