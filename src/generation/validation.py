"""Deterministic checks; these do not prove semantic entailment."""

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# The system prompt instructs the model to say plainly when the excerpts do not
# answer the question. Matching that is a heuristic, not a proof -- it is listed
# here so a reader can see exactly what counts as a refusal.
REFUSAL_MARKERS = (
    "nao encontrei",
    "nao contem",
    "nao permitem responder",
    "nao e possivel responder",
    "nao ha informacao",
    "nao ha trechos",
    "nao mencionam",
    "nao tratam",
    "nao abordam",
    "nao respondem",
    "trechos fornecidos nao",
)


def normalize(text: str) -> str:
    """Casefold, strip accents and collapse whitespace for robust matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def cited_indices(answer: str) -> set[int]:
    """Extract the passage numbers the answer cites, as in '[1]' or '[2][3]'."""
    return {int(n) for n in CITATION_PATTERN.findall(answer)}


def citations_within_range(answer: str, source_count: int) -> bool:
    """True when every citation points at a passage that was actually supplied.

    Models routinely cite '[7]' when handed five passages; that is a fabricated
    reference even if the prose around it happens to be right.
    """
    return all(1 <= n <= source_count for n in cited_indices(answer))


def looks_like_refusal(answer: str) -> bool:
    """True when the answer declines instead of asserting something."""
    normalized = normalize(answer)
    return any(marker in normalized for marker in REFUSAL_MARKERS)


@dataclass
class CitationCheck:
    valid: bool
    cited: list[int] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    semantic_support: str = "not_verified"


def numbers(text: str) -> set[Decimal]:
    return {Decimal(n.replace(",", ".")) for n in re.findall(r"(?<![\w])\d+(?:[.,]\d+)?", text)}


def check_citations(text: str, sources: list[Any]) -> CitationCheck:
    citations = cited_indices(text)
    by_index = {s.index: s for s in sources}
    valid = bool(text.strip()) and citations <= by_index.keys()
    valid = valid and (bool(citations) or looks_like_refusal(text))
    unsupported = set()
    # Associate claims with their trailing citations, never with all retrieved text.
    for match in re.finditer(r"([^\[\]]+)((?:\[\d+\]\s*)+)", text):
        claim, refs = match.groups()
        cited = [by_index[i] for i in cited_indices(refs) if i in by_index]
        evidence = " ".join(f"{s.text} {s.nro_reuniao} {s.data_publicacao}" for s in cited)
        unsupported.update(str(n) for n in numbers(claim) - numbers(evidence))
    return CitationCheck(
        valid=valid, cited=sorted(citations), unsupported_numbers=sorted(unsupported)
    )
