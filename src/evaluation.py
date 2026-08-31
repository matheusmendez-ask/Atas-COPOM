"""Evaluation of retrieval and answer grounding against a hand-authored golden set.

Measures three things the pipeline can otherwise only claim:

* whether the right passage is retrieved at all (hit@k, MRR);
* whether a generated answer states the facts it should and cites passages that
  actually exist;
* whether the system refuses questions the corpus cannot answer, which is the
  failure mode that sinks a RAG system in production.

Grading is deterministic on purpose. An LLM judge would add a second unreliable
system to assess the first, and a disagreement would not say which one erred.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.generation.answerer import CopomAnswerer

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "golden_set.json"

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


def first_matching_rank(passages: list[str], anchors: list[str]) -> int | None:
    """Return the 1-based rank of the first passage containing any anchor.

    Args:
        passages: Retrieved passage texts, best match first.
        anchors: Phrases from the golden set; any one of them counts as a hit.

    Returns:
        The rank, or None when no retrieved passage matches.
    """
    normalized_anchors = [normalize(a) for a in anchors]
    for rank, passage in enumerate(passages, start=1):
        haystack = normalize(passage)
        if any(anchor in haystack for anchor in normalized_anchors):
            return rank
    return None


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
class QuestionResult:
    """Outcome for a single golden-set question."""

    id: str
    kind: str
    question: str
    hit_rank: int | None = None
    retrieved_count: int = 0
    meeting_ok: bool | None = None
    random_baseline: float = 0.0
    answer: str | None = None
    facts_ok: bool | None = None
    citations_ok: bool | None = None
    refused: bool | None = None
    generation_error: str | None = None

    @property
    def retrieval_ok(self) -> bool:
        return self.hit_rank is not None


@dataclass
class EvaluationReport:
    """Aggregate metrics over a golden-set run."""

    results: list[QuestionResult] = field(default_factory=list)
    generated: bool = False

    @property
    def answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.kind == "answerable"]

    @property
    def traps(self) -> list[QuestionResult]:
        return [r for r in self.results if r.kind == "unanswerable"]

    def hit_at(self, k: int) -> float:
        """Fraction of answerable questions whose passage appears in the top k."""
        items = self.answerable
        if not items:
            return 0.0
        hits = sum(1 for r in items if r.hit_rank is not None and r.hit_rank <= k)
        return hits / len(items)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank: rewards ranking the right passage first."""
        items = self.answerable
        if not items:
            return 0.0
        return sum(1.0 / r.hit_rank for r in items if r.hit_rank) / len(items)

    @property
    def random_hit1_baseline(self) -> float:
        """hit@1 a random retriever would score, from the golden set's own metadata."""
        items = self.answerable
        if not items:
            return 0.0
        return sum(r.random_baseline for r in items) / len(items)

    @property
    def provenance_accuracy(self) -> float | None:
        """Share of questions whose expected meeting was among those retrieved."""
        checked = [r for r in self.answerable if r.meeting_ok is not None]
        if not checked:
            return None
        return sum(1 for r in checked if r.meeting_ok) / len(checked)

    def _rate(self, items: list[QuestionResult], attribute: str) -> float | None:
        graded = [r for r in items if getattr(r, attribute) is not None]
        if not graded:
            return None
        return sum(1 for r in graded if getattr(r, attribute)) / len(graded)

    @property
    def generation_failures(self) -> list[QuestionResult]:
        """Questions whose model call errored, so their grading is missing, not zero."""
        return [r for r in self.results if r.generation_error]

    @property
    def facts_accuracy(self) -> float | None:
        return self._rate(self.answerable, "facts_ok")

    @property
    def citation_validity(self) -> float | None:
        return self._rate(self.results, "citations_ok")

    @property
    def refusal_rate(self) -> float | None:
        """Share of unanswerable questions the system declined instead of inventing."""
        return self._rate(self.traps, "refused")


def load_golden_set(path: Path | None = None) -> dict[str, Any]:
    """Load the golden set from disk."""
    target = path or GOLDEN_SET_PATH
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def run_evaluation(
    answerer: CopomAnswerer,
    golden: dict[str, Any],
    limit: int = 5,
    generate: bool = False,
) -> EvaluationReport:
    """Run every golden-set question through retrieval, and optionally generation.

    Args:
        answerer: Provides retrieval and, when generating, the model call.
        golden: Parsed golden set document.
        limit: Passages to retrieve per question.
        generate: Also grade the generated answer. Requires LLM credentials.

    Returns:
        A report holding per-question outcomes and aggregate metrics.
    """
    report = EvaluationReport(generated=generate)

    for entry in golden.get("questions", []):
        sources = answerer.retrieve(entry["question"], limit=limit)
        authoring = entry.get("authoring", {})
        result = QuestionResult(
            id=entry["id"],
            kind=entry["kind"],
            question=entry["question"],
            retrieved_count=len(sources),
            random_baseline=authoring.get("random_hit1_baseline", 0.0),
        )

        if entry["kind"] == "answerable":
            result.hit_rank = first_matching_rank(
                [s.text for s in sources], entry["must_retrieve_any"]
            )
            expected_meeting = entry.get("expected_meeting")
            if expected_meeting is not None:
                result.meeting_ok = expected_meeting in {s.nro_reuniao for s in sources}

        if generate:
            try:
                answer = answerer.generate(entry["question"], sources)
            except Exception as err:
                # A quota exhausted halfway through must not discard the retrieval
                # results already gathered: grade what is gradable and say what broke.
                result.generation_error = f"{type(err).__name__}: {err}"
                report.results.append(result)
                continue
            result.answer = answer.text
            result.citations_ok = citations_within_range(answer.text, len(sources))
            if entry["kind"] == "unanswerable":
                result.refused = looks_like_refusal(answer.text)
            elif entry.get("answer_must_contain"):
                normalized = normalize(answer.text)
                result.facts_ok = all(
                    normalize(fact) in normalized for fact in entry["answer_must_contain"]
                )

        report.results.append(result)

    return report
