"""Unit tests for the retrieval and grounding evaluation harness."""

from src.evaluation import (
    EvaluationReport,
    QuestionResult,
    citations_within_range,
    cited_indices,
    first_matching_rank,
    load_golden_set,
    looks_like_refusal,
    run_evaluation,
)
from src.generation.answerer import Answer, Source


def make_source(index: int, text: str, nro: int = 280) -> Source:
    return Source(
        index=index,
        doc_id=f"copom_{nro}",
        chunk_id=f"copom_{nro}_chunk_{index:03d}",
        nro_reuniao=nro,
        titulo=f"{nro}a Reuniao",
        data_publicacao="2026-08-11",
        score=1.0 / index,
        text=text,
    )


class FakeAnswerer:
    """Answerer double: returns canned passages and a canned answer."""

    def __init__(self, sources: list[Source], answer_text: str = "resposta [1]"):
        self.sources = sources
        self.answer_text = answer_text

    def retrieve(self, question, limit=5, filter_year=None, filter_meeting=None):
        return self.sources[:limit]

    def generate(self, question, sources):
        return Answer(question=question, text=self.answer_text, sources=sources)


class TestMatching:
    def test_rank_is_the_position_of_the_first_matching_passage(self):
        passages = ["nada aqui", "tambem nao", "manter a taxa basica em 15,00% a.a."]

        assert first_matching_rank(passages, ["manter a taxa basica"]) == 3

    def test_matching_ignores_accents_and_case(self):
        passages = ["O Copom decidiu MANTER a taxa básica de juros"]

        assert first_matching_rank(passages, ["manter a taxa basica"]) == 1

    def test_absent_anchor_yields_no_rank(self):
        assert first_matching_rank(["texto qualquer"], ["frase ausente"]) is None


class TestCitations:
    def test_extracts_cited_passage_numbers(self):
        assert cited_indices("O Comitê manteve a taxa [1], por causa dos riscos [2][3].") == {
            1,
            2,
            3,
        }

    def test_citation_out_of_range_is_invalid(self):
        """Citing [7] when five passages were supplied is a fabricated reference."""
        assert citations_within_range("segundo o trecho [7]", source_count=5) is False

    def test_citations_within_supplied_range_are_valid(self):
        assert citations_within_range("conforme [1] e [5]", source_count=5) is True

    def test_answer_without_citations_is_not_flagged_as_out_of_range(self):
        assert citations_within_range("uma resposta sem citações", source_count=5) is True


class TestRefusal:
    def test_declining_answer_is_detected(self):
        assert looks_like_refusal("Os trechos fornecidos não permitem responder a essa pergunta.")

    def test_assertive_answer_is_not_a_refusal(self):
        assert not looks_like_refusal("O Copom reduziu a taxa básica para 14,00% a.a. [1].")


class TestReportMetrics:
    def test_hit_at_k_and_mrr_use_the_ranks(self):
        report = EvaluationReport(
            results=[
                QuestionResult(id="a", kind="answerable", question="q", hit_rank=1),
                QuestionResult(id="b", kind="answerable", question="q", hit_rank=3),
                QuestionResult(id="c", kind="answerable", question="q", hit_rank=None),
            ]
        )

        assert report.hit_at(1) == 1 / 3
        assert report.hit_at(3) == 2 / 3
        assert report.mrr == (1.0 + 1 / 3) / 3

    def test_traps_are_excluded_from_retrieval_metrics(self):
        report = EvaluationReport(
            results=[
                QuestionResult(id="a", kind="answerable", question="q", hit_rank=1),
                QuestionResult(id="t", kind="unanswerable", question="q", refused=True),
            ]
        )

        assert report.hit_at(1) == 1.0
        assert report.refusal_rate == 1.0


class TestRunEvaluation:
    def test_grades_retrieval_and_provenance(self):
        answerer = FakeAnswerer([make_source(1, "manter a taxa basica em 15,00% a.a.", nro=276)])
        golden = {
            "questions": [
                {
                    "id": "selic-276",
                    "question": "quanto?",
                    "kind": "answerable",
                    "must_retrieve_any": ["manter a taxa basica"],
                    "expected_meeting": 276,
                    "authoring": {"random_hit1_baseline": 0.076},
                }
            ]
        }

        report = run_evaluation(answerer, golden, limit=5)

        assert report.hit_at(1) == 1.0
        assert report.provenance_accuracy == 1.0
        assert report.random_hit1_baseline == 0.076

    def test_trap_answered_instead_of_refused_counts_against_the_system(self):
        answerer = FakeAnswerer(
            [make_source(1, "texto sobre juros")],
            answer_text="O Copom aprovou a regulação de bitcoin [1].",
        )
        golden = {
            "questions": [
                {
                    "id": "trap-bitcoin",
                    "question": "e sobre bitcoin?",
                    "kind": "unanswerable",
                    "absent_terms": ["bitcoin"],
                }
            ]
        }

        report = run_evaluation(answerer, golden, limit=5, generate=True)

        assert report.refusal_rate == 0.0
        assert report.citation_validity == 1.0

    def test_generation_failure_keeps_the_retrieval_results(self):
        """A quota exhausted mid-run must not discard the retrieval work already done."""

        class ThrottledAnswerer(FakeAnswerer):
            def generate(self, question, sources):
                raise RuntimeError("Error code: 429 - Too Many Requests")

        answerer = ThrottledAnswerer([make_source(1, "manter a taxa basica em 15,00% a.a.")])
        golden = {
            "questions": [
                {
                    "id": "selic",
                    "question": "quanto?",
                    "kind": "answerable",
                    "must_retrieve_any": ["manter a taxa basica"],
                }
            ]
        }

        report = run_evaluation(answerer, golden, limit=5, generate=True)

        assert report.hit_at(1) == 1.0, "a recuperacao continua medida"
        assert len(report.generation_failures) == 1
        assert "429" in report.generation_failures[0].generation_error
        assert report.facts_accuracy is None, "sem nota e diferente de nota zero"


class TestGoldenSetFile:
    """The shipped golden set must stay well-formed and self-describing."""

    def test_every_question_is_usable(self):
        golden = load_golden_set()
        questions = golden["questions"]

        assert len(questions) >= 10
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids)), "ids duplicados tornam o relatorio ambiguo"

        for entry in questions:
            assert entry["question"].strip()
            assert entry["kind"] in {"answerable", "unanswerable"}
            if entry["kind"] == "answerable":
                assert entry["must_retrieve_any"], entry["id"]
            else:
                assert entry["absent_terms"], entry["id"]

    def test_answerable_anchors_are_rare_enough_to_discriminate(self):
        """An anchor matching most of the corpus makes hit@1 nearly free."""
        golden = load_golden_set()

        for entry in golden["questions"]:
            if entry["kind"] != "answerable":
                continue
            baseline = entry["authoring"]["random_hit1_baseline"]
            assert baseline <= 0.10, f"{entry['id']}: ancora casa {baseline:.0%} do corpus"

    def test_traps_are_recorded_as_absent_from_the_corpus(self):
        golden = load_golden_set()
        traps = [q for q in golden["questions"] if q["kind"] == "unanswerable"]

        assert traps, "sem armadilhas nao se mede alucinacao"
        for entry in traps:
            assert entry["authoring"]["matching_chunks"] == 0
