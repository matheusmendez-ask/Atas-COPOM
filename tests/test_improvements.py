"""Regression tests for filtering, safe synchronization and auditable evaluation."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.evaluation import load_golden_set, run_evaluation
from src.evaluation_artifacts import digest, save_report
from src.generation.answerer import Answer, AnswerValidationError, CopomAnswerer, Source
from src.generation.validation import check_citations
from src.processing.chunker import AtaChunker
from src.vectorstore.filters import infer_meeting
from src.vectorstore.qdrant_manager import QdrantManager


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Na 278ª reunião, qual a decisão?", 278),
        ("ata 278", 278),
        ("reunião nº 278", 278),
        ("Na reunião 278 em 2026", 278),
        ("Na ata da 278ª reunião, quais as expectativas para 2026 e 2027?", 278),
        ("O que mudou da 278ª para a 279ª reunião?", None),
        ("Em 2026, qual a taxa de 14,25%?", None),
        ("Compare a 278ª reunião e a 279ª reunião", None),
        ("Reuniões 278 e 279", None),
        ("Ata 278 e 279", None),
        ("Entre a reunião 278 e a reunião 280", None),
        ("Da reunião 278 até 280", None),
        ("reunião 278 e reunião 279", None),
    ],
)
def test_explicit_meeting_inference(query, expected):
    assert infer_meeting(query) == expected


@pytest.fixture
def manager():
    dense = SimpleNamespace(
        model_name="fake",
        provider="fastembed",
        dimension=4,
        embed_texts=Mock(side_effect=lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts]),
        embed_query=lambda q: [1.0, 0.0, 0.0, 0.0],
    )
    sparse = SimpleNamespace(
        model_name="fake-sparse",
        language="portuguese",
        embed_documents=lambda texts: [([1], [1.0]) for _ in texts],
        embed_query=lambda q: ([1], [1.0]),
    )
    instance = QdrantManager(url=":memory:", embedding_generator=dense, sparse_embedder=sparse)
    yield instance
    instance.client.close()


@pytest.fixture
def doc(sample_bronze_record):
    return AtaChunker(chunk_size=100, chunk_overlap=15).process_record(sample_bronze_record)


def shrink(doc):
    result = doc.model_copy(deep=True)
    result.chunks = result.chunks[:1]
    result.chunks[0].text = "Texto revisado e menor."
    result.chunks[0].metadata.total_chunks = 1
    return result


def test_reindex_skips_unchanged_and_removes_obsolete_only_in_changed_document(manager, doc):
    other = doc.model_copy(deep=True)
    other.doc_id = "other"
    for chunk in other.chunks:
        chunk.metadata.doc_id = "other"
    assert manager.sync_documents([doc, other]).failed == 0
    manager.embedding_generator.embed_texts.reset_mock()
    result = manager.sync_documents([doc])
    assert result.skipped_points == len(doc.chunks)
    manager.embedding_generator.embed_texts.assert_not_called()
    result = manager.sync_documents([shrink(doc)])
    assert result.deleted_points == len(doc.chunks) - 1
    assert len(manager.payload_snapshot(doc.doc_id)) == 1
    assert len(manager.payload_snapshot("other")) == len(other.chunks)


def test_failed_update_keeps_old_chunks_and_retry_completes(manager, doc):
    manager.sync_documents([doc])
    original = manager.embedding_generator.embed_texts.side_effect
    manager.embedding_generator.embed_texts.side_effect = RuntimeError("offline")
    assert manager.sync_documents([shrink(doc)]).failed == 1
    assert len(manager.payload_snapshot(doc.doc_id)) == len(doc.chunks)
    manager.embedding_generator.embed_texts.side_effect = original
    assert manager.sync_documents([shrink(doc)]).deleted_points == len(doc.chunks) - 1


def test_incomplete_document_rejected_before_mutating(manager, doc):
    manager.sync_documents([doc])
    doc.chunks = doc.chunks[:1]  # total_chunks still declares the original count
    with pytest.raises(ValueError, match="Incomplete"):
        manager.sync_documents([doc])
    assert len(manager.payload_snapshot(doc.doc_id)) > 1


def test_changed_model_identity_reembeds_same_dimension(manager, doc):
    manager.sync_documents([doc])
    manager.embedding_generator.model_name = "another-model"
    assert manager.sync_documents([doc]).upserted_points == len(doc.chunks)


def test_metadata_filter_is_applied_to_both_branches_and_explicit_filter_wins(manager, doc):
    manager.sync_documents([doc])
    assert manager.search("decisão da reunião 279") == []
    assert manager.search("decisão da reunião 279", filter_meeting=280)
    assert manager.search("decisão da reunião 280", filter_year=1999) == []


def source(index=1, text="A Selic foi fixada em 14,25%."):
    return Source(index, "copom_269", "chunk", 269, "Ata", "2025-03-25", 1.0, text)


def test_citations_check_numbers_only_in_cited_evidence():
    sources = [source(1), source(2, "A taxa era 12,25%.")]
    assert check_citations("Selic de 14,25% [1].", sources).unsupported_numbers == []
    assert check_citations("Selic de 12,25% [1].", sources).unsupported_numbers == ["12.25"]
    assert not check_citations("Selic de 14,25% [7].", sources).valid
    assert not check_citations("A Selic foi mantida.", sources).valid
    assert not check_citations("", sources).valid
    assert check_citations("Não encontrei essa informação nos trechos.", sources).valid
    assert check_citations("Caiu para 14,25% [1].", sources).semantic_support == "not_verified"


def test_invalid_model_output_is_graded_instead_of_disappearing_from_metrics(tmp_path):
    answerer = CopomAnswerer(qdrant=object())
    answerer.retrieve = lambda *args, **kwargs: [source()]

    def generate(question, sources):
        raise AnswerValidationError(Answer(question, "A Selic era 14,25% [9].", sources))

    answerer.generate = generate
    golden = {
        "questions": [
            {
                "id": "q",
                "kind": "answerable",
                "question": "Selic?",
                "must_retrieve_any": ["14,25%"],
                "answer_must_contain": ["14,25%"],
                "authoring": {"random_hit1_baseline": 0.99},
            }
        ]
    }
    snapshot = [
        {"id": "1", "payload": {"text": "14,25%"}},
        {"id": "2", "payload": {"text": "outro"}},
    ]
    report = run_evaluation(
        answerer, golden, generate=True, corpus_payloads=[r["payload"] for r in snapshot]
    )
    assert not report.generation_failures
    assert report.citation_validity == 0
    assert report.results[0].citation_contract_ok is False
    assert report.random_hit1_baseline == 0.5
    output = save_report(report, golden, snapshot, tmp_path / "run.json", 5, {"test": True})
    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["corpus_sha256"] == digest(snapshot)
    assert body["results"][0]["retrieved_sources"][0]["text"]
    with pytest.raises(FileExistsError):
        save_report(report, golden, snapshot, output, 5, {"test": True})


def test_test_set_is_disjoint_from_development():
    from pathlib import Path

    dev = load_golden_set()
    test = load_golden_set(Path(__file__).resolve().parents[1] / "evaluation/test_set.json")
    assert dev["split"] == "development" and test["split"] == "test"
    assert not {q["id"] for q in dev["questions"]} & {q["id"] for q in test["questions"]}
    assert len(test["questions"]) == 10


def test_evaluate_cli_writes_actual_snapshot_and_refuses_overwrite(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from src.pipeline import app

    golden = {
        "questions": [
            {
                "id": "q",
                "kind": "answerable",
                "question": "Qual a taxa?",
                "must_retrieve_any": ["14,25%"],
            }
        ]
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(golden), encoding="utf-8")
    snapshot = [{"id": "1", "payload": {"text": "14,25%"}}]
    answerer = SimpleNamespace(
        qdrant=SimpleNamespace(payload_snapshot=lambda: snapshot),
        retrieve=lambda *args, **kwargs: [source()],
    )
    monkeypatch.setattr("src.pipeline.CopomAnswerer", lambda: answerer)
    monkeypatch.setattr("src.pipeline.runtime_metadata", lambda: {"test": True})
    output = tmp_path / "results.json"
    args = ["evaluate", "--retrieval-only", "--golden-set", str(path), "--output", str(output)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    saved = output.read_bytes()
    body = json.loads(saved)
    assert body["metrics"]["hit_at_1"] == 1
    assert body["corpus"] == snapshot
    assert CliRunner().invoke(app, args).exit_code != 0
    assert output.read_bytes() == saved


def test_demo_prepare_sequences_services_before_pipeline(tmp_path, monkeypatch):
    from scripts import demo

    calls = []
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(demo.sys, "argv", ["demo.py", "--prepare"])
    monkeypatch.setattr(demo.subprocess, "run", lambda command, **kwargs: calls.append(command))
    from unittest.mock import MagicMock

    monkeypatch.setattr(demo.urllib.request, "urlopen", lambda *args, **kwargs: MagicMock())
    demo.main()
    assert calls[-3] == ["docker", "compose", "up", "-d", "qdrant", "phoenix"]
    assert calls[-2][-3:] == ["-m", "src.pipeline", "run-all"]
    assert calls[-1][-2:] == ["-m", "src.web"]


def test_demo_never_indexes_when_docker_fails(tmp_path, monkeypatch):
    from scripts import demo

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "docker":
            raise demo.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(demo.sys, "argv", ["demo.py", "--prepare"])
    monkeypatch.setattr(demo.subprocess, "run", run)
    with pytest.raises(demo.subprocess.CalledProcessError):
        demo.main()
    assert not any("src.pipeline" in call for call in calls)
