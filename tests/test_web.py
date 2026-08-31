"""Tests for the local web interface."""

import pytest
from fastapi.testclient import TestClient

from src.generation.answerer import Answer, GenerationUnavailableError, Source
from src.web.app import create_app


def make_source(index: int = 1, text: str = "O Copom decidiu manter a Selic.") -> Source:
    return Source(
        index=index,
        doc_id="copom_280",
        chunk_id=f"copom_280_chunk_{index:03d}",
        nro_reuniao=280,
        titulo="280a Reuniao",
        data_publicacao="2026-08-11",
        score=0.87,
        text=text,
    )


class FakeAnswerer:
    """Stands in for CopomAnswerer so no model or Qdrant is needed."""

    def __init__(self, sources=None, text="O Copom manteve a taxa [1].", error=None):
        self.sources = sources if sources is not None else [make_source()]
        self.text = text
        self.error = error
        self.qdrant = type("Q", (), {"count_points": staticmethod(lambda: 575)})()

    def retrieve(self, question, limit=5, filter_year=None, filter_meeting=None):
        return self.sources[:limit]

    def generate(self, question, sources):
        if self.error:
            raise self.error
        return Answer(question=question, text=self.text, sources=sources, model="fake/model")


@pytest.fixture
def client():
    return TestClient(create_app(FakeAnswerer()))


class TestStatus:
    def test_reports_index_size_so_the_page_can_state_its_limits(self, client):
        body = client.get("/api/status").json()

        assert body["indexed_chunks"] == 575
        assert body["collection"]
        assert "generation_ready" in body


class TestAsk:
    def test_answer_carries_the_passages_it_was_allowed_to_use(self, client):
        body = client.post("/api/ask", json={"question": "o que decidiram?"}).json()

        assert body["answer"] == "O Copom manteve a taxa [1]."
        assert body["refused"] is False
        assert len(body["sources"]) == 1
        assert body["sources"][0]["nro_reuniao"] == 280
        assert body["sources"][0]["text"]

    def test_refusal_is_flagged_as_an_outcome_not_an_error(self):
        app = create_app(FakeAnswerer(text="Os trechos fornecidos não contêm essa informação."))

        response = TestClient(app).post("/api/ask", json={"question": "e sobre bitcoin?"})

        assert response.status_code == 200, "recusar não é falhar"
        assert response.json()["refused"] is True

    def test_no_passages_means_refused_without_inventing(self):
        app = create_app(FakeAnswerer(sources=[], text="qualquer coisa"))

        body = TestClient(app).post("/api/ask", json={"question": "pergunta"}).json()

        assert body["refused"] is True
        assert body["sources"] == []

    def test_missing_credentials_surfaces_as_service_unavailable(self):
        app = create_app(FakeAnswerer(error=GenerationUnavailableError("LLM_API_KEY is not set")))

        response = TestClient(app).post("/api/ask", json={"question": "pergunta"})

        assert response.status_code == 503
        assert "LLM_API_KEY" in response.json()["detail"]

    @pytest.mark.parametrize("question", ["", "ab"])
    def test_too_short_a_question_is_rejected_before_any_model_call(self, client, question):
        assert client.post("/api/ask", json={"question": question}).status_code == 422


class TestPage:
    def test_root_serves_the_interface(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "Atas do Copom" in response.text
        assert "/api/ask" in response.text
