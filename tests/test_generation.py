"""Unit tests for grounded answer generation over retrieved Copom passages."""

from types import SimpleNamespace

import pytest

from src.config import settings
from src.generation.answerer import (
    Answer,
    CopomAnswerer,
    GenerationUnavailableError,
    Source,
    retrieval_span_attributes,
)


def make_source(index: int = 1, text: str = "O Copom decidiu manter a Selic.", nro: int = 280):
    """Build a retrieved passage without touching Qdrant."""
    return Source(
        index=index,
        doc_id=f"copom_{nro}",
        chunk_id=f"copom_{nro}_chunk_001",
        nro_reuniao=nro,
        titulo=f"{nro}a Reuniao",
        data_publicacao="2026-08-11",
        score=0.9,
        text=text,
    )


class FakeLLMClient:
    """Stand-in for the OpenAI-compatible client, counting calls."""

    def __init__(self, content: str = "resposta", prompt_tokens: int = 10, completion: int = 5):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion
        self.calls = 0
        self.last_kwargs: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            ),
        )


class TestPromptBuilding:
    """The prompt is what keeps the answer grounded, so it is asserted directly."""

    def test_excerpts_are_numbered_for_citation(self):
        sources = [make_source(1, "Primeiro trecho.", 280), make_source(2, "Segundo trecho.", 279)]

        messages = CopomAnswerer.build_messages("O que foi decidido?", sources)

        user = messages[-1]["content"]
        assert "[1]" in user and "Primeiro trecho." in user
        assert "[2]" in user and "Segundo trecho." in user
        assert "O que foi decidido?" in user

    def test_system_prompt_forbids_answering_outside_the_excerpts(self):
        messages = CopomAnswerer.build_messages("pergunta", [make_source()])

        assert messages[0]["role"] == "system"
        system = messages[0]["content"]
        assert "EXCLUSIVAMENTE" in system
        assert "[1]" in system, "o formato de citacao precisa ser ensinado ao modelo"


class TestCredentials:
    """Generation must fail loudly rather than degrade, mirroring the embedding lesson."""

    def test_generate_without_api_key_raises_actionable_error(self, monkeypatch):
        monkeypatch.setattr(settings, "LLM_API_KEY", None)
        answerer = CopomAnswerer(qdrant=object())

        with pytest.raises(GenerationUnavailableError, match="LLM_API_KEY"):
            answerer.generate("pergunta", [make_source()])


class TestGeneration:
    def test_returns_answer_text_sources_and_token_usage(self):
        client = FakeLLMClient(content="O Copom manteve a Selic [1].", prompt_tokens=120)
        answerer = CopomAnswerer(qdrant=object(), client=client, model="fake/model")
        sources = [make_source()]

        answer = answerer.generate("por que manteve?", sources)

        assert isinstance(answer, Answer)
        assert answer.text == "O Copom manteve a Selic [1]."
        assert answer.sources == sources
        assert answer.model == "fake/model"
        assert answer.prompt_tokens == 120
        assert answer.completion_tokens == 5
        assert client.last_kwargs["model"] == "fake/model"

    def test_rate_limited_call_is_retried(self):
        """Evaluating the golden set fires many requests fast; 429 must not abort it."""

        class Throttled(Exception):
            status_code = 429

        class FlakyClient(FakeLLMClient):
            attempts = 0

            def _create(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise Throttled("Too Many Requests")
                return super()._create(**kwargs)

        client = FlakyClient(content="respondeu na segunda tentativa [1]")
        answerer = CopomAnswerer(qdrant=object(), client=client)
        answerer._complete.retry.wait = lambda *a, **kw: 0  # não dorme no teste

        answer = answerer.generate("pergunta", [make_source()])

        assert client.attempts == 2
        assert answer.text == "respondeu na segunda tentativa [1]"

    def test_non_retryable_error_is_not_retried(self):
        """A malformed request or bad key must fail at once instead of burning quota."""

        class BadRequest(Exception):
            status_code = 400

        class BrokenClient(FakeLLMClient):
            def _create(self, **kwargs):
                self.calls += 1
                raise BadRequest("Bad Request")

        client = BrokenClient()
        answerer = CopomAnswerer(qdrant=object(), client=client)

        with pytest.raises(BadRequest):
            answerer.generate("pergunta", [make_source()])
        assert client.calls == 1

    def test_without_retrieved_passages_the_model_is_never_called(self):
        client = FakeLLMClient()
        answerer = CopomAnswerer(qdrant=object(), client=client)

        answer = answerer.generate("pergunta", [])

        assert client.calls == 0
        assert answer.sources == []
        assert "não" in answer.text.lower()

    def test_retrieve_numbers_passages_from_qdrant_payloads(self):
        class FakeQdrant:
            def search(self, query, limit, filter_year=None, filter_meeting=None):
                return [
                    {
                        "id": "uuid-a",
                        "score": 0.81,
                        "payload": {
                            "doc_id": "copom_280",
                            "chunk_id": "copom_280_chunk_003",
                            "nro_reuniao": 280,
                            "titulo": "280a Reuniao",
                            "data_publicacao": "2026-08-11",
                            "text": "Trecho recuperado.",
                        },
                    }
                ]

        answerer = CopomAnswerer(qdrant=FakeQdrant())

        sources = answerer.retrieve("pergunta", limit=1)

        assert [s.index for s in sources] == [1]
        assert sources[0].doc_id == "copom_280"
        assert sources[0].nro_reuniao == 280
        assert sources[0].text == "Trecho recuperado."


class TestObservabilityAttributes:
    """Phoenix only renders a RAG trace if the OpenInference conventions are emitted."""

    def test_retrieval_attributes_follow_openinference_conventions(self):
        attrs = retrieval_span_attributes(
            [make_source(1, "Trecho A."), make_source(2, "Trecho B.")]
        )

        assert attrs["openinference.span.kind"] == "RETRIEVER"
        assert attrs["retrieval.documents.0.document.id"] == "copom_280_chunk_001"
        assert attrs["retrieval.documents.0.document.content"] == "Trecho A."
        assert attrs["retrieval.documents.1.document.content"] == "Trecho B."
        assert attrs["retrieval.documents.0.document.score"] == 0.9

    def test_answer_exposes_llm_span_attributes(self):
        answer = Answer(
            question="q",
            text="resposta",
            sources=[make_source()],
            model="fake/model",
            prompt_tokens=100,
            completion_tokens=20,
        )

        attrs = answer.llm_span_attributes()

        assert attrs["openinference.span.kind"] == "LLM"
        assert attrs["llm.model_name"] == "fake/model"
        assert attrs["llm.token_count.prompt"] == 100
        assert attrs["llm.token_count.completion"] == 20
        assert attrs["llm.token_count.total"] == 120
        assert attrs["output.value"] == "resposta"
