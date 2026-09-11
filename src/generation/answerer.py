"""Grounded answer generation over passages retrieved from the Gold layer.

Closes the RAG loop: retrieval alone returns passages, this turns them into an
answer that cites which passage each claim came from. Talks to any
OpenAI-compatible chat completions endpoint (NVIDIA NIM by default), so
switching provider is configuration rather than code.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.generation.validation import check_citations
from src.vectorstore.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Você é um analista que responde perguntas sobre as atas do Copom, \
o Comitê de Política Monetária do Banco Central do Brasil.

Regras que você deve seguir sem exceção:
1. Responda EXCLUSIVAMENTE com base nos trechos numerados fornecidos. Não recorra a \
conhecimento próprio nem a informações fora dos trechos.
2. Cite a origem de cada afirmação com o número do trecho entre colchetes, como [1] ou [2][3].
3. Se os trechos não contiverem a resposta, diga isso claramente e não especule.
4. Responda em português do Brasil, de forma objetiva e sem repetir a pergunta."""

NO_CONTEXT_ANSWER = (
    "Não encontrei trechos relevantes nas atas indexadas para responder a essa pergunta."
)


class GenerationUnavailableError(RuntimeError):
    """Raised when answer generation cannot run: no credentials or missing extra."""


class AnswerValidationError(GenerationUnavailableError):
    """Preserve a rejected model output for evaluation, without serving it as an answer."""

    def __init__(self, answer: "Answer") -> None:
        self.answer = answer
        super().__init__(
            "A resposta não passou na validação de citações: texto vazio, "
            "afirmação sem fonte ou referência inexistente. Tente reformular a pergunta."
        )


# Rate limits and transient server faults are worth another attempt; a malformed
# request or a bad key is not, and retrying those only wastes quota.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _is_retryable(error: BaseException) -> bool:
    """Decide by HTTP status, so no provider-specific exception import is needed."""
    return getattr(error, "status_code", None) in RETRYABLE_STATUS


@dataclass
class Source:
    """A retrieved passage, numbered as presented to the model and to the user."""

    index: int
    doc_id: str
    chunk_id: str
    nro_reuniao: int
    titulo: str
    data_publicacao: str
    score: float
    text: str
    source_url: str = ""


@dataclass
class Answer:
    """A generated answer together with the passages it was allowed to use."""

    question: str
    text: str
    sources: list[Source] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def llm_span_attributes(self) -> dict[str, Any]:
        """OpenInference attributes so Phoenix renders this as an LLM span."""
        return {
            "openinference.span.kind": "LLM",
            "llm.model_name": self.model,
            "llm.token_count.prompt": self.prompt_tokens,
            "llm.token_count.completion": self.completion_tokens,
            "llm.token_count.total": self.total_tokens,
            "input.value": self.question,
            "output.value": self.text,
        }


def retrieval_span_attributes(sources: list[Source]) -> dict[str, Any]:
    """OpenInference attributes so Phoenix renders retrieval as a RETRIEVER span."""
    attributes: dict[str, Any] = {
        "openinference.span.kind": "RETRIEVER",
        "retrieval.documents.count": len(sources),
    }
    for position, source in enumerate(sources):
        prefix = f"retrieval.documents.{position}.document"
        attributes[f"{prefix}.id"] = source.chunk_id
        attributes[f"{prefix}.score"] = source.score
        attributes[f"{prefix}.content"] = source.text
    return attributes


class CopomAnswerer:
    """Retrieves Copom passages and answers a question strictly from them."""

    def __init__(
        self,
        qdrant: QdrantManager | None = None,
        client: Any | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.qdrant = qdrant if qdrant is not None else QdrantManager()
        self.model = model or settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        self._client = client

    def _get_client(self) -> Any:
        """Build the OpenAI-compatible client, refusing to run without credentials."""
        if self._client is not None:
            return self._client

        if not settings.LLM_API_KEY:
            raise GenerationUnavailableError(
                "LLM_API_KEY is not set, so no answer can be generated. Export it (or add it "
                f"to .env) with a key valid for {settings.LLM_BASE_URL}. Use the 'query' "
                "command for retrieval-only inspection, which needs no credentials."
            )

        try:
            import openai
        except ImportError as err:  # pragma: no cover - exercised by the extras install
            raise GenerationUnavailableError(
                "The 'openai' package is required for answer generation and is an optional "
                'extra. Install it with: pip install -e ".[openai]"'
            ) from err

        self._client = openai.OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
        return self._client

    def retrieve(
        self,
        question: str,
        limit: int = 5,
        filter_year: int | None = None,
        filter_meeting: int | None = None,
    ) -> list[Source]:
        """Search the Gold layer and number the passages for citation."""
        results = self.qdrant.search(
            query=question,
            limit=limit,
            filter_year=filter_year,
            filter_meeting=filter_meeting,
        )

        sources: list[Source] = []
        for position, result in enumerate(results, start=1):
            payload = result.get("payload") or {}
            sources.append(
                Source(
                    index=position,
                    doc_id=payload.get("doc_id", ""),
                    chunk_id=payload.get("chunk_id", ""),
                    nro_reuniao=payload.get("nro_reuniao", 0),
                    titulo=payload.get("titulo", ""),
                    data_publicacao=payload.get("data_publicacao", ""),
                    score=float(result.get("score") or 0.0),
                    text=payload.get("text", ""),
                    source_url=payload.get("source_url", ""),
                )
            )
        return sources

    @staticmethod
    def build_messages(question: str, sources: list[Source]) -> list[dict[str, str]]:
        """Assemble the chat messages, numbering each excerpt so it can be cited."""
        excerpts = "\n\n".join(
            f"[{source.index}] (Reunião {source.nro_reuniao}, {source.data_publicacao})\n"
            f"{source.text}"
            for source in sources
        )
        user_prompt = (
            f"Trechos das atas do Copom:\n\n{excerpts}\n\n"
            f"Pergunta: {question}\n\n"
            "Responda usando apenas os trechos acima, citando os números correspondentes."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _complete(self, client: Any, messages: list[dict[str, str]]) -> Any:
        """Call the chat endpoint, backing off through rate limits.

        Evaluating the golden set fires one request per question in quick
        succession, which is exactly what a free tier throttles.
        """
        return client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

    def generate(self, question: str, sources: list[Source]) -> Answer:
        """Produce an answer grounded in the given passages.

        Args:
            question: The user's natural language question.
            sources: Passages from :meth:`retrieve`, already numbered.

        Returns:
            The answer plus the passages the model was allowed to use.

        Raises:
            GenerationUnavailableError: No API key configured, or the optional
                'openai' extra is not installed.
        """
        if not sources:
            logger.info("No passages retrieved; skipping the model call entirely.")
            return Answer(question=question, text=NO_CONTEXT_ANSWER, model=self.model)

        client = self._get_client()
        response = self._complete(client, self.build_messages(question, sources))

        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        answer = Answer(
            question=question,
            text=text,
            sources=sources,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

        if not check_citations(text, sources).valid:
            raise AnswerValidationError(answer)
        return answer
