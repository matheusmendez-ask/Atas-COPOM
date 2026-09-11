"""Local web interface over the RAG pipeline.

A thin shell around :class:`CopomAnswerer`: it adds no retrieval or generation
logic of its own. What it does add is making the grounding visible, which is the
part a plain chat window hides. Every answer ships with the passages it was
allowed to use, and a refusal is rendered as a first-class outcome rather than
as a failure.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient

from src.config import settings
from src.evaluation import looks_like_refusal
from src.generation.answerer import CopomAnswerer, GenerationUnavailableError
from src.generation.validation import check_citations
from src.vectorstore.filters import infer_meeting

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    """A question from the browser."""

    question: str = Field(..., min_length=3, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)
    meeting: int | None = Field(default=None, ge=1, le=999)
    year: int | None = Field(default=None, ge=1990, le=2100)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class SourceOut(BaseModel):
    """One retrieved passage, as the interface displays it."""

    index: int
    doc_id: str
    nro_reuniao: int
    data_publicacao: str
    score: float
    text: str
    source_url: str = ""


class AskResponse(BaseModel):
    """An answer plus everything needed to audit it."""

    question: str
    answer: str
    refused: bool = Field(
        ...,
        description="The corpus does not answer this. A correct outcome, not an error.",
    )
    sources: list[SourceOut] = Field(default_factory=list)
    model: str = ""
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    citation_check: dict[str, Any] = Field(default_factory=dict)
    applied_meeting: int | None = None
    applied_year: int | None = None


class StatusResponse(BaseModel):
    """What the index currently holds, so the interface can show its own limits."""

    collection: str
    indexed_chunks: int | None
    index_state: str = "ready"
    embedding_model: str
    llm_model: str
    generation_ready: bool


def create_app(answerer: CopomAnswerer | None = None) -> FastAPI:
    """Build the application.

    Args:
        answerer: Injected in tests; built lazily in production so importing this
            module never loads a 2.2 GB embedding model.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(title="Atas do Copom", docs_url="/api/docs", redoc_url=None)
    state: dict[str, Any] = {"answerer": answerer}

    def get_answerer() -> CopomAnswerer:
        if state["answerer"] is None:
            state["answerer"] = CopomAnswerer()
        return state["answerer"]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        """Report index size so the interface can state what it does and does not cover."""
        index_state = "ready"
        try:
            if state["answerer"] is not None:
                manager = state["answerer"].qdrant
                indexed = (
                    manager.client.count(manager.collection_name, exact=True).count
                    if hasattr(manager, "client")
                    else manager.count_points()
                )
            else:
                # Inspect the service without loading the embedding models.
                with QdrantClient(
                    url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=2
                ) as client:
                    indexed = (
                        client.count(settings.QDRANT_COLLECTION_NAME, exact=True).count
                        if client.collection_exists(settings.QDRANT_COLLECTION_NAME)
                        else 0
                    )
            if indexed == 0:
                index_state = "empty"
        except Exception:
            indexed = None
            index_state = "unavailable"
        return StatusResponse(
            collection=settings.QDRANT_COLLECTION_NAME,
            indexed_chunks=indexed,
            index_state=index_state,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            llm_model=settings.LLM_MODEL,
            generation_ready=bool(settings.LLM_API_KEY),
        )

    @app.post("/api/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        """Answer a question strictly from the indexed minutes."""
        started = time.perf_counter()
        try:
            answerer = get_answerer()
            sources = answerer.retrieve(
                request.question,
                limit=request.limit,
                filter_year=request.year,
                filter_meeting=request.meeting,
            )
            answer = answerer.generate(request.question, sources)
        except GenerationUnavailableError as err:
            # Missing credentials is a configuration problem the operator must
            # see, not something to paper over with an empty answer.
            raise HTTPException(status_code=503, detail=str(err)) from err
        except Exception as err:
            logger.error("RAG request failed: %s", type(err).__name__)
            raise HTTPException(
                status_code=503,
                detail="Consulta indisponível. Confira o índice e o serviço de geração e tente novamente.",
            ) from err

        return AskResponse(
            question=request.question,
            citation_check=asdict(check_citations(answer.text, sources)),
            applied_meeting=request.meeting
            if request.meeting is not None
            else infer_meeting(request.question),
            applied_year=request.year,
            answer=answer.text,
            refused=not sources or looks_like_refusal(answer.text),
            sources=[
                SourceOut(
                    index=s.index,
                    doc_id=s.doc_id,
                    nro_reuniao=s.nro_reuniao,
                    data_publicacao=s.data_publicacao,
                    score=s.score,
                    text=s.text,
                    source_url=official_source_url(s.source_url),
                )
                for s in sources
            ],
            model=answer.model,
            total_tokens=answer.total_tokens,
            elapsed_seconds=round(time.perf_counter() - started, 2),
        )

    return app


def official_source_url(url: str) -> str:
    """Link only to official HTTPS sources."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        return (
            url
            if parsed.scheme == "https"
            and (hostname == "bcb.gov.br" or hostname.endswith(".bcb.gov.br"))
            and not parsed.username
            and not parsed.password
            else ""
        )
    except ValueError:
        return ""


def build_default_app() -> FastAPI:
    """Start the page even when Qdrant is unavailable; expose status separately."""
    return create_app()
