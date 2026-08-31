"""Local web interface over the RAG pipeline.

A thin shell around :class:`CopomAnswerer`: it adds no retrieval or generation
logic of its own. What it does add is making the grounding visible, which is the
part a plain chat window hides. Every answer ships with the passages it was
allowed to use, and a refusal is rendered as a first-class outcome rather than
as a failure.
"""

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.evaluation import looks_like_refusal
from src.generation.answerer import CopomAnswerer, GenerationUnavailableError
from src.vectorstore.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    """A question from the browser."""

    question: str = Field(..., min_length=3, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)


class SourceOut(BaseModel):
    """One retrieved passage, as the interface displays it."""

    index: int
    doc_id: str
    nro_reuniao: int
    data_publicacao: str
    score: float
    text: str


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


class StatusResponse(BaseModel):
    """What the index currently holds, so the interface can show its own limits."""

    collection: str
    indexed_chunks: int
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
        try:
            indexed = get_answerer().qdrant.count_points()
        except Exception as err:  # pragma: no cover - depends on a live Qdrant
            logger.warning(f"Could not count indexed points: {err}")
            indexed = 0
        return StatusResponse(
            collection=settings.QDRANT_COLLECTION_NAME,
            indexed_chunks=indexed,
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            llm_model=settings.LLM_MODEL,
            generation_ready=bool(settings.LLM_API_KEY),
        )

    @app.post("/api/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        """Answer a question strictly from the indexed minutes."""
        started = time.perf_counter()
        answerer = get_answerer()

        sources = answerer.retrieve(request.question, limit=request.limit)
        try:
            answer = answerer.generate(request.question, sources)
        except GenerationUnavailableError as err:
            # Missing credentials is a configuration problem the operator must
            # see, not something to paper over with an empty answer.
            raise HTTPException(status_code=503, detail=str(err)) from err

        return AskResponse(
            question=request.question,
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
                )
                for s in sources
            ],
            model=answer.model,
            total_tokens=answer.total_tokens,
            elapsed_seconds=round(time.perf_counter() - started, 2),
        )

    return app


def build_default_app() -> FastAPI:
    """Entry point for `python -m src.web`, with a Qdrant reachability check."""
    manager = QdrantManager()
    if not manager.client.collection_exists(manager.collection_name):
        logger.warning(
            f"Collection '{manager.collection_name}' does not exist yet. Run "
            "'python -m src.pipeline run-all' before asking anything."
        )
    return create_app(CopomAnswerer(qdrant=manager))
