"""Global configuration management for the COPOM RAG Lakehouse pipeline.

Uses Pydantic v2 BaseSettings to validate and manage environment variables
and operational parameters across Bronze, Silver, and Gold layers.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Base Paths
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Root directory of the project.",
    )
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data",
        description="Root directory for data lakehouse layers.",
    )
    BRONZE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "bronze",
        description="Directory for raw ingested Bronze JSON data.",
    )
    SILVER_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "silver",
        description="Directory for cleaned and chunked Silver data.",
    )

    # Ingestion / BCB API Configuration
    BCB_API_BASE_URL: str = Field(
        default="https://www.bcb.gov.br/api/servico/sitebcb",
        description="Base URL for Banco Central do Brasil official site API.",
    )
    BCB_ODATA_URL: str = Field(
        default="https://olinda.bcb.gov.br/olinda/servico/Publicacoes/versao/v1/odata/PublicacoesComTexto",
        description="OData endpoint URL for BCB publications.",
    )
    BCB_REQUEST_TIMEOUT: int = Field(
        default=30,
        description="HTTP request timeout in seconds.",
    )
    BCB_MAX_RETRIES: int = Field(
        default=4,
        description="Maximum retry attempts for resilient HTTP requests.",
    )
    BCB_BACKOFF_FACTOR: float = Field(
        default=1.5,
        description="Exponential backoff factor for retries.",
    )
    DEFAULT_INGEST_LIMIT: int = Field(
        default=15,
        description="Default number of recent Copom minutes to fetch.",
    )

    # Text Processing & Chunking Configuration
    CHUNK_SIZE: int = Field(
        default=800,
        description="Target maximum chunk size in tokens.",
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        description="Token overlap between consecutive chunks.",
    )
    TOKENIZER_MODEL: str = Field(
        default="cl100k_base",
        description="Tokenizer model for accurate token calculation (e.g. tiktoken cl100k_base).",
    )

    # Vector Store & Embedding Configuration
    EMBEDDING_PROVIDER: Literal["fastembed", "openai"] = Field(
        default="fastembed",
        description="Embedding provider to use: 'fastembed' (local/free) or 'openai'.",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="intfloat/multilingual-e5-large",
        description=(
            "Embedding model name. The corpus is entirely in Portuguese and retrieval is "
            "asymmetric, so the model must be both multilingual and retrieval-trained. "
            "Benchmarked over 102 chunks from 16 meetings (hit@1 on 10 questions): "
            "multilingual-e5-large 9/10, BAAI/bge-small-en-v1.5 9/10, "
            "paraphrase-multilingual-MiniLM-L12-v2 6/10. The MiniLM is multilingual but "
            "trained for symmetric paraphrase similarity, which costs it 3 questions. "
            "Costs a 2.2 GB model download on first use."
        ),
    )
    EMBEDDING_DIMENSION: int = Field(
        default=1024,
        description=(
            "Vector dimension, which must match the model: 1024 for multilingual-e5-large, "
            "384 for paraphrase-multilingual-MiniLM-L12-v2 and bge-small, 1536 for OpenAI "
            "text-embedding-3-small. Changing it requires recreating the Qdrant collection."
        ),
    )
    OPENAI_API_KEY: str | None = Field(
        default=None,
        description="OpenAI API Key if using OpenAI embeddings.",
    )
    SPARSE_MODEL_NAME: str = Field(
        default="Qdrant/bm25",
        description=(
            "Sparse (lexical) model for hybrid retrieval. BM25 matches literal terms, "
            "which dense vectors are poor at: near-identical passages differing only in "
            "figures are indistinguishable to an embedding but not to BM25."
        ),
    )
    BM25_LANGUAGE: str = Field(
        default="portuguese",
        description=(
            "Language for BM25 stemming and stopword removal. The corpus is in "
            "Portuguese; 'english' would leave Portuguese stopwords in the index."
        ),
    )
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant service URL or ':memory:' for in-memory testing.",
    )
    QDRANT_API_KEY: str | None = Field(
        default=None,
        description="Optional API key for authenticated Qdrant instances.",
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="copom_minutes",
        description="Target Qdrant collection name for Gold layer vector search.",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for generating embeddings and upserting vectors.",
    )

    # Answer Generation (RAG) Configuration
    LLM_BASE_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description=(
            "OpenAI-compatible chat completions endpoint. Defaults to NVIDIA NIM. "
            "Point it elsewhere to switch provider without code changes: "
            "https://api.openai.com/v1, https://openrouter.ai/api/v1, "
            "or http://localhost:11434/v1 for a local Ollama."
        ),
    )
    LLM_MODEL: str = Field(
        default="moonshotai/kimi-k3",
        description="Chat model identifier, as named by the configured endpoint.",
    )
    LLM_API_KEY: str | None = Field(
        default=None,
        description=(
            "API key for LLM_BASE_URL (an 'nvapi-...' key for NVIDIA NIM). Answer "
            "generation fails loudly when unset rather than degrading silently."
        ),
    )
    LLM_TEMPERATURE: float = Field(
        default=0.2,
        description="Sampling temperature. Kept low because answers must track the sources.",
    )

    # Observability & Arize Phoenix Configuration
    ENABLE_PHOENIX: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing with Arize Phoenix.",
    )
    PHOENIX_COLLECTOR_ENDPOINT: str = Field(
        default="http://localhost:6006/v1/traces",
        description="Arize Phoenix OTLP HTTP trace collector endpoint.",
    )
    PHOENIX_PROJECT_NAME: str = Field(
        default="copom-rag-lakehouse",
        description="Project identifier in Arize Phoenix dashboard.",
    )

    def ensure_directories(self) -> None:
        """Create required lakehouse directory hierarchy if not present."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.BRONZE_DIR.mkdir(parents=True, exist_ok=True)
        self.SILVER_DIR.mkdir(parents=True, exist_ok=True)


# Singleton instance for pipeline-wide access
settings = Settings()
