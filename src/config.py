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
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name.",
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="Vector dimension for embeddings (384 for bge-small, 1536 for text-embedding-3-small).",
    )
    OPENAI_API_KEY: str | None = Field(
        default=None,
        description="OpenAI API Key if using OpenAI embeddings.",
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
