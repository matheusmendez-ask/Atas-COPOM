"""Batch embedding generator with support for FastEmbed (local) and OpenAI.

Provides rate-limiting handling, batching, and vector normalization.
"""

import logging
from typing import Literal

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the configured embedding provider cannot produce vectors."""


class EmbeddingGenerator:
    """Generates dense vector embeddings for text chunks in batches."""

    def __init__(
        self,
        provider: Literal["fastembed", "openai"] | None = None,
        model_name: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.provider = provider or settings.EMBEDDING_PROVIDER
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model = None

        self._initialize_provider()

    def _initialize_provider(self) -> None:
        """Load the configured provider, refusing to substitute a different one.

        A wrong-but-working embedder is worse than none: the pipeline would index
        vectors unrelated to the configured model, and the only symptom -- poor
        search results -- points away from the real cause.
        """
        if self.provider == "openai":
            self._initialize_openai()
        else:
            self._initialize_fastembed()

    def _initialize_fastembed(self) -> None:
        """Load the local ONNX model named by EMBEDDING_MODEL_NAME, or fail."""
        try:
            from fastembed import TextEmbedding
        except ImportError as err:  # pragma: no cover - fastembed is a core dependency
            raise EmbeddingUnavailableError(
                "The 'fastembed' package is required for local embeddings. "
                'Reinstall the project: pip install -e ".[dev]"'
            ) from err

        logger.info(f"Initializing FastEmbed model: {self.model_name}")
        try:
            self._model = TextEmbedding(model_name=self.model_name)
        except Exception as err:
            raise EmbeddingUnavailableError(
                f"Could not load the FastEmbed model '{self.model_name}': {err}. Fix "
                "EMBEDDING_MODEL_NAME (and EMBEDDING_DIMENSION to match it) instead of "
                "indexing with a model other than the one configured."
            ) from err

    def _initialize_openai(self) -> None:
        """Build the OpenAI embeddings client, or fail with what is missing."""
        if not settings.OPENAI_API_KEY:
            raise EmbeddingUnavailableError(
                "EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set. Set the key, "
                "or set EMBEDDING_PROVIDER=fastembed to embed locally."
            )

        try:
            import openai
        except ImportError as err:  # pragma: no cover - exercised by the extras install
            raise EmbeddingUnavailableError(
                "The 'openai' package is an optional extra required by "
                'EMBEDDING_PROVIDER=openai. Install it with: pip install -e ".[openai]"'
            ) from err

        try:
            self._openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        except Exception as err:
            raise EmbeddingUnavailableError(
                f"Could not initialize the OpenAI embeddings client: {err}"
            ) from err
        logger.info(f"OpenAI embedding client initialized with model: {self.model_name}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    def _embed_openai_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI Embeddings API with retry."""
        response = self._openai_client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        return [[float(x) for x in item.embedding] for item in response.data]

    def _embed_fastembed_batch(
        self, texts: list[str], *, as_query: bool = False
    ) -> list[list[float]]:
        """Embed a batch of texts using local FastEmbed ONNX runtime.

        Retrieval is asymmetric: a short question is matched against long passages.
        ``query_embed``/``passage_embed`` are model-specific hooks -- e5-style models
        require "query: "/"passage: " prefixes, which FastEmbed applies here, while
        symmetric models treat both as a plain embed. Going through them keeps the
        choice of model a pure configuration change.
        """
        if self._model is None:
            raise EmbeddingUnavailableError(
                "No embedding model is loaded, so no vector can be produced. "
                "Check EMBEDDING_MODEL_NAME and EMBEDDING_PROVIDER."
            )

        embed = self._model.query_embed if as_query else self._model.passage_embed
        return [[float(x) for x in vec] for vec in embed(texts)]

    def embed_texts(self, texts: list[str], *, as_query: bool = False) -> list[list[float]]:
        """Generate dense vector embeddings for a list of strings in batches.

        Args:
            texts: List of text strings to embed.
            as_query: Embed as search queries rather than as indexable passages.

        Returns:
            List of vector embeddings (list of floats).
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug(
                f"Generating embeddings for batch of {len(batch)} items ({i}/{len(texts)})"
            )

            if self.provider == "openai":
                batch_vectors = self._embed_openai_batch(batch)
            else:
                batch_vectors = self._embed_fastembed_batch(batch, as_query=as_query)

            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a single search query."""
        results = self.embed_texts([query], as_query=True)
        return results[0] if results else []
