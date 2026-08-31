"""Batch embedding generator with support for FastEmbed (local) and OpenAI.

Provides rate-limiting handling, batching, and vector normalization.
"""

import logging
from typing import Literal

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)


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
        """Initialize the underlying embedding model based on configuration."""
        if self.provider == "fastembed":
            try:
                from fastembed import TextEmbedding

                logger.info(f"Initializing FastEmbed model: {self.model_name}")
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as err:
                logger.warning(
                    f"Could not load FastEmbed model '{self.model_name}': {err}. "
                    "Falling back to default BAAI/bge-small-en-v1.5 or mock."
                )
                try:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding()
                except Exception as inner_err:
                    logger.error(f"FastEmbed initialization failed: {inner_err}")
                    self._model = None

        elif self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY not set. Falling back to FastEmbed provider.")
                self.provider = "fastembed"
                self._initialize_provider()
            else:
                try:
                    import openai

                    self._openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                    logger.info(
                        f"OpenAI Embedding client initialized with model: {self.model_name}"
                    )
                except Exception as err:
                    logger.error(f"Failed to initialize OpenAI client: {err}")
                    self._model = None

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
            # Fallback deterministic vector generator for headless/isolated testing
            return self._generate_fallback_vectors(texts)

        embed = self._model.query_embed if as_query else self._model.passage_embed
        return [[float(x) for x in vec] for vec in embed(texts)]

    def _generate_fallback_vectors(self, texts: list[str]) -> list[list[float]]:
        """Deterministic pseudo-embedding for testing environments without ONNX/C-libs."""
        import hashlib
        import math

        results: list[list[float]] = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            vec = []
            for i in range(self.dimension):
                byte_val = seed[i % len(seed)]
                val = (byte_val / 255.0) * 2.0 - 1.0 + (i * 0.001)
                vec.append(val)
            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / norm for x in vec])
        return results

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

            if self.provider == "openai" and hasattr(self, "_openai_client"):
                batch_vectors = self._embed_openai_batch(batch)
            else:
                batch_vectors = self._embed_fastembed_batch(batch, as_query=as_query)

            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a single search query."""
        results = self.embed_texts([query], as_query=True)
        return results[0] if results else []
