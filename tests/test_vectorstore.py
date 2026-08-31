"""Unit and integration tests for Gold layer vector embeddings and Qdrant operations."""

from unittest.mock import patch

import pytest
from qdrant_client.http import models

from src.config import settings
from src.processing.chunker import AtaChunker
from src.vectorstore.embeddings import EmbeddingGenerator, EmbeddingUnavailableError
from src.vectorstore.qdrant_manager import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantManager,
)


class TestEmbeddings:
    """Test suite for embedding generation and batching."""

    def test_embedding_generator_output(self):
        """Verify embedding generator produces expected vector shape."""
        generator = EmbeddingGenerator(provider="fastembed")
        texts = ["Taxa Selic mantida em 10.50%", "Balanço de riscos para a inflação"]
        vectors = generator.embed_texts(texts)

        assert len(vectors) == 2
        assert len(vectors[0]) == generator.dimension
        assert isinstance(vectors[0][0], float)

    def test_embed_single_query(self):
        """Verify single query vector generation."""
        generator = EmbeddingGenerator(provider="fastembed")
        query_vec = generator.embed_query("projeções de inflação")
        assert len(query_vec) == generator.dimension

    def test_unloadable_model_fails_instead_of_substituting_another(self):
        """Silently swapping in a different model would degrade search with no error."""
        with (
            patch("fastembed.TextEmbedding", side_effect=RuntimeError("modelo inexistente")),
            pytest.raises(EmbeddingUnavailableError, match="nao-existe/modelo"),
        ):
            EmbeddingGenerator(provider="fastembed", model_name="nao-existe/modelo")

    def test_openai_provider_without_key_fails_instead_of_switching(self, monkeypatch):
        """Asking for OpenAI and silently getting FastEmbed hides a config error."""
        monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

        with pytest.raises(EmbeddingUnavailableError, match="OPENAI_API_KEY"):
            EmbeddingGenerator(provider="openai")

    def test_embedding_without_a_loaded_model_raises(self):
        """No model must mean no vectors -- never pseudo-vectors derived from a hash."""
        generator = EmbeddingGenerator(provider="fastembed")
        generator._model = None

        with pytest.raises(EmbeddingUnavailableError):
            generator.embed_texts(["texto de uma ata"])

    def test_query_and_passage_take_distinct_fastembed_hooks(self):
        """Retrieval is asymmetric: prefix-aware models need the query/passage split."""
        calls: list[str] = []

        class FakeModel:
            def query_embed(self, texts, **kwargs):
                calls.append("query")
                return [[0.1] * 8 for _ in texts]

            def passage_embed(self, texts, **kwargs):
                calls.append("passage")
                return [[0.2] * 8 for _ in texts]

        generator = EmbeddingGenerator(provider="fastembed")
        generator._model = FakeModel()

        generator.embed_texts(["Trecho de uma ata do Copom."])
        generator.embed_query("qual foi a decisão sobre a Selic?")

        assert calls == ["passage", "query"]


class TestQdrantManager:
    """Test suite for Qdrant vector index management and idempotent upsert."""

    def test_deterministic_point_id_generation(self):
        """Ensure UUIDv5 point ID generation is deterministic and unique."""
        id1 = QdrantManager.generate_point_id("copom_280", "copom_280_chunk_001")
        id2 = QdrantManager.generate_point_id("copom_280", "copom_280_chunk_001")
        id3 = QdrantManager.generate_point_id("copom_280", "copom_280_chunk_002")

        # Same inputs produce identical UUID
        assert id1 == id2
        # Different chunk produces different UUID
        assert id1 != id3

    def test_collection_carries_both_dense_and_sparse_vectors(
        self, in_memory_qdrant: QdrantManager
    ):
        """Hybrid retrieval needs both halves stored in the same collection."""
        in_memory_qdrant.ensure_collection()

        params = in_memory_qdrant.client.get_collection(
            in_memory_qdrant.collection_name
        ).config.params

        assert DENSE_VECTOR_NAME in params.vectors
        assert SPARSE_VECTOR_NAME in params.sparse_vectors
        # Without the IDF modifier BM25 would ignore term rarity altogether.
        assert params.sparse_vectors[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF

    def test_bm25_separates_passages_that_differ_only_in_figures(
        self, in_memory_qdrant: QdrantManager, sample_bronze_record
    ):
        """The case dense retrieval could not solve: same prose, different numbers."""
        base = AtaChunker(chunk_size=100).process_record(sample_bronze_record).chunks[0]
        template = "As expectativas de inflação apuradas pela pesquisa Focus situam-se em {}."
        variants = []
        for index, figures in enumerate(("4,9% e 4,0%", "5,0% e 4,2%", "5,5% e 4,5%")):
            variant = base.model_copy(deep=True)
            variant.text = template.format(figures)
            variant.chunk_id = f"copom_280_chunk_{index:03d}"
            variant.metadata.chunk_id = variant.chunk_id
            variants.append(variant)
        in_memory_qdrant.upsert_chunks(variants)

        results = in_memory_qdrant.search("expectativas do Focus de 5,0% e 4,2%", limit=3)

        assert "5,0% e 4,2%" in results[0]["payload"]["text"]

    def test_legacy_collection_without_named_vectors_is_rejected(
        self, in_memory_qdrant: QdrantManager
    ):
        """A pre-hybrid collection must say so, not fail with an opaque Qdrant error."""
        in_memory_qdrant.client.create_collection(
            collection_name=in_memory_qdrant.collection_name,
            vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
        )

        with pytest.raises(ValueError, match="predates hybrid search"):
            in_memory_qdrant.ensure_collection(vector_size=8)

    def test_search_on_missing_collection_returns_nothing(self, in_memory_qdrant: QdrantManager):
        """Nothing indexed yet is an ordinary state, not a traceback."""
        assert in_memory_qdrant.search("qualquer pergunta", limit=3) == []

    def test_rejects_collection_built_for_another_model(self, in_memory_qdrant: QdrantManager):
        """Swapping to a model of another width must fail loudly, not half-index."""
        in_memory_qdrant.ensure_collection()
        in_memory_qdrant.embedding_generator.dimension = 8

        with pytest.raises(ValueError, match="Recreate the collection"):
            in_memory_qdrant.ensure_collection()

    def test_in_memory_collection_creation_and_upsert(
        self, in_memory_qdrant: QdrantManager, sample_bronze_record
    ):
        """Test full vector indexing lifecycle using an in-memory Qdrant instance."""
        chunker = AtaChunker(chunk_size=100)
        silver_doc = chunker.process_record(sample_bronze_record)

        # Upsert chunks into in-memory Qdrant
        summary = in_memory_qdrant.upsert_chunks(silver_doc.chunks, batch_size=2)

        assert summary.total_chunks == len(silver_doc.chunks)
        assert summary.upserted_points == len(silver_doc.chunks)
        assert summary.failed == 0

        # Verify point count
        total_points = in_memory_qdrant.count_points()
        assert total_points == len(silver_doc.chunks)

        # Re-upserting should be completely idempotent
        summary_reupsert = in_memory_qdrant.upsert_chunks(silver_doc.chunks, batch_size=2)
        assert summary_reupsert.upserted_points == len(silver_doc.chunks)
        # Total points should remain the exact same (no duplicates)
        assert in_memory_qdrant.count_points() == total_points

    def test_semantic_search_with_filters(
        self, in_memory_qdrant: QdrantManager, sample_bronze_record
    ):
        """Test semantic search retrieval and metadata filtering."""
        chunker = AtaChunker(chunk_size=100)
        silver_doc = chunker.process_record(sample_bronze_record)
        in_memory_qdrant.upsert_chunks(silver_doc.chunks)

        # Perform semantic search
        results = in_memory_qdrant.search("taxa Selic e balanço de riscos", limit=2)
        assert len(results) >= 1
        assert "score" in results[0]
        assert "payload" in results[0]
        assert results[0]["payload"]["doc_id"] == "copom_280"

        # Search with matching filter
        results_filtered = in_memory_qdrant.search(
            "inflação", limit=2, filter_year=2026, filter_meeting=280
        )
        assert len(results_filtered) >= 1

        # Search with non-matching filter
        results_empty = in_memory_qdrant.search("inflação", limit=2, filter_year=1999)
        assert len(results_empty) == 0
