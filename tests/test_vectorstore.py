"""Unit and integration tests for Gold layer vector embeddings and Qdrant operations."""

import pytest

from src.processing.chunker import AtaChunker
from src.vectorstore.embeddings import EmbeddingGenerator
from src.vectorstore.qdrant_manager import QdrantManager


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
