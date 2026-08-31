"""Unit and integration tests for Silver layer processing and chunking."""

from src.processing.chunker import AtaChunker, TokenCounter
from src.processing.cleaner import AtaCleaner
from src.processing.schemas import ChunkPayload


class TestCleaner:
    """Test suite for HTML sanitization and text normalization."""

    def test_clean_html_strips_tags_and_preserves_structure(self, sample_raw_html):
        """Ensure tags are removed but section titles and paragraph structure remain."""
        cleaner = AtaCleaner()
        cleaned = cleaner.clean_html(sample_raw_html)

        assert "<div" not in cleaned
        assert "<p" not in cleaned
        assert "A) Atualização da conjuntura" in cleaned
        assert "taxa Selic em 10,50% a.a." in cleaned
        # Verify no excessive blank lines
        assert "\n\n\n" not in cleaned

    def test_normalize_spaces_and_entities(self):
        """Test normalization of whitespace and special entities."""
        cleaner = AtaCleaner()
        raw = "<p>O&nbsp;&nbsp;Copom  decidiu &atilde; manter a meta.</p>"
        cleaned = cleaner.clean_html(raw)
        assert cleaned == "O Copom decidiu ã manter a meta."


class TestChunker:
    """Test suite for token counting, recursive chunking, and metadata enrichment."""

    def test_token_counter(self):
        """Verify token counting functionality."""
        counter = TokenCounter()
        text = "O Banco Central do Brasil divulgou a ata do Copom."
        count = counter.count(text)
        assert count > 0

    def test_chunker_processing_and_enrichment(self, sample_bronze_record, temp_lakehouse_dirs):
        """Test transforming a Bronze record into Silver document with rich chunks."""
        silver_dir = temp_lakehouse_dirs["silver"]
        chunker = AtaChunker(chunk_size=50, chunk_overlap=10, silver_dir=silver_dir)

        doc = chunker.process_record(sample_bronze_record)

        assert doc.doc_id == "copom_280"
        assert doc.ano == 2026
        assert doc.mes == 8
        assert len(doc.chunks) >= 1

        # Inspect first chunk metadata
        first_chunk = doc.chunks[0]
        assert isinstance(first_chunk, ChunkPayload)
        assert first_chunk.metadata.doc_id == "copom_280"
        assert first_chunk.metadata.chunk_index == 0
        assert first_chunk.metadata.total_chunks == len(doc.chunks)
        assert first_chunk.metadata.nro_reuniao == 280
        assert first_chunk.metadata.ano == 2026
        assert first_chunk.metadata.mes == 8
        assert len(first_chunk.metadata.chunk_hash) == 64
        assert first_chunk.metadata.token_count > 0

    def test_save_and_load_silver_documents(self, sample_bronze_record, temp_lakehouse_dirs):
        """Test writing to partitioned Silver disk storage and reloading."""
        silver_dir = temp_lakehouse_dirs["silver"]
        chunker = AtaChunker(chunk_size=100, silver_dir=silver_dir)

        doc = chunker.process_record(sample_bronze_record)
        saved_path = chunker.save_silver_document(doc)

        assert saved_path.exists()
        assert "year=2026/month=08" in str(saved_path).replace("\\", "/")

        loaded_docs = chunker.load_all_silver_documents()
        assert len(loaded_docs) == 1
        assert loaded_docs[0].doc_id == "copom_280"
        assert len(loaded_docs[0].chunks) == len(doc.chunks)
