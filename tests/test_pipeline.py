"""Unit and integration tests for CLI pipeline orchestrator."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.ingestion.collector import IngestionSummary
from src.pipeline import app
from src.processing.schemas import SilverDocument
from src.vectorstore.qdrant_manager import UpsertSummary

runner = CliRunner()


class TestPipelineCLI:
    """Test suite for Typer CLI commands."""

    def test_cli_ingest_command(self):
        """Test 'ingest' command with mocked collector."""
        mock_summary = IngestionSummary(
            total_catalog_items=2,
            new_ingested=2,
            skipped_existing=0,
            failed=0,
        )
        with patch("src.pipeline.BronzeCollector.run", return_value=mock_summary):
            result = runner.invoke(app, ["ingest", "--limit", "2"])
            assert result.exit_code == 0
            assert "Bronze Layer Ingestion Summary" in result.stdout
            assert "Total Catalog Items Found" in result.stdout

    def test_cli_transform_command(self, sample_bronze_record):
        """Test 'transform' command with mock Bronze documents."""
        mock_silver_doc = SilverDocument(
            doc_id="copom_280",
            nro_reuniao=280,
            titulo="280ª Reunião",
            data_publicacao="2026-08-11",
            ano=2026,
            mes=8,
            clean_text="Clean text",
            content_hash="abc",
            total_tokens=100,
            chunks=[],
        )
        with (
            patch(
                "src.pipeline.BronzeCollector.load_all_records", return_value=[sample_bronze_record]
            ),
            patch("src.pipeline.AtaChunker.process_all_bronze", return_value=[mock_silver_doc]),
        ):
            result = runner.invoke(app, ["transform"])
            assert result.exit_code == 0
            assert "Silver Layer Processing Summary" in result.stdout

    def test_cli_index_command(self):
        """Test 'index' command with mock Silver documents and Qdrant."""
        mock_upsert_summary = UpsertSummary(
            collection_name="copom_minutes",
            total_chunks=5,
            upserted_points=5,
            batch_count=1,
            failed=0,
        )
        mock_doc = MagicMock()
        mock_doc.chunks = [MagicMock()]

        with (
            patch("src.pipeline.AtaChunker.load_all_silver_documents", return_value=[mock_doc]),
            patch("src.pipeline.QdrantManager.upsert_chunks", return_value=mock_upsert_summary),
            patch("src.pipeline.QdrantManager.count_points", return_value=5),
        ):
            result = runner.invoke(app, ["index", "--batch-size", "10"])
            assert result.exit_code == 0
            assert "Gold Layer Vector Indexing Summary" in result.stdout

    def test_cli_query_command(self):
        """Test 'query' command with mocked search results."""
        mock_results = [
            {
                "id": "123",
                "score": 0.89,
                "payload": {
                    "doc_id": "copom_280",
                    "chunk_id": "copom_280_chunk_001",
                    "nro_reuniao": 280,
                    "titulo": "280ª Reunião",
                    "data_publicacao": "2026-08-11",
                    "token_count": 120,
                    "char_count": 500,
                    "text": "O Copom avalia o cenário econômico global e doméstico...",
                },
            }
        ]
        with patch("src.pipeline.QdrantManager.search", return_value=mock_results):
            result = runner.invoke(app, ["query", "inflação e taxa Selic", "--limit", "1"])
            assert result.exit_code == 0
            assert "Semantic Search Query" in result.stdout
            assert "Result #1" in result.stdout
            assert "copom_280" in result.stdout
