"""Unit and integration tests for Bronze layer ingestion and schemas."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.bcb_client import BCBClient, BCBClientError
from src.ingestion.collector import BronzeCollector
from src.ingestion.schemas import BronzeAtaRecord, RawAtaDetail, RawAtaItem


class TestIngestionSchemas:
    """Test suite for Pydantic v2 schemas in Ingestion layer."""

    def test_raw_ata_item_validation(self, sample_catalog_response):
        """Verify parsing and field aliasing for raw catalog items."""
        item_data = sample_catalog_response["conteudo"][0]
        item = RawAtaItem.model_validate(item_data)
        assert item.nro_reuniao == 280
        assert "280ª Reunião" in item.titulo
        assert item.data_publicacao == "2026-08-11"

    def test_bronze_record_factory_and_hash(self, sample_raw_html):
        """Verify SHA-256 computation and Hive partition generation."""
        record = BronzeAtaRecord.create(
            nro_reuniao=280,
            titulo="280ª Reunião",
            data_publicacao="2026-08-11",
            raw_content=sample_raw_html,
            source_url="https://example.com/280",
        )
        assert record.doc_id == "copom_280"
        assert record.ano == 2026
        assert record.mes == 8
        assert len(record.content_hash) == 64
        assert record.short_hash == record.content_hash[:8]

    def test_unparseable_publication_date_fails_instead_of_guessing(self, sample_raw_html):
        """Guessing today's date would file the document in the wrong Hive partition."""
        with pytest.raises(ValueError, match="data_publicacao"):
            BronzeAtaRecord.create(
                nro_reuniao=280,
                titulo="280a Reuniao",
                data_publicacao="data invalida",
                raw_content=sample_raw_html,
                source_url="https://example.com/280",
            )

    @pytest.mark.parametrize("texto_ata", ["   ", "", None])
    def test_detail_contract_rejects_minutes_without_text(self, texto_ata):
        """Older meetings ship only a PDF: the API returns null or an empty string."""
        payload = {
            "nroReuniao": 230,
            "titulo": "230a Reuniao",
            "dataPublicacao": "2020-05-12",
            "textoAta": texto_ata,
        }

        with pytest.raises(ValueError, match="textoAta is empty or null"):
            RawAtaDetail.model_validate(payload)

    def test_detail_contract_parses_the_real_api_shape(self, sample_detail_response):
        """The camelCase payload from the details endpoint must satisfy the contract."""
        detail = RawAtaDetail.model_validate(sample_detail_response["conteudo"][0])

        assert detail.nro_reuniao == 280
        assert detail.data_publicacao == "2026-08-11"
        assert detail.url_pdf_ata is not None
        assert "A) Atualização" in detail.texto_ata

    def test_invalid_month_validation(self, sample_raw_html):
        """Ensure month validation enforces 1-12 bounds."""
        with pytest.raises(ValueError, match="Month must be between 1 and 12"):
            BronzeAtaRecord(
                doc_id="copom_999",
                nro_reuniao=999,
                titulo="Test",
                data_publicacao="2026-15-01",
                ano=2026,
                mes=15,  # Invalid month
                source_url="http://example.com",
                raw_content=sample_raw_html,
                content_hash="abc123",
            )


class TestBCBClient:
    """Test suite for BCB API client resilience and requests."""

    def test_fetch_atas_list_success(self, sample_catalog_response):
        """Test successful catalog retrieval."""
        client = BCBClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_catalog_response

        with patch.object(client.session, "get", return_value=mock_resp):
            items = client.fetch_atas_list(quantidade=2)
            assert len(items) == 2
            assert items[0]["nroReuniao"] == 280

    def test_fetch_ata_details_success(self, sample_detail_response):
        """Test successful detail retrieval."""
        client = BCBClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_detail_response

        with patch.object(client.session, "get", return_value=mock_resp):
            detail = client.fetch_ata_details(280)
            assert detail is not None
            assert detail["nroReuniao"] == 280
            assert "A) Atualização" in detail["textoAta"]

    def test_client_retry_on_server_error(self):
        """Test that client retries on HTTP 500 error."""
        client = BCBClient(max_retries=2, backoff_factor=0.01)
        mock_err_resp = MagicMock()
        mock_err_resp.status_code = 500

        with patch.object(client.session, "get", return_value=mock_err_resp) as mock_get:
            with pytest.raises(BCBClientError, match="HTTP 500"):
                client.fetch_atas_list(quantidade=1)
            # Should have retried up to max_retries
            assert mock_get.call_count == 2


class TestBronzeCollector:
    """Test suite for idempotent Bronze layer data collection."""

    def test_idempotent_ingestion_flow(
        self, temp_lakehouse_dirs, sample_catalog_response, sample_detail_response
    ):
        """Test first ingestion writes file, second ingestion skips."""
        bronze_dir = temp_lakehouse_dirs["bronze"]
        mock_client = MagicMock(spec=BCBClient)
        mock_client.base_url = "https://mock.bcb.gov.br"
        mock_client.fetch_atas_list.return_value = sample_catalog_response["conteudo"][:1]
        mock_client.fetch_ata_details.return_value = sample_detail_response["conteudo"][0]

        collector = BronzeCollector(client=mock_client, bronze_dir=bronze_dir)

        # Run 1: Should ingest 1 new file
        summary_1 = collector.run(limit=1)
        assert summary_1.total_catalog_items == 1
        assert summary_1.new_ingested == 1
        assert summary_1.skipped_existing == 0

        # Check file on disk
        files = collector.list_all_bronze_files()
        assert len(files) == 1
        assert "year=2026/month=08" in str(files[0]).replace("\\", "/")

        # Run 2: Should detect existing hash and skip
        summary_2 = collector.run(limit=1)
        assert summary_2.total_catalog_items == 1
        assert summary_2.new_ingested == 0
        assert summary_2.skipped_existing == 1

    def test_select_current_versions_prefers_latest_ingestion(self):
        """A revised publication must win over the stale one it superseded."""
        meeting_280 = {
            "nro_reuniao": 280,
            "titulo": "280a Reuniao",
            "data_publicacao": "2026-08-11",
            "source_url": "https://example.com/280",
        }
        stale = BronzeAtaRecord.create(
            raw_content="<p>Versao original da ata, retificada depois pelo BCB. #5</p>",
            **meeting_280,
        ).model_copy(update={"ingested_at": "2026-08-11T09:00:00+00:00"})
        current = BronzeAtaRecord.create(
            raw_content="<p>Versao retificada pelo BCB. #5</p>",
            **meeting_280,
        ).model_copy(update={"ingested_at": "2026-08-25T09:00:00+00:00"})
        other_doc = BronzeAtaRecord.create(
            nro_reuniao=279,
            titulo="279a Reuniao",
            data_publicacao="2026-06-24",
            raw_content="<p>Ata da 279a reuniao.</p>",
            source_url="https://example.com/279",
        )

        # Precondition of the original defect: the stale version's hash sorts last,
        # so promoting in filename order would pick exactly the wrong record.
        assert stale.content_hash > current.content_hash

        selected = BronzeCollector.select_current_versions([stale, current, other_doc])

        # One record per document, and for copom_280 it is the revision.
        assert [record.doc_id for record in selected] == ["copom_279", "copom_280"]
        assert selected[1].content_hash == current.content_hash
