"""Idempotent Bronze Lakehouse Collector for Copom publications.

Handles fetching raw data from the BCB client, computing SHA-256 integrity hashes,
and writing partitioned Hive JSON files (year=YYYY/month=MM/).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.ingestion.bcb_client import BCBClient
from src.ingestion.schemas import BronzeAtaRecord, RawAtaItem

logger = logging.getLogger(__name__)


@dataclass
class IngestionSummary:
    """Summary metrics of an ingestion run."""

    total_catalog_items: int = 0
    new_ingested: int = 0
    skipped_existing: int = 0
    failed: int = 0
    records: list[BronzeAtaRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BronzeCollector:
    """Collector responsible for idempotent ingestion into the Bronze lakehouse layer."""

    def __init__(
        self,
        client: BCBClient | None = None,
        bronze_dir: Path | None = None,
    ) -> None:
        self.client = client or BCBClient()
        self.bronze_dir = bronze_dir or settings.BRONZE_DIR
        self.bronze_dir.mkdir(parents=True, exist_ok=True)

    def is_already_ingested(self, record: BronzeAtaRecord) -> bool:
        """Check if a record with the same doc_id and content hash already exists on disk."""
        canonical_path = record.get_canonical_filepath(self.bronze_dir)
        if canonical_path.exists():
            return True

        # Also check if any file in the partition matches the doc_id and short hash
        partition_dir = record.get_hive_partition_path(self.bronze_dir)
        if partition_dir.exists():
            matches = list(partition_dir.glob(f"{record.doc_id}_{record.short_hash}.json"))
            if matches:
                return True

        return False

    def save_record(self, record: BronzeAtaRecord) -> Path:
        """Save a Bronze record to its Hive-partitioned file path."""
        partition_dir = record.get_hive_partition_path(self.bronze_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)

        target_file = record.get_canonical_filepath(self.bronze_dir)
        # Write JSON with formatted indentation for auditability
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(record.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved Bronze record: {target_file.name} to {partition_dir}")
        return target_file

    def ingest_meeting(
        self, nro_reuniao: int, raw_catalog_item: dict | None = None
    ) -> BronzeAtaRecord | None:
        """Ingest a single meeting by its meeting number."""
        details = self.client.fetch_ata_details(nro_reuniao)
        if not details:
            logger.warning(f"Could not retrieve meeting #{nro_reuniao} from BCB API.")
            return None

        texto_ata = details.get("textoAta") or ""
        titulo = details.get("titulo") or f"Reunião #{nro_reuniao}"
        data_pub = details.get("dataPublicacao") or ""
        data_ref = details.get("dataReferencia")
        url_pdf = details.get("urlPdfAta")

        if not texto_ata.strip():
            logger.warning(f"Meeting #{nro_reuniao} has empty text content.")
            return None

        record = BronzeAtaRecord.create(
            nro_reuniao=nro_reuniao,
            titulo=titulo,
            data_publicacao=data_pub,
            raw_content=texto_ata,
            source_url=f"{self.client.base_url}/copom/atas_detalhes?nro_reuniao={nro_reuniao}",
            data_referencia=data_ref,
            url_pdf=url_pdf,
            metadata={
                "catalog_item": raw_catalog_item,
                "api_endpoint": "sitebcb/copom/atas_detalhes",
            },
        )
        return record

    def run(self, limit: int | None = None) -> IngestionSummary:
        """Execute idempotent ingestion of the latest Copom minutes.

        Args:
            limit: Maximum number of catalog items to process.

        Returns:
            IngestionSummary with counts and records.
        """
        fetch_limit = limit or settings.DEFAULT_INGEST_LIMIT
        logger.info(f"Starting Bronze ingestion for up to {fetch_limit} Copom meetings.")

        summary = IngestionSummary()

        try:
            catalog = self.client.fetch_atas_list(quantidade=fetch_limit)
        except Exception as err:
            logger.error(f"Failed to retrieve Copom catalog: {err}")
            summary.errors.append(str(err))
            return summary

        summary.total_catalog_items = len(catalog)

        for item in catalog:
            try:
                raw_item = RawAtaItem.model_validate(item)
                if raw_item.nro_reuniao is None:
                    logger.warning(f"Skipping catalog item without nro_reuniao: {item}")
                    continue

                record = self.ingest_meeting(raw_item.nro_reuniao, raw_catalog_item=item)
                if not record:
                    summary.failed += 1
                    continue

                if self.is_already_ingested(record):
                    logger.info(
                        f"Skipping already ingested document {record.doc_id} "
                        f"(Hash: {record.short_hash}) in partition {record.ano}/{record.mes:02d}"
                    )
                    summary.skipped_existing += 1
                    summary.records.append(record)
                else:
                    self.save_record(record)
                    summary.new_ingested += 1
                    summary.records.append(record)

            except Exception as err:
                error_msg = f"Error processing meeting item {item}: {err}"
                logger.error(error_msg, exc_info=True)
                summary.failed += 1
                summary.errors.append(error_msg)

        logger.info(
            f"Bronze Ingestion completed. Total: {summary.total_catalog_items}, "
            f"New: {summary.new_ingested}, Skipped: {summary.skipped_existing}, "
            f"Failed: {summary.failed}"
        )
        return summary

    def list_all_bronze_files(self) -> list[Path]:
        """List all Bronze JSON files in the Hive partition structure."""
        return sorted(list(self.bronze_dir.glob("year=*/month=*/*.json")))

    def load_all_records(self) -> list[BronzeAtaRecord]:
        """Load and parse all Bronze records stored on disk."""
        records: list[BronzeAtaRecord] = []
        for file_path in self.list_all_bronze_files():
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    records.append(BronzeAtaRecord.model_validate(data))
            except Exception as err:
                logger.error(f"Error loading Bronze file {file_path}: {err}")
        return records
