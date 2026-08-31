"""Pydantic v2 schemas and data contracts for the Bronze Ingestion layer."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class RawAtaItem(BaseModel):
    """Schema representing an item from the BCB Copom API catalog list."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    nro_reuniao: int | None = Field(
        default=None,
        alias="nroReuniao",
        description="Meeting number of the Copom session.",
    )
    titulo: str = Field(
        ...,
        alias="titulo",
        description="Title of the Copom publication.",
    )
    data_publicacao: str | None = Field(
        default=None,
        alias="dataPublicacao",
        description="Publication date (ISO string or YYYY-MM-DD).",
    )
    data_referencia: str | None = Field(
        default=None,
        alias="dataReferencia",
        description="Reference date of the meeting.",
    )


class RawAtaDetail(BaseModel):
    """Schema representing the full detailed payload for a Copom meeting."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    nro_reuniao: int = Field(
        ...,
        alias="nroReuniao",
        description="Meeting number.",
    )
    titulo: str = Field(
        ...,
        alias="titulo",
        description="Meeting title.",
    )
    data_publicacao: str = Field(
        ...,
        alias="dataPublicacao",
        description="Publication date.",
    )
    data_referencia: str | None = Field(
        default=None,
        alias="dataReferencia",
        description="Reference date.",
    )
    url_pdf_ata: str | None = Field(
        default=None,
        alias="urlPdfAta",
        description="URL to the official PDF publication.",
    )
    texto_ata: str = Field(
        ...,
        alias="textoAta",
        description="Full raw HTML or text content of the Copom minutes.",
    )


class BronzeAtaRecord(BaseModel):
    """Canonical contract for raw records stored in the Bronze Data Lakehouse layer."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(
        ...,
        description="Unique identifier for the document (e.g. 'copom_280').",
    )
    nro_reuniao: int = Field(
        ...,
        description="Copom meeting sequence number.",
    )
    titulo: str = Field(
        ...,
        description="Publication title.",
    )
    data_publicacao: str = Field(
        ...,
        description="Publication date in YYYY-MM-DD or ISO format.",
    )
    data_referencia: str | None = Field(
        default=None,
        description="Meeting reference date.",
    )
    ano: int = Field(
        ...,
        description="Publication year for Hive partitioning.",
    )
    mes: int = Field(
        ...,
        description="Publication month (1-12) for Hive partitioning.",
    )
    url_pdf: str | None = Field(
        default=None,
        description="Link to original official PDF.",
    )
    source_url: str = Field(
        ...,
        description="Source endpoint or URI where document was retrieved.",
    )
    raw_content: str = Field(
        ...,
        description="Full raw unmodified content (HTML or plain text).",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash of the raw_content for data integrity and idempotency.",
    )
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="UTC timestamp of ingestion.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional raw metadata from source.",
    )

    @field_validator("mes")
    @classmethod
    def validate_month_range(cls, v: int) -> int:
        """Ensure month is between 1 and 12."""
        if not 1 <= v <= 12:
            raise ValueError(f"Month must be between 1 and 12, got {v}")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def short_hash(self) -> str:
        """Return the first 8 characters of the SHA-256 hash."""
        return self.content_hash[:8]

    def get_hive_partition_path(self, base_bronze_dir: Path) -> Path:
        """Generate the standard Hive partition directory path: year=YYYY/month=MM/."""
        month_str = f"{self.mes:02d}"
        return base_bronze_dir / f"year={self.ano}" / f"month={month_str}"

    def get_canonical_filename(self) -> str:
        """Generate the standard file name: copom_{doc_id}_{short_hash}.json."""
        return f"{self.doc_id}_{self.short_hash}.json"

    def get_canonical_filepath(self, base_bronze_dir: Path) -> Path:
        """Generate the full canonical filepath in the Bronze lakehouse."""
        return self.get_hive_partition_path(base_bronze_dir) / self.get_canonical_filename()

    @classmethod
    def create(
        cls,
        nro_reuniao: int,
        titulo: str,
        data_publicacao: str,
        raw_content: str,
        source_url: str,
        data_referencia: str | None = None,
        url_pdf: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "BronzeAtaRecord":
        """Factory method to compute hash, extract year/month, and construct record."""
        # Calculate SHA-256 hash of raw content
        hasher = hashlib.sha256()
        hasher.update(raw_content.encode("utf-8"))
        content_hash = hasher.hexdigest()

        # Parse date to extract year and month
        pub_date_clean = data_publicacao.strip()
        try:
            if "T" in pub_date_clean:
                dt = datetime.fromisoformat(pub_date_clean.replace("Z", "+00:00"))
            elif "/" in pub_date_clean:
                parts = pub_date_clean.split("/")
                if len(parts[0]) == 4:
                    dt = datetime.strptime(pub_date_clean, "%Y/%m/%d")
                else:
                    dt = datetime.strptime(pub_date_clean, "%d/%m/%Y")
            else:
                dt = datetime.strptime(pub_date_clean[:10], "%Y-%m-%d")
            ano = dt.year
            mes = dt.month
        except Exception:
            # Fallback to current year/month if unparseable
            now = datetime.now(UTC)
            ano = now.year
            mes = now.month

        doc_id = f"copom_{nro_reuniao}"

        return cls(
            doc_id=doc_id,
            nro_reuniao=nro_reuniao,
            titulo=titulo,
            data_publicacao=data_publicacao,
            data_referencia=data_referencia,
            ano=ano,
            mes=mes,
            url_pdf=url_pdf,
            source_url=source_url,
            raw_content=raw_content,
            content_hash=content_hash,
            metadata=metadata or {},
        )
