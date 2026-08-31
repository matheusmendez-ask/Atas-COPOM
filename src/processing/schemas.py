"""Pydantic v2 schemas and data contracts for the Silver Processing layer."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Normalized metadata associated with each text chunk for semantic filtering and RAG."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(
        ...,
        description="Parent document identifier (e.g. 'copom_280').",
    )
    chunk_id: str = Field(
        ...,
        description="Unique deterministic chunk identifier (e.g. 'copom_280_chunk_001').",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-indexed sequence position of the chunk within the document.",
    )
    total_chunks: int = Field(
        ...,
        ge=1,
        description="Total number of chunks produced from the parent document.",
    )
    nro_reuniao: int = Field(
        ...,
        description="Copom meeting number.",
    )
    titulo: str = Field(
        ...,
        description="Title of the Copom meeting.",
    )
    data_publicacao: str = Field(
        ...,
        description="Publication date (YYYY-MM-DD or ISO format).",
    )
    data_referencia: str | None = Field(
        default=None,
        description="Reference date.",
    )
    ano: int = Field(
        ...,
        description="Publication year.",
    )
    mes: int = Field(
        ...,
        ge=1,
        le=12,
        description="Publication month.",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash of the parent raw document.",
    )
    chunk_hash: str = Field(
        ...,
        description="SHA-256 hash of this specific chunk text.",
    )
    token_count: int = Field(
        ...,
        ge=0,
        description="Exact or estimated token count of the chunk text.",
    )
    char_count: int = Field(
        ...,
        ge=0,
        description="Total character length of the chunk text.",
    )
    source_url: str = Field(
        ...,
        description="Source endpoint URL of the document.",
    )


class ChunkPayload(BaseModel):
    """Complete chunk representation ready for Vector Store indexing."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier.",
    )
    text: str = Field(
        ...,
        description="Cleaned, normalized textual content of the chunk.",
    )
    metadata: ChunkMetadata = Field(
        ...,
        description="Rich metadata schema for filtering and context reconstruction.",
    )

    @property
    def embedding_text(self) -> str:
        """The text actually embedded: the passage prefixed with its document identity.

        Copom minutes are formulaic -- the same sections in near-identical prose every
        meeting -- so passage vectors from different meetings come out nearly collinear.
        ``nro_reuniao`` lives in the Qdrant payload, which filters but never embeds,
        leaving a question like "the 280th meeting" nothing to match on: retrieval
        returned the right topic from the wrong document.

        Measured on the golden set (11 answerable questions, 66 chunks):
        hit@1 27% -> 45%, hit@5 45% -> 64%, MRR 0.341 -> 0.491, provenance 40% -> 80%.

        Only the vector carries the prefix; ``text`` stays clean for display and for
        the answer prompt, which already labels each excerpt with meeting and date.
        """
        return (
            f"Ata da {self.metadata.nro_reuniao}a reunião do Copom, "
            f"publicada em {self.metadata.data_publicacao}.\n\n{self.text}"
        )

    @classmethod
    def create(
        cls,
        doc_id: str,
        nro_reuniao: int,
        titulo: str,
        data_publicacao: str,
        data_referencia: str | None,
        ano: int,
        mes: int,
        parent_hash: str,
        source_url: str,
        text: str,
        chunk_index: int,
        total_chunks: int,
        token_count: int,
    ) -> "ChunkPayload":
        """Factory method to construct a validated ChunkPayload."""
        chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"

        # Calculate chunk-level SHA-256 hash
        hasher = hashlib.sha256()
        hasher.update(text.encode("utf-8"))
        chunk_hash = hasher.hexdigest()

        metadata = ChunkMetadata(
            doc_id=doc_id,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            nro_reuniao=nro_reuniao,
            titulo=titulo,
            data_publicacao=data_publicacao,
            data_referencia=data_referencia,
            ano=ano,
            mes=mes,
            content_hash=parent_hash,
            chunk_hash=chunk_hash,
            token_count=token_count,
            char_count=len(text),
            source_url=source_url,
        )
        return cls(chunk_id=chunk_id, text=text, metadata=metadata)


class SilverDocument(BaseModel):
    """Canonical model for a cleaned and chunked document in the Silver layer."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(
        ...,
        description="Document identifier.",
    )
    nro_reuniao: int = Field(
        ...,
        description="Copom meeting number.",
    )
    titulo: str = Field(
        ...,
        description="Meeting title.",
    )
    data_publicacao: str = Field(
        ...,
        description="Publication date.",
    )
    data_referencia: str | None = Field(
        default=None,
        description="Reference date.",
    )
    ano: int = Field(
        ...,
        description="Year for partitioning.",
    )
    mes: int = Field(
        ...,
        description="Month for partitioning.",
    )
    clean_text: str = Field(
        ...,
        description="Full sanitized text representation of the document.",
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hash of the parent raw document.",
    )
    total_tokens: int = Field(
        ...,
        ge=0,
        description="Total token count for the entire sanitized document.",
    )
    chunks: list[ChunkPayload] = Field(
        default_factory=list,
        description="List of semantic chunks generated from the document.",
    )
    processed_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="UTC timestamp of Silver transformation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transformation metadata.",
    )

    def get_silver_filepath(self, base_silver_dir: Path) -> Path:
        """Generate the Silver layer Hive partition filepath."""
        month_str = f"{self.mes:02d}"
        partition = base_silver_dir / f"year={self.ano}" / f"month={month_str}"
        partition.mkdir(parents=True, exist_ok=True)
        return partition / f"{self.doc_id}_silver.json"
