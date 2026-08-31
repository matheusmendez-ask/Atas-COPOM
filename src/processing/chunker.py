"""Semantic chunking and metadata enrichment module for Copom minutes."""

import json
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion.schemas import BronzeAtaRecord
from src.processing.cleaner import AtaCleaner
from src.processing.schemas import ChunkPayload, SilverDocument

logger = logging.getLogger(__name__)


class TokenCounter:
    """Helper for counting tokens using TikToken with fallback estimation."""

    def __init__(self, model_name: str = "cl100k_base") -> None:
        self.model_name = model_name
        self._encoder = None
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(model_name)
        except Exception as err:
            logger.warning(
                f"Could not load tiktoken encoding '{model_name}': {err}. Using heuristic token counter."
            )

    def count(self, text: str) -> int:
        """Count tokens in text string."""
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        # Fallback heuristic: ~4 characters per token
        return max(1, len(text) // 4)


class AtaChunker:
    """Splits sanitized documents into semantically coherent chunks with rich metadata."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        cleaner: AtaCleaner | None = None,
        silver_dir: Path | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.cleaner = cleaner or AtaCleaner()
        self.silver_dir = silver_dir or settings.SILVER_DIR
        self.token_counter = TokenCounter(model_name=settings.TOKENIZER_MODEL)

        # Initialize recursive character text splitter with token length function
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.token_counter.count,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
            keep_separator=True,
        )

    def process_record(self, bronze_record: BronzeAtaRecord) -> SilverDocument:
        """Sanitize, chunk, and enrich a Bronze record, creating a SilverDocument.

        Args:
            bronze_record: Validated raw record from Bronze layer.

        Returns:
            SilverDocument with populated chunk list and normalized metadata.
        """
        clean_text = self.cleaner.clean_html(bronze_record.raw_content)
        total_tokens = self.token_counter.count(clean_text)

        # Split text into raw chunks
        raw_chunks = self.text_splitter.split_text(clean_text)
        if not raw_chunks and clean_text:
            raw_chunks = [clean_text]

        total_chunks = len(raw_chunks)
        chunks: list[ChunkPayload] = []

        for idx, chunk_str in enumerate(raw_chunks):
            chunk_tokens = self.token_counter.count(chunk_str)
            chunk_payload = ChunkPayload.create(
                doc_id=bronze_record.doc_id,
                nro_reuniao=bronze_record.nro_reuniao,
                titulo=bronze_record.titulo,
                data_publicacao=bronze_record.data_publicacao,
                data_referencia=bronze_record.data_referencia,
                ano=bronze_record.ano,
                mes=bronze_record.mes,
                parent_hash=bronze_record.content_hash,
                source_url=bronze_record.source_url,
                text=chunk_str,
                chunk_index=idx,
                total_chunks=total_chunks,
                token_count=chunk_tokens,
            )
            chunks.append(chunk_payload)

        silver_doc = SilverDocument(
            doc_id=bronze_record.doc_id,
            nro_reuniao=bronze_record.nro_reuniao,
            titulo=bronze_record.titulo,
            data_publicacao=bronze_record.data_publicacao,
            data_referencia=bronze_record.data_referencia,
            ano=bronze_record.ano,
            mes=bronze_record.mes,
            clean_text=clean_text,
            content_hash=bronze_record.content_hash,
            total_tokens=total_tokens,
            chunks=chunks,
            metadata={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "tokenizer": settings.TOKENIZER_MODEL,
            },
        )
        return silver_doc

    def save_silver_document(self, doc: SilverDocument) -> Path:
        """Write SilverDocument to partitioned disk storage."""
        target_path = doc.get_silver_filepath(self.silver_dir)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Silver document with {len(doc.chunks)} chunks to {target_path}")
        return target_path

    def process_all_bronze(self, bronze_records: list[BronzeAtaRecord]) -> list[SilverDocument]:
        """Process a collection of Bronze records into Silver documents."""
        silver_docs: list[SilverDocument] = []
        for record in bronze_records:
            try:
                doc = self.process_record(record)
                self.save_silver_document(doc)
                silver_docs.append(doc)
            except Exception as err:
                logger.error(
                    f"Failed to process Bronze record {record.doc_id}: {err}", exc_info=True
                )
        return silver_docs

    def load_all_silver_documents(self) -> list[SilverDocument]:
        """Load all existing Silver documents from disk."""
        silver_docs: list[SilverDocument] = []
        for file_path in sorted(list(self.silver_dir.glob("year=*/month=*/*_silver.json"))):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    silver_docs.append(SilverDocument.model_validate(data))
            except Exception as err:
                logger.error(f"Error loading Silver file {file_path}: {err}")
        return silver_docs
