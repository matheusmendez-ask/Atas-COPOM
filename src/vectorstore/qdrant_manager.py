"""Qdrant Vector Store Manager for idempotent collection management, batch upserts, and semantic search."""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings
from src.processing.schemas import ChunkPayload
from src.vectorstore.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class UpsertSummary:
    """Summary of vector upsert operation."""

    collection_name: str
    total_chunks: int = 0
    upserted_points: int = 0
    batch_count: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class QdrantManager:
    """Manager for Qdrant vector database operations."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ) -> None:
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

        # Initialize Qdrant Client (supports http URL or ':memory:' for tests)
        if self.url == ":memory:":
            logger.info("Initializing in-memory Qdrant instance.")
            self.client = QdrantClient(location=":memory:", check_compatibility=False)
        else:
            logger.info(f"Connecting to Qdrant at {self.url}")
            self.client = QdrantClient(url=self.url, api_key=self.api_key, check_compatibility=False)

    @staticmethod
    def generate_point_id(doc_id: str, chunk_id: str) -> str:
        """Generate a deterministic UUIDv5 from document and chunk identifiers.

        Ensures that repeated upserts of the exact same chunk yield identical point IDs.
        """
        seed = f"{doc_id}:{chunk_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    def ensure_collection(self, vector_size: int | None = None) -> None:
        """Create Qdrant collection if it does not exist with Cosine distance metric."""
        dim = vector_size or self.embedding_generator.dimension
        try:
            collections_response = self.client.get_collections()
            existing_names = [col.name for col in collections_response.collections]

            if self.collection_name not in existing_names:
                logger.info(
                    f"Creating collection '{self.collection_name}' with vector dimension {dim} and Cosine distance."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=10000,
                    ),
                )
                # Create payload indexes for frequent filtering fields
                self._create_payload_indexes()
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as err:
            logger.error(f"Error checking or creating collection '{self.collection_name}': {err}")
            raise

    def _create_payload_indexes(self) -> None:
        """Create field indexes in Qdrant for optimized metadata filtering."""
        fields_to_index = [
            ("ano", models.PayloadSchemaType.INTEGER),
            ("mes", models.PayloadSchemaType.INTEGER),
            ("nro_reuniao", models.PayloadSchemaType.INTEGER),
            ("doc_id", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_type in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as err:
                logger.warning(f"Could not create index for field '{field_name}': {err}")

    def upsert_chunks(
        self, chunks: list[ChunkPayload], batch_size: int | None = None
    ) -> UpsertSummary:
        """Embed and upsert chunks into Qdrant in idempotent batches.

        Args:
            chunks: List of ChunkPayload objects from the Silver layer.
            batch_size: Batch size for embeddings and upsert requests.

        Returns:
            UpsertSummary detailing indexed points.
        """
        self.ensure_collection()

        summary = UpsertSummary(collection_name=self.collection_name, total_chunks=len(chunks))
        if not chunks:
            return summary

        bs = batch_size or settings.EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), bs):
            chunk_batch = chunks[i : i + bs]
            texts = [c.text for c in chunk_batch]

            try:
                # Generate embeddings for current batch
                vectors = self.embedding_generator.embed_texts(texts)

                points: list[models.PointStruct] = []
                for chunk, vector in zip(chunk_batch, vectors, strict=False):
                    point_id = self.generate_point_id(chunk.metadata.doc_id, chunk.chunk_id)
                    payload = {
                        "text": chunk.text,
                        **chunk.metadata.model_dump(),
                    }
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                    )

                # Upsert into Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                summary.upserted_points += len(points)
                summary.batch_count += 1
                logger.info(
                    f"Upserted batch {summary.batch_count} ({len(points)} vectors) into '{self.collection_name}'"
                )

            except Exception as err:
                error_msg = f"Failed to upsert batch {summary.batch_count + 1}: {err}"
                logger.error(error_msg, exc_info=True)
                summary.failed += len(chunk_batch)
                summary.errors.append(error_msg)

        return summary

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        filter_year: int | None = None,
        filter_meeting: int | None = None,
    ) -> list[dict[str, Any]]:
        """Perform semantic search against indexed Copom minutes.

        Args:
            query: Natural language query.
            limit: Top-k results to return.
            score_threshold: Minimum cosine similarity score.
            filter_year: Optional filter for publication year.
            filter_meeting: Optional filter for specific meeting number.

        Returns:
            List of matching records with score, text, and metadata.
        """
        query_vector = self.embedding_generator.embed_query(query)
        if not query_vector:
            return []

        # Build query filters
        conditions = []
        if filter_year is not None:
            conditions.append(
                models.FieldCondition(key="ano", match=models.MatchValue(value=filter_year))
            )
        if filter_meeting is not None:
            conditions.append(
                models.FieldCondition(
                    key="nro_reuniao", match=models.MatchValue(value=filter_meeting)
                )
            )

        query_filter = models.Filter(must=conditions) if conditions else None

        # Search Qdrant using unified query_points API
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        formatted_results = []
        for res in search_results.points:
            formatted_results.append(
                {
                    "id": res.id,
                    "score": float(res.score) if res.score is not None else 0.0,
                    "payload": res.payload,
                }
            )
        return formatted_results

    def count_points(self) -> int:
        """Count total vectors indexed in the collection."""
        try:
            res = self.client.count(collection_name=self.collection_name, exact=True)
            return res.count
        except (UnexpectedResponse, Exception) as err:
            logger.warning(f"Could not retrieve point count: {err}")
            return 0
