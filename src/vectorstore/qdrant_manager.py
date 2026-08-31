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
from src.vectorstore.embeddings import EmbeddingGenerator, SparseEmbedder

logger = logging.getLogger(__name__)

# Named vectors: one collection holds both halves of the hybrid index.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# Fusion works on the union of both candidate lists, so each branch is asked for
# more than the caller wants: a passage ranked 20th by BM25 can still win overall.
# Depth matters for DBSF (x4 -> x10 moved hit@1 from 32% to 36%) but not for RRF.
PREFETCH_MULTIPLIER = 10

# Distribution-Based Score Fusion, chosen by measurement rather than by argument.
# RRF was the a-priori pick -- it fuses ranks, so it needs no normalisation between
# cosine (0-1) and BM25 (unbounded), and has no weight to tune wrong. But on the
# golden set RRF dropped provenance from 100% to 83%: fusing by position promotes
# passages both retrievers agree on and demotes one that only dense found, undoing
# the gain from embedding document identity. DBSF regressed nothing.
#
#   golden set, 22 answerable   hit@1  hit@3  hit@5    MRR  provenance
#   dense only                    32%    50%    64%  0.433        100%
#   hybrid RRF   (prefetch x4)    32%    68%    77%  0.515         83%
#   hybrid DBSF  (prefetch x10)   36%    59%    77%  0.508        100%
FUSION_METHOD = models.Fusion.DBSF


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
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.sparse_embedder = sparse_embedder or SparseEmbedder()

        # Initialize Qdrant Client (supports http URL or ':memory:' for tests)
        if self.url == ":memory:":
            logger.info("Initializing in-memory Qdrant instance.")
            self.client = QdrantClient(location=":memory:", check_compatibility=False)
        else:
            logger.info(f"Connecting to Qdrant at {self.url}")
            self.client = QdrantClient(
                url=self.url, api_key=self.api_key, check_compatibility=False
            )

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
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=dim,
                            distance=models.Distance.COSINE,
                        )
                    },
                    # IDF is computed by the server across the collection; without
                    # this modifier BM25 scores would ignore term rarity entirely.
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=10000,
                    ),
                )
                # Create payload indexes for frequent filtering fields
                self._create_payload_indexes()
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
                self._assert_dimension_matches(dim)
        except Exception as err:
            logger.error(f"Error checking or creating collection '{self.collection_name}': {err}")
            raise

    def _assert_dimension_matches(self, expected_dim: int) -> None:
        """Fail loudly when an existing collection was built for another model.

        Switching ``EMBEDDING_MODEL_NAME`` to a model of a different width would
        otherwise surface as a batch of failed upserts buried in the summary. The
        collection has to be dropped and rebuilt, which is a deliberate act.
        """
        info = self.client.get_collection(self.collection_name)
        params = info.config.params.vectors

        if not isinstance(params, dict):
            # Collections created before hybrid search hold a single anonymous
            # vector. Qdrant would reject the named queries with an opaque error,
            # so name the actual problem here.
            raise ValueError(
                f"Collection '{self.collection_name}' predates hybrid search: it holds one "
                "anonymous vector instead of the named 'dense' and 'bm25' vectors. Drop it "
                "and reindex (transform + index), or point QDRANT_COLLECTION_NAME elsewhere."
            )

        dense_params = params.get(DENSE_VECTOR_NAME)
        if dense_params is None:
            raise ValueError(
                f"Collection '{self.collection_name}' has no '{DENSE_VECTOR_NAME}' vector "
                f"(found: {sorted(params)}). Drop it and reindex."
            )

        actual_dim = getattr(dense_params, "size", None)
        if actual_dim is not None and actual_dim != expected_dim:
            raise ValueError(
                f"Collection '{self.collection_name}' stores {actual_dim}-dimensional vectors "
                f"but the configured model produces {expected_dim}. Recreate the collection "
                f"(or set QDRANT_COLLECTION_NAME to a new one) and reindex."
            )

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
            # Embed the contextualized form so the vector carries which meeting the
            # passage came from; the payload below still stores the clean text.
            texts = [c.embedding_text for c in chunk_batch]

            try:
                # Generate embeddings for current batch
                vectors = self.embedding_generator.embed_texts(texts)
                sparse_vectors = self.sparse_embedder.embed_documents(texts)

                points: list[models.PointStruct] = []
                for chunk, vector, sparse in zip(chunk_batch, vectors, sparse_vectors, strict=True):
                    point_id = self.generate_point_id(chunk.metadata.doc_id, chunk.chunk_id)
                    payload = {
                        "text": chunk.text,
                        **chunk.metadata.model_dump(),
                    }
                    indices, values = sparse
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={
                                DENSE_VECTOR_NAME: vector,
                                SPARSE_VECTOR_NAME: models.SparseVector(
                                    indices=indices, values=values
                                ),
                            },
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
        if not self.client.collection_exists(self.collection_name):
            # Nothing indexed yet is an ordinary state, not a crash: both the
            # 'query' and 'ask' commands already tell the user to run 'index'.
            logger.warning(
                f"Collection '{self.collection_name}' does not exist, so the search returns "
                "nothing. Run the 'index' command, or check QDRANT_COLLECTION_NAME."
            )
            return []

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

        sparse_indices, sparse_values = self.sparse_embedder.embed_query(query)

        # Hybrid: dense and BM25 each produce a ranking, fused by Reciprocal Rank
        # Fusion. RRF combines positions rather than scores, so it needs no
        # normalisation between cosine (0-1) and BM25 (unbounded) -- and it has no
        # weight to tune wrong.
        candidates = limit * PREFETCH_MULTIPLIER
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=candidates,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using=SPARSE_VECTOR_NAME,
                    limit=candidates,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=FUSION_METHOD),
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
