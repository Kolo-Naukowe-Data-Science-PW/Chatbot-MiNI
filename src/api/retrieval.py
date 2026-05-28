import logging
import os
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)
from sentence_transformers import CrossEncoder

from src.ingestion.embedder import Embedder
from src.ingestion.vector_db import COLLECTION_NAME, load_vector_db
from src.utils.paths import get_data_dir

logger = logging.getLogger(__name__)

DATABASE_PATH = os.environ.get("QDRANT_DIR", get_data_dir("qdrant_db"))

logger.info("Loading Embedder model for retrieval...")
embedder = Embedder()
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
logger.info("Embedder and reranker loaded.")

_qdrant_client: QdrantClient | None = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        logger.info("Initialising Qdrant client (singleton) at %s", DATABASE_PATH)
        _qdrant_client = load_vector_db(DATABASE_PATH)
    return _qdrant_client


def _get_sparse_vector(query: str) -> SparseVector:
    result = list(sparse_model.embed([query]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def get_top_k_chunks(
    query: str,
    top_k: int = 30,
    use_rerank: bool = True,
) -> list[dict[str, Any]]:
    logger.info(
        "Starting hybrid retrieval for top %d chunks (rerank=%s). Query: '%s'",
        top_k,
        use_rerank,
        query,
    )

    try:
        client: QdrantClient = _get_qdrant_client()

        dense_vector = embedder.generate_embeddings([query])[0]
        sparse_vector = _get_sparse_vector(query)

        # Fetch twice as many candidates for re-ranking
        candidates = top_k * 2
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=candidates * 3,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=candidates * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=candidates,
            with_payload=True,
        )

        candidates_list = [
            {
                "text_chunk": point.payload.get("text", ""),
                "source_url": point.payload.get("url", "Unknown Source"),
            }
            for point in results.points
        ]

        if not candidates_list:
            return []

        if not use_rerank:
            logger.info("Reranking skipped — returning top %d RRF results.", top_k)
            return candidates_list[:top_k]

        # Score all chunks and return top_k by reranker score, no per-URL cap.
        pairs = [(query, c["text_chunk"]) for c in candidates_list]
        scores = reranker.predict(pairs)

        structured_results = [
            chunk
            for _, chunk in sorted(
                zip(scores, candidates_list, strict=False),
                key=lambda x: x[0],
                reverse=True,
            )
        ][:top_k]

        logger.info(
            "Retrieved %d candidates, re-ranked to top %d.",
            len(candidates_list),
            len(structured_results),
        )
        return structured_results

    except Exception as e:
        logger.error("Failed during chunk retrieval: %s", e, exc_info=True)
        return []
