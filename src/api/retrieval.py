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


def _get_sparse_vector(query: str) -> SparseVector:
    result = list(sparse_model.embed([query]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def get_top_k_chunks(query: str, top_k: int = 30) -> list[dict[str, Any]]:
    logger.info(
        "Starting hybrid retrieval for top %d chunks. Query: '%s'", top_k, query
    )

    try:
        client: QdrantClient = load_vector_db(DATABASE_PATH)

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

        # Re-rank with cross-encoder
        pairs = [(query, c["text_chunk"]) for c in candidates_list]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(scores, candidates_list), key=lambda x: x[0], reverse=True
        )
        structured_results = [item for _, item in ranked[:top_k]]

        logger.info(
            "Retrieved %d candidates, re-ranked to top %d.",
            len(candidates_list),
            len(structured_results),
        )
        return structured_results

    except Exception as e:
        logger.error("Failed during chunk retrieval: %s", e, exc_info=True)
        return []
