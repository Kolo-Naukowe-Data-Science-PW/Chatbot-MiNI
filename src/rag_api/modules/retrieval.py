import logging
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector,
    NamedSparseVector,
    SparseVector,
    FusionQuery,
    Fusion,
    Prefetch,
)
from fastembed import SparseTextEmbedding

from src.data_ingest.modules.embedder import Embedder
from src.data_ingest.modules.vector_db import load_vector_db, COLLECTION_NAME
from src.utils.paths import get_data_dir

logger = logging.getLogger(__name__)

DATABASE_PATH = os.environ.get("QDRANT_DIR", get_data_dir("qdrant_db"))

logger.info("Loading Embedder model for retrieval...")
embedder = Embedder()
sparse_model = SparseTextEmbedding(model_name="Prithivida/Splade_PP_en_v1")
logger.info("Embedder loaded.")


def _get_sparse_vector(query: str) -> SparseVector:
    result = list(sparse_model.embed([query]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def get_top_k_chunks(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    logger.info("Starting hybrid retrieval for top %d chunks. Query: '%s'", top_k, query)

    try:
        client: QdrantClient = load_vector_db(DATABASE_PATH)

        dense_vector = embedder.generate_embeddings([query])[0]
        sparse_vector = _get_sparse_vector(query)

        # Hybrid search z RRF (Reciprocal Rank Fusion)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=NamedVector(name="dense", vector=dense_vector),
                    limit=top_k * 3,  # więcej kandydatów do fuzji
                ),
                Prefetch(
                    query=NamedSparseVector(
                        name="sparse",
                        vector=sparse_vector,
                    ),
                    limit=top_k * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        structured_results = [
            {
                "text_chunk": point.payload.get("text", ""),
                "source_url": point.payload.get("url", "Unknown Source"),
            }
            for point in results.points
        ]

        logger.info("Successfully retrieved %d results.", len(structured_results))
        return structured_results

    except Exception as e:
        logger.error("Failed during chunk retrieval: %s", e, exc_info=True)
        return []