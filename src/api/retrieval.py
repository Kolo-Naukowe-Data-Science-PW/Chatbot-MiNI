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

_EN_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_PL_RERANKER_MODEL = os.environ.get("POLISH_RERANKER_MODEL", "clarin-knext/herbert-base-reranker")

logger.info("Loading Embedder model for retrieval...")
embedder = Embedder()
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder(_EN_RERANKER_MODEL)
logger.info("Embedder and reranker loaded.")

_qdrant_client: QdrantClient | None = None
_polish_reranker: CrossEncoder | None = None


def _get_polish_reranker() -> CrossEncoder:
    global _polish_reranker
    if _polish_reranker is None:
        logger.info("Loading Polish cross-encoder (lazy): %s", _PL_RERANKER_MODEL)
        _polish_reranker = CrossEncoder(_PL_RERANKER_MODEL)
    return _polish_reranker


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
    query: str, top_k: int = 30, use_rerank: bool = True, use_cascade: bool = False
) -> list[dict[str, Any]]:
    logger.info(
        "Starting hybrid retrieval for top %d chunks (rerank=%s, cascade=%s). Query: '%s'",
        top_k, use_rerank, use_cascade, query,
    )

    try:
        client: QdrantClient = _get_qdrant_client()

        dense_vector = embedder.generate_embeddings([query])[0]
        sparse_vector = _get_sparse_vector(query)

        # Cascade fetches 4× candidates (120 for top_k=30), single rerank fetches 2× (60)
        rrf_candidates = top_k * 4 if use_cascade else top_k * 2
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=rrf_candidates * 3,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=rrf_candidates * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=rrf_candidates,
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

        if use_cascade:
            # Stage 1: English CE — 120 → 60
            pairs = [(query, c["text_chunk"]) for c in candidates_list]
            scores_en = reranker.predict(pairs)
            stage1 = [
                item for _, item in
                sorted(zip(scores_en, candidates_list), key=lambda x: x[0], reverse=True)
            ][:top_k * 2]

            # Stage 2: Polish CE — 60 → 30
            pairs2 = [(query, c["text_chunk"]) for c in stage1]
            scores_pl = _get_polish_reranker().predict(pairs2)
            structured_results = [
                item for _, item in
                sorted(zip(scores_pl, stage1), key=lambda x: x[0], reverse=True)
            ][:top_k]

            logger.info(
                "Cascade rerank: %d RRF → %d (EN-CE) → %d (PL-CE).",
                len(candidates_list), len(stage1), len(structured_results),
            )
        else:
            pairs = [(query, c["text_chunk"]) for c in candidates_list]
            scores = reranker.predict(pairs)
            structured_results = [
                item for _, item in
                sorted(zip(scores, candidates_list), key=lambda x: x[0], reverse=True)
            ][:top_k]

            logger.info(
                "Retrieved %d candidates, re-ranked to top %d.",
                len(candidates_list), len(structured_results),
            )

        return structured_results

    except Exception as e:
        logger.error("Failed during chunk retrieval: %s", e, exc_info=True)
        return []
