import logging
import os
import re
from collections import defaultdict
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


def _tokenize(text: str) -> set[str]:
    """Lightweight Polish-aware tokenizer — extracts words ≥2 chars (no spacy needed)."""
    return set(re.findall(r'\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{2,}\b', text.lower()))


def _bm25_prefilter(
    query: str, candidates: list[dict], min_overlap: int = 1
) -> list[dict]:
    """Drop chunks with zero keyword-overlap with the query before cross-encoder."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return candidates
    filtered = [
        c for c in candidates
        if len(query_tokens & _tokenize(c["text_chunk"])) >= min_overlap
    ]
    # Safety: never return fewer than 5 candidates
    return filtered if len(filtered) >= 5 else candidates


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
    use_url_aggregation: bool = False,
    use_bm25_prefilter: bool = False,
) -> list[dict[str, Any]]:
    logger.info(
        "Starting hybrid retrieval for top %d chunks "
        "(rerank=%s, url_agg=%s, bm25_pre=%s). Query: '%s'",
        top_k, use_rerank, use_url_aggregation, use_bm25_prefilter, query,
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

        rerank_input = candidates_list
        if use_bm25_prefilter:
            rerank_input = _bm25_prefilter(query, candidates_list)
            logger.info(
                "BM25 pre-filter: %d → %d candidates.",
                len(candidates_list), len(rerank_input),
            )

        pairs = [(query, c["text_chunk"]) for c in rerank_input]
        scores = reranker.predict(pairs)

        if use_url_aggregation:
            # Per-URL max score — better ranking when many chunks share the same source
            url_best: dict[str, tuple[float, dict]] = {}
            for score, chunk in zip(scores, rerank_input):
                url = chunk["source_url"]
                if url not in url_best or score > url_best[url][0]:
                    url_best[url] = (score, chunk)
            structured_results = [
                chunk for _, chunk in
                sorted(url_best.values(), key=lambda x: x[0], reverse=True)
            ][:top_k]
        else:
            ranked = sorted(zip(scores, rerank_input), key=lambda x: x[0], reverse=True)
            structured_results = [item for _, item in ranked[:top_k]]

        logger.info(
            "Retrieved %d candidates, re-ranked to top %d.",
            len(rerank_input),
            len(structured_results),
        )
        return structured_results

    except Exception as e:
        logger.error("Failed during chunk retrieval: %s", e, exc_info=True)
        return []
