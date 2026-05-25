"""
Retrieval runner for RAG ablation experiments.

Exposes a single public function:
    retrieve(query, config) -> list[dict]

Each returned dict has keys: text_chunk, source_url, score.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    Prefetch,
    SparseVector,
)

from src.api.retrieval import (
    _get_qdrant_client,
    _get_sparse_vector,
    embedder,
    reranker,
)
from src.evaluation.pipeline_config import PipelineConfig
from src.ingestion.vector_db import COLLECTION_NAME
from src.utils.paths import get_data_dir

logger = logging.getLogger(__name__)

COLLECTION_NAME_CHUNKS = "mini_chunks"

_chunks_client: QdrantClient | None = None


def _get_chunks_qdrant_client() -> QdrantClient:
    global _chunks_client
    if _chunks_client is None:
        chunks_dir = os.environ.get("QDRANT_CHUNKS_DIR", get_data_dir("qdrant_chunks_db"))
        _chunks_client = QdrantClient(path=chunks_dir)
    return _chunks_client


def _client_for(collection: str) -> QdrantClient:
    return _get_chunks_qdrant_client() if collection == COLLECTION_NAME_CHUNKS else _get_qdrant_client()

# Top-N URLs retrieved in stage 1 of two-stage retrieval
_TWO_STAGE_TOP_URLS = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dense_vector(query: str) -> list[float]:
    return embedder.generate_embeddings([query])[0]


def _build_prefetch(dense_vec: list[float], sparse_vec: SparseVector, limit: int) -> list[Prefetch]:
    return [
        Prefetch(query=dense_vec, using="dense", limit=limit * 3),
        Prefetch(query=sparse_vec, using="sparse", limit=limit * 3),
    ]


def _hybrid_query(
    query: str,
    collection: str,
    n: int,
    url_filter: Filter | None = None,
) -> list[dict[str, Any]]:
    """RRF-fused hybrid retrieval from *collection*, returning up to *n* results."""
    client = _client_for(collection)
    dense_vec = _dense_vector(query)
    sparse_vec = _get_sparse_vector(query)

    results = client.query_points(
        collection_name=collection,
        prefetch=_build_prefetch(dense_vec, sparse_vec, n),
        query=FusionQuery(fusion=Fusion.RRF),
        limit=n,
        with_payload=True,
        query_filter=url_filter,
    )
    return [
        {
            "text_chunk": p.payload.get("text", ""),
            "source_url": p.payload.get("url", "Unknown Source"),
            "score": float(p.score) if p.score is not None else 0.0,
        }
        for p in results.points
    ]


def _dense_query(
    query: str,
    collection: str,
    n: int,
    url_filter: Filter | None = None,
) -> list[dict[str, Any]]:
    """Pure dense (cosine) retrieval from *collection*, returning up to *n* results."""
    client = _client_for(collection)
    dense_vec = _dense_vector(query)

    results = client.query_points(
        collection_name=collection,
        query=dense_vec,
        using="dense",
        limit=n,
        with_payload=True,
        query_filter=url_filter,
    )
    return [
        {
            "text_chunk": p.payload.get("text", ""),
            "source_url": p.payload.get("url", "Unknown Source"),
            "score": float(p.score) if p.score is not None else 0.0,
        }
        for p in results.points
    ]


def _rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """Cross-encoder rerank *candidates*, return top *top_n* by score."""
    if not candidates:
        return []
    pairs = [(query, c["text_chunk"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [
        {**c, "score": float(s)}
        for s, c in ranked[:top_n]
    ]


def _collection_for_content(content_type: str) -> str:
    return COLLECTION_NAME if content_type == "facts" else COLLECTION_NAME_CHUNKS


def _check_collection_exists(collection: str) -> bool:
    try:
        return _client_for(collection).collection_exists(collection)
    except Exception as exc:
        logger.warning("Could not check collection '%s': %s", collection, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, config: PipelineConfig) -> list[dict]:
    """
    Run retrieval according to *config* for *query*.

    Parameters
    ----------
    query:
        Raw user query (rewriting is applied internally if config.use_query_rewrite).
    config:
        PipelineConfig describing the retrieval variant.

    Returns
    -------
    list[dict]
        Each dict: {text_chunk: str, source_url: str, score: float}
    """
    # --- Query rewriting ---
    retrieval_query = query
    if config.use_query_rewrite:
        try:
            from src.api.query_rewriter import rewrite_query  # lazy import
            retrieval_query = rewrite_query(query)
        except Exception as exc:
            logger.warning("Query rewriting failed (%s), using original.", exc)

    # --- Choose collection ---
    collection = _collection_for_content(config.content_type)

    if collection == COLLECTION_NAME_CHUNKS and not _check_collection_exists(COLLECTION_NAME_CHUNKS):
        logger.warning(
            "Collection '%s' does not exist — returning empty list. "
            "Run src/ingestion/ingest_chunks.py first.",
            COLLECTION_NAME_CHUNKS,
        )
        return []

    # --- Two-stage retrieval ---
    if config.two_stage:
        return _two_stage_retrieve(retrieval_query, config)

    # --- Standard retrieval ---
    return _standard_retrieve(retrieval_query, config, collection)


# ---------------------------------------------------------------------------
# Standard retrieval path
# ---------------------------------------------------------------------------

def _standard_retrieve(
    retrieval_query: str,
    config: PipelineConfig,
    collection: str,
) -> list[dict]:
    # Fetch more candidates than needed so reranker has something to work with.
    fetch_n = config.n_retrieve * 2

    if config.retrieval_mode == "dense_only":
        candidates = _dense_query(retrieval_query, collection, fetch_n)
    else:
        candidates = _hybrid_query(retrieval_query, collection, fetch_n)

    # Trim to n_retrieve
    candidates = candidates[: config.n_retrieve]

    if not candidates:
        return []

    if not config.use_rerank:
        logger.info("Reranking disabled — returning top %d RRF results.", config.n_final)
        return candidates[: config.n_final]

    if config.rerank_steps == 2:
        return _two_step_rerank(retrieval_query, candidates, config)

    # Single-step rerank
    return _rerank_candidates(retrieval_query, candidates, config.n_final)


def _two_step_rerank(
    query: str,
    candidates: list[dict],
    config: PipelineConfig,
) -> list[dict]:
    """60 candidates -> rerank -> 30 -> rerank -> 15."""
    mid = config.n_retrieve // 2  # e.g. 60//2 = 30
    step1 = _rerank_candidates(query, candidates, mid)
    step2 = _rerank_candidates(query, step1, config.n_final)
    return step2


# ---------------------------------------------------------------------------
# Two-stage retrieval path
# ---------------------------------------------------------------------------

def _two_stage_retrieve(
    retrieval_query: str,
    config: PipelineConfig,
) -> list[dict]:
    """
    Stage 1: retrieve from mini_docs (facts) to find top-N URLs by max score.
    Stage 2: retrieve from the target collection filtered to those URLs, rerank.
    """
    # Stage 1 always uses mini_docs regardless of content_type
    stage1_n = config.n_retrieve * 2
    stage1_candidates = _hybrid_query(retrieval_query, COLLECTION_NAME, stage1_n)

    # Aggregate: max score per URL
    url_max_score: dict[str, float] = {}
    for c in stage1_candidates:
        url = c["source_url"]
        if url not in url_max_score or c["score"] > url_max_score[url]:
            url_max_score[url] = c["score"]

    # Pick top URLs
    top_urls = sorted(url_max_score, key=lambda u: url_max_score[u], reverse=True)
    top_urls = top_urls[:_TWO_STAGE_TOP_URLS]

    if not top_urls:
        logger.warning("Two-stage retrieval: no URLs found in stage 1.")
        return []

    logger.info("Two-stage stage 1: selected %d URLs.", len(top_urls))

    # Build Qdrant filter for stage 2
    url_filter = Filter(
        must=[FieldCondition(key="url", match=MatchAny(any=top_urls))]
    )

    # Stage 2 collection
    stage2_collection = _collection_for_content(config.two_stage_content)

    if stage2_collection == COLLECTION_NAME_CHUNKS and not _check_collection_exists(COLLECTION_NAME_CHUNKS):
        logger.warning(
            "Collection '%s' does not exist — returning empty list.",
            COLLECTION_NAME_CHUNKS,
        )
        return []

    # Stage 2 retrieval within the filtered URLs
    stage2_fetch = config.n_retrieve * 2
    stage2_candidates = _hybrid_query(
        retrieval_query, stage2_collection, stage2_fetch, url_filter=url_filter
    )
    stage2_candidates = stage2_candidates[: config.n_retrieve]

    if not stage2_candidates:
        return []

    if not config.use_rerank:
        return stage2_candidates[: config.n_final]

    return _rerank_candidates(retrieval_query, stage2_candidates, config.n_final)
