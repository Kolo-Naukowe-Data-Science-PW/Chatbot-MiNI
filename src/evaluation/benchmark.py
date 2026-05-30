import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import log2
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _get_top_k_chunks(query: str, top_k: int, **kwargs) -> list:
    """Lazy proxy — loads the retrieval module (and ML models) only on first call."""
    from src.api.retrieval import get_top_k_chunks  # noqa: PLC0415

    global _get_top_k_chunks  # replace self with the real function after first load

    def _real_get_top_k_chunks(q: str, top_k: int, **kw) -> list:
        return get_top_k_chunks(q, top_k=top_k, **kw)

    _get_top_k_chunks = _real_get_top_k_chunks
    return get_top_k_chunks(query, top_k=top_k, **kwargs)


def _rewrite_query(query: str) -> str:
    """Lazy proxy — loads the query rewriter (and OpenRouter client) only on first call."""
    from src.api.query_rewriter import rewrite_query  # noqa: PLC0415

    global _rewrite_query  # noqa: PLW0603
    _rewrite_query = rewrite_query
    return rewrite_query(query)


# logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum relevance score to treat a retrieved URL as "relevant" in binary
# metrics (Hit, Precision, Recall, F1, MAP, MRR).
# At the default value of 0.25, exact matches (1.0) and direct parents (0.75)
# are counted as hits; children (0.5625) and more distant relatives are not.
# Raise this to 0.5 to exclude child pages.
RELEVANCE_THRESHOLD: float = 0.25

# MRRw parameters (Metryka_chatbot.pdf)
# α: weight for a URL that is deeper (more specific) than the gold by 1 level.
#    α^d applied for d levels deeper.  0 < β < α < 1.
# β: weight for a URL that is shallower (more general) than the gold by 1 level.
#    β^|d| applied for |d| levels shallower.
MRRW_ALPHA: float = 0.8
MRRW_BETA: float = 0.4


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class EvalRow:
    query: str
    target_url: str  # single ground-truth URL per question


# ── URL normalisation and deduplication ──────────────────────────────────────


def normalize_url(url: str) -> str:
    """
    Strips trailing slashes and removes query/fragment components so that
    semantically equivalent URLs are treated as the same source.
    """
    cleaned = url.strip()
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    normalized_path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc, normalized_path, "", ""))


_DB_URLS_CACHE: frozenset[str] | None = None


def _get_db_urls() -> frozenset[str]:
    """Load all unique normalized URLs stored in the Qdrant collection (cached)."""
    global _DB_URLS_CACHE
    if _DB_URLS_CACHE is not None:
        return _DB_URLS_CACHE

    from src.api.retrieval import _get_qdrant_client  # noqa: PLC0415
    from src.ingestion.vector_db import COLLECTION_NAME  # noqa: PLC0415

    client = _get_qdrant_client()
    urls: set[str] = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1000,
            with_payload=["url"],
            with_vectors=False,
        )
        for p in points:
            raw = (p.payload or {}).get("url", "")
            normed = normalize_url(raw)
            if normed:
                urls.add(normed)
        if next_offset is None:
            break
        offset = next_offset

    _DB_URLS_CACHE = frozenset(urls)
    return _DB_URLS_CACHE


def is_covered(target_url: str, db_urls: frozenset[str]) -> bool:
    """True if target URL or its direct parent is present in the DB."""
    if not target_url:
        return False
    if target_url in db_urls:
        return True
    t = urlsplit(target_url)
    t_path = t.path.rstrip("/")
    if "/" in t_path[1:]:
        parent_path = t_path.rsplit("/", 1)[0] or "/"
        parent = normalize_url(urlunsplit((t.scheme, t.netloc, parent_path, "", "")))
        if parent in db_urls:
            return True
    return False


def unique_preserve_order(items: list[str]) -> list[str]:
    """Returns a deduplicated list while preserving the original order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


# ── Hierarchical URL relevance scoring ───────────────────────────────────────


def hierarchical_relevance(retrieved_url: str, target_url: str) -> float:
    """
    Returns a graded relevance score for a retrieved URL relative to the
    single ground-truth target URL, based on the URL path hierarchy:

        Exact match                          -> 1.00
        Direct parent (1 level above)        -> 0.75
        Grandparent (2 levels above)         -> 0.5625
        Great-grandparent (3 levels above)   -> 0.4219  etc.
        Direct child (1 level below)         -> 0.5625
        Grandchild (2 levels below)          -> 0.4219  etc.
        Unrelated path or different origin   -> 0.00

    Algorithm:
    1. Compare scheme and netloc -- different origins always yield 0.
    2. Determine ancestor / descendant relationship from the URL path.
    3. Score = 0.75^depth, where depth is the difference in path depth.
       For descendants, depth is incremented by 1 (one step stricter than
       ancestors, reflecting that a child page is a less reliable source
       than a parent page for a query about the parent).
    """
    if not retrieved_url or not target_url:
        return 0.0

    t = urlsplit(target_url)
    r = urlsplit(retrieved_url)

    # Different origins are never relevant
    if t.scheme != r.scheme or t.netloc != r.netloc:
        return 0.0

    t_path = t.path.rstrip("/")
    r_path = r.path.rstrip("/")

    if t_path == r_path:
        return 1.0

    # retrieved URL is an ancestor (parent, grandparent, ...) of the target
    if t_path.startswith(r_path + "/"):
        depth = t_path.count("/") - r_path.count("/")
        return 0.75**depth

    # retrieved URL is a descendant (child, grandchild, ...) of the target
    if r_path.startswith(t_path + "/"):
        depth = r_path.count("/") - t_path.count("/")
        return 0.75 ** (depth + 1)

    return 0.0


# ── MRRw: weighted MRR with depth-aware link accuracy ────────────────────────
#
# Reference: Metryka_chatbot.pdf (B. Gawlik, 2026-03-13)
#
# Unlike hierarchical_relevance (which uses a fixed 0.5^depth decay and treats
# child URLs as worse than parent URLs), MRRw distinguishes:
#   - deeper (more specific) links → penalised less   (weight α^d)
#   - shallower (more general) links → penalised more  (weight β^|d|)
# This reflects the intuition that returning a more specific page is better
# than returning a too-general one.


def _url_depth_difference(retrieved_url: str, target_url: str) -> int | None:
    """
    Return the signed depth difference d between retrieved_url and target_url,
    or None if the two URLs are not on the same path (unrelated).

    d > 0  → retrieved is deeper (more specific) than target by d levels
    d = 0  → exact path match
    d < 0  → retrieved is shallower (more general) than target by |d| levels

    Only ancestor / descendant relationships count; unrelated sibling paths
    (same depth but different branch) return None.
    """
    if not retrieved_url or not target_url:
        return None

    t = urlsplit(target_url)
    r = urlsplit(retrieved_url)

    if t.scheme != r.scheme or t.netloc != r.netloc:
        return None

    t_path = t.path.rstrip("/")
    r_path = r.path.rstrip("/")

    if t_path == r_path:
        return 0

    # retrieved is an ancestor (shallower): target starts with retrieved path
    if t_path.startswith(r_path + "/"):
        return -(t_path.count("/") - r_path.count("/"))  # negative → shallower

    # retrieved is a descendant (deeper): retrieved starts with target path
    if r_path.startswith(t_path + "/"):
        return r_path.count("/") - t_path.count("/")  # positive → deeper

    return None  # unrelated branch


def _mrr_weight(d: int, alpha: float = MRRW_ALPHA, beta: float = MRRW_BETA) -> float:
    """
    Accuracy weight w(d) from Metryka_chatbot.pdf:
        w(0)   = 1.0
        w(d>0) = alpha^d   (deeper is penalised less than shallower)
        w(d<0) = beta^|d|  (shallower is penalised more)
    """
    if d == 0:
        return 1.0
    if d > 0:
        return alpha**d
    return beta ** abs(d)


def mrr_weighted_single(
    retrieved_urls: list[str],
    target_url: str,
    alpha: float = MRRW_ALPHA,
    beta: float = MRRW_BETA,
) -> float:
    """
    Compute the MRRw score S for a single query (Metryka_chatbot.pdf, eq. 3).

    S = max over all returned links i of  w(d_i) / r_i

    where r_i is the 1-based rank and d_i is the depth difference.
    Unrelated links (d_i = None) are skipped.  Returns 0.0 if no link matches.
    """
    best = 0.0
    for rank, url in enumerate(retrieved_urls, start=1):
        d = _url_depth_difference(url, target_url)
        if d is None:
            continue
        s = _mrr_weight(d, alpha, beta) / rank
        if s > best:
            best = s
    return best


def mrr_weighted(
    gold: list["EvalRow"],
    k: int,
    alpha: float = MRRW_ALPHA,
    beta: float = MRRW_BETA,
) -> float:
    """
    Compute MRRw over the full evaluation set (Metryka_chatbot.pdf, eq. 4).

    MRRw = (1/N) * Σ S_j

    Parameters
    ----------
    gold : list[EvalRow]
        Evaluation set (query + target URL pairs).
    k : int
        Rank cut-off — only the top-k retrieved URLs are considered.
    alpha : float
        Weight decay for URLs deeper than the target (default 0.8).
    beta : float
        Weight decay for URLs shallower than the target (default 0.4).

    Returns
    -------
    float
        MRRw in [0, 1].
    """
    scores = []
    for row in gold:
        retrieved_chunks = _get_top_k_chunks(row.query, top_k=k)
        urls = unique_preserve_order(
            [
                normalize_url(c.get("source_url", ""))
                for c in retrieved_chunks
                if c.get("source_url", "")
            ]
        )
        scores.append(mrr_weighted_single(urls[:k], row.target_url, alpha, beta))
    return mean(scores) if scores else 0.0


# ── Gold set loading ──────────────────────────────────────────────────────────


def load_gold(path: str) -> list[EvalRow]:
    """
    Reads a CSV evaluation set.  Accepted column names (case-insensitive):
      query   : query / pytanie / question
      url     : relevant_urls / strona / url / link

    If a URL cell contains multiple URLs separated by '|', only the first
    one is used, since this evaluation assumes exactly one target per query.
    """

    def normalize_row(raw: dict[str, str | None]) -> dict[str, str]:
        return {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}

    rows: list[EvalRow] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = normalize_row(r)
            query = row.get("query") or row.get("pytanie") or row.get("question", "")
            raw_url = (
                row.get("relevant_urls")
                or row.get("strona")
                or row.get("url")
                or row.get("link")
                or ""
            )
            target_url = normalize_url(raw_url.split("|")[0])
            if not query or not target_url:
                continue
            rows.append(EvalRow(query=query, target_url=target_url))
    return rows


# ── Metric implementations ────────────────────────────────────────────────────
#
# Shared parameter conventions across all functions:
#   rel_scores  -- ordered list of relevance scores for unique retrieved URLs
#   k           -- rank cut-off
#   threshold   -- binarisation threshold (default: RELEVANCE_THRESHOLD)
#   total_rel   -- total number of relevant documents in the gold set
#                  (always 1 because each query has exactly one target URL)


def hit_at_k(
    rel_scores: list[float], k: int, threshold: float = RELEVANCE_THRESHOLD
) -> float:
    """Hit@k: 1 if at least one of the top-k results is relevant, else 0."""
    return 1.0 if any(s >= threshold for s in rel_scores[:k]) else 0.0


def mrr_at_k(
    rel_scores: list[float], k: int, threshold: float = RELEVANCE_THRESHOLD
) -> float:
    """MRR@k: reciprocal rank of the first relevant result within top-k."""
    for rank, score in enumerate(rel_scores[:k], start=1):
        if score >= threshold:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    rel_scores: list[float],
    k: int,
    total_rel: int,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Recall@k: fraction of all relevant documents found within top-k."""
    if total_rel == 0:
        return 0.0
    hits = sum(1 for s in rel_scores[:k] if s >= threshold)
    return hits / total_rel


def precision_at_k(
    rel_scores: list[float], k: int, threshold: float = RELEVANCE_THRESHOLD
) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    topk = rel_scores[:k]
    if not topk:
        return 0.0
    hits = sum(1 for s in topk if s >= threshold)
    return hits / len(topk)


def f_measure_at_k(
    rel_scores: list[float],
    k: int,
    total_rel: int,
    beta: float = 1.0,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """F_beta@k: weighted harmonic mean of Precision@k and Recall@k."""
    p = precision_at_k(rel_scores, k, threshold)
    r = recall_at_k(rel_scores, k, total_rel, threshold)
    denom = (beta**2) * p + r
    if denom == 0.0:
        return 0.0
    return (1 + beta**2) * p * r / denom


def dcg_at_k(rel_scores: list[float], k: int) -> float:
    """DCG@k using graded relevance: gain = 2^rel - 1."""
    return sum(
        (2**rel - 1) / log2(rank + 1)
        for rank, rel in enumerate(rel_scores[:k], start=1)
    )


def ideal_dcg_at_k(rel_scores: list[float], k: int) -> float:
    """IDCG@k: DCG achieved by the ideal (descending) ordering."""
    return dcg_at_k(sorted(rel_scores, reverse=True), k)


def ndcg_at_k(rel_scores: list[float], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k."""
    idcg = ideal_dcg_at_k(rel_scores, k)
    return 0.0 if idcg == 0.0 else dcg_at_k(rel_scores, k) / idcg


def average_precision_at_k(
    rel_scores: list[float],
    k: int,
    total_rel: int,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """
    AP@k: average of Precision@i over every position i (1 <= i <= k) where
    the i-th document is relevant.  Normalised by total_rel so that a
    system that retrieves the only relevant document at rank 1 scores 1.0.
    """
    if total_rel == 0:
        return 0.0
    ap, hits = 0.0, 0
    for i, score in enumerate(rel_scores[:k], start=1):
        if score >= threshold:
            hits += 1
            ap += hits / i
    return ap / total_rel


def r_precision(
    rel_scores: list[float], total_rel: int, threshold: float = RELEVANCE_THRESHOLD
) -> float:
    """R-Precision: Precision@R where R = |gold relevant set|."""
    if total_rel == 0:
        return 0.0
    hits = sum(1 for s in rel_scores[:total_rel] if s >= threshold)
    return hits / total_rel


# def source_diversity_at_k(retrieved: list[str], k: int) -> float:
#     """
#     Fraction of distinct sources among the top-k chunks.
#     1.0 means every slot comes from a different URL; lower values indicate
#     that duplicate chunks consume context-window budget unnecessarily.
#     """
#     topk = retrieved[:k]
#     if not topk:
#         return 0.0
#     return len(set(topk)) / len(topk)


# ── Precision-Recall curve helpers ───────────────────────────────────────────


def pr_curve_points(
    rel_scores: list[float], total_rel: int, threshold: float = RELEVANCE_THRESHOLD
) -> tuple[list[float], list[float]]:
    """
    Computes raw (recall, precision) pairs at every rank position.
    The full ranking is used (not truncated to k) so that the curve can
    reach recall = 1.0 whenever the relevant document is present.
    """
    recalls, precisions = [], []
    hits = 0
    for i, score in enumerate(rel_scores, start=1):
        if score >= threshold:
            hits += 1
        precisions.append(hits / i)
        recalls.append(hits / total_rel if total_rel > 0 else 0.0)
    return recalls, precisions


def interpolated_precision_at_levels(
    recalls: list[float],
    precisions: list[float],
    levels: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """
    Computes interpolated precision at standard recall levels:
        p_interp(r) = max_{r' >= r} p(r')

    Defaults to the 11-point scale [0.0, 0.1, ..., 1.0].
    """
    if levels is None:
        levels = [i / 10 for i in range(11)]
    interp = [
        max(
            (p for rec, p in zip(recalls, precisions, strict=False) if rec >= r),
            default=0.0,
        )
        for r in levels
    ]
    return levels, interp


# ── Results container ─────────────────────────────────────────────────────────


@dataclass
class MetricsAtK:
    k: int
    hit: float
    mrr: float
    mrr_weighted: float  # MRRw (Metryka_chatbot.pdf) — depth-aware weighted MRR
    recall: float
    precision: float
    f1: float
    ndcg: float
    map_score: float  # MAP@k
    r_prec: float  # R-Precision (rank-cutoff independent)
    # source_diversity: float
    # source_redundancy: float


# ── Evaluation loop ───────────────────────────────────────────────────────────


def evaluate(
    gold: list[EvalRow],
    k: int,
    use_rewrite: bool = False,
) -> tuple[MetricsAtK, list[tuple[list[float], list[float]]]]:
    """
    Evaluates the retriever on the gold set for a given rank cut-off k.

    Returns:
        metrics   -- averaged MetricsAtK instance
        pr_curves -- per-query list of (recall_points, precision_points)
    """
    hits, mrrs, mrrws, recalls, precisions = [], [], [], [], []
    f1s, ndcgs, aps, r_precs = [], [], [], []
    pr_curves: list[tuple[list[float], list[float]]] = []

    # Each query has exactly one ground-truth URL, so total_rel is always 1.
    total_rel = 1

    for index, row in enumerate(gold):
        retrieval_q = _rewrite_query(row.query) if use_rewrite else row.query
        retrieved_chunks = _get_top_k_chunks(retrieval_q, top_k=k)
        raw_urls = [normalize_url(c.get("source_url", "")) for c in retrieved_chunks]
        raw_urls = [u for u in raw_urls if u]
        unique_urls = unique_preserve_order(raw_urls)

        # Graded relevance scores for existing metrics (hierarchical, symmetric decay)
        rel_scores = [
            hierarchical_relevance(url, row.target_url) for url in unique_urls
        ]

        # MRRw score for this query (depth-aware, asymmetric α/β weights)
        mrrws.append(mrr_weighted_single(unique_urls[:k], row.target_url))

        if index < 5:
            print(f"Sample {index + 1}: {row.query}")
            print(f"  Target:     {row.target_url}")
            print(
                f"  Retrieved {len(raw_urls)} chunks "
                f"-> {len(unique_urls)} unique URLs:"
            )
            for url, score in zip(unique_urls, rel_scores, strict=False):
                count = Counter(raw_urls)[url]
                d = _url_depth_difference(url, row.target_url)
                print(f"    [rel={score:.2f}, d={d}] {url} (x{count})")
            print("-" * 50)

        hits.append(hit_at_k(rel_scores, k))
        mrrs.append(mrr_at_k(rel_scores, k))
        recalls.append(recall_at_k(rel_scores, k, total_rel))
        precisions.append(precision_at_k(rel_scores, k))
        f1s.append(f_measure_at_k(rel_scores, k, total_rel))
        ndcgs.append(ndcg_at_k(rel_scores, k))
        aps.append(average_precision_at_k(rel_scores, k, total_rel))
        r_precs.append(r_precision(rel_scores, total_rel))

        rc, pr = pr_curve_points(rel_scores, total_rel)
        pr_curves.append((rc, pr))

    def avg(lst: list[float]) -> float:
        return mean(lst) if lst else 0.0

    metrics = MetricsAtK(
        k=k,
        hit=avg(hits),
        mrr=avg(mrrs),
        mrr_weighted=avg(mrrws),
        recall=avg(recalls),
        precision=avg(precisions),
        f1=avg(f1s),
        ndcg=avg(ndcgs),
        map_score=avg(aps),
        r_prec=avg(r_precs),
    )
    return metrics, pr_curves


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_pr_curves(
    all_pr_curves: dict[int, list[tuple[list[float], list[float]]]],
    output_path: str = "pr_curves.png",
) -> None:
    """
    Produces a two-panel figure:
      Panel A -- mean interpolated P-R curves for every value of k.
      Panel B -- raw vs. interpolated mean P-R curve for the largest k,
                 illustrating the monotone-envelope property of
                 interpolated precision.
    """
    recall_levels = [i / 10 for i in range(11)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: interpolated curves for all k values
    ax = axes[0]
    for color, (k, curves) in zip(COLORS, sorted(all_pr_curves.items()), strict=False):
        mean_interp: list[float] = []
        for level_idx in range(len(recall_levels)):
            level_vals = []
            for rc, pr in curves:
                _, interp = interpolated_precision_at_levels(rc, pr, recall_levels)
                level_vals.append(interp[level_idx])
            mean_interp.append(mean(level_vals) if level_vals else 0.0)

        ax.plot(
            recall_levels,
            mean_interp,
            marker="o",
            markersize=5,
            label=f"k={k}",
            color=color,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated Precision")
    ax.set_title("Mean interpolated P-R curve (all k)")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Panel B: raw vs. interpolated for the largest k
    ax2 = axes[1]
    max_k = max(all_pr_curves)
    curves = all_pr_curves[max_k]

    # Average raw precision at each recall level (nearest-neighbour lookup)
    mean_raw_pr: dict[float, list[float]] = {r: [] for r in recall_levels}
    for rc, pr in curves:
        for level in recall_levels:
            closest = min(
                ((abs(r - level), p) for r, p in zip(rc, pr, strict=False)),
                default=(None, 0.0),
                key=lambda x: x[0],
            )
            mean_raw_pr[level].append(closest[1])

    raw_means = [mean(mean_raw_pr[r]) if mean_raw_pr[r] else 0.0 for r in recall_levels]

    mean_interp = []
    for level_idx in range(len(recall_levels)):
        vals = []
        for rc, pr in curves:
            _, interp = interpolated_precision_at_levels(rc, pr, recall_levels)
            vals.append(interp[level_idx])
        mean_interp.append(mean(vals) if vals else 0.0)

    ax2.plot(
        recall_levels,
        raw_means,
        marker="x",
        linestyle="--",
        color=COLORS[0],
        label="Raw Precision (avg)",
        alpha=0.7,
    )
    ax2.step(
        recall_levels,
        mean_interp,
        where="post",
        marker="o",
        markersize=5,
        color=COLORS[1],
        label="Interpolated Precision (avg)",
    )

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Raw vs. interpolated P-R (k={max_k})")
    ax2.legend()
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved P-R curve plot -> {output_path}")


def plot_metrics_summary(
    all_metrics: list[MetricsAtK],
    output_path: str = "metrics_summary.png",
) -> None:
    """Grouped bar chart comparing all metrics across values of k."""
    metric_labels = ["Hit", "MRR", "MRRw", "Recall", "Precision", "F1", "nDCG", "MAP"]
    attr_names = [
        "hit",
        "mrr",
        "mrr_weighted",
        "recall",
        "precision",
        "f1",
        "ndcg",
        "map_score",
    ]

    ks = [m.k for m in all_metrics]
    x = range(len(metric_labels))
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (m, color) in enumerate(zip(all_metrics, COLORS, strict=False)):
        vals = [getattr(m, attr) for attr in attr_names]
        offsets = [xi + i * bar_width for xi in x]
        ax.bar(offsets, vals, width=bar_width, label=f"k={m.k}", color=color)

    tick_pos = [xi + bar_width * (len(ks) - 1) / 2 for xi in x]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Metric value")
    ax.set_title("RAG retrieval metrics across rank cut-offs")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics summary plot -> {output_path}")


# ── Filtered gold loader (wymagany kontekst = 0) ─────────────────────────────


def load_gold_filtered(path: str) -> list[EvalRow]:
    """
    Load from questions_with_links.csv, keeping only rows where
    "wymagany kontekst" == "0" (exact match after stripping whitespace).

    These are the only questions that have a reliable gold URL — other
    categories (1, 2, 3, 0*, 0**, ?) either require personal context or
    lack a definitive source link.
    """
    rows: list[EvalRow] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            norm = {
                (k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k
            }
            if norm.get("wymagany kontekst") != "0":
                continue
            query = norm.get("pytanie") or norm.get("query") or norm.get("question", "")
            raw_url = (
                norm.get("strona")
                or norm.get("relevant_urls")
                or norm.get("url")
                or norm.get("link", "")
            )
            target_url = normalize_url(raw_url.split("|")[0])
            if not query or not target_url:
                continue
            rows.append(EvalRow(query=query, target_url=target_url))
    return rows


# ── Chatbot API helper ────────────────────────────────────────────────────────


def _call_chat_api(
    api_url: str, query: str, timeout: int = 60
) -> tuple[str, list[str]]:
    """POST query to /chat and return (answer_text, source_urls)."""
    data = json.dumps({"query": query, "language": "pl"}).encode("utf-8")
    req = Request(
        api_url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return (body.get("answer") or "").strip(), body.get("sources") or []
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Chat API error: {exc}") from exc


# ── CSV-based evaluation ──────────────────────────────────────────────────────


def evaluate_to_csv(
    gold: list[EvalRow],
    ks: list[int],
    output_dir: Path,
    api_url: str | None = None,
    answers_csv: str | None = None,
    timeout: int = 60,
    use_rewrite: bool = False,
    metric_mode: str = "both",
    check_coverage: bool = False,
    use_rerank: bool = True,
) -> dict[str, float]:
    """
    Evaluate the chatbot on gold and write two output files:
      - eval_per_query_<timestamp>.csv  — one row per query with selected metrics
      - eval_summary_<timestamp>.csv    — single-row averages (also as .json)

    Parameters
    ----------
    use_rewrite : bool
        When True, each query is rewritten via rewrite_query() before retrieval,
        matching the production /chat pipeline. Has no effect when api_url is given
        (the /chat endpoint already rewrites queries internally).
    metric_mode : str
        "standard" — compute only fixed-k metrics (hit@k, mrr@k, etc.)
        "adaptive" — compute only adaptive-k metrics (hit_adaptive, mrr_adaptive, etc.)
        "both" (default) — compute both standard and adaptive metrics

    Columns per query (varies by metric_mode):
      Standard: query | retrieval_query | chatbot_answer | chatbot_links | gold_link |
                hit@k | mrr@k | ... | r_prec
      Adaptive: query | retrieval_query | chatbot_answer | chatbot_links | gold_link |
                hit_adaptive | mrr_adaptive | ... | r_prec_adaptive
      Both: all of the above

    Adaptive metrics are computed at k = actual number of retrieved links per query.
    Standard metrics use fixed k values from the 'ks' parameter.
    """
    max_k = max(ks)
    total_rel = 1
    per_query_rows: list[dict] = []
    accum: dict[str, list[float]] = {}
    covered_hits: dict[str, list[float]] = {}
    covered_list: list[bool] = []

    db_urls: frozenset[str] | None = None
    if check_coverage:
        print("Loading DB URL index for coverage check...")
        db_urls = _get_db_urls()
        print(f"  Found {len(db_urls)} unique URLs in DB.\n")

    # Load pre-generated answers and sources if provided
    answers_map: dict[str, tuple[str, list[str]]] = {}
    if answers_csv:
        print(f"Loading pre-generated answers from {answers_csv}...")
        with open(answers_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_dict in reader:
                q = row_dict.get("pytanie", "").strip()
                ans = row_dict.get("odpowiedz_wygenerowana", "").strip()
                links_json = row_dict.get("zwrocone_linki", "[]")
                try:
                    links_data = json.loads(links_json)
                    # Extract URLs from ranked list format: [{"rank": 1, "url": "..."}, ...]
                    if isinstance(links_data, list):
                        if links_data and isinstance(links_data[0], dict):
                            links = [item.get("url", "") for item in links_data]
                        else:
                            links = links_data
                    else:
                        links = []
                except json.JSONDecodeError:
                    links = []
                if q:
                    answers_map[q] = (ans, links)
        print(f"  Loaded answers for {len(answers_map)} questions.\n")

    for idx, row in enumerate(gold):
        if answers_csv and row.query in answers_map:
            # Use pre-generated answers and sources
            answer, raw_sources = answers_map[row.query]
            retrieval_q = row.query
        elif api_url:
            try:
                answer, raw_sources = _call_chat_api(api_url, row.query, timeout)
            except RuntimeError as exc:
                print(f"  WARNING [{idx + 1}]: {exc}")
                answer, raw_sources = "", []
            retrieval_q = row.query  # rewriting happens inside /chat
        else:
            answer = ""
            retrieval_q = _rewrite_query(row.query) if use_rewrite else row.query
            chunks = _get_top_k_chunks(retrieval_q, top_k=max_k, use_rerank=use_rerank)
            raw_sources = [c.get("source_url", "") for c in chunks]

        sources = unique_preserve_order([normalize_url(u) for u in raw_sources if u])
        rel_scores = [hierarchical_relevance(u, row.target_url) for u in sources]

        csv_row: dict = {
            "query": row.query,
            "retrieval_query": retrieval_q,
            "chatbot_answer": answer,
            "chatbot_links": ";".join(sources),
            "gold_link": row.target_url,
        }

        for k in ks:
            metrics_k = {
                f"hit@{k}": hit_at_k(rel_scores, k),
                f"mrr@{k}": mrr_at_k(rel_scores, k),
                f"mrrw@{k}": mrr_weighted_single(sources[:k], row.target_url),
                f"recall@{k}": recall_at_k(rel_scores, k, total_rel),
                f"precision@{k}": precision_at_k(rel_scores, k),
                f"f1@{k}": f_measure_at_k(rel_scores, k, total_rel),
                f"ndcg@{k}": ndcg_at_k(rel_scores, k),
                f"map@{k}": average_precision_at_k(rel_scores, k, total_rel),
            }
            csv_row.update(metrics_k)
            for col, val in metrics_k.items():
                accum.setdefault(col, []).append(val)

        # Only add standard metrics if mode is not "adaptive-only"
        if metric_mode in ("standard", "both"):
            r_prec_val = r_precision(rel_scores, total_rel)
            csv_row["r_prec"] = r_prec_val
            accum.setdefault("r_prec", []).append(r_prec_val)

        # Only add adaptive metrics if mode is not "standard-only"
        if metric_mode in ("adaptive", "both"):
            # Adaptive metrics: all metrics computed at k = actual number of retrieved links
            adaptive_k = len(sources)
            csv_row["hit_adaptive"] = hit_at_k(rel_scores, adaptive_k)
            csv_row["mrr_adaptive"] = mrr_at_k(rel_scores, adaptive_k)
            csv_row["mrrw_adaptive"] = mrr_weighted_single(
                sources[:adaptive_k], row.target_url
            )
            csv_row["recall_adaptive"] = recall_at_k(rel_scores, adaptive_k, total_rel)
            csv_row["precision_adaptive"] = precision_at_k(rel_scores, adaptive_k)
            csv_row["f1_adaptive"] = f_measure_at_k(rel_scores, adaptive_k, total_rel)
            csv_row["ndcg_adaptive"] = ndcg_at_k(rel_scores, adaptive_k)
            csv_row["map_adaptive"] = average_precision_at_k(
                rel_scores, adaptive_k, total_rel
            )
            csv_row["r_prec_adaptive"] = r_precision(
                rel_scores, total_rel
            )  # k-independent

            for col in [
                "hit_adaptive",
                "mrr_adaptive",
                "mrrw_adaptive",
                "recall_adaptive",
                "precision_adaptive",
                "f1_adaptive",
                "ndcg_adaptive",
                "map_adaptive",
                "r_prec_adaptive",
            ]:
                accum.setdefault(col, []).append(csv_row[col])

        # Coverage-corrected hit@k
        if db_urls is not None:
            covered = is_covered(row.target_url, db_urls)
            covered_list.append(covered)
            csv_row["covered"] = covered
            if covered:
                for k in ks:
                    covered_hits.setdefault(f"cch@{k}", []).append(
                        hit_at_k(rel_scores, k)
                    )

        per_query_rows.append(csv_row)
        print(f"  [{idx + 1}/{len(gold)}] {row.query[:70]}")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Per-query CSV
    per_query_path = output_dir / f"eval_per_query_{ts}.csv"
    if per_query_rows:
        fieldnames = list(per_query_rows[0].keys())
        with open(per_query_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)
        print(f"\nSaved per-query results  -> {per_query_path}")

    # Summary (mean of every metric column)
    summary = {col: mean(vals) for col, vals in accum.items() if vals}

    # Coverage-corrected summary
    if db_urls is not None and covered_list:
        summary["coverage_rate"] = sum(covered_list) / len(covered_list)
        for key, vals in covered_hits.items():
            summary[key] = mean(vals) if vals else 0.0

    summary_csv_path = output_dir / f"eval_summary_{ts}.csv"
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow({k: f"{v:.6f}" for k, v in summary.items()})
    print(f"Saved summary CSV        -> {summary_csv_path}")

    summary_json_path = output_dir / f"eval_summary_{ts}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: round(v, 6) for k, v in summary.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved summary JSON       -> {summary_json_path}")

    print("\n=== Summary ===")
    for col, val in sorted(summary.items()):
        print(f"  {col:20s}: {val:.4f}")

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiNIonek retrieval benchmark — per-query CSV output."
    )
    parser.add_argument(
        "--input-csv",
        default="src/evaluation/data/questions_with_links.csv",
        help=(
            "Evaluation CSV. Use questions_with_links.csv (default) to auto-filter "
            "wymagany kontekst=0, or questions_filtered.csv for the pre-filtered set."
        ),
    )
    parser.add_argument(
        "--api-url",
        default=None,
        metavar="URL",
        help=(
            "Chatbot /chat endpoint, e.g. http://localhost:8000/chat. "
            "When given, each query is sent to the full chatbot pipeline and the "
            "returned answer + sources are used for evaluation. "
            "When omitted, the retrieval module is called directly (no answer text)."
        ),
    )
    parser.add_argument(
        "--answers-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to CSV with pre-generated answers from generate_chatbot_answers.py. "
            "Expected columns: pytanie, odpowiedz_wygenerowana, zwrocone_linki. "
            "When provided, uses stored sources instead of calling API or retrieval."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="src/evaluation/results",
        help="Directory where CSV / JSON / plot files are saved.",
    )
    parser.add_argument(
        "--ks",
        default="3,5,7,10",
        help="Comma-separated rank cut-offs (default: 3,5,7,10).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for /chat calls (only used with --api-url).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip P-R curve and bar-chart plots.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help=(
            "Rewrite each query via rewrite_query() before retrieval, matching the "
            "production /chat pipeline. Has no effect when --api-url is given "
            "(the /chat endpoint already rewrites queries internally). "
            "Requires OPENROUTER_API_KEY to be set."
        ),
    )
    parser.add_argument(
        "--metric-mode",
        choices=["standard", "adaptive", "both"],
        default="both",
        help="Metrics to compute: 'standard' (fixed k only), 'adaptive' (adaptive k only), or 'both' (default).",
    )
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help=(
            "Compute coverage-corrected Hit@k (cch@k): hit rate restricted to queries "
            "whose target URL (or direct parent) is present in the Qdrant DB. "
            "Also adds 'coverage_rate' and 'covered' column to output."
        ),
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help=(
            "Skip cross-encoder reranking — return raw RRF results instead. "
            "Useful for ablation: compare with vs. without reranker."
        ),
    )
    args = parser.parse_args()

    ks = [int(k.strip()) for k in args.ks.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation set — filter for wymagany kontekst=0 when using the full CSV
    input_path = Path(args.input_csv)
    if input_path.name == "questions_with_links.csv":
        gold = load_gold_filtered(args.input_csv)
        source_label = "questions_with_links.csv (wymagany kontekst = 0)"
    else:
        gold = load_gold(args.input_csv)
        source_label = args.input_csv

    print(f"Loaded {len(gold)} evaluation queries from {source_label}\n")

    if args.rewrite:
        print(
            "Query rewriting ENABLED — each query will be rewritten before retrieval.\n"
        )
    if args.no_rerank:
        print("Cross-encoder reranking DISABLED — using raw RRF output.\n")

    # CSV-based evaluation (per-query + summary)
    evaluate_to_csv(
        gold,
        ks,
        output_dir,
        api_url=args.api_url,
        answers_csv=args.answers_csv,
        timeout=args.timeout,
        use_rewrite=args.rewrite,
        metric_mode=args.metric_mode,
        check_coverage=args.check_coverage,
        use_rerank=not args.no_rerank,
    )

    # Plots use the legacy evaluate() loop (calls retrieval directly per k)
    if not args.no_plots:
        print("\nGenerating plots (direct retrieval per k)...")
        all_metrics: list[MetricsAtK] = []
        all_pr_curves: dict[int, list[tuple[list[float], list[float]]]] = {}

        for k in ks:
            m, pr_curves = evaluate(gold, k, use_rewrite=args.rewrite)
            all_metrics.append(m)
            all_pr_curves[k] = pr_curves
            print(
                f"  k={k}: Hit={m.hit:.4f} MRR={m.mrr:.4f} MRRw={m.mrr_weighted:.4f} "
                f"nDCG={m.ndcg:.4f} MAP={m.map_score:.4f}"
            )

        plot_pr_curves(all_pr_curves, str(output_dir / "pr_curves.png"))
        plot_metrics_summary(all_metrics, str(output_dir / "metrics_summary.png"))


if __name__ == "__main__":
    main()
