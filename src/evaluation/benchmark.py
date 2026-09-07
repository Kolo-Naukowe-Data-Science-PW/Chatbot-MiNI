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

    global _get_top_k_chunks

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


# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum relevance score to treat a retrieved URL as "relevant" in binary
# metrics (Hit, Precision, Recall, F1, MAP, MRR).
# The threshold is STRICT (>), matching the definition:
#   Z_rel^τ(q) = { z ∈ Z : rel(z, q) > τ }
# At τ=0.25: exact matches (1.0) and direct parents (0.75) are counted as hits;
# children (0.5625) are not.  Raise to 0.5 to also exclude child pages.
RELEVANCE_THRESHOLD: float = 0.25

# MRRw parameters (Metryka_chatbot.pdf)
MRRW_ALPHA: float = 0.8
MRRW_BETA: float = 0.4


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class EvalRow:
    query: str
    target_url: str


# ── URL normalisation and deduplication ──────────────────────────────────────


def normalize_url(url: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    normalized_path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc, normalized_path, "", ""))


_DB_URLS_CACHE: frozenset[str] | None = None


def _get_db_urls() -> frozenset[str]:
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
        Direct parent (1 level above)        -> 0.75^1  = 0.75
        Grandparent (2 levels above)         -> 0.75^2  = 0.5625
        Direct child (1 level below)         -> 0.75^2  = 0.5625
        Grandchild (2 levels below)          -> 0.75^3  = 0.4219
        Unrelated path or different origin   -> 0.00

    Per the definition:
        rel(z, q) = 0.75^d      if z is an ancestor at depth d
        rel(z, q) = 0.75^(d+1) if z is a descendant at depth d
    """
    if not retrieved_url or not target_url:
        return 0.0

    t = urlsplit(target_url)
    r = urlsplit(retrieved_url)

    if t.scheme != r.scheme or t.netloc != r.netloc:
        return 0.0

    t_path = t.path.rstrip("/")
    r_path = r.path.rstrip("/")

    if t_path == r_path:
        return 1.0

    # retrieved URL is an ancestor (parent, grandparent, …) of the target
    if t_path.startswith(r_path + "/"):
        depth = t_path.count("/") - r_path.count("/")
        return 0.75**depth

    # retrieved URL is a descendant (child, grandchild, …) of the target
    if r_path.startswith(t_path + "/"):
        depth = r_path.count("/") - t_path.count("/")
        return 0.75 ** (depth + 1)

    return 0.0


# ── MRRw ─────────────────────────────────────────────────────────────────────


def _url_depth_difference(retrieved_url: str, target_url: str) -> int | None:
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

    if t_path.startswith(r_path + "/"):
        return -(t_path.count("/") - r_path.count("/"))

    if r_path.startswith(t_path + "/"):
        return r_path.count("/") - t_path.count("/")

    return None


def _mrr_weight(d: int, alpha: float = MRRW_ALPHA, beta: float = MRRW_BETA) -> float:
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
    def normalize_row(raw: dict[str, str | None]) -> dict[str, str]:
        return {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}

    def make_row(row: dict[str, str]) -> EvalRow | None:
        query = row.get("query") or row.get("pytanie") or row.get("question", "")
        raw_url = (
            row.get("relevant_urls")
            or row.get("strona")
            or row.get("url")
            or row.get("link")
            or row.get("źródła")
            or row.get("zrodla")
            or ""
        )
        raw_url = raw_url.splitlines()[0] if raw_url else ""
        target_url = normalize_url(raw_url.split("|")[0])
        if not query or not target_url:
            return None
        return EvalRow(query=query, target_url=target_url)

    def notebooklm_source_map(jsonl_path: Path) -> dict[str, str]:
        if not jsonl_path.exists():
            return {}
        source_by_query: dict[str, str] = {}
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                row = normalize_row({str(k): str(v) for k, v in record.items()})
                query = (
                    row.get("pytanie") or row.get("query") or row.get("question", "")
                )
                raw_url = row.get("źródła") or row.get("zrodla") or ""
                raw_url = raw_url.splitlines()[0] if raw_url else ""
                target_url = normalize_url(raw_url.split("|")[0])
                if query and target_url:
                    source_by_query[query] = target_url
        return source_by_query

    rows: list[EvalRow] = []
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                row = normalize_row({str(k): str(v) for k, v in record.items()})
                eval_row = make_row(row)
                if eval_row:
                    rows.append(eval_row)
        return rows

    unresolved_queries: list[str] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        f.seek(0)
        delimiter = "|" if "|" in header else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for r in reader:
            row = normalize_row(r)
            eval_row = make_row(row)
            if eval_row:
                rows.append(eval_row)
            else:
                query = (
                    row.get("query") or row.get("pytanie") or row.get("question", "")
                )
                if query:
                    unresolved_queries.append(query)

    if unresolved_queries:
        source_by_query = notebooklm_source_map(
            input_path.with_name("final_notebooklm_QA.jsonl")
        )
        seen = {row.query for row in rows}
        for query in unresolved_queries:
            target_url = source_by_query.get(query, "")
            if query not in seen and target_url:
                rows.append(EvalRow(query=query, target_url=target_url))
                seen.add(query)

    return rows


# ── Metric implementations ────────────────────────────────────────────────────
#
# Threshold semantics: a URL is "relevant" iff rel(z, q) > τ  (strict ">")
# matching the definition  Z_rel^τ(q) = { z ∈ Z : rel(z,q) > τ }.


def _is_relevant(score: float, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """Strict threshold check: rel(z,q) > τ."""
    # FIX: was ">=" throughout the codebase; the definition uses strict ">"
    return score > threshold


def hit_at_k(
    rel_scores: list[float], k: int, threshold: float = RELEVANCE_THRESHOLD
) -> float:
    """Hit@k: 1 if at least one of the top-k results is relevant, else 0."""
    return 1.0 if any(_is_relevant(s, threshold) for s in rel_scores[:k]) else 0.0


def mrr_at_k(
    rel_scores: list[float], k: int, threshold: float = RELEVANCE_THRESHOLD
) -> float:
    """MRR@k: reciprocal rank of the first relevant result within top-k."""
    for rank, score in enumerate(rel_scores[:k], start=1):
        if _is_relevant(score, threshold):
            return 1.0 / rank
    return 0.0


def recall_at_k(
    rel_scores: list[float],
    k: int,
    total_rel: int,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """
    Recall@k ≈ Hit@k per the specification.

    The spec states:
        Recall@k(q) = Hit@k(q)
    because computing |Z_rel^τ(q)| over all Z is too expensive; instead
    Z_rel^τ is approximated by restricting it to the top-k retrieved documents.
    With a single gold URL per query, this equals Hit@k exactly.
    """
    # FIX: was hits/total_rel; the spec approx. Recall@k = Hit@k
    return hit_at_k(rel_scores, k, threshold)


def precision_at_k(
    rel_scores: list[float], k: int, threshold: float = RELEVANCE_THRESHOLD
) -> float:
    """
    Precision@k: fraction of the top-k slots that are relevant.

    Definition: |Z_top-k ∩ Z_rel| / |Z_top-k| = hits / k.
    The denominator is always k (the requested cut-off), not the number of
    actually retrieved documents — even if fewer than k were returned.
    """
    # FIX: was hits/len(topk); definition divides by k (fixed denominator)
    hits = sum(1 for s in rel_scores[:k] if _is_relevant(s, threshold))
    return hits / k if k > 0 else 0.0


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
    AP@k (modern definition):
        AP(q) = (1 / |K_rel(q)|) * Σ_{i ∈ K_rel^(k)(q)} Precision@i(q)

    where K_rel^(k)(q) = { i ≤ k : z_i is relevant }.
    Normalised by the number of relevant documents actually found within
    the top-k (|K_rel^(k)(q)|), not by total_rel.

    Edge cases:
    - If no relevant document is found in the top-k, AP@k = 0.
    - With a single gold URL per query, |K_rel^(k)| is either 0 or 1,
      so AP@k collapses to Precision@rank_of_first_hit (or 0).
    """
    # FIX: was normalised by total_rel; definition normalises by |K_rel(q)| —
    # the count of relevant documents retrieved, not the total gold set size.
    ap, hits = 0.0, 0
    for i, score in enumerate(rel_scores[:k], start=1):
        if _is_relevant(score, threshold):
            hits += 1
            ap += hits / i
    if hits == 0:
        return 0.0
    return ap / hits  # normalise by |K_rel^(k)(q)|


def r_precision(
    rel_scores: list[float], total_rel: int, threshold: float = RELEVANCE_THRESHOLD
) -> float:
    """R-Precision: Precision@R where R = |gold relevant set|."""
    if total_rel == 0:
        return 0.0
    hits = sum(1 for s in rel_scores[:total_rel] if _is_relevant(s, threshold))
    return hits / total_rel


# ── Precision-Recall curve helpers ───────────────────────────────────────────


def pr_curve_points(
    rel_scores: list[float], total_rel: int, threshold: float = RELEVANCE_THRESHOLD
) -> tuple[list[float], list[float]]:
    """
    Computes raw (recall, precision) pairs at every rank position.
    Uses the corrected precision denominator (fixed k) and strict threshold.
    """
    recalls, precisions = [], []
    hits = 0
    for i, score in enumerate(rel_scores, start=1):
        if _is_relevant(score, threshold):
            hits += 1
        precisions.append(hits / i)  # P@i always divides by i (the rank)
        recalls.append(hits / total_rel if total_rel > 0 else 0.0)
    return recalls, precisions


def interpolated_precision_at_levels(
    recalls: list[float],
    precisions: list[float],
    levels: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """
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
    mrr_weighted: float
    recall: float
    precision: float
    f1: float
    ndcg: float
    map_score: float
    r_prec: float


# ── Evaluation loop ───────────────────────────────────────────────────────────


def evaluate(
    gold: list[EvalRow],
    k: int,
    use_rewrite: bool = False,
) -> tuple[MetricsAtK, list[tuple[list[float], list[float]]]]:
    hits, mrrs, mrrws, recalls, precisions = [], [], [], [], []
    f1s, ndcgs, aps, r_precs = [], [], [], []
    pr_curves: list[tuple[list[float], list[float]]] = []

    total_rel = 1  # each query has exactly one gold URL

    for index, row in enumerate(gold):
        retrieval_q = _rewrite_query(row.query) if use_rewrite else row.query
        retrieved_chunks = _get_top_k_chunks(retrieval_q, top_k=k)
        raw_urls = [normalize_url(c.get("source_url", "")) for c in retrieved_chunks]
        raw_urls = [u for u in raw_urls if u]
        unique_urls = unique_preserve_order(raw_urls)

        rel_scores = [
            hierarchical_relevance(url, row.target_url) for url in unique_urls
        ]

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
    recall_levels = [i / 10 for i in range(11)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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

    ax2 = axes[1]
    max_k = max(all_pr_curves)
    curves = all_pr_curves[max_k]

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


# ── Filtered gold loader ──────────────────────────────────────────────────────


def load_gold_filtered(path: str) -> list[EvalRow]:
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
            answer, raw_sources = answers_map[row.query]
            retrieval_q = row.query
        elif api_url:
            try:
                answer, raw_sources = _call_chat_api(api_url, row.query, timeout)
            except RuntimeError as exc:
                print(f"  WARNING [{idx + 1}]: {exc}")
                answer, raw_sources = "", []
            retrieval_q = row.query
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

        if metric_mode in ("standard", "both"):
            r_prec_val = r_precision(rel_scores, total_rel)
            csv_row["r_prec"] = r_prec_val
            accum.setdefault("r_prec", []).append(r_prec_val)

        if metric_mode in ("adaptive", "both"):
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
            csv_row["r_prec_adaptive"] = r_precision(rel_scores, total_rel)

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

    per_query_path = output_dir / f"eval_per_query_{ts}.csv"
    if per_query_rows:
        fieldnames = list(per_query_rows[0].keys())
        with open(per_query_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)
        print(f"\nSaved per-query results  -> {per_query_path}")

    summary = {col: mean(vals) for col, vals in accum.items() if vals}

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
    )
    parser.add_argument("--api-url", default=None, metavar="URL")
    parser.add_argument("--answers-csv", default=None, metavar="PATH")
    parser.add_argument("--output-dir", default="src/evaluation/results")
    parser.add_argument("--ks", default="3,5,7,10")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument(
        "--metric-mode",
        choices=["standard", "adaptive", "both"],
        default="both",
    )
    parser.add_argument("--check-coverage", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    ks = [int(k.strip()) for k in args.ks.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_csv)
    if input_path.name == "questions_with_links.csv":
        gold = load_gold_filtered(args.input_csv)
        source_label = "questions_with_links.csv (wymagany kontekst = 0)"
    else:
        gold = load_gold(args.input_csv)
        source_label = args.input_csv

    print(f"Loaded {len(gold)} evaluation queries from {source_label}\n")

    if args.rewrite:
        print("Query rewriting ENABLED\n")
    if args.no_rerank:
        print("Cross-encoder reranking DISABLED\n")

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
