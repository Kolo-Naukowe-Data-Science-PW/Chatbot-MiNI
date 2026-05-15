"""
benchmark_hier.py — Hierarchically-aware retrieval benchmark
=============================================================

Variant of benchmark.py that rewards partial URL matches more generously.

Two key differences from benchmark.py:
  1. RELEVANCE_THRESHOLD = 0.25  (was 0.5)
     → direct child URLs (one level deeper than gold) now count as hits
       in all binary metrics (Hit, MRR, Recall, Precision, F1, MAP).
  2. Soft / graded metric variants (prefixed ``soft_``) are added as extra
     columns to the output CSV.  Unlike threshold-based metrics these never
     snap to 0 or 1 — they return the actual fractional relevance score,
     so returning a parent page earns 0.5 and returning a child page earns
     0.25 in every metric, rather than a hard 0.

Use-case this addresses
-----------------------
The test set sometimes contains a general page URL
(e.g. https://ww2.mini.pw.edu.pl/wydzial/regulaminy/)
while the chatbot correctly returns a link to a specific document
on that page (e.g. .../regulaminy/regulamin-studiow.pdf).
With RELEVANCE_THRESHOLD = 0.5 the chatbot would score 0 on all binary
metrics for that query.  With threshold = 0.25 (a direct child → score 0.25)
it counts as a hit.  The soft_ metrics give it exactly 0.25 continuous credit.

Relevance scoring (unchanged from benchmark.py)
------------------------------------------------
    Exact match                    → 1.00
    Direct parent (1 level up)     → 0.50
    Grandparent   (2 levels up)    → 0.25
    Direct child  (1 level down)   → 0.25   ← now counts as a hit
    Grandchild    (2 levels down)  → 0.125  ← still NOT a hit (below 0.25)
    Unrelated / different origin   → 0.00
"""
import argparse
import csv
import json
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import log2
from pathlib import Path
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _get_top_k_chunks(query: str, top_k: int) -> list:
    """Lazy proxy — loads the retrieval module (and ML models) only on first call."""
    from src.api.retrieval import get_top_k_chunks  # noqa: PLC0415
    global _get_top_k_chunks  # noqa: PLW0603
    _get_top_k_chunks = lambda q, top_k: get_top_k_chunks(q, top_k=top_k)  # noqa: E731
    return get_top_k_chunks(query, top_k=top_k)


def _rewrite_query(query: str) -> str:
    """Lazy proxy — loads the query rewriter (and OpenRouter client) only on first call."""
    from src.api.query_rewriter import rewrite_query  # noqa: PLC0415
    global _rewrite_query  # noqa: PLW0603
    _rewrite_query = rewrite_query
    return rewrite_query(query)


# ── Constants ─────────────────────────────────────────────────────────────────

# Lowered threshold vs benchmark.py (0.5 → 0.25) so that direct child URLs
# (relevance = 0.25) count as hits in binary metrics.
RELEVANCE_THRESHOLD: float = 0.25

# MRRw parameters (unchanged from benchmark.py)
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


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


# ── Hierarchical URL relevance scoring (unchanged from benchmark.py) ─────────

def hierarchical_relevance(retrieved_url: str, target_url: str) -> float:
    """
    Returns a graded relevance score:
        exact match            → 1.00
        direct parent          → 0.50
        grandparent            → 0.25
        direct child           → 0.25
        grandchild             → 0.125
        unrelated / different  → 0.00
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

    if t_path.startswith(r_path + "/"):
        depth = t_path.count("/") - r_path.count("/")
        return 0.5 ** depth

    if r_path.startswith(t_path + "/"):
        depth = r_path.count("/") - t_path.count("/")
        return 0.5 ** (depth + 1)

    return 0.0


# ── MRRw (depth-aware weighted MRR, unchanged from benchmark.py) ─────────────

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
        return alpha ** d
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


# ── Standard threshold-based metric implementations ──────────────────────────
# (same logic as benchmark.py, but applied with RELEVANCE_THRESHOLD = 0.25)

def hit_at_k(rel_scores: list[float], k: int,
             threshold: float = RELEVANCE_THRESHOLD) -> float:
    """Hit@k: 1 if at least one of the top-k results is relevant, else 0."""
    return 1.0 if any(s >= threshold for s in rel_scores[:k]) else 0.0


def mrr_at_k(rel_scores: list[float], k: int,
             threshold: float = RELEVANCE_THRESHOLD) -> float:
    """MRR@k: reciprocal rank of the first relevant result within top-k."""
    for rank, score in enumerate(rel_scores[:k], start=1):
        if score >= threshold:
            return 1.0 / rank
    return 0.0


def recall_at_k(rel_scores: list[float], k: int, total_rel: int,
                threshold: float = RELEVANCE_THRESHOLD) -> float:
    if total_rel == 0:
        return 0.0
    hits = sum(1 for s in rel_scores[:k] if s >= threshold)
    return hits / total_rel


def precision_at_k(rel_scores: list[float], k: int,
                   threshold: float = RELEVANCE_THRESHOLD) -> float:
    topk = rel_scores[:k]
    if not topk:
        return 0.0
    hits = sum(1 for s in topk if s >= threshold)
    return hits / len(topk)


def f_measure_at_k(rel_scores: list[float], k: int, total_rel: int,
                   beta: float = 1.0,
                   threshold: float = RELEVANCE_THRESHOLD) -> float:
    p = precision_at_k(rel_scores, k, threshold)
    r = recall_at_k(rel_scores, k, total_rel, threshold)
    denom = (beta ** 2) * p + r
    if denom == 0.0:
        return 0.0
    return (1 + beta ** 2) * p * r / denom


def dcg_at_k(rel_scores: list[float], k: int) -> float:
    return sum(
        (2 ** rel - 1) / log2(rank + 1)
        for rank, rel in enumerate(rel_scores[:k], start=1)
    )


def ideal_dcg_at_k(rel_scores: list[float], k: int) -> float:
    return dcg_at_k(sorted(rel_scores, reverse=True), k)


def ndcg_at_k(rel_scores: list[float], k: int) -> float:
    idcg = ideal_dcg_at_k(rel_scores, k)
    return 0.0 if idcg == 0.0 else dcg_at_k(rel_scores, k) / idcg


def average_precision_at_k(rel_scores: list[float], k: int, total_rel: int,
                           threshold: float = RELEVANCE_THRESHOLD) -> float:
    if total_rel == 0:
        return 0.0
    ap, hits = 0.0, 0
    for i, score in enumerate(rel_scores[:k], start=1):
        if score >= threshold:
            hits += 1
            ap += hits / i
    return ap / total_rel


def r_precision(rel_scores: list[float], total_rel: int,
                threshold: float = RELEVANCE_THRESHOLD) -> float:
    if total_rel == 0:
        return 0.0
    hits = sum(1 for s in rel_scores[:total_rel] if s >= threshold)
    return hits / total_rel


# ── NEW: Soft / graded metric variants ───────────────────────────────────────
# These never threshold — the graded relevance score is used directly, so
# returning a parent earns 0.5 credit and returning a child earns 0.25 credit
# in every metric.

def hit_soft_at_k(rel_scores: list[float], k: int) -> float:
    """Soft Hit@k: max graded relevance in top-k (1.0 exact, 0.5 parent, 0.25 child)."""
    return max(rel_scores[:k], default=0.0)


def mrr_soft_at_k(rel_scores: list[float], k: int) -> float:
    """Soft MRR@k: max of (relevance_score / rank) across all top-k results with rel > 0."""
    return max(
        (score / rank
         for rank, score in enumerate(rel_scores[:k], start=1)
         if score > 0.0),
        default=0.0,
    )


def recall_soft_at_k(rel_scores: list[float], k: int, total_rel: int) -> float:
    """Soft Recall@k: sum of graded relevance in top-k, capped at 1.0.
    Interpretation: what fraction of the 'total relevance mass' was recovered."""
    if total_rel == 0:
        return 0.0
    return min(sum(rel_scores[:k]), float(total_rel)) / total_rel


def precision_soft_at_k(rel_scores: list[float], k: int) -> float:
    """Soft Precision@k: mean graded relevance of top-k results."""
    topk = rel_scores[:k]
    return sum(topk) / len(topk) if topk else 0.0


def f1_soft_at_k(rel_scores: list[float], k: int, total_rel: int) -> float:
    """Soft F1@k: harmonic mean of soft precision and soft recall."""
    p = precision_soft_at_k(rel_scores, k)
    r = recall_soft_at_k(rel_scores, k, total_rel)
    denom = p + r
    return 0.0 if denom == 0.0 else 2 * p * r / denom


def ap_soft_at_k(rel_scores: list[float], k: int, total_rel: int) -> float:
    """Soft AP@k: graded-relevance average precision.
    At each rank i where rel_scores[i] > 0, adds (running_soft_precision@i) * rel_scores[i].
    This rewards systems that place higher-relevance results earlier.
    """
    if total_rel == 0:
        return 0.0
    ap, cumulative = 0.0, 0.0
    for i, score in enumerate(rel_scores[:k], start=1):
        cumulative += score
        if score > 0.0:
            ap += (cumulative / i) * score
    return ap / total_rel


# ── Precision-Recall curve helpers (unchanged) ───────────────────────────────

def pr_curve_points(rel_scores: list[float], total_rel: int,
                    threshold: float = RELEVANCE_THRESHOLD
                    ) -> tuple[list[float], list[float]]:
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
    if levels is None:
        levels = [i / 10 for i in range(11)]
    interp = [
        max((p for rec, p in zip(recalls, precisions) if rec >= r), default=0.0)
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
    # Soft / graded variants
    hit_soft: float
    mrr_soft: float
    recall_soft: float
    precision_soft: float
    f1_soft: float
    map_soft: float


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
        gold: list[EvalRow],
        k: int,
        use_rewrite: bool = False,
) -> tuple[MetricsAtK, list[tuple[list[float], list[float]]]]:
    hits, mrrs, mrrws, recalls, precisions = [], [], [], [], []
    f1s, ndcgs, aps, r_precs = [], [], [], []
    hit_softs, mrr_softs, recall_softs = [], [], []
    precision_softs, f1_softs, ap_softs = [], [], []
    pr_curves: list[tuple[list[float], list[float]]] = []

    total_rel = 1

    for index, row in enumerate(gold):
        retrieval_q = _rewrite_query(row.query) if use_rewrite else row.query
        retrieved_chunks = _get_top_k_chunks(retrieval_q, top_k=k)
        raw_urls = [
            normalize_url(c.get("source_url", ""))
            for c in retrieved_chunks
        ]
        raw_urls = [u for u in raw_urls if u]
        unique_urls = unique_preserve_order(raw_urls)

        rel_scores = [
            hierarchical_relevance(url, row.target_url)
            for url in unique_urls
        ]

        mrrws.append(mrr_weighted_single(unique_urls[:k], row.target_url))

        if index < 5:
            print(f"Sample {index + 1}: {row.query}")
            print(f"  Target:     {row.target_url}")
            print(
                f"  Retrieved {len(raw_urls)} chunks "
                f"-> {len(unique_urls)} unique URLs:"
            )
            for url, score in zip(unique_urls, rel_scores):
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

        hit_softs.append(hit_soft_at_k(rel_scores, k))
        mrr_softs.append(mrr_soft_at_k(rel_scores, k))
        recall_softs.append(recall_soft_at_k(rel_scores, k, total_rel))
        precision_softs.append(precision_soft_at_k(rel_scores, k))
        f1_softs.append(f1_soft_at_k(rel_scores, k, total_rel))
        ap_softs.append(ap_soft_at_k(rel_scores, k, total_rel))

        rc, pr = pr_curve_points(rel_scores, total_rel)
        pr_curves.append((rc, pr))

    avg = lambda lst: mean(lst) if lst else 0.0
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
        hit_soft=avg(hit_softs),
        mrr_soft=avg(mrr_softs),
        recall_soft=avg(recall_softs),
        precision_soft=avg(precision_softs),
        f1_soft=avg(f1_softs),
        map_soft=avg(ap_softs),
    )
    return metrics, pr_curves


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_pr_curves(
        all_pr_curves: dict[int, list[tuple[list[float], list[float]]]],
        output_path: str = "pr_curves_hier.png",
) -> None:
    recall_levels = [i / 10 for i in range(11)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for color, (k, curves) in zip(COLORS, sorted(all_pr_curves.items())):
        mean_interp: list[float] = []
        for level_idx in range(len(recall_levels)):
            level_vals = []
            for rc, pr in curves:
                _, interp = interpolated_precision_at_levels(rc, pr, recall_levels)
                level_vals.append(interp[level_idx])
            mean_interp.append(mean(level_vals) if level_vals else 0.0)
        ax.plot(recall_levels, mean_interp, marker="o", markersize=5,
                label=f"k={k}", color=color)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated Precision")
    ax.set_title("Mean interpolated P-R curve (all k) — hierarchical threshold=0.25")
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
                ((abs(r - level), p) for r, p in zip(rc, pr)),
                default=(None, 0.0),
                key=lambda x: x[0],
            )
            mean_raw_pr[level].append(closest[1])

    raw_means = [mean(mean_raw_pr[r]) if mean_raw_pr[r] else 0.0
                 for r in recall_levels]

    mean_interp = []
    for level_idx in range(len(recall_levels)):
        vals = []
        for rc, pr in curves:
            _, interp = interpolated_precision_at_levels(rc, pr, recall_levels)
            vals.append(interp[level_idx])
        mean_interp.append(mean(vals) if vals else 0.0)

    ax2.plot(recall_levels, raw_means, marker="x", linestyle="--",
             color=COLORS[0], label="Raw Precision (avg)", alpha=0.7)
    ax2.step(recall_levels, mean_interp, where="post",
             marker="o", markersize=5, color=COLORS[1],
             label="Interpolated Precision (avg)")

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
        output_path: str = "metrics_summary_hier.png",
) -> None:
    """Grouped bar chart: standard metrics (solid) vs soft metrics (hatched)."""
    std_labels  = ["Hit",      "MRR",      "MRRw",          "Recall",      "Precision",      "F1",      "nDCG", "MAP"]
    std_attrs   = ["hit",      "mrr",      "mrr_weighted",   "recall",      "precision",      "f1",      "ndcg", "map_score"]
    soft_labels = ["Hit_soft", "MRR_soft", None,             "Recall_soft", "Precision_soft", "F1_soft", None,   "MAP_soft"]
    soft_attrs  = ["hit_soft", "mrr_soft", None,             "recall_soft", "precision_soft", "f1_soft", None,   "map_soft"]

    x = range(len(std_labels))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, (m, color) in enumerate(zip(all_metrics, COLORS)):
        std_vals  = [getattr(m, attr) for attr in std_attrs]
        soft_vals = [getattr(m, attr) if attr else None for attr in soft_attrs]

        offsets = [xi + i * bar_width * 2 for xi in x]
        ax.bar(offsets, std_vals, width=bar_width, label=f"k={m.k}", color=color)
        soft_plot = [v if v is not None else 0.0 for v in soft_vals]
        ax.bar([o + bar_width for o in offsets], soft_plot, width=bar_width,
               color=color, alpha=0.45, hatch="//", label=f"k={m.k} soft")

    tick_pos = [xi + bar_width * (len(all_metrics) - 0.5) for xi in x]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(std_labels)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Metric value")
    ax.set_title("Hierarchical benchmark — standard (solid) vs soft/graded (hatched)")
    ax.legend(ncol=4, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics summary plot -> {output_path}")


# ── Gold set loaders (unchanged from benchmark.py) ───────────────────────────

def load_gold(path: str) -> list[EvalRow]:
    def normalize_row(raw: dict[str, str | None]) -> dict[str, str]:
        return {
            k.strip().lower(): (v or "").strip()
            for k, v in raw.items()
            if k
        }

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


def load_gold_filtered(path: str) -> list[EvalRow]:
    """Load questions_with_links.csv, keeping only wymagany kontekst = 0."""
    rows: list[EvalRow] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            norm = {
                (k or "").strip().lower(): (v or "").strip()
                for k, v in r.items()
                if k
            }
            if norm.get("wymagany kontekst") != "0":
                continue
            query = (
                norm.get("pytanie")
                or norm.get("query")
                or norm.get("question", "")
            )
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


# ── Chatbot API helper (unchanged) ────────────────────────────────────────────

def _call_chat_api(api_url: str, query: str, timeout: int = 60) -> tuple[str, list[str]]:
    data = json.dumps({"query": query, "language": "pl"}).encode("utf-8")
    req = Request(
        api_url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return (body.get("answer") or "").strip(), body.get("sources") or []
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Chat API error: {exc}") from exc


# ── CSV-based evaluation ──────────────────────────────────────────────────────

def evaluate_to_csv(
    gold: list[EvalRow],
    ks: list[int],
    output_dir: Path,
    api_url: str | None = None,
    timeout: int = 60,
    use_rewrite: bool = False,
) -> None:
    """
    Same as benchmark.py's evaluate_to_csv but adds soft_ metric columns.

    Standard columns (threshold=0.25):
      query | retrieval_query | chatbot_answer | chatbot_links | gold_link |
      hit@k | mrr@k | mrrw@k | recall@k | precision@k | f1@k | ndcg@k |
      map@k | r_prec | hit_adaptive | mrr_adaptive

    Extra soft columns (no threshold, continuous relevance):
      soft_hit@k | soft_mrr@k | soft_recall@k | soft_precision@k |
      soft_f1@k  | soft_map@k

    When use_rewrite is True, each query is rewritten via rewrite_query() before
    retrieval (only in direct-retrieval mode; /chat already rewrites internally).
    """
    max_k = max(ks)
    total_rel = 1
    per_query_rows: list[dict] = []
    accum: dict[str, list[float]] = {}

    for idx, row in enumerate(gold):
        if api_url:
            try:
                answer, raw_sources = _call_chat_api(api_url, row.query, timeout)
            except RuntimeError as exc:
                print(f"  WARNING [{idx + 1}]: {exc}")
                answer, raw_sources = "", []
            retrieval_q = row.query  # rewriting happens inside /chat
        else:
            answer = ""
            retrieval_q = _rewrite_query(row.query) if use_rewrite else row.query
            chunks = _get_top_k_chunks(retrieval_q, top_k=max_k)
            raw_sources = [c.get("source_url", "") for c in chunks]

        sources = unique_preserve_order(
            [normalize_url(u) for u in raw_sources if u]
        )
        rel_scores = [hierarchical_relevance(u, row.target_url) for u in sources]

        hit_adaptive = 1.0 if any(s >= RELEVANCE_THRESHOLD for s in rel_scores) else 0.0
        mrr_adaptive = mrr_at_k(rel_scores, len(sources))

        csv_row: dict = {
            "query": row.query,
            "retrieval_query": retrieval_q,
            "chatbot_answer": answer,
            "chatbot_links": ";".join(sources),
            "gold_link": row.target_url,
        }

        for k in ks:
            # Standard threshold-based metrics (threshold=0.25)
            metrics_k = {
                f"hit@{k}":       hit_at_k(rel_scores, k),
                f"mrr@{k}":       mrr_at_k(rel_scores, k),
                f"mrrw@{k}":      mrr_weighted_single(sources[:k], row.target_url),
                f"recall@{k}":    recall_at_k(rel_scores, k, total_rel),
                f"precision@{k}": precision_at_k(rel_scores, k),
                f"f1@{k}":        f_measure_at_k(rel_scores, k, total_rel),
                f"ndcg@{k}":      ndcg_at_k(rel_scores, k),
                f"map@{k}":       average_precision_at_k(rel_scores, k, total_rel),
            }
            csv_row.update(metrics_k)
            for col, val in metrics_k.items():
                accum.setdefault(col, []).append(val)

            # Soft / graded metric variants (no threshold)
            soft_k = {
                f"soft_hit@{k}":       hit_soft_at_k(rel_scores, k),
                f"soft_mrr@{k}":       mrr_soft_at_k(rel_scores, k),
                f"soft_recall@{k}":    recall_soft_at_k(rel_scores, k, total_rel),
                f"soft_precision@{k}": precision_soft_at_k(rel_scores, k),
                f"soft_f1@{k}":        f1_soft_at_k(rel_scores, k, total_rel),
                f"soft_map@{k}":       ap_soft_at_k(rel_scores, k, total_rel),
            }
            csv_row.update(soft_k)
            for col, val in soft_k.items():
                accum.setdefault(col, []).append(val)

        r_prec_val = r_precision(rel_scores, total_rel)
        csv_row["r_prec"] = r_prec_val
        accum.setdefault("r_prec", []).append(r_prec_val)

        csv_row["hit_adaptive"] = hit_adaptive
        csv_row["mrr_adaptive"] = mrr_adaptive
        accum.setdefault("hit_adaptive", []).append(hit_adaptive)
        accum.setdefault("mrr_adaptive", []).append(mrr_adaptive)

        per_query_rows.append(csv_row)
        print(f"  [{idx + 1}/{len(gold)}] {row.query[:70]}")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    per_query_path = output_dir / f"eval_hier_per_query_{ts}.csv"
    if per_query_rows:
        fieldnames = list(per_query_rows[0].keys())
        with open(per_query_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)
        print(f"\nSaved per-query results  -> {per_query_path}")

    summary = {col: mean(vals) for col, vals in accum.items() if vals}

    summary_csv_path = output_dir / f"eval_hier_summary_{ts}.csv"
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow({k: f"{v:.6f}" for k, v in summary.items()})
    print(f"Saved summary CSV        -> {summary_csv_path}")

    summary_json_path = output_dir / f"eval_hier_summary_{ts}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({k: round(v, 6) for k, v in summary.items()}, f,
                  ensure_ascii=False, indent=2)
    print(f"Saved summary JSON       -> {summary_json_path}")

    print("\n=== Summary ===")
    for col, val in sorted(summary.items()):
        print(f"  {col:25s}: {val:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MiNIonek hierarchical retrieval benchmark — "
            "threshold=0.25 + soft/graded metric variants."
        )
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
        help="Chatbot /chat endpoint. When given, calls the full pipeline.",
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

    print(f"Loaded {len(gold)} evaluation queries from {source_label}")
    print(f"RELEVANCE_THRESHOLD = {RELEVANCE_THRESHOLD} "
          f"(child URLs count as hits)\n")

    if args.rewrite:
        print("Query rewriting ENABLED — each query will be rewritten before retrieval.\n")

    evaluate_to_csv(gold, ks, output_dir, api_url=args.api_url, timeout=args.timeout,
                    use_rewrite=args.rewrite)

    if not args.no_plots:
        print("\nGenerating plots (direct retrieval per k)...")
        all_metrics: list[MetricsAtK] = []
        all_pr_curves: dict[int, list[tuple[list[float], list[float]]]] = {}

        for k in ks:
            m, pr_curves = evaluate(gold, k, use_rewrite=args.rewrite)
            all_metrics.append(m)
            all_pr_curves[k] = pr_curves
            print(f"  k={k}: Hit={m.hit:.4f} Hit_soft={m.hit_soft:.4f} "
                  f"MRR={m.mrr:.4f} MRRw={m.mrr_weighted:.4f} "
                  f"nDCG={m.ndcg:.4f} MAP={m.map_score:.4f}")

        plot_pr_curves(all_pr_curves, str(output_dir / "pr_curves_hier.png"))
        plot_metrics_summary(all_metrics, str(output_dir / "metrics_summary_hier.png"))


if __name__ == "__main__":
    main()
