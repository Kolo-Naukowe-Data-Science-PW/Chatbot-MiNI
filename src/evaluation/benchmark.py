import csv
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
from dataclasses import dataclass
from math import log2
from pathlib import Path
from statistics import mean
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.retrieval import get_top_k_chunks

#logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum relevance score to treat a retrieved URL as "relevant" in binary
# metrics (Hit, Precision, Recall, F1, MAP, MRR).
# At the default value of 0.5, exact matches (1.0) and direct parents (0.5)
# are counted as hits; children (0.25) and more distant relatives are not.
# Lower this to 0.25 to also credit child pages.
RELEVANCE_THRESHOLD: float = 0.5

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
        Direct parent (1 level above)        -> 0.50
        Grandparent (2 levels above)         -> 0.25
        Great-grandparent (3 levels above)   -> 0.125  etc.
        Direct child (1 level below)         -> 0.25
        Grandchild (2 levels below)          -> 0.125  etc.
        Unrelated path or different origin   -> 0.00

    Algorithm:
    1. Compare scheme and netloc -- different origins always yield 0.
    2. Determine ancestor / descendant relationship from the URL path.
    3. Score = 0.5^depth, where depth is the difference in path depth.
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
        return 0.5 ** depth

    # retrieved URL is a descendant (child, grandchild, ...) of the target
    if r_path.startswith(t_path + "/"):
        depth = r_path.count("/") - t_path.count("/")
        return 0.5 ** (depth + 1)

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
        return alpha ** d
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
        retrieved_chunks = get_top_k_chunks(row.query, top_k=k)
        urls = unique_preserve_order([
            normalize_url(c.get("source_url", ""))
            for c in retrieved_chunks
            if c.get("source_url", "")
        ])
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


# ── Metric implementations ────────────────────────────────────────────────────
#
# Shared parameter conventions across all functions:
#   rel_scores  -- ordered list of relevance scores for unique retrieved URLs
#   k           -- rank cut-off
#   threshold   -- binarisation threshold (default: RELEVANCE_THRESHOLD)
#   total_rel   -- total number of relevant documents in the gold set
#                  (always 1 because each query has exactly one target URL)


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
    """Recall@k: fraction of all relevant documents found within top-k."""
    if total_rel == 0:
        return 0.0
    hits = sum(1 for s in rel_scores[:k] if s >= threshold)
    return hits / total_rel


def precision_at_k(rel_scores: list[float], k: int,
                   threshold: float = RELEVANCE_THRESHOLD) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    topk = rel_scores[:k]
    if not topk:
        return 0.0
    hits = sum(1 for s in topk if s >= threshold)
    return hits / len(topk)


def f_measure_at_k(rel_scores: list[float], k: int, total_rel: int,
                   beta: float = 1.0,
                   threshold: float = RELEVANCE_THRESHOLD) -> float:
    """F_beta@k: weighted harmonic mean of Precision@k and Recall@k."""
    p = precision_at_k(rel_scores, k, threshold)
    r = recall_at_k(rel_scores, k, total_rel, threshold)
    denom = (beta ** 2) * p + r
    if denom == 0.0:
        return 0.0
    return (1 + beta ** 2) * p * r / denom


def dcg_at_k(rel_scores: list[float], k: int) -> float:
    """DCG@k using graded relevance: gain = 2^rel - 1."""
    return sum(
        (2 ** rel - 1) / log2(rank + 1)
        for rank, rel in enumerate(rel_scores[:k], start=1)
    )


def ideal_dcg_at_k(rel_scores: list[float], k: int) -> float:
    """IDCG@k: DCG achieved by the ideal (descending) ordering."""
    return dcg_at_k(sorted(rel_scores, reverse=True), k)


def ndcg_at_k(rel_scores: list[float], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k."""
    idcg = ideal_dcg_at_k(rel_scores, k)
    return 0.0 if idcg == 0.0 else dcg_at_k(rel_scores, k) / idcg


def average_precision_at_k(rel_scores: list[float], k: int, total_rel: int,
                           threshold: float = RELEVANCE_THRESHOLD) -> float:
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


def r_precision(rel_scores: list[float], total_rel: int,
                threshold: float = RELEVANCE_THRESHOLD) -> float:
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

def pr_curve_points(rel_scores: list[float], total_rel: int,
                    threshold: float = RELEVANCE_THRESHOLD
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
    mrr_weighted: float     # MRRw (Metryka_chatbot.pdf) — depth-aware weighted MRR
    recall: float
    precision: float
    f1: float
    ndcg: float
    map_score: float        # MAP@k
    r_prec: float           # R-Precision (rank-cutoff independent)
    # source_diversity: float
    # source_redundancy: float


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
        gold: list[EvalRow],
        k: int,
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
        retrieved_chunks = get_top_k_chunks(row.query, top_k=k)
        raw_urls = [
            normalize_url(c.get("source_url", ""))
            for c in retrieved_chunks
        ]
        raw_urls = [u for u in raw_urls if u]
        unique_urls = unique_preserve_order(raw_urls)

        # Graded relevance scores for existing metrics (hierarchical, symmetric decay)
        rel_scores = [
            hierarchical_relevance(url, row.target_url)
            for url in unique_urls
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
        output_path: str = "metrics_summary.png",
) -> None:
    """Grouped bar chart comparing all metrics across values of k."""
    metric_labels = ["Hit", "MRR", "MRRw", "Recall", "Precision", "F1", "nDCG", "MAP"]
    attr_names    = ["hit", "mrr", "mrr_weighted", "recall", "precision", "f1", "ndcg", "map_score"]

    ks = [m.k for m in all_metrics]
    x = range(len(metric_labels))
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (m, color) in enumerate(zip(all_metrics, COLORS)):
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    gold_path = "src/evaluation/data/questions_filtered.csv"
    ks = [ 3, 5,7, 10]

    gold = load_gold(gold_path)
    print(f"Loaded {len(gold)} evaluation queries\n")

    all_metrics: list[MetricsAtK] = []
    all_pr_curves: dict[int, list[tuple[list[float], list[float]]]] = {}

    for k in ks:
        m, pr_curves = evaluate(gold, k)
        all_metrics.append(m)
        all_pr_curves[k] = pr_curves

        print(f"=== k={k} {'=' * 30}")
        print(f"  Hit@{k}:          {m.hit:.4f}")
        print(f"  MRR@{k}:          {m.mrr:.4f}")
        print(f"  MRRw@{k}:         {m.mrr_weighted:.4f}  (α={MRRW_ALPHA}, β={MRRW_BETA})")
        print(f"  Recall@{k}:       {m.recall:.4f}")
        print(f"  Precision@{k}:    {m.precision:.4f}")
        print(f"  F1@{k}:           {m.f1:.4f}")
        print(f"  nDCG@{k}:         {m.ndcg:.4f}")
        print(f"  MAP@{k}:          {m.map_score:.4f}")
        print(f"  R-Precision:      {m.r_prec:.4f}  (rank-cutoff independent)")
        print()

    plot_pr_curves(all_pr_curves)
    plot_metrics_summary(all_metrics)


if __name__ == "__main__":
    main()
