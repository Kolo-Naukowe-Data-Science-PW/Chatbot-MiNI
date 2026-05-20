"""
Statistical tests for comparing two model variants (model A vs model B).

Implements:
  1. Wilcoxon Signed-Rank Test — non-parametric comparison of paired metrics
  2. Goodman–Kruskal γ (gamma) — ordinal association between ranks
  3. Kappa Coefficient — stability of retriever across multiple runs

Usage:
    python -m evaluation.statistical_comparison \\
        --model_a_csv eval_per_query_20260515T100000.csv \\
        --model_b_csv eval_per_query_20260515T110000.csv \\
        --metric mrr@10 \\
        --output_dir src/evaluation/data/stats

"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ComparisonResult:
    """Container for statistical test results."""

    metric: str
    n_queries: int
    mean_A: float
    mean_B: float
    mean_diff: float
    std_diff: float

    # Wilcoxon test
    wilcoxon_stat: float
    wilcoxon_pvalue: float
    wilcoxon_sig: bool  # p <= 0.05

    # Goodman-Kruskal gamma
    gamma_coeff: float
    gamma_concordant: int
    gamma_discordant: int

    # Kappa (if multi-run data available)
    kappa_avg: float | None = None
    kappa_min: float | None = None
    kappa_max: float | None = None


def load_csv_metrics(csv_path: Path, metric_cols: list[str]) -> dict[str, list[float]]:
    """
    Load metrics from eval_per_query CSV.

    Returns:
        {metric_name: [score_1, score_2, ...]}
    """
    metrics = {col: [] for col in metric_cols}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in metric_cols:
                if col in row:
                    try:
                        metrics[col].append(float(row[col]))
                    except (ValueError, TypeError):
                        metrics[col].append(0.0)

    return metrics


def calculate_adaptive_metrics(csv_path: Path) -> dict[str, list[float]]:
    """
    Calculate metrics using adaptive k = actual number of retrieved links.

    For each query, computes:
      - hit_adaptive: 1 if gold_link is in chatbot_links, 0 otherwise
      - mrr_adaptive: 1/rank if found, 0 otherwise (rank = position in list)
      - rank_adaptive: position of gold link (999 if not found)
      - n_links: number of retrieved links

    Returns:
        {
            'hit_adaptive': [0/1 per query],
            'mrr_adaptive': [1/rank per query],
            'rank_adaptive': [rank per query],
            'n_links': [count per query],
        }
    """
    hits = []
    mrrs = []
    ranks = []
    n_links_list = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources_str = row.get("chatbot_links", "")
            gold_link = row.get("gold_link", "").strip()

            n_links = count_retrieved_links(sources_str)
            n_links_list.append(n_links)

            # Parse retrieved URLs
            sources = [u.strip() for u in sources_str.split(";") if u.strip()]

            # Normalize URLs for comparison
            sources_normalized = [normalize_url(u) for u in sources]
            gold_normalized = normalize_url(gold_link)

            # Check if gold link is in retrieved links
            if gold_normalized in sources_normalized:
                rank = sources_normalized.index(gold_normalized) + 1  # 1-indexed
                hits.append(1.0)
                mrrs.append(1.0 / rank)
                ranks.append(rank)
            else:
                hits.append(0.0)
                mrrs.append(0.0)
                ranks.append(999)  # not retrieved

    return {
        "hit_adaptive": hits,
        "mrr_adaptive": mrrs,
        "rank_adaptive": ranks,
        "n_links": n_links_list,
    }


def extract_rank_from_sources(sources_str: str) -> int:
    """
    Extract rank (position) of gold URL from chatbot_links.

    If gold URL not in sources, return rank = inf (represented as max_int).
    """
    if not sources_str or not sources_str.strip():
        return 999  # not retrieved within top-k

    urls = [u.strip() for u in sources_str.split(";") if u.strip()]
    return len(urls) if urls else 999


def count_retrieved_links(sources_str: str) -> int:
    """Count actual number of retrieved links for a query."""
    if not sources_str or not sources_str.strip():
        return 0
    return len([u.strip() for u in sources_str.split(";") if u.strip()])


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing trailing slashes and query/fragment components.
    Ensures semantically equivalent URLs are treated as the same source.
    """
    cleaned = url.strip()
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    normalized_path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc, normalized_path, "", ""))


def extract_ranks_from_csv(csv_path: Path) -> np.ndarray:
    """
    Extract rank of gold URL in each query from CSV.

    Reads chatbot_links and counts position of first occurrence.
    """
    ranks = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources = row.get("chatbot_links", "")
            rank = extract_rank_from_sources(sources)
            ranks.append(rank)

    return np.array(ranks)


def wilcoxon_test(
    differences: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, bool]:
    """
    Wilcoxon Signed-Rank Test.

    Tests H0: E[d_q] = 0 (no difference between models).

    Args:
        differences: per-query differences d_q = score_A(q) - score_B(q)
        alpha: significance level (default 0.05)

    Returns:
        (test_statistic, p_value, is_significant)
    """
    # Remove zero differences (tied pairs)
    nonzero = differences[differences != 0]

    if len(nonzero) < 2:
        return np.nan, 1.0, False

    stat, pvalue = stats.wilcoxon(nonzero, alternative="two-sided")
    is_sig = pvalue <= alpha

    return float(stat), float(pvalue), is_sig


def goodman_kruskal_gamma(
    ranks_A: np.ndarray, ranks_B: np.ndarray
) -> tuple[float, int, int]:
    """
    Goodman–Kruskal γ (gamma) Coefficient.

    Measures ordinal association between ranks assigned by two models.
    γ = (C - D) / (C + D), where C = concordant pairs, D = discordant pairs.

    Args:
        ranks_A: per-query ranks from model A
        ranks_B: per-query ranks from model B

    Returns:
        (gamma_coeff, n_concordant, n_discordant)
    """
    n = len(ranks_A)
    concordant = 0
    discordant = 0

    # For each pair of queries (i, j) where i < j
    for i, j in combinations(range(n), 2):
        rank_diff_A = ranks_A[i] - ranks_A[j]
        rank_diff_B = ranks_B[i] - ranks_B[j]

        # Both increase or both decrease → concordant
        if rank_diff_A * rank_diff_B > 0:
            concordant += 1
        # One increases, other decreases → discordant
        elif rank_diff_A * rank_diff_B < 0:
            discordant += 1

    if concordant + discordant == 0:
        gamma = 0.0
    else:
        gamma = (concordant - discordant) / (concordant + discordant)

    return float(gamma), int(concordant), int(discordant)


def kappa_agreement(
    retrieval_sets_list: list[list[set[str]]], queries: list[int] | None = None
) -> dict[str, float]:
    """
    Kappa Coefficient of Agreement.

    Measures stability: whether repeated runs return the same documents.
    Computes κ for all pairs of runs, then averages.

    Args:
        retrieval_sets_list: list of [run_i, run_j, ...] where each run is
                            list of sets; each set is top-k URLs for one query
        queries: optional query IDs; if None, assumes sequential indexing

    Returns:
        {
            'kappa_mean': average κ across all query/run pairs,
            'kappa_min': minimum κ,
            'kappa_max': maximum κ,
            'n_runs': number of runs,
            'n_queries': number of unique queries,
        }
    """
    if not retrieval_sets_list or len(retrieval_sets_list) < 2:
        return {}

    n_runs = len(retrieval_sets_list)
    n_queries = len(retrieval_sets_list[0])

    kappas = []

    # For each query
    for q_idx in range(n_queries):
        # For each pair of runs
        for run_i, run_j in combinations(range(n_runs), 2):
            set_i = retrieval_sets_list[run_i][q_idx]
            set_j = retrieval_sets_list[run_j][q_idx]

            # Universe
            universe = set_i | set_j

            if not universe:
                continue

            # Agreement: URLs in both or in neither
            agreed = len(
                (set_i & set_j) | (set_i.symmetric_difference(set_j) - set_i - set_j)
            )
            # Disagreement: URLs in one but not the other
            disagreed = len(set_i.symmetric_difference(set_j))

            p_o = agreed / len(universe) if universe else 0.0

            # Expected agreement by chance
            prob_i = len(set_i) / len(universe)
            prob_j = len(set_j) / len(universe)
            p_e = prob_i * prob_j + (1 - prob_i) * (1 - prob_j)

            # Cohen's kappa
            if p_e < 1.0:
                kappa = (p_o - p_e) / (1 - p_e)
            else:
                kappa = 1.0 if p_o == p_e else 0.0

            kappas.append(kappa)

    if not kappas:
        return {}

    return {
        "kappa_mean": float(np.mean(kappas)),
        "kappa_min": float(np.min(kappas)),
        "kappa_max": float(np.max(kappas)),
        "n_runs": n_runs,
        "n_queries": n_queries,
        "n_kappas": len(kappas),
    }


def compare_models(
    csv_A: Path,
    csv_B: Path,
    metrics: list[str],
    alpha: float = 0.05,
    test_type: str = "wilcoxon",
    use_adaptive_k: bool = False,
) -> list[ComparisonResult]:
    """
    Compare two models across metrics using specified statistical test.

    Args:
        csv_A: path to eval_per_query CSV for model A
        csv_B: path to eval_per_query CSV for model B
        metrics: list of metric column names (e.g., ['mrr@10'] or ['hit_adaptive'])
        alpha: significance level for statistical tests
        test_type: which test to perform: 'wilcoxon', 'gamma', or 'kappa'
        use_adaptive_k: if True, expect adaptive k metrics already in CSV (hit_adaptive, mrr_adaptive, etc.)
                       If False, expect fixed k metrics (hit@k, mrr@k, etc.)

    Returns:
        list of ComparisonResult objects
    """
    # Load metrics from both CSVs
    metrics_A = load_csv_metrics(csv_A, metrics)
    metrics_B = load_csv_metrics(csv_B, metrics)

    # Extract ranks (needed for gamma and kappa tests)
    ranks_A = extract_ranks_from_csv(csv_A)
    ranks_B = extract_ranks_from_csv(csv_B)

    results = []

    for metric in metrics:
        if metric not in metrics_A or metric not in metrics_B:
            print(f"  WARNING: Metric '{metric}' not found in data. Skipping.")
            continue

        scores_A = np.array(metrics_A[metric])
        scores_B = np.array(metrics_B[metric])

        if len(scores_A) != len(scores_B) or len(scores_A) == 0:
            print(f"  WARNING: Metric '{metric}' has mismatched lengths. Skipping.")
            continue

        differences = scores_A - scores_B

        # Initialize result with common fields
        result_dict = {
            "metric": metric,
            "n_queries": len(scores_A),
            "mean_A": float(np.mean(scores_A)),
            "mean_B": float(np.mean(scores_B)),
            "mean_diff": float(np.mean(differences)),
            "std_diff": float(np.std(differences, ddof=1)),
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "wilcoxon_sig": None,
            "gamma_coeff": None,
            "gamma_concordant": None,
            "gamma_discordant": None,
            "kappa_avg": None,
            "kappa_min": None,
            "kappa_max": None,
        }

        # Perform only the selected test type
        if test_type == "wilcoxon":
            wilcox_stat, wilcox_p, is_sig = wilcoxon_test(differences, alpha)
            result_dict["wilcoxon_stat"] = wilcox_stat
            result_dict["wilcoxon_pvalue"] = wilcox_p
            result_dict["wilcoxon_sig"] = is_sig

        elif test_type == "gamma":
            gamma, concordant, discordant = goodman_kruskal_gamma(ranks_A, ranks_B)
            result_dict["gamma_coeff"] = gamma
            result_dict["gamma_concordant"] = concordant
            result_dict["gamma_discordant"] = discordant

        elif test_type == "kappa":
            # For kappa, we need retrieval sets from sources
            # This requires parsing the HTML/JSON sources from both CSVs
            kappa_result = kappa_agreement([ranks_A, ranks_B])
            if kappa_result:
                result_dict["kappa_avg"] = kappa_result.get("kappa_mean")
                result_dict["kappa_min"] = kappa_result.get("kappa_min")
                result_dict["kappa_max"] = kappa_result.get("kappa_max")

        result = ComparisonResult(**result_dict)
        results.append(result)

    return results


def format_result(result: ComparisonResult, test_type: str = "wilcoxon") -> str:
    """Format ComparisonResult for human-readable output, showing only selected test type."""
    lines = [
        f"\n{'='*70}",
        f"Metric: {result.metric}",
        f"{'='*70}",
        f"  Queries:              {result.n_queries}",
        f"  Model A (mean):       {result.mean_A:.6f}",
        f"  Model B (mean):       {result.mean_B:.6f}",
        f"  Difference (A - B):   {result.mean_diff:+.6f}",
        f"  Difference (std):     {result.std_diff:.6f}",
    ]

    if test_type == "wilcoxon":
        significance_marker = "✓ SIGNIFICANT" if result.wilcoxon_sig else "n.s."
        lines.extend(
            [
                "\n  Wilcoxon Signed-Rank Test (α=0.05, two-tailed):",
                f"    Test statistic:     {result.wilcoxon_stat:.4f}",
                f"    p-value:            {result.wilcoxon_pvalue:.6f} {significance_marker}",
            ]
        )

    elif test_type == "gamma":
        lines.extend(
            [
                "\n  Goodman–Kruskal γ Coefficient:",
                f"    γ:                  {result.gamma_coeff:+.4f}",
                f"    Concordant pairs:   {result.gamma_concordant}",
                f"    Discordant pairs:   {result.gamma_discordant}",
            ]
        )

    elif test_type == "kappa":
        if result.kappa_avg is not None:
            lines.extend(
                [
                    "\n  Kappa Agreement (stability):",
                    f"    κ (mean):           {result.kappa_avg:.4f}",
                    f"    κ (min):            {result.kappa_min:.4f}",
                    f"    κ (max):            {result.kappa_max:.4f}",
                ]
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical comparison of two model variants (A vs B)."
    )
    parser.add_argument(
        "--model_a_csv",
        type=Path,
        required=True,
        help="Path to eval_per_query CSV for model A.",
    )
    parser.add_argument(
        "--model_b_csv",
        type=Path,
        required=True,
        help="Path to eval_per_query CSV for model B.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mrr@10",
        help="Single metric to compare (e.g., mrr@10, hit@10, hit_adaptive, mrr_adaptive).",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["wilcoxon", "gamma", "kappa"],
        default="wilcoxon",
        help="Statistical test to perform: wilcoxon (default), gamma, or kappa.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default 0.05).",
    )
    parser.add_argument(
        "--use-adaptive-k",
        action="store_true",
        help=(
            "Use adaptive k (actual # of returned links per query) instead of fixed k. "
            "Expects *_adaptive metrics in CSV (hit_adaptive, mrr_adaptive, etc.). "
            "Useful when number of returned links varies per query."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("src/evaluation/data"),
        help="Output directory for results.",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.model_a_csv.exists():
        print(f"ERROR: Model A CSV not found: {args.model_a_csv}", file=sys.stderr)
        sys.exit(1)

    if not args.model_b_csv.exists():
        print(f"ERROR: Model B CSV not found: {args.model_b_csv}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metric name based on adaptive k setting
    metric = args.metric.strip()
    if args.use_adaptive_k:
        # Auto-convert metric names to adaptive versions if not already
        if metric.endswith("_adaptive"):
            pass  # already adaptive
        elif "@" in metric:
            base_metric = metric.split("@")[0]
            metric = f"{base_metric}_adaptive"

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON: Model A vs Model B")
    print("=" * 70)
    print(f"Model A CSV:       {args.model_a_csv}")
    print(f"Model B CSV:       {args.model_b_csv}")
    print(f"Metric:            {metric}")
    print(f"Test type:         {args.test_type.upper()}")
    print(f"Significance:      α = {args.alpha}")
    if args.use_adaptive_k:
        print("Mode:              ADAPTIVE k (reading *_adaptive metrics from CSV)")
    print("=" * 70)

    # Run comparison
    results = compare_models(
        args.model_a_csv,
        args.model_b_csv,
        [metric],  # wrap in list for compatibility
        args.alpha,
        test_type=args.test_type,
        use_adaptive_k=args.use_adaptive_k,
    )

    # Print results
    for result in results:
        print(format_result(result, test_type=args.test_type))

    # Save results as JSON
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_json = args.output_dir / f"statistical_comparison_{ts}.json"

    json_data = {
        "timestamp": ts,
        "model_a_csv": str(args.model_a_csv),
        "model_b_csv": str(args.model_b_csv),
        "metric": metric,
        "test_type": args.test_type,
        "alpha": args.alpha,
        "use_adaptive_k": args.use_adaptive_k,
        "results": [
            {
                "metric": r.metric,
                "n_queries": r.n_queries,
                "mean_A": r.mean_A,
                "mean_B": r.mean_B,
                "mean_diff": r.mean_diff,
                "std_diff": r.std_diff,
                "wilcoxon_statistic": r.wilcoxon_stat,
                "wilcoxon_pvalue": r.wilcoxon_pvalue,
                "wilcoxon_significant": r.wilcoxon_sig,
                "gamma_coeff": r.gamma_coeff,
                "gamma_concordant": r.gamma_concordant,
                "gamma_discordant": r.gamma_discordant,
                "kappa_mean": r.kappa_avg,
                "kappa_min": r.kappa_min,
                "kappa_max": r.kappa_max,
            }
            for r in results
        ],
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to: {output_json}\n")


if __name__ == "__main__":
    main()
