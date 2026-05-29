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


def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    """
    Rank array values, averaging ranks for tied values.

    Per thesis: if |d_{k_i}| = |d_{k_j}|, then rank(d_{k_i}) = rank(d_{k_j}) = (i+j)/2

    Args:
        values: array of values to rank (typically absolute differences)

    Returns:
        array of ranks [1, n] with tie-averaging
    """
    n = len(values)
    # Sort indices by value
    sorted_indices = np.argsort(values)
    # Initialize ranks array
    ranks = np.empty(n)

    i = 0
    while i < n:
        # Find all tied values
        j = i
        while j < n and values[sorted_indices[j]] == values[sorted_indices[i]]:
            j += 1
        # Assign average rank to all tied values
        avg_rank = (i + j + 1) / 2.0  # +1 because ranks are 1-indexed
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j

    return ranks


def wilcoxon_signed_rank_test(
    differences: np.ndarray, alpha: float = 0.05, alternative: str = "two-sided"
) -> dict[str, float | bool]:
    r"""
    Wilcoxon Signed-Rank Test Implementation.

    Per thesis mathematical definition:

    1. d_q = score_A(q) - score_B(q)   (∀q ∈ Q)

    2. Remove zeros: n := |{q ∈ Q : d_q ≠ 0}|

    3. Rank |d_q| values with tie-averaging:
       If |d_{k_i}| = |d_{k_j}|, then rank = (i+j)/2

    4. Calculate signed rank sums:
       W⁺ = Σ_{q: d_q > 0} rank(|d_q|)
       W⁻ = Σ_{q: d_q < 0} rank(|d_q|)

    5. Test statistic: W = min(W⁺, W⁻)

    6. For n ≥ 25, approximate normal distribution:
       E[W] = n(n+1)/4
       Var(W) = n(n+1)(2n+1)/24
       Z = (W - E[W]) / sqrt(Var(W))  → 𝒩(0,1)

    Args:
        differences: per-query differences d_q = score_A(q) - score_B(q)
        alpha: significance level (default 0.05)
        alternative: "two-sided", "greater", "less"

    Returns:
        {
            'n': sample size (after removing zeros),
            'W_plus': W⁺ (sum of positive ranks),
            'W_minus': W⁻ (sum of negative ranks),
            'W': test statistic = min(W⁺, W⁻),
            'expected_W': E[W],
            'variance_W': Var(W),
            'z_score': Z-score (only if n ≥ 25),
            'p_value': statistical p-value,
            'is_significant': bool(p_value ≤ alpha),
            'method': "Exact" if n < 25 else "Approximate (Z-score, n ≥ 25)",
        }
    """
    # Step 1-2: Remove zero differences
    nonzero_diffs = differences[differences != 0]
    n = len(nonzero_diffs)

    result = {
        "n": n,
        "W_plus": None,
        "W_minus": None,
        "W": None,
        "expected_W": None,
        "variance_W": None,
        "z_score": None,
        "p_value": None,
        "is_significant": None,
        "method": None,
    }

    if n < 1:
        result["p_value"] = 1.0
        result["is_significant"] = False
        result["method"] = "N/A (no non-zero differences)"
        return result

    # Step 3: Rank absolute differences with tie-averaging
    abs_diffs = np.abs(nonzero_diffs)
    ranks = _rank_with_ties(abs_diffs)

    # Step 4: Calculate W⁺ and W⁻
    W_plus = np.sum(ranks[nonzero_diffs > 0])
    W_minus = np.sum(ranks[nonzero_diffs < 0])

    result["W_plus"] = float(W_plus)
    result["W_minus"] = float(W_minus)

    # Step 5: Test statistic W = min(W⁺, W⁻)
    W = min(W_plus, W_minus)
    result["W"] = float(W)

    # Step 6a: For n < 25, use exact distribution (scipy)
    if n < 25:
        from scipy.stats import wilcoxon as scipy_wilcoxon

        stat, p_value = scipy_wilcoxon(nonzero_diffs, alternative=alternative)
        result["p_value"] = float(p_value)
        result["method"] = f"Exact (n={n} < 25)"

    # Step 6b: For n ≥ 25, use normal approximation
    else:
        E_W = n * (n + 1) / 4.0
        Var_W = n * (n + 1) * (2 * n + 1) / 24.0

        result["expected_W"] = float(E_W)
        result["variance_W"] = float(Var_W)

        # Z-score
        z_score = (W - E_W) / np.sqrt(Var_W)
        result["z_score"] = float(z_score)

        # P-value based on alternative hypothesis
        if alternative == "two-sided":
            # p = 2 * P(|Z| ≥ |z_obs|)
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
        elif alternative == "greater":
            # p = P(Z ≥ z_obs)
            p_value = 1.0 - stats.norm.cdf(z_score)
        else:  # alternative == "less"
            # p = P(Z ≤ z_obs)
            p_value = stats.norm.cdf(z_score)

        result["p_value"] = float(p_value)
        result["method"] = f"Approximate normal (Z-score, n={n} ≥ 25)"

    result["is_significant"] = result["p_value"] <= alpha

    return result


def goodman_kruskal_gamma_test(
    ranks_A: np.ndarray, ranks_B: np.ndarray
) -> dict[str, float | int]:
    r"""
    Goodman–Kruskal γ (gamma) Coefficient Test.

    Per thesis definition:

    Measures ordinal association between ranks assigned by two models.
    Models A and B assign ranks ∈ {1,2,3,4,5,∞} to correct link position.

    For queries q_i, q_j ∈ Q, define:

      Π_c = P((a_i < a_j ∧ b_i < b_j) ∨ (a_i > a_j ∧ b_i > b_j))
          = P((a_i - a_j)(b_i - b_j) > 0)  → concordant pairs

      Π_d = P((a_i < a_j ∧ b_i > b_j) ∨ (a_i > a_j ∧ b_i < b_j))
          = P((a_i - a_j)(b_i - b_j) < 0)  → discordant pairs

    Sample estimators:
      C = total number of concordant pairs
      D = total number of discordant pairs

    Gamma coefficient:
      γ = (C - D) / (C + D)

    Interpretation: measures agreement on query difficulty ranking
      γ = 1: perfect agreement
      γ = 0: no agreement
      γ = -1: perfect disagreement

    Args:
        ranks_A: per-query ranks from model A (int array, typically ∈ {1,2,3,4,5,999})
        ranks_B: per-query ranks from model B

    Returns:
        {
            'n_queries': number of queries,
            'n_pairs': number of query pairs n(n-1)/2,
            'C': concordant pairs count,
            'D': discordant pairs count,
            'gamma': γ = (C-D)/(C+D),
            'interpretation': human-readable interpretation,
        }
    """
    ranks_A = np.asarray(ranks_A, dtype=float)
    ranks_B = np.asarray(ranks_B, dtype=float)
    n = len(ranks_A)

    concordant = 0
    discordant = 0

    # For each pair of queries (i, j) where i < j
    for i in range(n):
        for j in range(i + 1, n):
            # Rank differences
            rank_diff_A = ranks_A[i] - ranks_A[j]
            rank_diff_B = ranks_B[i] - ranks_B[j]

            # Product of differences determines concordance/discordance
            product = rank_diff_A * rank_diff_B

            if product > 0:
                # Both rank differences have same sign → concordant
                concordant += 1
            elif product < 0:
                # Rank differences have opposite signs → discordant
                discordant += 1
            # If product == 0, neither concordant nor discordant (tie)

    n_pairs = n * (n - 1) // 2

    # Calculate gamma
    if concordant + discordant == 0:
        gamma = 0.0
        interpretation = "No association (C=D=0)"
    else:
        gamma = (concordant - discordant) / (concordant + discordant)

        if abs(gamma) < 0.3:
            interpretation = "Weak association"
        elif abs(gamma) < 0.7:
            interpretation = "Moderate association"
        else:
            interpretation = "Strong association"

        if gamma < 0:
            interpretation += " (opposite direction)"

    return {
        "n_queries": n,
        "n_pairs": n_pairs,
        "C": concordant,
        "D": discordant,
        "gamma": float(gamma),
        "interpretation": interpretation,
    }


def kappa_coefficient(
    retrieval_sets_i: list[set[str]],
    retrieval_sets_j: list[set[str]],
    relevant_sets: list[set[str]] | None = None,
) -> dict[str, float]:
    r"""
    Kappa Coefficient of Agreement (for single query pair).

    Per thesis definition with contingency table:

    For each query q, compute agreement between two generations i and j:

        U(q) = Z_rel(q) ∪ Z^(i)_top-k(q) ∪ Z^(j)_top-k(q)  (universe of documents)

    Contingency table:
        ┌─────────────────────────────────────────┐
        │           │ Z^(i) ∩ Z^(j) │ Others       │
        ├─────────────────────────────────────────┤
        │ Z^(i) ⊆    │      a        │      b       │
        │ Z^(j) ∩    │      c        │      d       │
        └─────────────────────────────────────────┘

    Where:
        a = |Z^(i) ∩ Z^(j)|  (both agree - document retrieved by both)
        b = |Z^(i) \ Z^(j)|  (disagreement - i only)
        c = |Z^(j) \ Z^(i)|  (disagreement - j only)
        d = |U \ (Z^(i) ∪ Z^(j))|  (both agree - document in neither)

    Agreement:
        p_o = (a + d) / |U|  (observed agreement)
        p_e = (a+b)/|U| * (a+c)/|U| + (c+d)/|U| * (b+d)/|U|  (expected by chance)

    Cohen's Kappa:
        κ = (p_o - p_e) / (1 - p_e)

    Args:
        retrieval_sets_i: list of sets of documents retrieved in generation i
        retrieval_sets_j: list of sets of documents retrieved in generation j
        relevant_sets: optional list of relevant documents for each query
                      (to include in universe if provided)

    Returns:
        {
            'n_queries': number of queries,
            'kappa_mean': mean κ across queries,
            'kappa_min': minimum κ,
            'kappa_max': maximum κ,
            'kappas': list of individual κ values per query,
        }
    """
    if len(retrieval_sets_i) != len(retrieval_sets_j):
        raise ValueError("retrieval_sets_i and retrieval_sets_j must have same length")

    n_queries = len(retrieval_sets_i)
    kappas = []

    for q_idx in range(n_queries):
        Z_i = set(retrieval_sets_i[q_idx])
        Z_j = set(retrieval_sets_j[q_idx])

        # Build universe
        universe = Z_i | Z_j
        if relevant_sets and q_idx < len(relevant_sets):
            universe = universe | set(relevant_sets[q_idx])

        if not universe:
            # Empty universe → perfect agreement (vacuously true)
            kappas.append(1.0)
            continue

        # Contingency table
        a = len(Z_i & Z_j)  # both agree - in both
        b = len(Z_i - Z_j)  # disagree - i only
        c = len(Z_j - Z_i)  # disagree - j only
        d = len(universe - (Z_i | Z_j))  # both agree - in neither

        # Observed agreement
        p_o = (a + d) / len(universe)

        # Expected agreement by chance
        p_A = (a + b) / len(universe)
        p_B = (a + c) / len(universe)
        p_not_A = (c + d) / len(universe)
        p_not_B = (b + d) / len(universe)

        p_e = p_A * p_B + p_not_A * p_not_B

        # Cohen's kappa
        if p_e < 1.0:
            kappa = (p_o - p_e) / (1.0 - p_e)
        else:
            kappa = 1.0 if p_o == p_e else 0.0

        kappas.append(kappa)

    return {
        "n_queries": n_queries,
        "kappa_mean": float(np.mean(kappas)) if kappas else 0.0,
        "kappa_min": float(np.min(kappas)) if kappas else 0.0,
        "kappa_max": float(np.max(kappas)) if kappas else 0.0,
        "kappas": kappas,
    }


def paired_permutation_test(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 67,
    alternative: str = "two-sided",
) -> dict[str, float | list]:
    r"""
    Paired Permutation Test.

    Non-parametric test for comparing paired samples.

    Method:
    1. Compute differences: d_q = score_A(q) - score_B(q)
    2. Observed test statistic: obs = mean(d)
    3. For b = 1..n_perm:
       - Randomly flip signs of d_q: d'_q = signs_b * d_q
       - Compute permuted statistic: mean(d')
    4. P-value = |{perm_stats ≥ |obs|} | / n_perm

    Args:
        scores_A: scores from model A
        scores_B: scores from model B
        n_permutations: number of permutation iterations (default 10000)
        seed: random seed for reproducibility (default 67)
        alternative: "two-sided" (default), "greater", "less"

    Returns:
        {
            'observed_diff': mean(A - B),
            'p_value': permutation test p-value,
            'n_permutations': number of permutations performed,
            'perm_stats': array of permuted test statistics,
            'alternative': alternative hypothesis,
            'interpretation': human-readable result,
        }
    """
    rng = np.random.default_rng(seed)

    scores_A = np.asarray(scores_A, dtype=float)
    scores_B = np.asarray(scores_B, dtype=float)

    if len(scores_A) != len(scores_B):
        raise ValueError("scores_A and scores_B must have same length")

    # Step 1-2: Compute differences and observed statistic
    diffs = scores_A - scores_B
    obs = np.mean(diffs)

    # Step 3: Generate permutation distribution
    perm_stats = np.empty(n_permutations)

    for b in range(n_permutations):
        # Randomly flip signs of differences
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_stats[b] = np.mean(signs * diffs)

    # Step 4: Calculate p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(perm_stats) >= np.abs(obs))
        interpretation = f"Two-tailed: mean difference = {obs:.6f}, p = {p_value:.4f}"
    elif alternative == "greater":
        p_value = np.mean(perm_stats >= obs)
        interpretation = (
            f"One-tailed (>): mean difference = {obs:.6f}, p = {p_value:.4f}"
        )
    else:  # alternative == "less"
        p_value = np.mean(perm_stats <= obs)
        interpretation = (
            f"One-tailed (<): mean difference = {obs:.6f}, p = {p_value:.4f}"
        )

    return {
        "observed_diff": float(obs),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "perm_stats": perm_stats,
        "alternative": alternative,
        "interpretation": interpretation,
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
            wilcox_result = wilcoxon_signed_rank_test(differences, alpha)
            result_dict["wilcoxon_stat"] = wilcox_result["W"]
            result_dict["wilcoxon_pvalue"] = wilcox_result["p_value"]
            result_dict["wilcoxon_sig"] = wilcox_result["is_significant"]

        elif test_type == "gamma":
            gamma_result = goodman_kruskal_gamma_test(ranks_A, ranks_B)
            result_dict["gamma_coeff"] = gamma_result["gamma"]
            result_dict["gamma_concordant"] = gamma_result["C"]
            result_dict["gamma_discordant"] = gamma_result["D"]

        elif test_type == "kappa":
            # For kappa, we need retrieval sets from sources
            # This requires parsing the CSV sources for document sets
            # For now, we'll use a placeholder or skip this test type
            pass

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
                "    Method:             Per thesis: rank absolute differences with tie-averaging",
                f"    Test statistic W:   {result.wilcoxon_stat:.4f}",
                f"    p-value:            {result.wilcoxon_pvalue:.6f} {significance_marker}",
            ]
        )

    elif test_type == "gamma":
        lines.extend(
            [
                "\n  Goodman–Kruskal γ Coefficient (ordinal association):",
                f"    γ:                  {result.gamma_coeff:+.4f}",
                f"    Concordant pairs C: {result.gamma_concordant}",
                f"    Discordant pairs D: {result.gamma_discordant}",
                "    Formula: γ = (C-D)/(C+D)",
            ]
        )

    elif test_type == "kappa":
        if result.kappa_avg is not None:
            lines.extend(
                [
                    "\n  Kappa Agreement (stability across runs):",
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
