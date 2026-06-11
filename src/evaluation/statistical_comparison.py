"""
Statistical tests for comparing two model variants (model A vs model B).

Aligns with thesis definitions (Chapter: Statistical testing and pairwise comparison).

Implements:
  1. Paired T-Test               — parametric comparison (assumes normality)
  2. Paired Permutation Test     — non-parametric resampling-based test
  3. Wilcoxon Signed-Rank Test  — non-parametric comparison using rank information

For each query q ∈ Q, paired difference is computed as:
  d_q = score_A(q) - score_B(q)

Hypotheses (same for all tests):
  - Two-tailed: H0: E[d_q] = 0,  H1: E[d_q] ≠ 0
  - Upper-tailed (greater): H0: E[d_q] ≤ 0,  H1: E[d_q] > 0  (Model A > Model B)
  - Lower-tailed (less): H0: E[d_q] ≥ 0,  H1: E[d_q] < 0  (Model A < Model B)

Text CSV format (per query):
  pytanie, odpowiedz_wygenerowana, odpowiedz_ref,
  bleu, bleu_1, bleu_2, bleu_3, bleu_4,
  rouge_1_r/p/f, rouge_2_r/p/f, rouge_l_r/p/f, rouge_w_r/p/f, rouge_s_r/p/f,
  meteor,
  bertscore_precision_base, bertscore_recall_base, bertscore_f1_base

Retrieval CSV format (per query):
  pytanie, ..., retrieved_count, missing_answer,
  hit_adaptive, mrr_adaptive, mrrw_adaptive, recall_adaptive,
  precision_adaptive, f1_adaptive, ndcg_adaptive, map_adaptive,
  r_prec_adaptive

Usage:
    python statistical_comparison.py \\
        --model_a_csv eval_A.csv \\
        --model_b_csv eval_B.csv \\
        --metric bertscore_f1_base \\
        --test wilcoxon permutation ttest \\
        --output_dir results/

    # Run all tests on multiple metrics:
    python statistical_comparison.py \\
        --model_a_text_csv text_A.csv \\
        --model_a_retrieval_csv retrieval_A.csv \\
        --model_b_text_csv text_B.csv \\
        --model_b_retrieval_csv retrieval_B.csv \\
        --metric all \\
        --test all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Known metric columns in the CSV
# ---------------------------------------------------------------------------
TEXT_METRICS: list[str] = [
    "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "rouge_1_r", "rouge_1_p", "rouge_1_f",
    "rouge_2_r", "rouge_2_p", "rouge_2_f",
    "rouge_l_r", "rouge_l_p", "rouge_l_f",
    "rouge_w_r", "rouge_w_p", "rouge_w_f",
    "rouge_s_r", "rouge_s_p", "rouge_s_f",
    "meteor",
    "bertscore_precision_base", "bertscore_recall_base", "bertscore_f1_base",
]

RETRIEVAL_METRICS: list[str] = [
    "retrieved_count",
    "missing_answer",
    "hit_adaptive",
    "mrr_adaptive",
    "mrrw_adaptive",
    "recall_adaptive",
    "precision_adaptive",
    "f1_adaptive",
    "ndcg_adaptive",
    "map_adaptive",
    "r_prec_adaptive",
]

LLM_JUDGE_METRICS: list[str] = [
    "usefulness",
    "accuracy",
    "conciseness",
]

AVAILABLE_METRICS: list[str] = TEXT_METRICS + RETRIEVAL_METRICS + LLM_JUDGE_METRICS

ALL_TESTS = ["wilcoxon", "permutation", "ttest"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WilcoxonResult:
    n: int
    W_plus: float
    W_minus: float
    W: float
    expected_W: float | None
    variance_W: float | None
    z_score: float | None
    p_value: float
    is_significant: bool
    method: str


@dataclass
class PermutationResult:
    observed_diff: float
    p_value: float
    n_permutations: int
    alternative: str
    is_significant: bool


@dataclass
class TTestResult:
    n: int
    mean_A: float
    mean_B: float
    std_A: float
    std_B: float
    mean_diff: float
    std_diff: float
    t_statistic: float
    degrees_of_freedom: int
    p_value: float
    is_significant: bool
    alternative: str
    ci_lower: float
    ci_upper: float


@dataclass
class MetricComparison:
    metric: str
    n_queries: int
    mean_A: float
    mean_B: float
    mean_diff: float
    std_diff: float
    wilcoxon: WilcoxonResult | None = None
    permutation: PermutationResult | None = None
    ttest: TTestResult | None = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_metrics(csv_path: Path, metric_cols: list[str]) -> dict[str, list[float]]:
    """
    Load metric columns from a per-query CSV.

    Returns {metric_name: [score_per_query, ...]}.
    Missing or non-numeric values are replaced with NaN so they can be
    detected and skipped later.
    """
    result: dict[str, list[float]] = {col: [] for col in metric_cols}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in metric_cols:
                raw = row.get(col, "").strip()
                try:
                    result[col].append(float(raw))
                except (ValueError, TypeError):
                    result[col].append(float("nan"))

    return result


def load_metrics_from_sources(
    csv_paths: list[Path],
    metric_cols: list[str],
) -> dict[str, list[float]]:
    """
    Load metric columns from one or more per-query CSVs.

    This lets a single model be represented by both text-metric and
    retrieval-metric files. Each requested metric must be present in exactly
    one of the provided files for that model.
    """
    result: dict[str, list[float]] = {}
    owners: dict[str, Path] = {}

    for csv_path in csv_paths:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            present_metrics = [col for col in metric_cols if col in fieldnames]

            for col in present_metrics:
                if col in owners:
                    raise ValueError(
                        f"Metric '{col}' appears in both {owners[col]} and {csv_path}."
                    )
                owners[col] = csv_path
                result[col] = []

            for row in reader:
                for col in present_metrics:
                    raw = row.get(col, "").strip()
                    try:
                        result[col].append(float(raw))
                    except (ValueError, TypeError):
                        result[col].append(float("nan"))

    missing = [col for col in metric_cols if col not in result]
    if missing:
        joined = ", ".join(missing)
        paths = ", ".join(str(p) for p in csv_paths)
        raise ValueError(f"Missing metric columns in provided CSVs ({paths}): {joined}")

    return result


def available_metric_columns(csv_paths: list[Path]) -> set[str]:
    """Return supported metric columns present in the provided CSV files."""
    columns: set[str] = set()
    for csv_path in csv_paths:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns.update((reader.fieldnames or []))
    return columns & set(AVAILABLE_METRICS)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    """
    Rank array values (ascending), averaging ranks for tied values.

    Per thesis: if |d_{k_i}| = |d_{k_j}|, then rank = (i + j) / 2.
    """
    n = len(values)
    sorted_idx = np.argsort(values)
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and values[sorted_idx[j]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[sorted_idx[k]] = avg_rank
        i = j

    return ranks


def wilcoxon_signed_rank_test(
    differences: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    r"""
    Wilcoxon Signed-Rank Test (non-parametric).

    Per thesis definition (Chapter Statistical testing):
    1. d_q = score_A(q) − score_B(q)
    2. Remove zeros; n = |{q : d_q ≠ 0}|
    3. Rank |d_q| with tie-averaging
    4. W⁺ = Σ_{d_q>0} rank(|d_q|),  W⁻ = Σ_{d_q<0} rank(|d_q|)

    For n < 25: use exact critical value tables (via scipy)
    For n ≥ 25: normal approximation using:
      E[W⁺] = n(n+1)/4
      Var(W⁺) = (1/24) n(n+1)(2n+1) - (1/48) Σ_j t_j(t_j+1)(t_j-1)
      Z = (W* - E[W⁺]) / √(Var(W⁺))

    The variance term is corrected for ties in |d_q|: ranks of tied absolute
    differences are averaged, t_j is the number of values tied at the j-th of
    the m unique assigned ranks (Lehmann 1975). When there are no ties, m = n
    and t_j = 1 for all j, reducing to Var(W⁺) = n(n+1)(2n+1)/24.

    where W* is selected based on alternative hypothesis:
    - Two-tailed: W* = min(W⁺, W⁻)  → two-tailed p-value
    - Upper-tailed (greater, A > B): W* = W⁺  → P(Z ≥ z_obs)
    - Lower-tailed (less, A < B): W* = W⁻  → P(Z ≤ z_obs)
    """
    nonzero = differences[differences != 0]
    n = len(nonzero)

    if n < 1:
        return WilcoxonResult(
            n=0, W_plus=0.0, W_minus=0.0, W=0.0,
            expected_W=None, variance_W=None, z_score=None,
            p_value=1.0, is_significant=False,
            method="N/A (no non-zero differences)",
        )

    abs_diffs = np.abs(nonzero)
    ranks = _rank_with_ties(abs_diffs)

    W_plus  = float(np.sum(ranks[nonzero > 0]))
    W_minus = float(np.sum(ranks[nonzero < 0]))
    W = min(W_plus, W_minus)

    if n < 25:
        from scipy.stats import wilcoxon as _scipy_wilcoxon
        _, p_value = _scipy_wilcoxon(nonzero, alternative=alternative)
        return WilcoxonResult(
            n=n, W_plus=W_plus, W_minus=W_minus, W=W,
            expected_W=None, variance_W=None, z_score=None,
            p_value=float(p_value), is_significant=float(p_value) <= alpha,
            method=f"Exact distribution (n={n} < 25)",
        )

    # For n >= 25: normal approximation
    E_W_plus = n * (n + 1) / 4.0

    # Tie-corrected variance (Lehmann 1975):
    #   V = (1/24) n(n+1)(2n+1) - (1/48) sum_j t_j(t_j+1)(t_j-1)
    # where t_j is the size of the j-th group of tied |d_q| values.
    _, tie_counts = np.unique(abs_diffs, return_counts=True)
    tie_correction = float(np.sum(tie_counts * (tie_counts + 1) * (tie_counts - 1)))
    Var_W_plus = (n * (n + 1) * (2 * n + 1)) / 24.0 - tie_correction / 48.0

    # Select W* based on alternative hypothesis
    if alternative == "two-sided":
        W_star = W  # min(W+, W-)
    elif alternative == "greater":
        # H1: E[d] > 0 => Model A > Model B => most d_q > 0 => W+ large
        W_star = W_plus
    else:  # "less"
        # H1: E[d] < 0 => Model A < Model B => most d_q < 0 => W- large
        W_star = W_minus

    z = (W_star - E_W_plus) / np.sqrt(Var_W_plus)

    # Compute p-value based on alternative hypothesis
    if alternative == "two-sided":
        # p = P(|Z| >= |z_obs|)
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        # p = P(Z >= z_obs) = 1 - CDF(z)
        p_value = 1.0 - stats.norm.cdf(z)
    else:  # "less"
        # p = P(Z <= z_obs) = CDF(z)
        p_value = stats.norm.cdf(z)

    return WilcoxonResult(
        n=n, W_plus=W_plus, W_minus=W_minus, W=W,
        expected_W=E_W_plus, variance_W=Var_W_plus, z_score=float(z),
        p_value=float(p_value), is_significant=float(p_value) <= alpha,
        method=f"Normal approximation (n={n} ≥ 25)",
    )


def paired_permutation_test(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 67,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> PermutationResult:
    r"""
    Paired Permutation Test (non-parametric).

    Per thesis definition (Chapter Statistical testing, Section Permutation test):

    For paired samples score_A and score_B, test whether E[d_q] differs from 0,
    where d_q = score_A(q) - score_B(q).

    Hypotheses:
    - Two-tailed: H0: E[d_q] = 0,  H1: E[d_q] ≠ 0
    - Upper-tailed: H0: E[d_q] = 0,  H1: E[d_q] > 0  (Model A > Model B)
    - Lower-tailed: H0: E[d_q] = 0,  H1: E[d_q] < 0  (Model A < Model B)

    Test statistic: T_0 = (1/n) Σ d_q = mean(d)

    For each of N permutations, draw random signs c_{q,j} ∈ {-1, +1} with
    equal probability and compute: T_j = (1/n) Σ c_{q,j} * d_q

    P-value computation:
    - Two-tailed: ASL = (# permutations where |T_j| ≥ |T_0|) / N
    - Upper-tailed: ASL = (# permutations where T_j ≥ T_0) / N
    - Lower-tailed: ASL = (# permutations where T_j ≤ T_0) / N

    Reject H0 if ASL ≤ α.
    """
    rng = np.random.default_rng(seed)
    diffs = scores_A - scores_B
    obs   = float(np.mean(diffs))

    perm_stats = np.array([
        np.mean(rng.choice([-1.0, 1.0], size=len(diffs)) * diffs)
        for _ in range(n_permutations)
    ])

    if alternative == "two-sided":
        # p = (# |T_j| >= |T_0|) / N
        p_value = float(np.mean(np.abs(perm_stats) >= abs(obs)))
    elif alternative == "greater":
        # p = (# T_j >= T_0) / N
        p_value = float(np.mean(perm_stats >= obs))
    else:  # "less"
        # p = (# T_j <= T_0) / N
        p_value = float(np.mean(perm_stats <= obs))

    return PermutationResult(
        observed_diff=obs,
        p_value=p_value,
        n_permutations=n_permutations,
        alternative=alternative,
        is_significant=p_value <= alpha,
    )


def paired_ttest(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TTestResult:
    r"""
    Paired T-Test (parametric).

    Per thesis definition (Chapter Statistical testing, Section Pairwise t-test):

    Hypotheses:
    - Two-tailed: H0: E[d_q] = 0,  H1: E[d_q] ≠ 0
    - Upper-tailed: H0: E[d_q] = 0,  H1: E[d_q] > 0  (Model A > Model B)
    - Lower-tailed: H0: E[d_q] = 0,  H1: E[d_q] < 0  (Model A < Model B)

    For each query q, paired difference:
      d_q = score_A(q) - score_B(q)

    Test statistic:
      t = (mean_diff * √n) / s_d
      where mean_diff = (1/n) Σ d_q,  s_d² = (1/(n-1)) Σ(d_q - mean_diff)²

    Degrees of freedom: df = n - 1

    Validity assumption: differences should be approximately normally distributed
    (typically satisfied for n ≥ 30 by Central Limit Theorem). Non-normality is
    acceptable if skewness and excess kurtosis remain in [-3, 3].

    Also reports:
      - Standard deviation of each model's scores (std_A, std_B)
      - Standard deviation of paired differences (std_diff)
      - 95% confidence interval for mean difference
    """
    n = len(scores_A)
    diffs = scores_A - scores_B

    mean_A   = float(np.mean(scores_A))
    mean_B   = float(np.mean(scores_B))
    std_A    = float(np.std(scores_A, ddof=1))
    std_B    = float(np.std(scores_B, ddof=1))
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1))

    # t = (mean_diff * sqrt(n)) / s_d
    se = std_diff / np.sqrt(n)
    t_stat = mean_diff / se if se > 0 else 0.0
    df = n - 1

    # p-value via scipy for accuracy
    t_res = stats.ttest_rel(scores_A, scores_B, alternative=alternative)
    p_value = float(t_res.pvalue)

    # 95% confidence interval for the mean difference
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return TTestResult(
        n=n,
        mean_A=mean_A,
        mean_B=mean_B,
        std_A=std_A,
        std_B=std_B,
        mean_diff=mean_diff,
        std_diff=std_diff,
        t_statistic=float(t_stat),
        degrees_of_freedom=df,
        p_value=p_value,
        is_significant=p_value <= alpha,
        alternative=alternative,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
    )


# ---------------------------------------------------------------------------
# Main comparison driver
# ---------------------------------------------------------------------------

def compare_models(
    csv_A: Path | list[Path],
    csv_B: Path | list[Path],
    metrics: list[str],
    tests: list[str],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    n_permutations: int = 10_000,
    perm_seed: int = 67,
) -> list[MetricComparison]:
    """
    Load both CSVs and run selected tests for each metric.
    """
    csvs_A = [csv_A] if isinstance(csv_A, Path) else csv_A
    csvs_B = [csv_B] if isinstance(csv_B, Path) else csv_B
    data_A = load_metrics_from_sources(csvs_A, metrics)
    data_B = load_metrics_from_sources(csvs_B, metrics)

    results: list[MetricComparison] = []

    for metric in metrics:
        arr_A = np.array(data_A[metric], dtype=float)
        arr_B = np.array(data_B[metric], dtype=float)

        valid = ~(np.isnan(arr_A) | np.isnan(arr_B))
        arr_A, arr_B = arr_A[valid], arr_B[valid]

        n = len(arr_A)
        if n == 0:
            print(f"  WARNING: '{metric}' — no valid rows after NaN removal. Skipping.")
            continue
        if len(arr_A) != len(arr_B):
            print(f"  WARNING: '{metric}' — mismatched lengths. Skipping.")
            continue

        diffs = arr_A - arr_B

        comparison = MetricComparison(
            metric=metric,
            n_queries=n,
            mean_A=float(np.mean(arr_A)),
            mean_B=float(np.mean(arr_B)),
            mean_diff=float(np.mean(diffs)),
            std_diff=float(np.std(diffs, ddof=1)),
        )

        if "wilcoxon" in tests:
            comparison.wilcoxon = wilcoxon_signed_rank_test(diffs, alpha, alternative)

        if "permutation" in tests:
            comparison.permutation = paired_permutation_test(
                arr_A, arr_B, n_permutations, perm_seed, alpha, alternative
            )

        if "ttest" in tests:
            comparison.ttest = paired_ttest(arr_A, arr_B, alpha, alternative)

        results.append(comparison)

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _sig(flag: bool | None, alpha: float = 0.05) -> str:
    if flag is None:
        return ""
    return f"  ✓ SIGNIFICANT (α={alpha})" if flag else "  n.s."


def format_comparison(c: MetricComparison, alpha: float = 0.05) -> str:
    lines = [
        "",
        "=" * 72,
        f"Metric: {c.metric}",
        "=" * 72,
        f"  Queries:            {c.n_queries}",
        f"  Model A mean:       {c.mean_A:.6f}",
        f"  Model B mean:       {c.mean_B:.6f}",
        f"  Difference (A−B):   {c.mean_diff:+.6f}",
        f"  Std of differences: {c.std_diff:.6f}",
    ]

    if c.wilcoxon:
        w = c.wilcoxon
        lines += [
            "",
            "  ── Wilcoxon Signed-Rank Test ──────────────────────────────────",
            f"  Method:       {w.method}",
            f"  n (non-zero): {w.n}",
            f"  W⁺:           {w.W_plus:.4f}",
            f"  W⁻:           {w.W_minus:.4f}",
            f"  W (test stat):{w.W:.4f}",
        ]
        if w.z_score is not None:
            lines += [
                f"  E[W]:         {w.expected_W:.4f}",
                f"  Var(W):       {w.variance_W:.4f}",
                f"  Z-score:      {w.z_score:.4f}",
            ]
        lines.append(f"  p-value:      {w.p_value:.6f}{_sig(w.is_significant, alpha)}")

    if c.permutation:
        p = c.permutation
        lines += [
            "",
            "  ── Paired Permutation Test ────────────────────────────────────",
            f"  Permutations:    {p.n_permutations}",
            f"  Alternative:     {p.alternative}",
            f"  Observed diff:   {p.observed_diff:+.6f}",
            f"  p-value:         {p.p_value:.6f}{_sig(p.is_significant, alpha)}",
        ]

    if c.ttest:
        t = c.ttest
        lines += [
            "",
            "  ── Paired T-Test ──────────────────────────────────────────────",
            f"  n:               {t.n}",
            f"  Model A mean:    {t.mean_A:.6f}  (std = {t.std_A:.6f})",
            f"  Model B mean:    {t.mean_B:.6f}  (std = {t.std_B:.6f})",
            f"  Mean diff (A−B): {t.mean_diff:+.6f}  (std = {t.std_diff:.6f})",
            f"  t-statistic:     {t.t_statistic:.4f}",
            f"  Degrees of freedom: {t.degrees_of_freedom}",
            f"  Alternative:     {t.alternative}",
            f"  95% CI (A−B):    [{t.ci_lower:+.6f},  {t.ci_upper:+.6f}]",
            f"  p-value:         {t.p_value:.6f}{_sig(t.is_significant, alpha)}",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _result_to_dict(c: MetricComparison) -> dict:
    d: dict = {
        "metric": c.metric,
        "n_queries": c.n_queries,
        "mean_A": c.mean_A,
        "mean_B": c.mean_B,
        "mean_diff": c.mean_diff,
        "std_diff": c.std_diff,
    }
    if c.wilcoxon:
        w = c.wilcoxon
        d["wilcoxon"] = {
            "method": w.method, "n": w.n,
            "W_plus": w.W_plus, "W_minus": w.W_minus, "W": w.W,
            "expected_W": w.expected_W, "variance_W": w.variance_W,
            "z_score": w.z_score, "p_value": w.p_value,
            "is_significant": w.is_significant,
        }
    if c.permutation:
        p = c.permutation
        d["permutation"] = {
            "observed_diff": p.observed_diff, "p_value": p.p_value,
            "n_permutations": p.n_permutations, "alternative": p.alternative,
            "is_significant": p.is_significant,
        }
    if c.ttest:
        t = c.ttest
        d["ttest"] = {
            "n": t.n,
            "mean_A": t.mean_A,
            "mean_B": t.mean_B,
            "std_A": t.std_A,
            "std_B": t.std_B,
            "mean_diff": t.mean_diff,
            "std_diff": t.std_diff,
            "t_statistic": t.t_statistic,
            "degrees_of_freedom": t.degrees_of_freedom,
            "p_value": t.p_value,
            "is_significant": t.is_significant,
            "alternative": t.alternative,
            "ci_lower": t.ci_lower,
            "ci_upper": t.ci_upper,
        }
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical comparison of two model variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_a_csv", type=Path,
                        help="Per-query CSV for model A (backward-compatible alias for text metrics).")
    parser.add_argument("--model_b_csv", type=Path,
                        help="Per-query CSV for model B (backward-compatible alias for text metrics).")
    parser.add_argument("--model_a_text_csv", type=Path,
                        help="Per-query text-metrics CSV for model A.")
    parser.add_argument("--model_b_text_csv", type=Path,
                        help="Per-query text-metrics CSV for model B.")
    parser.add_argument("--model_a_retrieval_csv", type=Path,
                        help="Per-query retrieval-metrics CSV for model A.")
    parser.add_argument("--model_b_retrieval_csv", type=Path,
                        help="Per-query retrieval-metrics CSV for model B.")
    parser.add_argument("--model_a_llm_judge_csv", type=Path,
                        help="Per-query LLM judge metrics CSV for model A (usefulness, accuracy, conciseness).")
    parser.add_argument("--model_b_llm_judge_csv", type=Path,
                        help="Per-query LLM judge metrics CSV for model B (usefulness, accuracy, conciseness).")
    parser.add_argument(
        "--metric", nargs="+", default=["bertscore_f1_base"],
        metavar="METRIC",
        help=(
            "One or more metric column names to compare, or 'all'. "
            f"Available: {', '.join(AVAILABLE_METRICS)}. "
            "Default: bertscore_f1_base."
        ),
    )
    parser.add_argument(
        "--test", nargs="+", default=["wilcoxon"],
        choices=ALL_TESTS + ["all"],
        metavar="TEST",
        help=(
            f"Tests to run: {', '.join(ALL_TESTS)}, or 'all'. "
            "Default: wilcoxon."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default 0.05).")
    parser.add_argument(
        "--alternative", choices=["two-sided", "greater", "less"],
        default="two-sided",
        help="Alternative hypothesis direction (default: two-sided).",
    )
    parser.add_argument("--n_permutations", type=int, default=10_000,
                        help="Number of permutations (default 10000).")
    parser.add_argument("--perm_seed", type=int, default=67,
                        help="Random seed for permutation test (default 67).")
    parser.add_argument("--output_dir", type=Path, default=Path("results"),
                        help="Directory for JSON output (default: results/).")
    parser.add_argument("--list_metrics", action="store_true",
                        help="Print all available metric names and exit.")

    args = parser.parse_args()

    if args.list_metrics:
        print("Available text metrics:")
        for m in TEXT_METRICS:
            print(f"  {m}")
        print("\nAvailable retrieval metrics:")
        for m in RETRIEVAL_METRICS:
            print(f"  {m}")
        print("\nAvailable LLM Judge metrics:")
        for m in LLM_JUDGE_METRICS:
            print(f"  {m}")
        sys.exit(0)

    model_a_csvs = [
        p for p in [args.model_a_text_csv or args.model_a_csv, args.model_a_retrieval_csv, args.model_a_llm_judge_csv]
        if p is not None
    ]
    model_b_csvs = [
        p for p in [args.model_b_text_csv or args.model_b_csv, args.model_b_retrieval_csv, args.model_b_llm_judge_csv]
        if p is not None
    ]

    if not model_a_csvs:
        print("ERROR: Provide --model_a_csv or --model_a_text_csv/--model_a_retrieval_csv.", file=sys.stderr)
        sys.exit(1)
    if not model_b_csvs:
        print("ERROR: Provide --model_b_csv or --model_b_text_csv/--model_b_retrieval_csv.", file=sys.stderr)
        sys.exit(1)

    for label, paths in [("Model A", model_a_csvs), ("Model B", model_b_csvs)]:
        for path in paths:
            if not path.exists():
                print(f"ERROR: {label} CSV not found: {path}", file=sys.stderr)
                sys.exit(1)

    if "all" in args.metric:
        model_a_metrics = available_metric_columns(model_a_csvs)
        model_b_metrics = available_metric_columns(model_b_csvs)
        common_metrics = model_a_metrics & model_b_metrics
        metrics = [m for m in AVAILABLE_METRICS if m in common_metrics]
    else:
        metrics = list(dict.fromkeys(args.metric))

    if not metrics:
        print("ERROR: No common supported metric columns found in the provided CSVs.", file=sys.stderr)
        sys.exit(1)

    unknown_metrics = [m for m in metrics if m not in AVAILABLE_METRICS]
    if unknown_metrics:
        print(
            f"ERROR: Unknown metric(s): {', '.join(unknown_metrics)}. "
            "Use --list_metrics to see supported names.",
            file=sys.stderr,
        )
        sys.exit(1)

    tests = ALL_TESTS if "all" in args.test else list(dict.fromkeys(args.test))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("  STATISTICAL COMPARISON — Model A vs Model B")
    print("=" * 72)
    print(f"  Model A CSVs:  {', '.join(str(p) for p in model_a_csvs)}")
    print(f"  Model B CSVs:  {', '.join(str(p) for p in model_b_csvs)}")
    print(f"  Metrics:       {', '.join(metrics)}")
    print(f"  Tests:         {', '.join(tests)}")
    print(f"  α (alpha):     {args.alpha}")
    print(f"  Alternative:   {args.alternative}")
    if "permutation" in tests:
        print(f"  Permutations:  {args.n_permutations}  (seed={args.perm_seed})")
    print("=" * 72)

    comparisons = compare_models(
        csv_A=model_a_csvs,
        csv_B=model_b_csvs,
        metrics=metrics,
        tests=tests,
        alpha=args.alpha,
        alternative=args.alternative,
        n_permutations=args.n_permutations,
        perm_seed=args.perm_seed,
    )

    for c in comparisons:
        print(format_comparison(c, args.alpha))

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_json = args.output_dir / f"stats_{ts}.json"
    payload = {
        "timestamp": ts,
        "model_a_csvs": [str(p) for p in model_a_csvs],
        "model_b_csvs": [str(p) for p in model_b_csvs],
        "metrics": metrics,
        "tests": tests,
        "alpha": args.alpha,
        "alternative": args.alternative,
        "results": [_result_to_dict(c) for c in comparisons],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved → {out_json}\n")


if __name__ == "__main__":
    main()
