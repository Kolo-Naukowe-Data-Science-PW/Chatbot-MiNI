"""
Statistical tests for comparing two model variants (model A vs model B).

Implements:
  1. Wilcoxon Signed-Rank Test  — non-parametric comparison of paired metrics
  2. Goodman–Kruskal γ (gamma)  — ordinal association between score-based ranks
  3. Kappa Coefficient           — stability / agreement between two score vectors
  4. Paired Permutation Test     — non-parametric permutation test
  5. Paired T-Test               — parametric comparison of paired metrics

CSV format (per query):
  pytanie, odpowiedz_wygenerowana, odpowiedz_ref,
  bleu, bleu_1, bleu_2, bleu_3, bleu_4,
  rouge_1_r/p/f, rouge_2_r/p/f, rouge_l_r/p/f, rouge_w_r/p/f, rouge_s_r/p/f,
  meteor,
  bertscore_precision_base, bertscore_recall_base, bertscore_f1_base

Usage:
    python statistical_comparison.py \\
        --model_a_csv eval_A.csv \\
        --model_b_csv eval_B.csv \\
        --metric bertscore_f1_base \\
        --test wilcoxon permutation gamma kappa ttest \\
        --output_dir results/

    # Run all tests on multiple metrics:
    python statistical_comparison.py \\
        --model_a_csv eval_A.csv \\
        --model_b_csv eval_B.csv \\
        --metric bleu meteor bertscore_f1_base rouge_1_f rouge_l_f \\
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
AVAILABLE_METRICS: list[str] = [
    "bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "rouge_1_r", "rouge_1_p", "rouge_1_f",
    "rouge_2_r", "rouge_2_p", "rouge_2_f",
    "rouge_l_r", "rouge_l_p", "rouge_l_f",
    "rouge_w_r", "rouge_w_p", "rouge_w_f",
    "rouge_s_r", "rouge_s_p", "rouge_s_f",
    "meteor",
    "bertscore_precision_base", "bertscore_recall_base", "bertscore_f1_base",
]

ALL_TESTS = ["wilcoxon", "permutation", "gamma", "kappa", "ttest"]


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
class GammaResult:
    n_queries: int
    n_pairs: int
    C: int
    D: int
    gamma: float
    interpretation: str


@dataclass
class KappaResult:
    n_queries: int
    kappa_mean: float
    kappa_min: float
    kappa_max: float


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
    gamma: GammaResult | None = None
    kappa: KappaResult | None = None
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


def _scores_to_ordinal_ranks(scores: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Convert continuous scores in [0, 1] into ordinal rank labels {1, …, n_bins}.

    The mapping mirrors the retrieval convention used in the thesis:
      rank 1 = best (score closest to 1), rank n_bins = worst.
    Scores of exactly 0 are treated as "not retrieved" and mapped to n_bins+1
    (analogous to ∞ in the retrieval setting).

    This enables the Goodman-Kruskal γ test (designed for ordinal data) to be
    applied to continuous text-similarity metrics.
    """
    ranks = np.empty(len(scores), dtype=float)
    zero_mask = scores == 0.0
    non_zero = scores[~zero_mask]

    if len(non_zero) > 0:
        percentiles = np.linspace(0, 100, n_bins + 1)
        breakpoints = np.percentile(non_zero, percentiles)
        breakpoints = np.unique(breakpoints)
        bin_labels = np.digitize(non_zero, breakpoints[1:-1], right=False) + 1
        bin_labels = n_bins + 1 - bin_labels
        bin_labels = np.clip(bin_labels, 1, n_bins)
        ranks[~zero_mask] = bin_labels.astype(float)

    ranks[zero_mask] = float(n_bins + 1)
    return ranks


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
    Wilcoxon Signed-Rank Test.

      1. d_q = score_A(q) − score_B(q)
      2. Remove zeros; n = |{q : d_q ≠ 0}|
      3. Rank |d_q| with tie-averaging
      4. W⁺ = Σ_{d_q>0} rank(|d_q|),  W⁻ = Σ_{d_q<0} rank(|d_q|)
      5. W  = min(W⁺, W⁻)
      6. n < 25  → exact distribution (scipy)
         n ≥ 25  → normal approximation
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
            method=f"Exact (n={n} < 25)",
        )

    E_W   = n * (n + 1) / 4.0
    Var_W = n * (n + 1) * (2 * n + 1) / 24.0
    z     = (W - E_W) / np.sqrt(Var_W)

    if alternative == "two-sided":
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p_value = 1.0 - stats.norm.cdf(z)
    else:
        p_value = stats.norm.cdf(z)

    return WilcoxonResult(
        n=n, W_plus=W_plus, W_minus=W_minus, W=W,
        expected_W=E_W, variance_W=Var_W, z_score=float(z),
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
    Paired Permutation Test.

      1. d_q = score_A(q) − score_B(q);  obs = mean(d)
      2. For b = 1…B: randomly flip signs of d_q, compute mean(d')
      3. p = |{perm stats ≥ |obs|}| / B  (two-sided)
    """
    rng = np.random.default_rng(seed)
    diffs = scores_A - scores_B
    obs   = float(np.mean(diffs))

    perm_stats = np.array([
        np.mean(rng.choice([-1.0, 1.0], size=len(diffs)) * diffs)
        for _ in range(n_permutations)
    ])

    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(perm_stats) >= abs(obs)))
    elif alternative == "greater":
        p_value = float(np.mean(perm_stats >= obs))
    else:
        p_value = float(np.mean(perm_stats <= obs))

    return PermutationResult(
        observed_diff=obs,
        p_value=p_value,
        n_permutations=n_permutations,
        alternative=alternative,
        is_significant=p_value <= alpha,
    )


def goodman_kruskal_gamma(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    n_bins: int = 5,
) -> GammaResult:
    r"""
    Goodman–Kruskal γ coefficient.

      For each pair (q_i, q_j):
        concordant  if (a_i − a_j)(b_i − b_j) > 0
        discordant  if (a_i − a_j)(b_i − b_j) < 0

      γ = (C − D) / (C + D)
    """
    ranks_A = _scores_to_ordinal_ranks(scores_A, n_bins)
    ranks_B = _scores_to_ordinal_ranks(scores_B, n_bins)

    n = len(ranks_A)
    C = D = 0

    for i in range(n):
        for j in range(i + 1, n):
            product = (ranks_A[i] - ranks_A[j]) * (ranks_B[i] - ranks_B[j])
            if product > 0:
                C += 1
            elif product < 0:
                D += 1

    n_pairs = n * (n - 1) // 2

    if C + D == 0:
        gamma = 0.0
        interp = "No association (C = D = 0)"
    else:
        gamma = (C - D) / (C + D)
        strength = "Weak" if abs(gamma) < 0.3 else ("Moderate" if abs(gamma) < 0.7 else "Strong")
        direction = " (opposite direction)" if gamma < 0 else ""
        interp = f"{strength} association{direction}"

    return GammaResult(
        n_queries=n,
        n_pairs=n_pairs,
        C=C,
        D=D,
        gamma=float(gamma),
        interpretation=interp,
    )


def kappa_coefficient(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    n_bins: int = 5,
) -> KappaResult:
    r"""
    Kappa Coefficient of Agreement.

      For each query q, treat the ordinal-rank assignments by model A and B
      as two "retrievers" over a shared universe of rank classes {1,…,n_bins+1}.

      κ = (p_o − p_e) / (1 − p_e)

    The mean κ across all queries is reported.
    """
    ranks_A = _scores_to_ordinal_ranks(scores_A, n_bins)
    ranks_B = _scores_to_ordinal_ranks(scores_B, n_bins)

    universe_size = n_bins + 1
    kappas: list[float] = []

    for rA, rB in zip(ranks_A, ranks_B):
        agree = int(rA == rB)
        a = agree
        b = 1 - agree
        c = 1 - agree
        d = universe_size - a - b - c

        p_o = (a + d) / universe_size
        p_A     = (a + b) / universe_size
        p_B     = (a + c) / universe_size
        p_not_A = (c + d) / universe_size
        p_not_B = (b + d) / universe_size

        p_e = p_A * p_B + p_not_A * p_not_B

        kappa = (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else (1.0 if p_o == 1.0 else 0.0)
        kappas.append(kappa)

    return KappaResult(
        n_queries=len(kappas),
        kappa_mean=float(np.mean(kappas)),
        kappa_min=float(np.min(kappas)),
        kappa_max=float(np.max(kappas)),
    )


def paired_ttest(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TTestResult:
    r"""
    Paired T-Test (parametric).

    Assumes the differences d_q = score_A(q) − score_B(q) are approximately
    normally distributed (central limit theorem typically holds for n ≥ 30).

      H0: mean(d) = 0
      t  = mean(d) / (std(d) / sqrt(n))
      df = n − 1

    Also reports:
      - Standard deviation of each model's scores independently (std_A, std_B)
      - Standard deviation of the paired differences (std_diff)
      - 95% confidence interval for the mean difference

    Parameters
    ----------
    scores_A, scores_B : aligned per-query score arrays (same length, no NaNs)
    alpha              : significance level (default 0.05)
    alternative        : "two-sided" | "greater" | "less"
    """
    n = len(scores_A)
    diffs = scores_A - scores_B

    mean_A   = float(np.mean(scores_A))
    mean_B   = float(np.mean(scores_B))
    std_A    = float(np.std(scores_A, ddof=1))
    std_B    = float(np.std(scores_B, ddof=1))
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1))

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
    csv_A: Path,
    csv_B: Path,
    metrics: list[str],
    tests: list[str],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    n_permutations: int = 10_000,
    perm_seed: int = 67,
    gamma_bins: int = 5,
    kappa_bins: int = 5,
) -> list[MetricComparison]:
    """
    Load both CSVs and run selected tests for each metric.
    """
    data_A = load_metrics(csv_A, metrics)
    data_B = load_metrics(csv_B, metrics)

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

        if "gamma" in tests:
            comparison.gamma = goodman_kruskal_gamma(arr_A, arr_B, gamma_bins)

        if "kappa" in tests:
            comparison.kappa = kappa_coefficient(arr_A, arr_B, kappa_bins)

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

    if c.gamma:
        g = c.gamma
        lines += [
            "",
            "  ── Goodman–Kruskal γ ──────────────────────────────────────────",
            f"  Query pairs:     {g.n_pairs}",
            f"  Concordant C:    {g.C}",
            f"  Discordant D:    {g.D}",
            f"  γ = (C−D)/(C+D): {g.gamma:+.4f}",
            f"  Interpretation:  {g.interpretation}",
        ]

    if c.kappa:
        k = c.kappa
        lines += [
            "",
            "  ── Kappa Coefficient ──────────────────────────────────────────",
            f"  κ mean:          {k.kappa_mean:.4f}",
            f"  κ min:           {k.kappa_min:.4f}",
            f"  κ max:           {k.kappa_max:.4f}",
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
    if c.gamma:
        g = c.gamma
        d["gamma"] = {
            "n_pairs": g.n_pairs, "C": g.C, "D": g.D,
            "gamma": g.gamma, "interpretation": g.interpretation,
        }
    if c.kappa:
        k = c.kappa
        d["kappa"] = {
            "kappa_mean": k.kappa_mean,
            "kappa_min": k.kappa_min,
            "kappa_max": k.kappa_max,
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
        description="Statistical comparison of two model variants (text metrics).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model_a_csv", type=Path, required=True,
                        help="Per-query CSV for model A.")
    parser.add_argument("--model_b_csv", type=Path, required=True,
                        help="Per-query CSV for model B.")
    parser.add_argument(
        "--metric", nargs="+", default=["bertscore_f1_base"],
        metavar="METRIC",
        help=(
            "One or more metric column names to compare. "
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
    parser.add_argument("--gamma_bins", type=int, default=5,
                        help="Number of ordinal bins for gamma (default 5).")
    parser.add_argument("--kappa_bins", type=int, default=5,
                        help="Number of ordinal bins for kappa (default 5).")
    parser.add_argument("--output_dir", type=Path, default=Path("results"),
                        help="Directory for JSON output (default: results/).")
    parser.add_argument("--list_metrics", action="store_true",
                        help="Print all available metric names and exit.")

    args = parser.parse_args()

    if args.list_metrics:
        print("Available metrics:")
        for m in AVAILABLE_METRICS:
            print(f"  {m}")
        sys.exit(0)

    for label, path in [("Model A", args.model_a_csv), ("Model B", args.model_b_csv)]:
        if not path.exists():
            print(f"ERROR: {label} CSV not found: {path}", file=sys.stderr)
            sys.exit(1)

    tests = ALL_TESTS if "all" in args.test else list(dict.fromkeys(args.test))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("  STATISTICAL COMPARISON — Model A vs Model B")
    print("=" * 72)
    print(f"  Model A:       {args.model_a_csv}")
    print(f"  Model B:       {args.model_b_csv}")
    print(f"  Metrics:       {', '.join(args.metric)}")
    print(f"  Tests:         {', '.join(tests)}")
    print(f"  α (alpha):     {args.alpha}")
    print(f"  Alternative:   {args.alternative}")
    if "permutation" in tests:
        print(f"  Permutations:  {args.n_permutations}  (seed={args.perm_seed})")
    print("=" * 72)

    comparisons = compare_models(
        csv_A=args.model_a_csv,
        csv_B=args.model_b_csv,
        metrics=args.metric,
        tests=tests,
        alpha=args.alpha,
        alternative=args.alternative,
        n_permutations=args.n_permutations,
        perm_seed=args.perm_seed,
        gamma_bins=args.gamma_bins,
        kappa_bins=args.kappa_bins,
    )

    for c in comparisons:
        print(format_comparison(c, args.alpha))

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_json = args.output_dir / f"stats_{ts}.json"
    payload = {
        "timestamp": ts,
        "model_a_csv": str(args.model_a_csv),
        "model_b_csv": str(args.model_b_csv),
        "metrics": args.metric,
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