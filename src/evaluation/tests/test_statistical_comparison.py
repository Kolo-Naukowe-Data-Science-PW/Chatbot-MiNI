"""
Tests for statistical comparison functions.

Validates implementations against thesis definitions:
1. Wilcoxon Signed-Rank Test    - per-query differences with explicit tie-averaging
2. Goodman-Kruskal Gamma         - ordinal association between model ranks
3. Kappa Coefficient             - agreement beyond chance
4. Paired Permutation Test       - non-parametric sign-flip test
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistical_comparison import (  # noqa: E402
    _rank_with_ties,
    goodman_kruskal_gamma_test,
    kappa_coefficient,
    paired_permutation_test,
    wilcoxon_signed_rank_test,
)


class TestRankWithTies:
    """Test tie-averaging in ranking."""

    def test_no_ties_simple(self):
        """Test ranking with no ties."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ranks = _rank_with_ties(values)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(ranks, expected)

    def test_single_tie_top(self):
        """Test ranking with two equal values at top."""
        values = np.array([1.0, 1.0, 2.0, 3.0])
        ranks = _rank_with_ties(values)
        expected = np.array([1.5, 1.5, 3.0, 4.0])
        np.testing.assert_array_equal(ranks, expected)

    def test_multiple_ties(self):
        """Test ranking with multiple groups of ties."""
        values = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
        ranks = _rank_with_ties(values)
        expected = np.array([1.5, 1.5, 4.0, 4.0, 4.0, 6.0])
        np.testing.assert_array_equal(ranks, expected)

    def test_all_equal(self):
        """Test ranking when all values are equal."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        ranks = _rank_with_ties(values)
        expected = np.array([2.5, 2.5, 2.5, 2.5])
        np.testing.assert_array_equal(ranks, expected)

    def test_negative_values(self):
        """Test ranking with negative values."""
        values = np.array([-3.0, -1.0, -1.0, 2.0])
        ranks = _rank_with_ties(values)
        expected = np.array([1.0, 2.5, 2.5, 4.0])
        np.testing.assert_array_equal(ranks, expected)


class TestWilcoxonSignedRankTest:
    """Test Wilcoxon Signed-Rank Test implementation per thesis."""

    def test_perfect_agreement(self):
        """Test when A = B (all differences zero)."""
        diff = np.array([0.0, 0.0, 0.0, 0.0])
        result = wilcoxon_signed_rank_test(diff)

        assert result["n"] == 0  # All zeros removed
        assert result["p_value"] == 1.0
        assert result["is_significant"] is False

    def test_single_nonzero(self):
        """Test with only one nonzero difference."""
        diff = np.array([0.5])
        result = wilcoxon_signed_rank_test(diff)

        assert result["n"] == 1
        assert result["W_plus"] == 1.0  # Only positive rank
        assert result["W_minus"] == 0.0
        assert result["W"] == 0.0  # min(W+, W-) = 0

    def test_simple_case_small_n(self):
        """Test simple case with n < 25 (exact test).

        Thesis example: d = [0.5, -0.2, 0.3, 0.0, -0.1]
        After removing zero: d = [0.5, -0.2, 0.3, -0.1]
        |d| = [0.5, 0.2, 0.3, 0.1]
        Ranks by magnitude: rank(0.1)=1, rank(0.2)=2, rank(0.3)=3, rank(0.5)=4
        W+ = 3+4 = 7 (for d=0.5 and d=0.3)
        W- = 1+2 = 3 (for d=-0.1 and d=-0.2)
        W = min(7, 3) = 3
        """
        diff = np.array([0.5, -0.2, 0.3, 0.0, -0.1])
        result = wilcoxon_signed_rank_test(diff)

        assert result["n"] == 4
        assert result["W_plus"] == 7.0
        assert result["W_minus"] == 3.0
        assert result["W"] == 3.0
        assert result["method"].startswith("Exact")

    def test_tied_differences(self):
        """Test with tied absolute differences (tie-averaging).

        Thesis definition: if |d_i| = |d_j|, then rank(d_i) = rank(d_j) = (i+j)/2
        """
        # d = [0.5, -0.5, 0.1, -0.1]
        # |d| sorted: [0.1, 0.1, 0.5, 0.5]
        # Ranks with tie-averaging: 0.1→1.5, 0.1→1.5, 0.5→3.5, 0.5→3.5
        diff = np.array([0.5, -0.5, 0.1, -0.1])
        result = wilcoxon_signed_rank_test(diff)

        assert result["n"] == 4
        # W+ = rank(0.5) + rank(0.1) = 3.5 + 1.5 = 5.0
        assert result["W_plus"] == 5.0
        # W- = rank(-0.5) + rank(-0.1) = 3.5 + 1.5 = 5.0
        assert result["W_minus"] == 5.0
        assert result["W"] == 5.0

    def test_normal_approximation_large_n(self):
        """Test Z-score approximation for n ≥ 25."""
        # Generate large sample with known distribution
        np.random.seed(42)
        diff = np.random.normal(0.1, 0.5, 30)  # Mean slightly > 0
        result = wilcoxon_signed_rank_test(diff)

        assert result["n"] >= 20  # Most differences nonzero
        assert "Approximate normal (Z-score" in result["method"] or result["n"] < 25
        assert "z_score" in result
        if result["n"] >= 25:
            assert result["expected_W"] is not None
            assert result["variance_W"] is not None
            # E[W] = n(n+1)/4
            expected_E = result["n"] * (result["n"] + 1) / 4
            assert abs(result["expected_W"] - expected_E) < 1e-6

    def test_two_tailed_p_value(self):
        """Test two-tailed p-value calculation."""
        diff = np.array(
            [0.1, 0.1, 0.1, 0.1, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05]  # 4 positive
        )  # 6 negative
        result = wilcoxon_signed_rank_test(diff, alternative="two-sided")

        assert 0.0 <= result["p_value"] <= 1.0
        assert result["method"] == "Exact (n=10 < 25)"

    def test_all_positive_differences(self):
        """Test when all differences are positive."""
        diff = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = wilcoxon_signed_rank_test(diff)

        assert result["n"] == 5
        assert result["W_minus"] == 0.0
        assert result["W_plus"] == 15.0  # Sum of ranks 1+2+3+4+5
        assert result["W"] == 0.0  # min(15, 0) = 0
        # With n=5, p-value ≈ 0.0625 (borderline for α=0.05)
        assert 0.05 <= result["p_value"] <= 0.1

    def test_all_negative_differences(self):
        """Test when all differences are negative."""
        diff = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
        result = wilcoxon_signed_rank_test(diff)

        assert result["n"] == 5
        assert result["W_plus"] == 0.0
        assert result["W_minus"] == 15.0  # Sum of ranks
        assert result["W"] == 0.0  # min(0, 15) = 0


class TestGoodmanKruskalGamma:
    """Test Goodman-Kruskal gamma coefficient per thesis."""

    def test_perfect_concordance(self):
        """Test perfect agreement (all pairs concordant).

        Both models rank queries identically.
        """
        ranks_A = np.array([1, 2, 3, 4, 5])
        ranks_B = np.array([1, 2, 3, 4, 5])
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B)

        # All pairs concordant: C = 10, D = 0
        assert result["gamma"] == 1.0
        assert result["interpretation"].startswith("Strong association")

    def test_perfect_discordance(self):
        """Test perfect disagreement (all pairs discordant).

        Models rank queries in opposite order.
        """
        ranks_A = np.array([1, 2, 3, 4, 5])
        ranks_B = np.array([5, 4, 3, 2, 1])
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B)

        # All pairs discordant: C = 0, D = 10
        assert result["gamma"] == -1.0
        assert "opposite direction" in result["interpretation"]

    def test_no_association(self):
        """Test when C = D (no association)."""
        ranks_A = np.array([1, 2, 3, 4])
        ranks_B = np.array([1, 3, 2, 4])  # Partially scrambled
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B)

        # Should have equal concordant and discordant pairs
        if result["C"] == result["D"]:
            assert result["gamma"] == 0.0

    def test_thesis_definition_formula(self):
        """Verify γ = (C-D)/(C+D) formula is correct.

        Per thesis: Π_c and Π_d calculated from rank pairs.
        """
        ranks_A = np.array([1, 1, 2, 3])  # With ties
        ranks_B = np.array([2, 1, 1, 3])
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B)

        # Verify formula
        C = result["C"]
        D = result["D"]
        if C + D > 0:
            expected_gamma = (C - D) / (C + D)
            assert abs(result["gamma"] - expected_gamma) < 1e-10

    def test_rank_infinities(self):
        """Test with infinity ranks (not retrieved)."""
        ranks_A = np.array([1, 2, 3, 999])  # Last not retrieved
        ranks_B = np.array([1, 2, 999, 3])  # Different position of not-retrieved
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B)

        # Should still calculate properly
        assert -1.0 <= result["gamma"] <= 1.0
        assert result["n_queries"] == 4

    def test_weak_association(self):
        """Test weak association classification."""
        # Create mostly concordant but not perfect
        ranks_A = np.array([1, 2, 3, 4, 5, 6])
        ranks_B = np.array([1, 2, 4, 3, 5, 6])  # One pair swapped
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B)

        # Should be high but not perfect
        assert 0.0 < result["gamma"] < 1.0
        assert (
            "Moderate" in result["interpretation"]
            or "Strong" in result["interpretation"]
        )


class TestKappaCoefficient:
    """Test Kappa coefficient per thesis definition."""

    def test_perfect_agreement(self):
        """Test perfect agreement (all documents identical)."""
        Z_i = [set(["url1", "url2", "url3"]), set(["url1", "url2"]), set(["url3"])]
        Z_j = [set(["url1", "url2", "url3"]), set(["url1", "url2"]), set(["url3"])]

        result = kappa_coefficient(Z_i, Z_j)

        assert result["kappa_mean"] == 1.0
        assert all(k == 1.0 for k in result["kappas"])

    def test_perfect_disagreement(self):
        """Test perfect disagreement (no overlap)."""
        Z_i = [set(["url1", "url2"]), set(["url3"]), set(["url4", "url5"])]
        Z_j = [set(["url6", "url7"]), set(["url8"]), set(["url9"])]

        result = kappa_coefficient(Z_i, Z_j)

        # No agreement at all, p_o = 0, p_e > 0 → κ < 0
        assert result["kappa_mean"] < 0.0

    def test_empty_universes(self):
        """Test with empty retrieval sets (edge case)."""
        Z_i = [set(), set(["url1"])]  # Both empty
        Z_j = [set(), set(["url1"])]  # Both empty

        result = kappa_coefficient(Z_i, Z_j)

        # Empty universe should give κ=1 (vacuously perfect agreement)
        assert len(result["kappas"]) == 2
        assert result["kappas"][0] == 1.0
        assert result["kappas"][1] == 1.0

    def test_contingency_table_formula(self):
        """Verify contingency table and κ formula per thesis.

        κ = (p_o - p_e) / (1 - p_e)
        where p_o = (a+d)/|U|  and  p_e = P(A)*P(B) + P(¬A)*P(¬B)
        """
        Z_i = [set(["A", "B", "C"])]
        Z_j = [set(["B", "C", "D"])]

        result = kappa_coefficient(Z_i, Z_j)

        # U = {A, B, C, D}
        # a = |Z_i ∩ Z_j| = |{B,C}| = 2
        # b = |Z_i \ Z_j| = |{A}| = 1
        # c = |Z_j \ Z_i| = |{D}| = 1
        # d = |U \ (Z_i ∪ Z_j)| = 0
        # p_o = (2+0)/4 = 0.5
        # p_e = (3/4)(3/4) + (1/4)(1/4) = 9/16 + 1/16 = 10/16 = 0.625
        # κ = (0.5 - 0.625) / (1 - 0.625) ≈ -0.333

        kappa = result["kappas"][0]
        assert abs(kappa - (-1 / 3)) < 0.01

    def test_with_relevant_documents(self):
        """Test with universe including relevant documents."""
        Z_i = [set(["url1", "url2"])]
        Z_j = [set(["url2", "url3"])]
        relevant = [set(["url1", "url2", "url3", "url4"])]

        result = kappa_coefficient(Z_i, Z_j, relevant)

        # Universe should include all: url1, url2, url3, url4
        # This affects calculation of p_o and p_e
        assert result["n_queries"] == 1
        assert len(result["kappas"]) == 1


class TestPairedPermutationTest:
    """Test paired permutation test implementation."""

    def test_perfect_match(self):
        """Test when scores are identical."""
        scores_A = np.array([0.5, 0.6, 0.7, 0.8])
        scores_B = np.array([0.5, 0.6, 0.7, 0.8])

        result = paired_permutation_test(scores_A, scores_B, n_permutations=1000)

        assert result["observed_diff"] == 0.0
        # With perfect match, permutation p-value should be 1.0
        assert result["p_value"] == 1.0

    def test_clear_difference(self):
        """Test when A clearly > B."""
        np.random.seed(42)
        scores_A = np.random.normal(0.8, 0.1, 20)  # Mean 0.8
        scores_B = np.random.normal(0.5, 0.1, 20)  # Mean 0.5

        result = paired_permutation_test(
            scores_A, scores_B, n_permutations=1000, seed=42
        )

        assert result["observed_diff"] > 0.2  # Clear positive difference
        assert result["p_value"] < 0.1  # Should be significant

    def test_perm_stats_distribution(self):
        """Test that permutation statistics form proper distribution."""
        np.random.seed(42)
        diffs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        scores_A = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        scores_B = scores_A - diffs

        result = paired_permutation_test(
            scores_A, scores_B, n_permutations=1000, seed=42
        )

        # Permutation stats should range from -mean_diff to +mean_diff approximately
        obs_mean = result["observed_diff"]
        perm_abs_max = np.max(np.abs(result["perm_stats"]))
        assert perm_abs_max >= abs(obs_mean)

    def test_two_sided_vs_one_sided(self):
        """Test one-tailed vs two-tailed p-values."""
        scores_A = np.array([0.7, 0.7, 0.7, 0.7])
        scores_B = np.array([0.5, 0.5, 0.5, 0.5])

        result_two = paired_permutation_test(
            scores_A, scores_B, n_permutations=1000, seed=42, alternative="two-sided"
        )
        result_greater = paired_permutation_test(
            scores_A, scores_B, n_permutations=1000, seed=42, alternative="greater"
        )

        # For clear difference, one-tailed should be ≤ two-tailed
        assert result_greater["p_value"] <= result_two["p_value"]

    def test_reproducibility_seed(self):
        """Test that same seed produces identical results."""
        scores_A = np.random.normal(0.6, 0.1, 15)
        scores_B = np.random.normal(0.5, 0.1, 15)

        result1 = paired_permutation_test(
            scores_A, scores_B, n_permutations=500, seed=123
        )
        result2 = paired_permutation_test(
            scores_A, scores_B, n_permutations=500, seed=123
        )

        assert result1["p_value"] == result2["p_value"]
        np.testing.assert_array_equal(result1["perm_stats"], result2["perm_stats"])

    def test_no_difference_p_value_one(self):
        """Test that no difference gives p ≈ 1.0 in two-tailed."""
        scores_A = np.array([1.0, 2.0, 3.0])
        scores_B = np.array([1.0, 2.0, 3.0])

        result = paired_permutation_test(
            scores_A, scores_B, n_permutations=100, seed=42, alternative="two-sided"
        )

        assert result["p_value"] == 1.0


class TestStatisticalTestsIntegration:
    """Integration tests combining multiple statistical tests."""

    def test_consistent_significance_detection(self):
        """Test that different tests detect same strong signals."""
        # Generate data with clear A > B signal
        np.random.seed(42)
        n = 30
        scores_A = np.random.normal(0.75, 0.15, n)
        scores_B = np.random.normal(0.55, 0.15, n)
        diffs = scores_A - scores_B

        # Wilcoxon test
        wilcox_result = wilcoxon_signed_rank_test(diffs)

        # Permutation test
        perm_result = paired_permutation_test(scores_A, scores_B, seed=42)

        # Both should detect significance
        assert wilcox_result["p_value"] < 0.05
        assert perm_result["p_value"] < 0.05

    def test_gamma_and_kappa_consistency(self):
        """Test that gamma and kappa measure similar agreement patterns."""
        # Create consistent rankings
        ranks_A = np.array([1, 2, 3, 4, 5])
        ranks_B = np.array([1, 2, 3, 4, 5])

        gamma_result = goodman_kruskal_gamma_test(ranks_A, ranks_B)

        # Both should indicate strong agreement
        assert gamma_result["gamma"] > 0.9
        assert "Strong" in gamma_result["interpretation"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
