"""
Unit tests for statistical_comparison module.

Tests updated to match new implementations:
- wilcoxon_signed_rank_test() returns dict with W, W+, W-, p_value, etc.
- goodman_kruskal_gamma_test() returns dict with gamma, C, D, n_queries, interpretation
- kappa_coefficient() returns dict with kappa_mean, kappas, n_queries
"""

import unittest

import numpy as np

from src.evaluation.statistical_comparison import (
    extract_rank_from_sources,
    goodman_kruskal_gamma_test,
    kappa_coefficient,
    wilcoxon_signed_rank_test,
)


class TestStatisticalComparison(unittest.TestCase):
    """Test cases for statistical comparison functions."""

    def test_extract_rank_from_sources(self):
        """Test rank extraction from sources string."""
        # Exact match at position 1
        self.assertEqual(extract_rank_from_sources("url1;url2;url3"), 3)

        # Empty sources
        self.assertEqual(extract_rank_from_sources(""), 999)
        self.assertEqual(extract_rank_from_sources("   "), 999)

        # Single URL
        self.assertEqual(extract_rank_from_sources("url1"), 1)

        # With spacing
        self.assertEqual(extract_rank_from_sources("url1; url2 ; url3"), 3)

    def test_wilcoxon_signed_rank_test(self):
        """Test Wilcoxon signed-rank test returning dict."""
        # No difference
        diff_zero = np.array([0.0, 0.0, 0.0])
        result = wilcoxon_signed_rank_test(diff_zero)
        # With all zeros, n=0
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertFalse(result["is_significant"])

        # Clear difference (all positive)
        diff_positive = np.array([0.1, 0.2, 0.15, 0.25, 0.1])
        result = wilcoxon_signed_rank_test(diff_positive)
        self.assertIsInstance(result["p_value"], float)
        self.assertEqual(result["is_significant"], result["p_value"] <= 0.05)
        # All positive → W+ = sum of all ranks, W- = 0
        self.assertGreater(result["W_plus"], 0)
        self.assertEqual(result["W_minus"], 0.0)
        self.assertEqual(result["W"], 0.0)

        # Mixed differences
        diff_mixed = np.array([0.1, -0.05, 0.15, -0.02, 0.08])
        result = wilcoxon_signed_rank_test(diff_mixed)
        self.assertIsInstance(result["p_value"], float)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)
        # Should have both W+ and W-
        self.assertGreater(result["W_plus"], 0)
        self.assertGreater(result["W_minus"], 0)

    def test_goodman_kruskal_gamma_test(self):
        """Test Goodman-Kruskal gamma coefficient returning dict."""
        # Perfect concordance (both increasing)
        ranks_A = np.array([1, 2, 3, 4, 5])
        ranks_B = np.array([1, 2, 3, 4, 5])
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B)
        self.assertAlmostEqual(result["gamma"], 1.0, places=4)
        self.assertEqual(result["C"], 10)  # C(5,2) = 10
        self.assertEqual(result["D"], 0)
        self.assertIn("Strong association", result["interpretation"])

        # Perfect discordance (opposite)
        ranks_B_reversed = np.array([5, 4, 3, 2, 1])
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B_reversed)
        self.assertAlmostEqual(result["gamma"], -1.0, places=4)
        self.assertEqual(result["C"], 0)
        self.assertEqual(result["D"], 10)
        self.assertIn("opposite direction", result["interpretation"])

        # No correlation (random)
        ranks_B_random = np.array([3, 1, 4, 2, 5])
        result = goodman_kruskal_gamma_test(ranks_A, ranks_B_random)
        # gamma should be between -1 and 1
        self.assertGreaterEqual(result["gamma"], -1.0)
        self.assertLessEqual(result["gamma"], 1.0)
        self.assertEqual(result["C"] + result["D"], 10)  # Total pairs

    def test_kappa_coefficient(self):
        """Test kappa coefficient for retrieval stability with new signature."""
        # Same results across runs → high kappa
        run1_q1 = {"url1", "url2", "url3"}  # query 1
        run1_q2 = {"url4", "url5"}  # query 2

        run2_q1 = {"url1", "url2", "url3"}
        run2_q2 = {"url4", "url5"}

        # New signature: (retrieval_sets_i, retrieval_sets_j, relevant_sets=None)
        result = kappa_coefficient([run1_q1, run1_q2], [run2_q1, run2_q2])
        self.assertIn("kappa_mean", result)
        self.assertGreater(result["kappa_mean"], 0.9)
        self.assertEqual(result["n_queries"], 2)

        # Completely different results
        run3_q1 = {"url7", "url8", "url9"}
        run3_q2 = {"url10", "url11"}

        result = kappa_coefficient([run1_q1, run1_q2], [run3_q1, run3_q2])
        self.assertIn("kappa_mean", result)
        self.assertLessEqual(result["kappa_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
