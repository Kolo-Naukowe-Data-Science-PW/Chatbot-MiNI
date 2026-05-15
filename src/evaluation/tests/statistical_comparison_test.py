"""
Unit tests for statistical_comparison module.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.statistical_comparison import (
    extract_rank_from_sources,
    goodman_kruskal_gamma,
    kappa_agreement,
    wilcoxon_test,
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

    def test_wilcoxon_test(self):
        """Test Wilcoxon signed-rank test."""
        # No difference
        diff_zero = np.array([0.0, 0.0, 0.0])
        stat, pval, is_sig = wilcoxon_test(diff_zero)
        # With all zeros, Wilcoxon returns NaN
        self.assertTrue(np.isnan(stat) or pval == 1.0)
        
        # Clear difference (all positive)
        diff_positive = np.array([0.1, 0.2, 0.15, 0.25, 0.1])
        stat, pval, is_sig = wilcoxon_test(diff_positive)
        self.assertIsInstance(stat, float)
        self.assertIsInstance(pval, float)
        self.assertEqual(is_sig, pval <= 0.05)
        
        # Mixed differences
        diff_mixed = np.array([0.1, -0.05, 0.15, -0.02, 0.08])
        stat, pval, is_sig = wilcoxon_test(diff_mixed)
        self.assertIsInstance(pval, float)
        self.assertGreaterEqual(pval, 0.0)
        self.assertLessEqual(pval, 1.0)

    def test_goodman_kruskal_gamma(self):
        """Test Goodman-Kruskal gamma coefficient."""
        # Perfect concordance (both increasing)
        ranks_A = np.array([1, 2, 3, 4, 5])
        ranks_B = np.array([1, 2, 3, 4, 5])
        gamma, conc, disc = goodman_kruskal_gamma(ranks_A, ranks_B)
        self.assertAlmostEqual(gamma, 1.0, places=4)
        self.assertEqual(conc, 10)  # C(5,2) = 10
        self.assertEqual(disc, 0)
        
        # Perfect discordance (opposite)
        ranks_B_reversed = np.array([5, 4, 3, 2, 1])
        gamma, conc, disc = goodman_kruskal_gamma(ranks_A, ranks_B_reversed)
        self.assertAlmostEqual(gamma, -1.0, places=4)
        self.assertEqual(conc, 0)
        self.assertEqual(disc, 10)
        
        # No correlation (random)
        ranks_B_random = np.array([3, 1, 4, 2, 5])
        gamma, conc, disc = goodman_kruskal_gamma(ranks_A, ranks_B_random)
        # gamma should be between -1 and 1
        self.assertGreaterEqual(gamma, -1.0)
        self.assertLessEqual(gamma, 1.0)
        self.assertEqual(conc + disc, 10)  # Total pairs

    def test_kappa_agreement(self):
        """Test kappa coefficient for retrieval stability."""
        # Same results across runs → high kappa
        run1 = [
            {"url1", "url2", "url3"},  # query 1
            {"url4", "url5"},           # query 2
        ]
        run2 = [
            {"url1", "url2", "url3"},
            {"url4", "url5"},
        ]
        
        kappa_dict = kappa_agreement([run1, run2])
        self.assertIn('kappa_mean', kappa_dict)
        self.assertGreater(kappa_dict['kappa_mean'], 0.9)
        
        # Completely different results
        run3 = [
            {"url7", "url8", "url9"},
            {"url10", "url11"},
        ]
        
        kappa_dict = kappa_agreement([run1, run3])
        self.assertIn('kappa_mean', kappa_dict)
        self.assertLessEqual(kappa_dict['kappa_mean'], 0.5)


if __name__ == "__main__":
    unittest.main()
