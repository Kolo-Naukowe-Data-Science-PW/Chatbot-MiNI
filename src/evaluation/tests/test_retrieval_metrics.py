"""
Comprehensive test suite for retrieval evaluation metrics.

Tests verify implementations match mathematical definitions from:
  "Retrieval Evaluation in RAG Systems" (thesis chapter 3)

Metrics tested:
  - Hit@k: Binary relevance check
  - Precision@k: Fraction of top-k that are relevant
  - Recall@k: Fraction of all relevant docs found in top-k
  - MRR@k: Reciprocal rank of first relevant result
  - nDCG@k: Normalized Discounted Cumulative Gain (graded relevance)
  - MAP@k: Mean Average Precision

Mathematical definitions:
  Hit@k(q) = 1 if ∃z ∈ Z_top-k(q) ∩ Z_rel(q), else 0
  Precision@k(q) = |Z_top-k(q) ∩ Z_rel(q)| / k
  Recall@k(q) = |Z_top-k(q) ∩ Z_rel(q)| / |Z_rel(q)|
  MRR@k(q) = 1 / min{i ∈ K_rel^(k)(q)} (or 0 if no relevant)
  nDCG@k(q) = DCG@k(q) / IDCG@k(q)
  MAP@k(q) = (1/|K_rel(q)|) * Σ Precision@i(q) for i ∈ K_rel^(k)(q)

Usage:
    pytest src/evaluation/tests/test_retrieval_metrics.py -v
"""

import sys
from math import log2
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    average_precision_at_k,
    precision_at_k,
    recall_at_k,
    dcg_at_k,
    ideal_dcg_at_k,
    hierarchical_relevance,
    normalize_url,
)


class TestHitAtK:
    """Test Hit@k metric (binary: 1 if relevant result in top-k, else 0)."""

    def test_hit_at_k_found_at_rank_1(self):
        """Hit@k should be 1 when relevant item is at rank 1."""
        rel_scores = [1.0, 0.0, 0.0, 0.0]
        assert hit_at_k(rel_scores, k=1) == 1.0
        assert hit_at_k(rel_scores, k=3) == 1.0

    def test_hit_at_k_found_at_rank_k(self):
        """Hit@k should be 1 when relevant item is within top-k."""
        rel_scores = [0.0, 0.0, 1.0, 0.0]  # Relevant at rank 3
        assert hit_at_k(rel_scores, k=1) == 0.0
        assert hit_at_k(rel_scores, k=3) == 1.0
        assert hit_at_k(rel_scores, k=10) == 1.0

    def test_hit_at_k_not_found(self):
        """Hit@k should be 0 when no relevant item in top-k."""
        rel_scores = [0.0, 0.0, 0.0, 0.0]
        assert hit_at_k(rel_scores, k=1) == 0.0
        assert hit_at_k(rel_scores, k=3) == 0.0

    def test_hit_at_k_empty_list(self):
        """Hit@k should be 0 for empty retrieval list."""
        rel_scores = []
        assert hit_at_k(rel_scores, k=1) == 0.0

    def test_hit_at_k_threshold(self):
        """Hit@k should use relevance threshold (default 0.5)."""
        rel_scores = [0.4, 0.6, 0.0]  # Only score 0.6 exceeds threshold
        assert hit_at_k(rel_scores, k=1, threshold=0.5) == 0.0
        assert hit_at_k(rel_scores, k=2, threshold=0.5) == 1.0
        assert hit_at_k(rel_scores, k=3, threshold=0.4) == 1.0

    def test_hit_at_k_multiple_relevant(self):
        """Hit@k should be 1 if any relevant item exists in top-k."""
        rel_scores = [0.0, 1.0, 1.0, 0.0]  # Multiple relevant items
        assert hit_at_k(rel_scores, k=2) == 1.0
        assert hit_at_k(rel_scores, k=4) == 1.0


class TestMRRAtK:
    """Test MRR@k (reciprocal rank of first relevant result)."""

    def test_mrr_at_k_rank_1(self):
        """MRR@k should be 1.0 when relevant item is at rank 1."""
        rel_scores = [1.0, 0.0, 0.0]
        assert mrr_at_k(rel_scores, k=3) == 1.0

    def test_mrr_at_k_rank_2(self):
        """MRR@k should be 0.5 when relevant item is at rank 2."""
        rel_scores = [0.0, 1.0, 0.0]
        assert mrr_at_k(rel_scores, k=3) == 0.5

    def test_mrr_at_k_rank_5(self):
        """MRR@k should be 0.2 when relevant item is at rank 5."""
        rel_scores = [0.0, 0.0, 0.0, 0.0, 1.0]
        assert mrr_at_k(rel_scores, k=10) == 0.2

    def test_mrr_at_k_not_found(self):
        """MRR@k should be 0 when no relevant item in top-k."""
        rel_scores = [0.0, 0.0, 0.0]
        assert mrr_at_k(rel_scores, k=3) == 0.0

    def test_mrr_at_k_beyond_k(self):
        """MRR@k should be 0 if relevant item is beyond rank k."""
        rel_scores = [0.0, 0.0, 1.0, 0.0]  # Relevant at rank 3
        assert mrr_at_k(rel_scores, k=2) == 0.0

    def test_mrr_at_k_threshold(self):
        """MRR@k should use relevance threshold."""
        rel_scores = [0.3, 0.6, 0.0]  # Only 0.6 exceeds 0.5 threshold
        assert mrr_at_k(rel_scores, k=3, threshold=0.5) == 0.5

    def test_mrr_at_k_first_relevant_only(self):
        """MRR@k should return rank of first relevant only."""
        rel_scores = [0.0, 1.0, 1.0, 1.0]  # Multiple relevant
        assert mrr_at_k(rel_scores, k=4) == 0.5  # First is at rank 2


class TestNDCGAtK:
    """Test nDCG@k (normalized Discounted Cumulative Gain)."""

    def test_ndcg_perfect_ranking(self):
        """nDCG@k should be 1.0 for perfect ranking."""
        rel_scores = [1.0, 1.0, 1.0, 0.0]
        ndcg = ndcg_at_k(rel_scores, k=4)
        assert ndcg == pytest.approx(1.0, abs=0.01)

    def test_ndcg_reversed_ranking(self):
        """nDCG@k should be low for reversed ranking."""
        rel_scores = [0.0, 0.0, 1.0, 1.0]
        ndcg = ndcg_at_k(rel_scores, k=4)
        assert ndcg < 0.5

    def test_ndcg_single_relevant(self):
        """nDCG@k with single relevant document."""
        rel_scores = [1.0, 0.0, 0.0, 0.0]
        ndcg = ndcg_at_k(rel_scores, k=1)
        assert ndcg == pytest.approx(1.0, abs=0.01)

    def test_ndcg_no_relevant(self):
        """nDCG@k should be 0 when no relevant items."""
        rel_scores = [0.0, 0.0, 0.0]
        ndcg = ndcg_at_k(rel_scores, k=3)
        assert ndcg == 0.0

    def test_ndcg_graded_relevance(self):
        """nDCG@k should handle graded relevance scores."""
        rel_scores = [0.0, 0.5, 1.0, 0.25]
        ndcg = ndcg_at_k(rel_scores, k=4)
        assert 0.0 <= ndcg <= 1.0

    def test_ndcg_range_in_k(self):
        """nDCG@k should be in [0, 1] range."""
        rel_scores = [0.7, 0.3, 0.9, 0.1, 0.5]
        for k in [1, 2, 3, 4, 5]:
            ndcg = ndcg_at_k(rel_scores, k=k)
            assert 0.0 <= ndcg <= 1.0

    def test_ndcg_cutoff_effect(self):
        """nDCG@k should consider only top k results."""
        rel_scores = [1.0, 1.0, 1.0, 0.0, 0.0]
        ndcg_3 = ndcg_at_k(rel_scores, k=3)
        ndcg_5 = ndcg_at_k(rel_scores, k=5)
        assert ndcg_3 == pytest.approx(ndcg_5, abs=0.01)


class TestAveragePrecisionAtK:
    """Test AP@k (Average Precision@k)."""

    def test_ap_single_relevant_at_1(self):
        """AP@k with single relevant at rank 1."""
        rel_scores = [1.0, 0.0, 0.0]  # total_rel = 1
        ap_score = average_precision_at_k(rel_scores, k=3, total_rel=1)
        assert ap_score == pytest.approx(1.0, abs=0.01)

    def test_ap_single_relevant_at_2(self):
        """AP@k with single relevant at rank 2."""
        rel_scores = [0.0, 1.0, 0.0]
        ap_score = average_precision_at_k(rel_scores, k=3, total_rel=1)
        assert ap_score == pytest.approx(0.5, abs=0.01)

    def test_ap_multiple_relevant(self):
        """AP@k with multiple relevant documents."""
        rel_scores = [1.0, 0.0, 1.0, 0.0]  # Relevant at ranks 1 and 3
        # AP = (1/1 + 2/3) / 2 ≈ 0.833
        ap_score = average_precision_at_k(rel_scores, k=4, total_rel=2)
        assert ap_score == pytest.approx(0.833, abs=0.01)

    def test_ap_no_relevant(self):
        """AP@k should be 0 when no relevant items."""
        rel_scores = [0.0, 0.0, 0.0]
        ap_score = average_precision_at_k(rel_scores, k=3, total_rel=1)
        assert ap_score == 0.0

    def test_ap_beyond_k(self):
        """AP@k should only consider results up to rank k."""
        rel_scores = [0.0, 0.0, 1.0, 1.0]  # Relevant at ranks 3, 4
        # At k=2: no relevant items
        ap_score = average_precision_at_k(rel_scores, k=2, total_rel=2)
        assert ap_score == 0.0

    def test_ap_range_in_k(self):
        """AP@k should be in [0, 1] range."""
        rel_scores = [1.0, 0.0, 1.0, 1.0, 0.0]
        ap_score = average_precision_at_k(rel_scores, k=5, total_rel=3)
        assert 0.0 <= ap_score <= 1.0


class TestPrecisionAtK:
    """Test Precision@k (fraction of top-k that are relevant)."""

    def test_precision_all_relevant(self):
        """Precision@k should be 1.0 when all top-k are relevant."""
        rel_scores = [1.0, 1.0, 1.0, 0.0]
        assert precision_at_k(rel_scores, k=3) == 1.0

    def test_precision_half_relevant(self):
        """Precision@k should be 0.5 when half are relevant."""
        rel_scores = [1.0, 0.0, 1.0, 0.0]
        assert precision_at_k(rel_scores, k=4) == 0.5

    def test_precision_none_relevant(self):
        """Precision@k should be 0 when none are relevant."""
        rel_scores = [0.0, 0.0, 0.0]
        assert precision_at_k(rel_scores, k=3) == 0.0

    def test_precision_at_k_1(self):
        """Precision@1 depends on first result only."""
        rel_scores = [1.0, 0.0, 0.0]
        assert precision_at_k(rel_scores, k=1) == 1.0
        
        rel_scores = [0.0, 1.0, 1.0]
        assert precision_at_k(rel_scores, k=1) == 0.0

    def test_precision_threshold(self):
        """Precision@k should use relevance threshold."""
        rel_scores = [0.6, 0.3, 0.7, 0.2]
        assert precision_at_k(rel_scores, k=4, threshold=0.5) == 0.5  # 2/4


class TestRecallAtK:
    """Test Recall@k (fraction of relevant docs found in top-k)."""

    def test_recall_all_found(self):
        """Recall@k should be 1.0 when all relevant found."""
        rel_scores = [1.0, 1.0, 0.0, 0.0]  # 2 relevant, both in top-2
        assert recall_at_k(rel_scores, k=2, total_rel=2) == 1.0

    def test_recall_half_found(self):
        """Recall@k should be 0.5 when half found."""
        rel_scores = [1.0, 0.0, 0.0, 0.0]  # 2 relevant, 1 in top-3
        assert recall_at_k(rel_scores, k=3, total_rel=2) == 0.5

    def test_recall_none_found(self):
        """Recall@k should be 0 when none found."""
        rel_scores = [0.0, 0.0, 0.0]  # 2 relevant total, 0 found
        assert recall_at_k(rel_scores, k=3, total_rel=2) == 0.0

    def test_recall_empty_total_rel(self):
        """Recall@k should be 0 when total_rel=0."""
        rel_scores = [1.0, 1.0, 0.0]
        recall = recall_at_k(rel_scores, k=3, total_rel=0)
        assert recall == 0.0

    def test_recall_beyond_k(self):
        """Recall@k only considers top-k results."""
        rel_scores = [0.0, 0.0, 1.0, 1.0]  # Relevant only beyond k=2
        assert recall_at_k(rel_scores, k=2, total_rel=2) == 0.0
        assert recall_at_k(rel_scores, k=4, total_rel=2) == 1.0


class TestHierarchicalRelevance:
    """Test hierarchical URL relevance scoring."""

    def test_exact_url_match(self):
        """Exact URL match should get score 1.0."""
        retrieved = "https://example.com/path/to/page"
        target = "https://example.com/path/to/page"
        assert hierarchical_relevance(retrieved, target) == 1.0

    def test_exact_match_trailing_slash_ignored(self):
        """Trailing slashes should be ignored."""
        retrieved = "https://example.com/path/to/page/"
        target = "https://example.com/path/to/page"
        assert hierarchical_relevance(retrieved, target) == 1.0

    def test_child_page(self):
        """Child page should get score 0.5."""
        retrieved = "https://example.com/path/to/page/child"
        target = "https://example.com/path/to/page"
        assert hierarchical_relevance(retrieved, target) == 0.5

    def test_grandchild_page(self):
        """Grandchild should get lower score (0.5^2 = 0.25)."""
        retrieved = "https://example.com/path/to/page/child/grandchild"
        target = "https://example.com/path/to/page"
        assert hierarchical_relevance(retrieved, target) == pytest.approx(0.25, abs=0.01)

    def test_parent_page(self):
        """Parent page is even less relevant (not penalized, but asymmetric)."""
        retrieved = "https://example.com/path/to"
        target = "https://example.com/path/to/page"
        # Parent gets lower score than target (asymmetric)
        rel = hierarchical_relevance(retrieved, target)
        assert rel > 0.0 and rel < 0.5

    def test_different_domain(self):
        """Different domain should get score 0."""
        retrieved = "https://other.com/path/to/page"
        target = "https://example.com/path/to/page"
        assert hierarchical_relevance(retrieved, target) == 0.0

    def test_different_scheme(self):
        """Different scheme (http vs https) should be 0."""
        retrieved = "http://example.com/path"
        target = "https://example.com/path"
        assert hierarchical_relevance(retrieved, target) == 0.0

    def test_empty_urls(self):
        """Empty URLs should return 0."""
        assert hierarchical_relevance("", "https://example.com") == 0.0
        assert hierarchical_relevance("https://example.com", "") == 0.0


class TestURLNormalization:
    """Test URL normalization."""

    def test_normalize_remove_trailing_slash(self):
        """Trailing slashes should be removed."""
        assert normalize_url("https://example.com/path/") == "https://example.com/path"

    def test_normalize_remove_query(self):
        """Query parameters should be removed."""
        assert normalize_url("https://example.com/path?q=1") == "https://example.com/path"

    def test_normalize_remove_fragment(self):
        """Fragments should be removed."""
        assert normalize_url("https://example.com/path#section") == "https://example.com/path"

    def test_normalize_whitespace(self):
        """Whitespace should be trimmed."""
        assert normalize_url("  https://example.com/path  ") == "https://example.com/path"

    def test_normalize_empty_string(self):
        """Empty string should return empty."""
        assert normalize_url("") == ""


class TestRetrievalMetricsEdgeCases:
    """Test edge cases for retrieval metrics."""

    def test_empty_retrieval_list(self):
        """Metrics should handle empty retrieval."""
        rel_scores = []
        assert hit_at_k(rel_scores, k=1) == 0.0
        assert mrr_at_k(rel_scores, k=1) == 0.0
        assert precision_at_k(rel_scores, k=1) == 0.0

    def test_single_result(self):
        """Metrics should work with single result."""
        rel_scores = [1.0]
        assert hit_at_k(rel_scores, k=1) == 1.0
        assert mrr_at_k(rel_scores, k=1) == 1.0
        assert precision_at_k(rel_scores, k=1) == 1.0

    def test_large_k_beyond_results(self):
        """Metrics should handle k > number of results."""
        rel_scores = [1.0, 0.0, 0.0]
        assert hit_at_k(rel_scores, k=100) == 1.0
        assert mrr_at_k(rel_scores, k=100) == 1.0

    def test_fractional_relevance_scores(self):
        """Metrics should handle fractional [0, 1] relevance scores."""
        rel_scores = [0.7, 0.3, 0.9, 0.1]
        # These should not crash
        hit = hit_at_k(rel_scores, k=4, threshold=0.5)
        mrr = mrr_at_k(rel_scores, k=4, threshold=0.5)
        ndcg = ndcg_at_k(rel_scores, k=4)
        
        assert 0.0 <= hit <= 1.0
        assert 0.0 <= mrr <= 1.0
        assert 0.0 <= ndcg <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
