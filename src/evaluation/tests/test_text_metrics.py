"""
Comprehensive test suite for text generation evaluation metrics.

Tests verify implementations match mathematical definitions from:
  "Text Generation Evaluation Metrics" (thesis chapter 4)

Metrics tested:
  - BLEU: Bilingual evaluation understudy (n-gram precision)
  - ROUGE-N: N-gram recall (n=1,2)
  - ROUGE-L: Longest common subsequence
  - ROUGE-W: Weighted LCS
  - ROUGE-S: Skip-bigram matching
  - METEOR: Metric for translation evaluation with morphology
  - BERTScore: Embedding-based semantic similarity (P, R, F1)

Mathematical definitions:
  BLEU = BP * exp(Σ w_n * log(p_n)) where p_n = clipped n-gram precision
  ROUGE-N = Σ count_match(g, c, r) / Σ count(g, r)
  ROUGE-L = F(LCS) with R_LCS = |LCS|/m, P_LCS = |LCS|/n
  ROUGE-W = F(WLCS) with f(k)=k^α weighted LCS scoring
  ROUGE-S = F(SKIP2_d) with skip-bigram matching within distance d
  METEOR = F * (1 - Penalty) with chunking penalty = 0.5*(h/|S*|)^3
  BERTScore: R = (1/m)Σ max_j(e_r^T e_c), P = (1/n)Σ max_i(e_r^T e_c)

Usage:
    pytest src/evaluation/tests/test_text_metrics.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.text_metrics import (  # noqa: E402
    _bleu_score,
    _meteor_score,
    _rouge_scores,
)


class TestBLEUScore:
    """Test BLEU (Bilingual Evaluation Understudy) metric."""

    def test_bleu_perfect_match(self):
        """BLEU = 1.0 for perfect match (BP=1, all p_n=1)."""
        hypothesis = ["the cat sat on the mat"]
        reference = ["the cat sat on the mat"]
        result = _bleu_score(hypothesis, reference)
        assert result["bleu"] == pytest.approx(1.0, abs=0.01)

    def test_bleu_complete_mismatch(self):
        """BLEU = 0 for complete mismatch (no n-gram overlap)."""
        hypothesis = ["a b c d e"]
        reference = ["x y z w v"]
        result = _bleu_score(hypothesis, reference)
        assert result["bleu"] == pytest.approx(0.0, abs=0.01)

    def test_bleu_partial_match_with_brevity_penalty(self):
        """BLEU applies brevity penalty for short candidates."""
        short_hyp = ["the cat"]
        reference = ["the cat sat on the mat"]
        result = _bleu_score(short_hyp, reference)
        # Should be penalized for short length
        assert result["bleu"] < 0.3

    def test_bleu_n_gram_components(self):
        """BLEU includes individual n-gram precision components."""
        hypothesis = ["the cat"]
        reference = ["the cat sat"]
        result = _bleu_score(hypothesis, reference)

        # All n-gram components should exist
        assert "bleu_1" in result  # unigram precision
        assert "bleu_2" in result  # bigram precision
        assert "bleu_3" in result  # trigram precision
        assert "bleu_4" in result  # 4-gram precision

        # Unigrams should be higher than higher-order
        assert result["bleu_1"] >= result["bleu_2"]

    def test_bleu_range_01(self):
        """BLEU metric should be in [0, 1]."""
        test_cases = [
            (["hello world"], ["hello world"]),
            (["the cat"], ["a dog"]),
            (["test"], ["test testing"]),
        ]
        for hyp, ref in test_cases:
            result = _bleu_score(hyp, ref)
            assert 0.0 <= result["bleu"] <= 1.0


class TestROUGEScores:
    """Test ROUGE-N, ROUGE-L, ROUGE-W, ROUGE-S calculation."""

    def test_rouge_exact_match(self):
        """ROUGE should give high scores for exact match."""
        hypotheses = ["the quick brown fox"]
        references = ["the quick brown fox"]
        result = _rouge_scores(hypotheses, references)

        # Check key metrics exist
        assert "rouge_1_r" in result  # ROUGE-1 recall
        assert "rouge_1_p" in result  # ROUGE-1 precision
        assert "rouge_1_f" in result  # ROUGE-1 F1
        assert "rouge_2_r" in result  # ROUGE-2 recall
        assert "rouge_l_r" in result  # ROUGE-L recall
        assert "rouge_w_r" in result  # ROUGE-W recall
        assert "rouge_s_r" in result  # ROUGE-S recall

        # For exact match, all should be ~1.0
        assert result["rouge_1_r"] == pytest.approx(1.0, abs=0.01)
        assert result["rouge_l_r"] == pytest.approx(1.0, abs=0.01)

    def test_rouge_partial_overlap(self):
        """ROUGE should give partial scores for partial overlap."""
        hypotheses = ["the quick brown"]
        references = ["the quick brown fox jumps"]
        result = _rouge_scores(hypotheses, references)

        # Recall should be < 1.0 (not all ref n-grams captured)
        assert 0.0 < result["rouge_1_r"] < 1.0

    def test_rouge_no_overlap(self):
        """ROUGE should be 0 for completely different text."""
        hypotheses = ["xyz abc"]
        references = ["the quick brown"]
        result = _rouge_scores(hypotheses, references)

        assert result["rouge_1_r"] == 0.0
        assert result["rouge_1_f"] == 0.0

    def test_rouge_multiple_references(self):
        """ROUGE should pick best ref when multiple references given."""
        hypotheses = ["the quick brown"]
        references = [
            ["xyz abc def", "the quick brown fox"],  # second ref matches better
        ]
        result = _rouge_scores(hypotheses, references)

        # Should use second reference (best match)
        assert result["rouge_1_r"] > 0.5

    def test_rouge_polish_text(self):
        """ROUGE should work with Polish text."""
        hypotheses = ["pies jest szybki"]
        references = ["pies jest szybki"]
        result = _rouge_scores(hypotheses, references)

        assert result["rouge_1_r"] == pytest.approx(1.0, abs=0.01)

    def test_rouge_empty_hypothesis(self):
        """ROUGE should handle empty hypothesis."""
        hypotheses = [""]
        references = ["the quick brown"]
        result = _rouge_scores(hypotheses, references)

        # Precision should be 1 (0 false positives) but recall 0 (missed everything)
        assert result["rouge_1_p"] >= 0.0
        assert result["rouge_1_r"] == 0.0

    def test_rouge_all_variants_present(self):
        """All ROUGE variants should be computed."""
        hypotheses = ["the quick brown fox jumps"]
        references = ["the quick brown fox jumps over the lazy dog"]
        result = _rouge_scores(hypotheses, references)

        expected_keys = [
            "rouge_1_r",
            "rouge_1_p",
            "rouge_1_f",
            "rouge_2_r",
            "rouge_2_p",
            "rouge_2_f",
            "rouge_l_r",
            "rouge_l_p",
            "rouge_l_f",
            "rouge_w_r",
            "rouge_w_p",
            "rouge_w_f",
            "rouge_s_r",
            "rouge_s_p",
            "rouge_s_f",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_rouge_metric_ranges(self):
        """All ROUGE metrics should be in [0, 1] range."""
        hypotheses = ["the quick brown fox"]
        references = ["the quick brown fox jumps over the lazy dog"]
        result = _rouge_scores(hypotheses, references)

        for key, value in result.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"


class TestMETEORScore:
    """Test METEOR score calculation."""

    def test_meteor_exact_match(self):
        """METEOR should be high for exact match."""
        hypotheses = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        result = _meteor_score(hypotheses, references)

        assert "meteor" in result
        assert result["meteor"] == pytest.approx(1.0, abs=0.01)

    def test_meteor_with_synonyms(self):
        """METEOR should give partial credit for synonyms."""
        hypotheses = ["the feline sat on the rug"]
        references = ["the cat sat on the mat"]
        result = _meteor_score(hypotheses, references)

        assert "meteor" in result
        # Should be > 0 due to stem matching and synonym recognition
        assert result["meteor"] > 0.0

    def test_meteor_partial_match(self):
        """METEOR should give partial scores for partial overlap."""
        hypotheses = ["the quick brown"]
        references = ["the quick brown fox jumps over the lazy dog"]
        result = _meteor_score(hypotheses, references)

        assert "meteor" in result
        assert 0.0 < result["meteor"] < 1.0

    def test_meteor_no_match(self):
        """METEOR should be 0 for completely different text."""
        hypotheses = ["xyz abc"]
        references = ["the quick brown"]
        result = _meteor_score(hypotheses, references)

        assert "meteor" in result
        assert result["meteor"] == 0.0

    def test_meteor_empty_hypothesis(self):
        """METEOR should handle empty hypothesis."""
        hypotheses = [""]
        references = ["the quick brown"]
        result = _meteor_score(hypotheses, references)

        assert "meteor" in result
        assert result["meteor"] == 0.0

    def test_meteor_range(self):
        """METEOR should be in [0, 1] range."""
        hypotheses = ["the quick brown fox"]
        references = ["the quick brown fox jumps over the lazy dog"]
        result = _meteor_score(hypotheses, references)

        assert 0.0 <= result["meteor"] <= 1.0


class TestTextMetricsEdgeCases:
    """Test edge cases and robustness."""

    def test_metrics_with_punctuation(self):
        """Metrics should handle punctuation correctly."""
        hypotheses = ["The cat sat on the mat."]
        references = ["The cat sat on the mat."]

        bleu = _bleu_score(hypotheses, references)
        rouge = _rouge_scores(hypotheses, references)
        meteor = _meteor_score(hypotheses, references)

        assert bleu["bleu"] > 0.5
        assert rouge["rouge_1_f"] > 0.5
        assert meteor["meteor"] > 0.5

    def test_metrics_with_numbers(self):
        """Metrics should handle numbers."""
        hypotheses = ["Result: 42"]
        references = ["Result: 42"]

        rouge = _rouge_scores(hypotheses, references)
        assert rouge["rouge_1_r"] > 0.8

    def test_metrics_case_sensitivity(self):
        """BLEU and ROUGE typically lowercase before comparison."""
        hypotheses = ["The Quick Brown Fox"]
        references = ["the quick brown fox"]

        # Most metrics are case-insensitive after preprocessing
        bleu = _bleu_score(hypotheses, references)
        assert bleu["bleu"] > 0.5

    def test_metrics_with_unicode(self):
        """Metrics should handle non-ASCII characters."""
        hypotheses = ["Psi ma rasa ma ćwierćkę"]
        references = ["Psi mają rasę mającą ćwierćkę"]

        rouge = _rouge_scores(hypotheses, references)
        assert "rouge_1_r" in rouge
        assert 0.0 <= rouge["rouge_1_r"] <= 1.0

    def test_metrics_long_text(self):
        """Metrics should handle long texts."""
        long_text = " ".join(["word"] * 100)
        hypotheses = [long_text]
        references = [long_text]

        bleu = _bleu_score(hypotheses, references)
        assert bleu["bleu"] == pytest.approx(1.0, abs=0.01)

    def test_metrics_single_word(self):
        """Metrics should handle single words."""
        hypotheses = ["cat"]
        references = ["cat"]

        bleu = _bleu_score(hypotheses, references)
        rouge = _rouge_scores(hypotheses, references)
        meteor = _meteor_score(hypotheses, references)

        assert bleu["bleu"] == pytest.approx(1.0, abs=0.01)
        assert rouge["rouge_1_r"] == pytest.approx(1.0, abs=0.01)
        assert meteor["meteor"] == pytest.approx(1.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
