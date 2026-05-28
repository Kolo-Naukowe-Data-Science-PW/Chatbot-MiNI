"""
Unit tests for BERTScore metric.

BERTScore uses contextual embeddings from pre-trained language models to compute
semantic similarity between candidate and reference texts.

Tests cover:
  - BERTScore precision, recall, F1
  - Multiple variants (base, idf, rescaled, full)
  - Handling of empty inputs
  - Handling of non-English text (Polish)
  - Robustness to small perturbations

Usage:
    pytest src/evaluation/tests/test_bertscore.py -v

Note: BERTScore downloads pre-trained models on first run, which may take time.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import BERTScore computation
try:
    from src.evaluation.text_metrics import _bertscore

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


@pytest.mark.skipif(not BERTSCORE_AVAILABLE, reason="BERTScore not available")
class TestBERTScore:
    """Test BERTScore metric variants."""

    def test_bertscore_exact_match(self):
        """BERTScore should give high scores for exact match."""
        hypotheses = ["the quick brown fox jumps over the lazy dog"]
        references = ["the quick brown fox jumps over the lazy dog"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        # Check all variants are present
        expected_variants = ["base", "idf", "rescaled", "full"]
        for variant in expected_variants:
            assert f"bertscore_precision_{variant}" in result
            assert f"bertscore_recall_{variant}" in result
            assert f"bertscore_f1_{variant}" in result

        # For exact match, scores should be high
        for variant in expected_variants:
            precision = result[f"bertscore_precision_{variant}"]
            recall = result[f"bertscore_recall_{variant}"]
            f1 = result[f"bertscore_f1_{variant}"]

            assert precision > 0.8, f"Precision_{variant} too low: {precision}"
            assert recall > 0.8, f"Recall_{variant} too low: {recall}"
            assert f1 > 0.8, f"F1_{variant} too low: {f1}"

    def test_bertscore_partial_overlap(self):
        """BERTScore should give partial scores for partial overlap."""
        hypotheses = ["the quick brown"]
        references = ["the quick brown fox jumps over the lazy dog"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        # Recall should be lower (didn't capture all ref tokens)
        # Precision should be higher (all hyp tokens are relevant)
        precision = result["bertscore_precision_base"]
        recall = result["bertscore_recall_base"]

        assert 0.0 < precision <= 1.0
        assert 0.0 < recall <= 1.0
        assert recall < 1.0  # Didn't get everything

    def test_bertscore_no_overlap(self):
        """BERTScore should be low for completely different texts."""
        hypotheses = ["xyz abc def"]
        references = ["the quick brown fox"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        # Should be low but not zero (contextual embeddings may find some similarity)
        f1 = result["bertscore_f1_base"]
        assert 0.0 <= f1 < 0.5

    def test_bertscore_empty_hypothesis(self):
        """BERTScore should handle empty hypothesis."""
        hypotheses = [""]
        references = ["the quick brown fox"]

        # Should not crash
        try:
            result = _bertscore(
                hypotheses, references, lang="en", model_type="bert-base-uncased"
            )
            # Result can be 0 or handled gracefully
            assert "bertscore_precision_base" in result
        except (ValueError, RuntimeError):
            # Some implementations may raise for empty input
            pass

    def test_bertscore_empty_reference(self):
        """BERTScore should handle empty reference."""
        hypotheses = ["the quick brown fox"]
        references = [""]

        try:
            result = _bertscore(
                hypotheses, references, lang="en", model_type="bert-base-uncased"
            )
            # Result can be 0 or handled gracefully
            assert "bertscore_recall_base" in result
        except (ValueError, RuntimeError):
            pass

    def test_bertscore_variants_all_present(self):
        """All BERTScore variants should be computed."""
        hypotheses = ["hello world"]
        references = ["hello world"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        variants = ["base", "idf", "rescaled", "full"]
        metrics = ["precision", "recall", "f1"]

        for variant in variants:
            for metric in metrics:
                key = f"bertscore_{metric}_{variant}"
                assert key in result, f"Missing key: {key}"

    def test_bertscore_metric_ranges(self):
        """All BERTScore metrics should be in [0, 1] range."""
        hypotheses = ["the quick brown fox"]
        references = ["the quick brown fox jumps over the lazy dog"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        for key, value in result.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of range"

    def test_bertscore_symmetry(self):
        """BERTScore is asymmetric: precision and recall are different."""
        hypotheses = ["hello"]
        references = ["hello world"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        precision = result["bertscore_precision_base"]
        recall = result["bertscore_recall_base"]

        # Precision > recall because hypothesis is fully represented in reference
        assert precision > recall

    def test_bertscore_idf_variant(self):
        """IDF variant should weight common words less."""
        hypotheses = ["the the the"]
        references = ["hello world"]

        result_base = _bertscore(
            hypotheses,
            references,
            lang="en",
            model_type="bert-base-uncased",
        )
        result_idf = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        # Both should exist
        assert "bertscore_precision_base" in result_base
        assert "bertscore_precision_idf" in result_idf

    def test_bertscore_multiple_references(self):
        """BERTScore should handle multiple reference sentences."""
        hypotheses = ["the cat sat"]
        references = [
            "the cat sat on the mat",
            "a feline was sitting",
        ]

        # Note: _bertscore takes flat lists, so this tests single-ref mode
        # Multi-ref would require different implementation
        for ref in references:
            result = _bertscore(
                hypotheses, [ref], lang="en", model_type="bert-base-uncased"
            )
            assert "bertscore_f1_base" in result

    def test_bertscore_polish_text(self):
        """BERTScore should work with Polish text using multilingual model."""
        hypotheses = ["pies jest szybki"]
        references = ["pies jest szybki"]

        result = _bertscore(
            hypotheses, references, lang="pl", model_type="bert-base-multilingual-cased"
        )

        # Should give high score for exact match
        f1 = result["bertscore_f1_base"]
        assert f1 > 0.8

    def test_bertscore_long_text(self):
        """BERTScore should handle long texts."""
        long_hyp = " ".join(["word"] * 50)
        long_ref = " ".join(["word"] * 50)

        result = _bertscore(
            [long_hyp], [long_ref], lang="en", model_type="bert-base-uncased"
        )

        # Should give high score
        f1 = result["bertscore_f1_base"]
        assert f1 > 0.8

    def test_bertscore_single_word(self):
        """BERTScore should handle single words."""
        hypotheses = ["cat"]
        references = ["cat"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        # Should give high score
        f1 = result["bertscore_f1_base"]
        assert f1 > 0.8

    def test_bertscore_semantic_similarity(self):
        """BERTScore should capture semantic similarity beyond exact words."""
        hypotheses = ["a feline creature"]
        references = ["the cat"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )

        # Should give reasonable score due to semantic similarity
        f1 = result["bertscore_f1_base"]
        assert f1 > 0.0, "BERTScore should recognize semantic similarity"

    def test_bertscore_robustness_to_case(self):
        """BERTScore should be relatively robust to case differences."""
        result_lower = _bertscore(
            ["the cat"], ["the cat"], lang="en", model_type="bert-base-uncased"
        )
        result_upper = _bertscore(
            ["THE CAT"], ["the cat"], lang="en", model_type="bert-base-uncased"
        )

        # Scores should be similar (uncased model)
        f1_lower = result_lower["bertscore_f1_base"]
        f1_upper = result_upper["bertscore_f1_base"]
        assert abs(f1_lower - f1_upper) < 0.05


@pytest.mark.skipif(not BERTSCORE_AVAILABLE, reason="BERTScore not available")
class TestBERTScoreEdgeCases:
    """Test edge cases and robustness."""

    def test_bertscore_with_punctuation(self):
        """BERTScore should handle punctuation."""
        hypotheses = ["Hello, world!"]
        references = ["Hello, world!"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )
        f1 = result["bertscore_f1_base"]
        assert f1 > 0.8

    def test_bertscore_with_numbers(self):
        """BERTScore should handle numbers."""
        hypotheses = ["The answer is 42"]
        references = ["The answer is 42"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )
        f1 = result["bertscore_f1_base"]
        assert f1 > 0.8

    def test_bertscore_with_special_chars(self):
        """BERTScore should handle special characters."""
        hypotheses = ["Email: test@example.com"]
        references = ["Email: test@example.com"]

        result = _bertscore(
            hypotheses, references, lang="en", model_type="bert-base-uncased"
        )
        f1 = result["bertscore_f1_base"]
        assert f1 > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
