# Evaluation Test Suite

This folder contains tests for retrieval metrics, text-generation metrics, BERTScore helpers, statistical helpers, and legacy CSV/metrics utilities.

## Test Files

| File | Purpose | Notes |
|---|---|---|
| `test_retrieval_metrics.py` | Tests retrieval metrics from `benchmark.py`: Hit@k, MRR@k, nDCG@k, MAP/AP@k, Precision@k, Recall@k, hierarchical URL relevance, and URL normalization. | Active. |
| `test_text_metrics.py` | Tests BLEU, ROUGE variants, METEOR, and text metric edge cases from `text_metrics.py`. | Active; requires text metric dependencies. |
| `test_bertscore.py` | Tests BERTScore variants from `text_metrics.py`. | Skipped automatically if BERTScore is unavailable; may download models on first run. |
| `test_statistical_comparison.py` | Primary pytest suite for `_rank_with_ties`, Wilcoxon, gamma, kappa, and paired permutation helpers. | Active. |
| `statistical_comparison_test.py` | Older unittest-style statistical tests. | Legacy but still present. |
| `evaluation_test.py` | Legacy tests for `prepare_data.py` and `metrics.py`. | `metrics.py` currently has placeholder `bert_score()`, so this file may fail unless that legacy class is restored. |
| `conftest.py` | Pytest fixtures/configuration. | Active. |

## Running Tests

Run from the repository root:

```bash
export PYTHONPATH=src

pytest src/evaluation/tests/test_retrieval_metrics.py -v
pytest src/evaluation/tests/test_text_metrics.py -v
pytest src/evaluation/tests/test_statistical_comparison.py -v
```

To run BERTScore tests:

```bash
pytest src/evaluation/tests/test_bertscore.py -v
```

To avoid optional or legacy failures while checking the active core:

```bash
pytest \
  src/evaluation/tests/test_retrieval_metrics.py \
  src/evaluation/tests/test_text_metrics.py \
  src/evaluation/tests/test_statistical_comparison.py \
  -v
```

Run the full folder only when you are ready to handle optional dependencies and legacy expectations:

```bash
pytest src/evaluation/tests/ -v
```

## Current Test Coverage by Area

### Retrieval Metrics

`test_retrieval_metrics.py` covers:

- `hit_at_k`
- `mrr_at_k`
- `ndcg_at_k`
- `average_precision_at_k`
- `precision_at_k`
- `recall_at_k`
- `hierarchical_relevance`
- `normalize_url`
- empty inputs, single-result inputs, large `k`, and fractional relevance scores

### Text Metrics

`test_text_metrics.py` covers:

- BLEU exact, mismatch, brevity penalty, n-gram components, and range checks
- ROUGE exact/partial/no overlap, multiple references, Polish text, empty hypothesis, and all variants
- METEOR exact, synonyms, partial/no match, empty hypothesis, and range checks
- punctuation, numbers, case, Unicode, long text, and single-word edge cases

### BERTScore

`test_bertscore.py` covers:

- exact, partial, and no-overlap cases
- empty hypothesis/reference handling
- base, IDF, rescaled, and full variants
- metric ranges, precision/recall asymmetry, multiple references, Polish text, long text, single words, semantic similarity, case, punctuation, numbers, and special characters

These tests are skipped if the BERTScore dependency is not installed.

### Statistical Helpers

`test_statistical_comparison.py` covers (32 tests total):

- `_rank_with_ties` (5 tests: no ties, single tie, multiple ties, all equal, negative values)
- `wilcoxon_signed_rank_test` (8 tests: perfect agreement, single non-zero, small n, ties, normal approximation, two-tailed, all positive, all negative)
- `goodman_kruskal_gamma_test` (6 tests: perfect concordance, perfect discordance, no association, thesis formula, rank infinities, weak association)
- `kappa_coefficient` (5 tests: perfect agreement, perfect disagreement, contingency table, empty universe, mixed scenarios)
- `paired_permutation_test` (6 tests: perfect match, clear difference, distribution, reproducibility, alternatives, edge cases)
- integration consistency checks (2 tests)

## Troubleshooting

### BERTScore is skipped

Install optional dependencies:

```bash
pip install bert-score torch transformers
```

The first run may download model weights.

### Import errors

Make sure you are in the repository root and `PYTHONPATH` includes `src`:

```bash
cd /Users/wiktoriagrodzka/moja-strona/chatbot\ mini\ v2/Chatbot-MiNI
export PYTHONPATH=src
pytest src/evaluation/tests/test_retrieval_metrics.py -v
```

### Legacy `evaluation_test.py` fails

That file expects `metrics.py::Metrics.bert_score()` to return tensors. The current `metrics.py` class is a placeholder with `pass`, so this is a known legacy mismatch. Prefer the newer `text_metrics.py` tests for active BERTScore/text-metric behavior.

