# Implementation Summary

This file summarizes the current statistical-testing implementation in `statistical_comparison.py`.

## Current Status

Implemented as Python helpers:

- `_rank_with_ties`
- `wilcoxon_signed_rank_test`
- `goodman_kruskal_gamma_test`
- `kappa_coefficient`
- `paired_permutation_test`
- `compare_models`
- `format_result`

CLI support:

- `wilcoxon`: wired through `--test-type wilcoxon`.
- `gamma`: wired through `--test-type gamma`.
- `kappa`: accepted by the CLI, but not fully integrated in `compare_models`; the standalone function works for Python inputs.
- `paired_permutation_test`: Python helper only, not exposed by CLI.

## Important CLI Reality

The current CLI compares one metric at a time:

```bash
python -m evaluation.statistical_comparison \
  --model_a_csv path/to/model_a.csv \
  --model_b_csv path/to/model_b.csv \
  --metric mrr@10 \
  --test-type wilcoxon \
  --output_dir src/evaluation/data
```

It does not support `--metrics "mrr@10,hit@10"` and does not use `--output-dir`. The output flag is `--output_dir`.

`--use-adaptive-k` only maps metric names, for example `mrr@10` to `mrr_adaptive`. The adaptive metric column must already exist in both CSV files.

## Files

| File | Current role |
|---|---|
| `statistical_comparison.py` | Statistical functions plus CLI. |
| `tests/test_statistical_comparison.py` | Primary pytest coverage for statistical helpers. |
| `tests/statistical_comparison_test.py` | Older unittest-style coverage for some helpers. |
| `STATISTICAL_TESTS.md` | User-facing usage notes aligned with current CLI. |
| `STATISTICAL_TESTING.md` | Function-level reference aligned with current code. |
| `STATISTICAL_QUICK_REFERENCE.py` | Python quick-reference script/material, not Markdown. |

## Verification

Run from the repository root:

```bash
export PYTHONPATH=src
pytest src/evaluation/tests/test_statistical_comparison.py -v
```

This summary does not claim a fixed number of passing tests for every environment. Optional dependencies and local pytest configuration can affect results.
