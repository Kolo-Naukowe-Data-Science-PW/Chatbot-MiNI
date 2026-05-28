# Statistical Testing Reference

This document describes what is actually implemented in `statistical_comparison.py`.

The mathematical helpers are broader than the command-line interface. The Python functions include Wilcoxon, Goodman-Kruskal gamma, Kappa, and paired permutation tests. The CLI currently exposes Wilcoxon, gamma, and a partially wired kappa mode for one metric at a time.

## Function Reference

### `_rank_with_ties(values)`

Ranks values from `1..n` with average ranks for ties.

Example:

```python
_rank_with_ties(np.array([1.0, 1.0, 2.0]))
# array([1.5, 1.5, 3.0])
```

### `wilcoxon_signed_rank_test(differences, alpha=0.05, alternative="two-sided")`

Implements a paired Wilcoxon signed-rank test:

1. remove zero differences,
2. rank absolute differences with tie averaging,
3. compute `W_plus`, `W_minus`, and `W = min(W_plus, W_minus)`,
4. use SciPy exact Wilcoxon for `n < 25`,
5. use a normal approximation for `n >= 25`.

Returns a dict with fields such as `n`, `W_plus`, `W_minus`, `W`, `p_value`, `is_significant`, and `method`.

### `goodman_kruskal_gamma_test(ranks_A, ranks_B)`

Computes ordinal association:

`gamma = (C - D) / (C + D)`

where `C` is the number of concordant query pairs and `D` is the number of discordant query pairs.

Returns `n_queries`, `n_pairs`, `C`, `D`, `gamma`, and a text interpretation.

**Important:** This function operates on **rank arrays only** and is **independent of the chosen metric**. It measures agreement between two models on query difficulty ranking, regardless of which metric was used to compute the ranks. Works with any models or ranking schemes.

### `kappa_coefficient(retrieval_sets_i, retrieval_sets_j, relevant_sets=None)`

Computes Cohen-style agreement over retrieved document sets per query.

Returns `n_queries`, `kappa_mean`, `kappa_min`, `kappa_max`, and per-query `kappas`.

This function works as a Python helper. The CLI integration for `test-type=kappa` is currently incomplete.

**Important:** This function operates on **document sets only** and is **independent of the chosen metric**. It measures agreement between two retrieval models on which documents were returned, regardless of scoring metrics. Works with any systems that produce ranked lists.

### `paired_permutation_test(scores_A, scores_B, n_permutations=10000, seed=67, alternative="two-sided")`

Runs a paired sign-flip permutation test over two score arrays.

This function is available for Python usage but is not exposed by the CLI.

## CLI Reference

```bash
python -m evaluation.statistical_comparison \
  --model_a_csv path/to/model_a/eval_per_query_20260515T100000.csv \
  --model_b_csv path/to/model_b/eval_per_query_20260515T110000.csv \
  --metric mrr@10 \
  --test-type wilcoxon \
  --alpha 0.05 \
  --output_dir src/evaluation/data
```

Current CLI flags:

| Flag | Notes |
|---|---|
| `--model_a_csv` | Required. |
| `--model_b_csv` | Required. |
| `--metric` | One metric column only, default `mrr@10`. |
| `--test-type` | `wilcoxon`, `gamma`, or `kappa`. |
| `--alpha` | Used by Wilcoxon. |
| `--use-adaptive-k` | Rewrites `mrr@10` to `mrr_adaptive`, etc.; does not compute missing columns. |
| `--output_dir` | Output directory, default `src/evaluation/data`. |

The CLI writes `statistical_comparison_<ts>.json`.

## CSV Expectations

The CLI expects both inputs to be per-query benchmark CSVs with matching row order and the selected metric column present in both files.

For fixed-k comparison, use columns such as:

- `hit@10`
- `mrr@10`
- `mrrw@10`
- `recall@10`
- `precision@10`
- `f1@10`
- `ndcg@10`
- `map@10`
- `r_prec`

For adaptive-k comparison, use columns such as:

- `hit_adaptive`
- `mrr_adaptive`
- `mrrw_adaptive`
- `recall_adaptive`
- `precision_adaptive`
- `f1_adaptive`
- `ndcg_adaptive`
- `map_adaptive`
- `r_prec_adaptive`

## Python Examples

```python
import numpy as np
from src.evaluation.statistical_comparison import (
    wilcoxon_signed_rank_test,
    goodman_kruskal_gamma_test,
    paired_permutation_test,
)

scores_a = np.array([0.85, 0.78, 0.92, 0.70, 0.88])
scores_b = np.array([0.80, 0.82, 0.90, 0.75, 0.85])

diffs = scores_a - scores_b
wilcoxon = wilcoxon_signed_rank_test(diffs)
permutation = paired_permutation_test(scores_a, scores_b, n_permutations=5000)

ranks_a = np.array([1, 2, 1, 3, 1])
ranks_b = np.array([1, 1, 1, 2, 1])
gamma = goodman_kruskal_gamma_test(ranks_a, ranks_b)
```

## Tests

The current primary statistical test file is:

```bash
pytest src/evaluation/tests/test_statistical_comparison.py -v
```

There is also a legacy unittest-style file:

```bash
pytest src/evaluation/tests/statistical_comparison_test.py -v
```

Do not treat this document as a record that all tests passed in the current environment. Run pytest locally to verify the active environment and optional dependencies.

