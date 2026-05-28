# Statistical Tests

`statistical_comparison.py` contains statistical helpers for comparing two RAG/evaluation variants on the same question set.

There are two layers:

1. Python functions for Wilcoxon, Goodman-Kruskal gamma, Kappa, and paired permutation tests.
2. A CLI that currently compares two `eval_per_query_<ts>.csv` files for one metric at a time.

## Implemented Functions

| Function | Status | Purpose |
|---|---|---|
| `wilcoxon_signed_rank_test(...)` | implemented | Paired non-parametric test over per-query score differences. |
| `goodman_kruskal_gamma_test(...)` | implemented | Ordinal association between two rank arrays. |
| `kappa_coefficient(...)` | implemented | Agreement between two lists of retrieved document sets. |
| `paired_permutation_test(...)` | implemented | Sign-flip paired permutation test over two score arrays. |
| `compare_models(...)` | partially integrated | Loads metric columns from two CSVs and runs the selected CLI test type. |

Important limitation: `kappa_coefficient(...)` exists, but `compare_models(..., test_type="kappa")` is not fully wired into CSV parsing yet.

## CLI Usage

```bash
export PYTHONPATH=src

python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/results/model_a/eval_per_query_20260515T100000.csv \
  --model_b_csv src/evaluation/results/model_b/eval_per_query_20260515T110000.csv \
  --metric mrr@10 \
  --test-type wilcoxon \
  --alpha 0.05 \
  --output_dir src/evaluation/data
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--model_a_csv` | required | Path to the first `eval_per_query_<ts>.csv`. |
| `--model_b_csv` | required | Path to the second `eval_per_query_<ts>.csv`. |
| `--metric` | `mrr@10` | One metric column to compare. The CLI does not currently accept a comma-separated list. |
| `--test-type` | `wilcoxon` | One of `wilcoxon`, `gamma`, `kappa`. |
| `--alpha` | `0.05` | Significance threshold for Wilcoxon. |
| `--use-adaptive-k` | off | Converts a fixed-k metric name such as `mrr@10` to `mrr_adaptive`. The adaptive column must already exist in both CSVs. |
| `--output_dir` | `src/evaluation/data` | Output directory. The actual flag uses an underscore, not `--output-dir`. |

Output:

- terminal summary,
- `statistical_comparison_<ts>.json`.

## Fixed-k Example

```bash
python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/data/model_a/eval_per_query_20260515T100000.csv \
  --model_b_csv src/evaluation/data/model_b/eval_per_query_20260515T110000.csv \
  --metric mrr@10 \
  --test-type wilcoxon \
  --alpha 0.05 \
  --output_dir src/evaluation/data
```

To compare another metric, run the command again with a different `--metric`, for example `hit@10`, `ndcg@10`, `map@10`, or `mrrw@10`.

## Adaptive-k Example

`benchmark.py` can write adaptive metric columns when `--metric-mode adaptive` or `--metric-mode both` is used. The default is `both`, so current benchmark CSVs normally include adaptive columns.

```bash
python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/data/model_a/eval_per_query_20260515T100000.csv \
  --model_b_csv src/evaluation/data/model_b/eval_per_query_20260515T110000.csv \
  --metric mrr@10 \
  --use-adaptive-k \
  --output_dir src/evaluation/data
```

With `--use-adaptive-k`, the CLI changes `mrr@10` to `mrr_adaptive`. It does not recompute adaptive metrics from `chatbot_links`.

You can also pass the adaptive column directly:

```bash
python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/data/model_a/eval_per_query_20260515T100000.csv \
  --model_b_csv src/evaluation/data/model_b/eval_per_query_20260515T110000.csv \
  --metric hit_adaptive \
  --output_dir src/evaluation/data
```

## Test Types

### Wilcoxon Signed-Rank Test

Used for paired metric differences:

`d_q = score_A(q) - score_B(q)`

The implementation removes zero differences, ranks absolute differences with tie averaging, computes `W+`, `W-`, and `W = min(W+, W-)`, then returns a p-value.

### Goodman-Kruskal Gamma

Measures ordinal agreement between two rank arrays. In the CLI, ranks are extracted from CSV link columns, currently using `chatbot_links`.

Formula:

`gamma = (C - D) / (C + D)`

where `C` is the number of concordant query pairs and `D` is the number of discordant pairs.

**Note:** Metric-independent; operates on ranks only.

### Kappa

The standalone `kappa_coefficient(...)` function is fully implemented for Python usage, but the CLI path is incomplete: CSV retrieval-set parsing is not fully connected in `compare_models`. To use kappa from Python:

```python
kappa = kappa_coefficient(
    [set(["url1", "url2"]), set(["url3"])],
    [set(["url1", "url4"]), set(["url3"])],
)
```

**Note:** Metric-independent; operates on document sets only.

### Paired Permutation

`paired_permutation_test(...)` exists as a full Python implementation but is **not exposed by the CLI**. Use directly in Python:

```python
result = paired_permutation_test(scores_a, scores_b, n_permutations=10000, seed=67)
```

## Python Usage

```python
import numpy as np
from src.evaluation.statistical_comparison import (
    wilcoxon_signed_rank_test,
    goodman_kruskal_gamma_test,
    kappa_coefficient,
    paired_permutation_test,
)

scores_a = np.array([0.8, 0.7, 0.9])
scores_b = np.array([0.7, 0.8, 0.85])

wilcoxon = wilcoxon_signed_rank_test(scores_a - scores_b)
permutation = paired_permutation_test(scores_a, scores_b, n_permutations=1000)

ranks_a = np.array([1, 2, 3])
ranks_b = np.array([1, 3, 2])
gamma = goodman_kruskal_gamma_test(ranks_a, ranks_b)

kappa = kappa_coefficient(
    [{"url1", "url2"}, {"url3"}],
    [{"url1", "url4"}, {"url3"}],
)
```
