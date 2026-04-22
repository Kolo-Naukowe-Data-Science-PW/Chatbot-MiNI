# Evaluation

All evaluation scripts, metrics, and benchmark data for MiNIonek.

## Contents

| File / Folder | Purpose |
|---|---|
| `benchmark.py` | Main benchmark: runs retrieval on the gold eval set, computes Hit@k, MRR@k, nDCG@k, MAP@k, Precision-Recall curves. **Use this.** |
| `benchmark_v1.py` | Legacy simple benchmark (basic Hit@k / MRR, no hierarchical scoring). |
| `metrics.py` | BERTScore class for text-level answer quality evaluation. |
| `prepare_data.py` | CSV loading utilities used by the BERTScore pipeline. |
| `eval_with_playwright.py` | Playwright macro: automates asking questions through the live chatbot UI and collecting answers to a TSV. Requires a running deployment. |
| `llm_judge/judge.py` | LLM-as-a-judge: rates two chatbot answers on usefulness / accuracy / conciseness (1–5 scales) and picks the better variant. |
| `llm_judge/testpro_runner.py` | Batch runner: reads eval CSV, calls `/chat` twice per question with random model configs, runs judge, writes results to CSV. |
| `data/` | Evaluation datasets (see below). |

## Evaluation Data (`data/`)

| File | Description |
|---|---|
| `questions.csv` | Raw questions from student survey (3 columns of free-text answers) |
| `questions_cat.csv` | Questions with topic category labels |
| `questions_filtered.csv` | Filtered eval set: `query` + gold `strona` URL — **used by benchmark.py** |
| `questions_with_links.csv` | Full eval set with source links — used by testpro_runner |
| `test.csv` | Small fixture for unit tests |

## Running the main benchmark

The benchmark requires a running Qdrant instance with ingested data. Start the API first, or just run the retrieval module directly.

```bash
# From repo root
export PYTHONPATH=src
python -m evaluation.benchmark
```

Outputs:
- Metrics table in stdout (Hit@k, MRR@k, nDCG@k, MAP@k, R-Precision for k = 3, 5, 7, 10)
- `pr_curves.png` — mean interpolated Precision-Recall curves for each k
- `metrics_summary.png` — grouped bar chart of all metrics

## Running LLM-as-a-judge batch evaluation

Requires the `/chat` API to be running locally.

```bash
export PYTHONPATH=src
python -m evaluation.llm_judge.testpro_runner \
  --judge-model openai/gpt-4o \
  --input-csv src/evaluation/data/questions_with_links.csv \
  --output-csv src/data/feedback/llm_judge_results.csv \
  --limit 50
```

## Retrieval metrics explained

All metrics use **hierarchical URL relevance** — a retrieved URL that is a parent or child of the gold URL gets partial credit (not just exact match):

| Relationship | Score |
|---|---|
| Exact match | 1.00 |
| Parent URL (1 level up) | 0.50 |
| Grandparent (2 levels up) | 0.25 |
| Child URL (1 level down) | 0.25 |
| Grandchild (2 levels down) | 0.125 |
| Unrelated | 0.00 |

`RELEVANCE_THRESHOLD = 0.5` (configurable) determines what counts as a "hit" in binary metrics.
