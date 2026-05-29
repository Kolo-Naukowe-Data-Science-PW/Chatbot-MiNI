# Evaluation

Evaluation scripts, metrics, test data, and benchmark helpers for MiNIonek.

This folder contains two main evaluation paths:

1. retrieval/source evaluation: does the RAG pipeline return the right URLs?
2. answer-quality evaluation: does the chatbot produce useful, accurate answers?

## Contents

| File / Folder | Purpose |
|---|---|
| `benchmark.py` | Main retrieval benchmark. Computes fixed-k and adaptive-k metrics such as Hit, MRR, MRRw, Recall, Precision, F1, nDCG, MAP, R-Precision, and optional plots. |
| `run_experiment.py` | Ablation runner for named RAG pipeline variants from `pipeline_config.py`. Can evaluate `human`, `rag`, or `generated` testsets and optionally call an LLM judge. |
| `pipeline_config.py` | Dataclass and named pipeline variants: baseline, vector-only, no-rerank, two-step rerank, chunks, no-rewrite, and two-stage variants. |
| `retrieval_runner.py` | Retrieval implementation used by `run_experiment.py`; supports hybrid/dense retrieval, reranking, chunks vs facts, and two-stage retrieval. |
| `benchmark_v1.py` | Legacy simpler benchmark. |
| `benchmark_hier.py` | Older/alternate hierarchical benchmark utilities. |
| `text_metrics.py` | Text-generation metrics: BLEU, ROUGE-{1,2,L,W,S}, METEOR, and BERTScore variants. Compares generated answers with reference answers. |
| `statistical_comparison.py` | CLI for comparing two benchmark CSVs with one selected statistical test: Wilcoxon, Goodman-Kruskal gamma, or kappa mode. Kappa helper exists, but the CLI integration is currently incomplete. |
| `generate_golden_answers.py` | Generates `golden_answers.csv` with OpenRouter supermodels, without RAG context. |
| `prepare_data.py` | Legacy CSV loading helpers used by older tests/pipeline pieces. |
| `metrics.py` | Legacy placeholder class. `bert_score()` and `perplexity()` are currently not implemented here; use `text_metrics.py` for active text metrics. |
| `eval_with_playwright.py` | Browser macro for asking questions through the live chatbot UI and collecting answers. |
| `weighted_feedback.py` | Utilities for weighting feedback/evaluation rows. |
| `prepare_chunking_subset.py` | Helper for preparing subset data for chunking experiments. |
| `METRICS_SUMMARY.py` | Reference/summary material for metrics. |
| `STATISTICAL_QUICK_REFERENCE.py` | Quick-reference material for statistical tests. |
| `benchmark_checks/` | Focused checks for coverage, reranker, rewriter, BM25 prefilter, and URL aggregation. |
| `llm_judge/` | LLM-as-a-judge modules and batch runners. |
| `tests/` | Pytest tests for retrieval metrics, text metrics, BERTScore, and statistical comparison helpers. |
| `data/` | Evaluation datasets. |

## Evaluation Data (`data/`)

| File | Description |
|---|---|
| `questions.csv` | Raw questions from the student survey. |
| `questions_cat.csv` | Questions with topic category labels. |
| `questions_with_links.csv` | Full eval set with source links. `benchmark.py` automatically filters this file to `wymagany kontekst = 0`; `testpro_runner.py` can also read it. |
| `questions_filtered.csv` | Pre-filtered eval set with question and gold URL columns. |
| `QA_rag.csv` | RAG QA reference set used by text metrics and `run_experiment.py --testset rag`. |
| `final_notebooklm_QA.jsonl` | Generated QA set used by `run_experiment.py --testset generated`. |
| `golden_answers.csv` | Reference answers from GPT/Opus/Gemini generated without RAG context; used by `golden_judge_runner.py`. |
| `context_golden_answers.csv` | Optional reference answers generated with full scraped context; used by `context_golden_runner.py` if present. Format: `query, golden_answer[, gold_url, skip_reason]`. |
| `test.csv` | Small fixture for legacy unit tests. |

## Main Retrieval Benchmark

The main benchmark is `benchmark.py`. It can run in two modes:

- API mode: sends every query to a running `/chat` endpoint and evaluates the returned answer and sources.
- Direct retrieval mode: calls the local retrieval code directly, without generating answer text.

```bash
# From repo root
export PYTHONPATH=src

# Full chatbot pipeline through /chat
python -m evaluation.benchmark \
  --api-url http://localhost:8000/chat \
  --output-dir src/evaluation/results \
  --no-plots

# Direct retrieval mode
python -m evaluation.benchmark --no-plots
```

### Key Flags

| Flag | Default | Description |
|---|---|---|
| `--input-csv` | `src/evaluation/data/questions_with_links.csv` | Evaluation CSV. If the file name is `questions_with_links.csv`, only rows with `wymagany kontekst = 0` are used. |
| `--api-url` | none | Chatbot `/chat` endpoint. If omitted, direct retrieval mode is used. |
| `--output-dir` | `src/evaluation/results` | Directory for CSV, JSON, and plot outputs. |
| `--ks` | `3,5,7,10` | Rank cut-offs for fixed-k metrics. |
| `--timeout` | `60` | HTTP timeout for `/chat` calls. |
| `--no-plots` | off | Skip `pr_curves.png` and `metrics_summary.png`. |
| `--rewrite` | off | Rewrite each query before direct retrieval. Has no effect with `--api-url`, because `/chat` already handles rewriting internally. |
| `--metric-mode` | `both` | `standard`, `adaptive`, or `both`. Controls whether fixed-k metrics, adaptive-k metrics, or both are written. |
| `--check-coverage` | off | Adds `covered`, `coverage_rate`, and coverage-corrected `cch@k` metrics. |
| `--no-rerank` | off | Direct retrieval only: disables cross-encoder reranking and returns raw RRF results. |

### Benchmark Outputs

| File | Content |
|---|---|
| `eval_per_query_<ts>.csv` | One row per query with `query`, `retrieval_query`, `chatbot_answer`, `chatbot_links`, `gold_link`, and metric columns. |
| `eval_summary_<ts>.csv` | One-row mean summary of every metric column. |
| `eval_summary_<ts>.json` | Same summary as JSON. |
| `pr_curves.png` | Mean interpolated precision-recall curves, unless `--no-plots` is set. |
| `metrics_summary.png` | Grouped metric bar chart, unless `--no-plots` is set. |

Fixed-k columns include:

`hit@k`, `mrr@k`, `mrrw@k`, `recall@k`, `precision@k`, `f1@k`, `ndcg@k`, `map@k`, and `r_prec`.

Adaptive-k columns include:

`hit_adaptive`, `mrr_adaptive`, `mrrw_adaptive`, `recall_adaptive`, `precision_adaptive`, `f1_adaptive`, `ndcg_adaptive`, `map_adaptive`, and `r_prec_adaptive`.

Adaptive metrics are computed with `k = number of unique links actually returned for that query`.

## Retrieval Metrics

The main definitions live in `benchmark.py`.

All retrieval metrics use hierarchical URL relevance. A retrieved URL can receive partial credit if it is an ancestor or descendant of the gold URL:

| Relationship | Score |
|---|---:|
| Exact match | 1.00 |
| Parent URL, 1 level above | 0.50 |
| Grandparent, 2 levels above | 0.25 |
| Child URL, 1 level below | 0.25 |
| Grandchild, 2 levels below | 0.125 |
| Unrelated URL or different origin | 0.00 |

`RELEVANCE_THRESHOLD = 0.5` decides what counts as relevant for binary metrics such as Hit, MRR, Precision, Recall, F1, and MAP.

`MRRw` is a depth-aware weighted MRR. It uses:

- `alpha = 0.8` for URLs deeper/more specific than the gold URL,
- `beta = 0.4` for URLs shallower/more general than the gold URL.

## RAG Ablation Experiments

Use `run_experiment.py` when you want to compare pipeline variants defined in `pipeline_config.py`.

```bash
export PYTHONPATH=src

python -m evaluation.run_experiment \
  --variant baseline \
  --testset human \
  --output-dir src/data/experiments \
  --n-questions 50
```

Available testsets:

| Testset | Source |
|---|---|
| `human` | `data/questions_with_links.csv`, filtered to `wymagany kontekst = 0` |
| `rag` | `data/QA_rag.csv` |
| `generated` | `data/final_notebooklm_QA.jsonl` |

Available variants are in `pipeline_config.ALL_VARIANTS`:

`baseline`, `v1_vector_only`, `v2_no_rerank`, `v3_two_step_rerank`, `v4_chunks`, `v5_no_rewrite`, `v6_two_stage_facts`, `v7_two_stage_chunks`.

To run all variants:

```bash
python -m evaluation.run_experiment \
  --variant all \
  --testset human \
  --output-dir src/data/experiments
```

Optional LLM judging:

```bash
python -m evaluation.run_experiment \
  --variant baseline \
  --testset rag \
  --judge-model anthropic/claude-opus-4.7 \
  --output-dir src/data/experiments
```

## Text-Generation Metrics

Use `text_metrics.py` for active answer-text metrics.

```bash
export PYTHONPATH=src

python -m evaluation.text_metrics \
  --generated-csv path/to/generated_answers.csv \
  --reference-csv src/evaluation/data/QA_rag.csv \
  --output-dir src/evaluation/results \
  --lang pl \
  --bertscore-model allegro/herbert-base-cased
```

Expected columns:

| CSV | Required columns |
|---|---|
| generated CSV | `pytanie`, `odpowiedz` |
| reference CSV | `pytanie`, `odpowiedz` |

Outputs:

| File | Content |
|---|---|
| `text_metrics_per_query_<ts>.csv` | Per-question text metric details, currently including per-row ROUGE values. |
| `text_metrics_summary_<ts>.json` | Aggregate BLEU, ROUGE, METEOR, and BERTScore values. |
| `text_metrics_report_<ts>.md` | Markdown summary report. |

## Statistical Comparison

`statistical_comparison.py` compares two `eval_per_query_<ts>.csv` files for one selected metric at a time.

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

### Statistical CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--model_a_csv` | required | Path to model A per-query CSV. |
| `--model_b_csv` | required | Path to model B per-query CSV. |
| `--metric` | `mrr@10` | Single metric column to compare, e.g. `hit@10`, `mrr@10`, `ndcg@10`, `hit_adaptive`, `mrr_adaptive`. |
| `--test-type` | `wilcoxon` | One of `wilcoxon`, `gamma`, `kappa`. |
| `--alpha` | `0.05` | Significance level for Wilcoxon. |
| `--use-adaptive-k` | off | Converts fixed-k metric names like `mrr@10` to `mrr_adaptive`. The adaptive column must already exist in the CSV. |
| `--output_dir` | `src/evaluation/data` | Output directory. Note the underscore in the flag name. |

Important current limitations:

- The CLI accepts one metric via `--metric`, not a comma-separated `--metrics` list.
- `--use-adaptive-k` does not recompute adaptive metrics from old CSVs; it expects adaptive columns already written by `benchmark.py`.
- The kappa helper is implemented, but `test-type=kappa` is not fully wired into the CSV comparison flow yet.

Output:

`statistical_comparison_<ts>.json`, plus a human-readable terminal summary.

For mathematical background, see `STATISTICAL_TESTS.md`, `STATISTICAL_TESTING.md`, and `STATISTICAL_QUICK_REFERENCE.py`. Treat command examples in the Markdown notes as background material if they differ from the CLI above.

## LLM-as-a-Judge A/B Evaluation

`llm_judge/testpro_runner.py` compares two chatbot answers, A and B. It calls `/chat` twice per question with different `modelConfig` values, then uses `llm_judge/judge.py` to rate both answers.

```bash
export PYTHONPATH=src

python -m evaluation.llm_judge.testpro_runner \
  --judge-model openai/gpt-4o \
  --input-csv src/evaluation/data/questions_with_links.csv \
  --output-csv src/data/feedback/llm_judge_results.csv \
  --limit 50
```

`EXPERIMENT_DIM` controls what differs between A and B:

| Value | What changes | What stays fixed |
|---|---|---|
| `model` | two different models from `MODEL_POOL` | temperature, persona |
| `temperature` | low temperature vs high temperature | model, persona |
| `persona` | two different style instructions | model, temperature |

Baseline environment variables:

| Variable | Default | Used when |
|---|---|---|
| `EXPERIMENT_MODEL` | `openai/gpt-4o-mini` | `temperature`, `persona` |
| `EXPERIMENT_TEMP` | `0.2` | `model`, `persona` |
| `EXPERIMENT_PERSONA` | `0` | `model`, `temperature` |

## Golden Answers

`generate_golden_answers.py` generates reference answers with three OpenRouter models, without RAG context.

```bash
export PYTHONPATH=src
export OPENROUTER_API_KEY=...

python -m evaluation.generate_golden_answers --limit 10
python -m evaluation.generate_golden_answers --resume
python -m evaluation.generate_golden_answers --no-skip-personal
```

Output: `src/evaluation/data/golden_answers.csv`

| Column | Description |
|---|---|
| `query` | Evaluation question. |
| `gold_url` | Gold URL from the input CSV. |
| `golden_answer_gpt` | Answer from `openai/gpt-5.5`. |
| `golden_answer_opus` | Answer from `anthropic/claude-opus-4.7`. |
| `golden_answer_gemini` | Answer from `google/gemini-3.1-pro-preview-customtools`. |
| `skip_reason` | `session_context` for personal/schedule-dependent questions, otherwise empty. |

Questions with `skip_reason=session_context` are written with empty answer fields instead of being dropped.

## LLM Judge vs Golden Answers

`llm_judge/golden_judge_runner.py` compares the chatbot answer against one selected column from `golden_answers.csv`.

```bash
export PYTHONPATH=src

python -m evaluation.llm_judge.golden_judge_runner \
  --judge-model anthropic/claude-opus-4.7 \
  --golden-model opus \
  --limit 20

python -m evaluation.llm_judge.golden_judge_runner \
  --judge-model anthropic/claude-opus-4.7 \
  --resume
```

Output: `src/data/feedback/golden_judge_results.csv`

The judge prompt is intentionally asymmetric: the golden answer was generated without RAG, so honest uncertainty is not penalized. The chatbot answer is judged on whether it used its retrieved RAG context well.

## LLM Judge vs Context-Aware Golden Answers

`llm_judge/context_golden_runner.py` compares the chatbot against a reference answer generated with full scraped context.

Prepare a CSV with `query, golden_answer` and optionally `gold_url, skip_reason`, then run:

```bash
export PYTHONPATH=src

python -m evaluation.llm_judge.context_golden_runner \
  --judge-model anthropic/claude-opus-4.7

python -m evaluation.llm_judge.context_golden_runner \
  --golden-csv src/evaluation/data/my_context_golden.csv \
  --judge-model openai/gpt-5.5 \
  --limit 30 \
  --resume
```

Output: `src/data/feedback/context_golden_results.csv`

This judge prompt is symmetric because both answers are assumed to have access to the knowledge base.

## Benchmark Checks

Focused checks in `benchmark_checks/` use a running API.

```bash
export PYTHONPATH=src

# Check how many gold URLs exist in Qdrant
python -m evaluation.benchmark_checks.check_coverage \
  --api-base-url http://localhost:8000 \
  --show-missing

# Compare reranker on vs off
python -m evaluation.benchmark_checks.check_reranker \
  --api-base-url http://localhost:8000 \
  --rewrite
```

Other checks:

| File | Purpose |
|---|---|
| `check_rewriter.py` | Query rewriter ablation/check. |
| `check_bm25_prefilter.py` | BM25/sparse prefilter check. |
| `check_url_aggregation.py` | URL aggregation behavior check. |

## Tests

```bash
export PYTHONPATH=src

pytest src/evaluation/tests/ -v
pytest src/evaluation/tests/test_retrieval_metrics.py -v
pytest src/evaluation/tests/test_text_metrics.py -v
pytest src/evaluation/tests/test_statistical_comparison.py -v
```

Notes:

- BERTScore tests may require model downloads and can be skipped if optional dependencies are unavailable.
- `tests/evaluation_test.py` is legacy and expects `metrics.py` to implement `bert_score()`. That class is currently a placeholder, so prefer the newer tests around `text_metrics.py`.
