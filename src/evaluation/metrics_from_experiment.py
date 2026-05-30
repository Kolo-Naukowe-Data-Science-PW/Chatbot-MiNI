#!/usr/bin/env python3
"""
Generation metrics from run_experiment output: compare generated answers with reference.

Reads per_query.csv from run_experiment (with column: generated_answer),
joins with reference answers from QA_rag.csv,
computes BLEU, ROUGE, METEOR metrics.

Usage
-----
python -m evaluation.metrics_from_experiment \\
    --per-query-csv src/evaluation/results/baseline_rag_20260529T120000_per_query.csv \\
    --reference-csv src/evaluation/data/QA_rag.csv \\
    --output-dir src/evaluation/results
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _bleu_score(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Compute BLEU score variants."""
    import sacrebleu

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "bleu": result.score / 100,
        "bleu_1": result.precisions[0] / 100 if len(result.precisions) > 0 else 0.0,
        "bleu_2": result.precisions[1] / 100 if len(result.precisions) > 1 else 0.0,
        "bleu_3": result.precisions[2] / 100 if len(result.precisions) > 2 else 0.0,
        "bleu_4": result.precisions[3] / 100 if len(result.precisions) > 3 else 0.0,
    }


def _rouge_scores(
    hypotheses: list[str], references: list[str]
) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    from rouge import Rouge

    rouge = Rouge()
    pairs = [
        (h, r)
        for h, r in zip(hypotheses, references)
        if h.strip() and r.strip()
    ]
    if not pairs:
        return {
            "rouge1_p": 0.0,
            "rouge1_r": 0.0,
            "rouge1_f": 0.0,
            "rouge2_p": 0.0,
            "rouge2_r": 0.0,
            "rouge2_f": 0.0,
            "rougeL_p": 0.0,
            "rougeL_r": 0.0,
            "rougeL_f": 0.0,
        }

    hyps, refs = zip(*pairs)
    scores = rouge.get_scores(list(hyps), list(refs), avg=True)

    flat: dict[str, float] = {}
    for key, sub in scores.items():
        clean = key.replace("-", "")
        for metric, val in sub.items():
            flat[f"{clean}_{metric}"] = val
    return flat


def _meteor_score(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Compute METEOR score."""
    import nltk
    from nltk.translate import meteor_score

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    meteor_scores = []
    for hyp, ref in zip(hypotheses, references):
        if hyp.strip() and ref.strip():
            score = meteor_score.single_meteor_score(ref, hyp)
            meteor_scores.append(score)

    avg_meteor = (
        sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    )
    return {"meteor": avg_meteor}


def main():
    parser = argparse.ArgumentParser(
        description="Compute generation metrics from run_experiment output"
    )
    parser.add_argument(
        "--per-query-csv",
        type=Path,
        required=True,
        help="Path to *_per_query.csv from run_experiment",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("src/evaluation/data/QA_rag.csv"),
        help="Path to reference answers (default: QA_rag.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/evaluation/results"),
        help="Output directory for metrics JSON",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Load experiment results
    logger.info("Loading experiment results from %s", args.per_query_csv)
    experiment_df = pd.read_csv(args.per_query_csv)

    if "generated_answer" not in experiment_df.columns:
        logger.error(
            "Column 'generated_answer' not found in %s", args.per_query_csv
        )
        sys.exit(1)

    if "query" not in experiment_df.columns:
        logger.error("Column 'query' not found in %s", args.per_query_csv)
        sys.exit(1)

    # Load reference answers
    logger.info("Loading reference answers from %s", args.reference_csv)
    ref_df = pd.read_csv(args.reference_csv, sep="|", engine="python")
    ref_df.columns = ref_df.columns.str.strip()
    ref_df = ref_df.rename(columns=str.lower)

    if "pytanie" not in ref_df.columns or "odpowiedz" not in ref_df.columns:
        logger.error(
            "Reference CSV must have 'pytanie' and 'odpowiedz' columns"
        )
        sys.exit(1)

    # Join on query
    logger.info("Joining experiment results with reference answers")
    experiment_df.columns = experiment_df.columns.str.strip().str.lower()
    ref_df_joined = ref_df.rename(
        columns={"pytanie": "query", "odpowiedz": "reference_answer"}
    )[["query", "reference_answer"]]

    merged = experiment_df.merge(
        ref_df_joined, on="query", how="inner"
    )

    if len(merged) == 0:
        logger.warning(
            "No matching queries found between experiment and reference CSVs"
        )
        sys.exit(1)

    logger.info(
        "Matched %d queries (%d total in experiment)",
        len(merged),
        len(experiment_df),
    )

    # Extract lists
    generated = merged["generated_answer"].fillna("").tolist()
    references = merged["reference_answer"].fillna("").tolist()

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = {}

    try:
        metrics.update(_bleu_score(generated, references))
    except Exception as e:
        logger.warning("BLEU computation failed: %s", e)

    try:
        metrics.update(_rouge_scores(generated, references))
    except Exception as e:
        logger.warning("ROUGE computation failed: %s", e)

    try:
        metrics.update(_meteor_score(generated, references))
    except Exception as e:
        logger.warning("METEOR computation failed: %s", e)

    # Save results
    results = {
        "timestamp": ts,
        "per_query_csv": str(args.per_query_csv),
        "reference_csv": str(args.reference_csv),
        "n_matched": len(merged),
        "n_total": len(experiment_df),
        "metrics": metrics,
    }

    output_file = args.output_dir / f"generation_metrics_{ts}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_file)

    # Summary
    logger.info("=== Generation Metrics Summary ===")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
