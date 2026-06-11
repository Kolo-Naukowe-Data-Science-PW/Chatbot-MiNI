"""
Batch LLM-as-Judge evaluator: Compare two pre-computed answer CSV files.

Outputs:
  - llm_judge_metrics_model_a.csv: Per-query metrics for Model A
  - llm_judge_metrics_model_b.csv: Per-query metrics for Model B
  - llm_judge_results.json: Detailed results with better_variant, reasons, and summary

"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.evaluation.llm_judge.judge import judge_pair

# For catching OpenRouter rate limit errors
class TokenLimitError(Exception):
    """Raised when API token limit is exceeded"""
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_answers_csv(filepath: str) -> pd.DataFrame:
    """Load answers CSV and normalize column names."""
    df = pd.read_csv(filepath, encoding="utf-8")
    df.columns = df.columns.str.strip()
    
    # Try common column name variations
    if "pytanie" not in df.columns:
        # Try English variants
        for col in df.columns:
            if col.lower() in ["query", "question", "q"]:
                df = df.rename(columns={col: "pytanie"})
                break
    
    if "odpowiedz_wygenerowana" not in df.columns:
        for col in df.columns:
            if "answer" in col.lower() or "odpowiedz" in col.lower():
                df = df.rename(columns={col: "odpowiedz_wygenerowana"})
                break
    
    return df


def evaluate_pair(
    query: str,
    answer_a: str,
    answer_b: str,
    judge_model: str,
    language: str = "pl",
    max_retries: int = 2,
) -> Optional[dict]:
    """
    Evaluate two answers using LLM as a judge.
    
    Returns:
        Dict with scores or None if evaluation fails
        
    Raises:
        TokenLimitError: If API rate limit or token limit is exceeded
    """
    try:
        result = judge_pair(
            query,
            answer_a,
            answer_b,
            judge_model=judge_model,
            language=language,
            temperature=0.0,
            max_tokens=400,
            max_retries=max_retries,
        )
        return {
            "a_usefulness": result.variant_a.usefulness,
            "a_accuracy": result.variant_a.accuracy,
            "a_conciseness": result.variant_a.conciseness,
            "b_usefulness": result.variant_b.usefulness,
            "b_accuracy": result.variant_b.accuracy,
            "b_conciseness": result.variant_b.conciseness,
            "better_variant": result.better_variant,
            "reason": result.reason,
        }
    except Exception as e:
        error_str = str(e).lower()
        # Detect token limit or rate limit errors
        if any(x in error_str for x in ["rate_limit", "token", "quota", "limit", "429", "503"]):
            logger.error(f"Token/Rate limit error: {str(e)[:100]}")
            raise TokenLimitError(f"API limit exceeded: {str(e)}") from e
        else:
            logger.error(f"Error evaluating pair: {str(e)[:100]}")
            return None




def save_results(
    results: list,
    failed_count: int,
    output_dir: Path,
    judge_model: str,
    language: str,
    is_incomplete: bool = False,
) -> None:
    """
    Save evaluation results to CSV and JSON files.
    
    Args:
        results: List of successful evaluation results
        failed_count: Number of failed evaluations
        output_dir: Output directory path
        judge_model: Name of judge model used
        language: Language used for evaluation
        is_incomplete: Whether evaluation was interrupted/incomplete
    """
    if not results:
        logger.warning("No results to save - all evaluations failed")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    # Creating CSV files compatible with statistical_comparison.py

    # CSV for Model A metrics 
    df_a_metrics = df_results[
        ["pytanie", "a_usefulness", "a_accuracy", "a_conciseness"]
    ].copy()
    df_a_metrics.columns = ["pytanie", "usefulness", "accuracy", "conciseness"]
    
    csv_a_path = output_dir / "llm_judge_metrics_model_a.csv"
    df_a_metrics.to_csv(csv_a_path, index=False, encoding="utf-8")
    logger.info(f"✓ Saved Model A metrics: {csv_a_path}")
    
    # CSV for Model B metrics
    df_b_metrics = df_results[
        ["pytanie", "b_usefulness", "b_accuracy", "b_conciseness"]
    ].copy()
    df_b_metrics.columns = ["pytanie", "usefulness", "accuracy", "conciseness"]
    
    csv_b_path = output_dir / "llm_judge_metrics_model_b.csv"
    df_b_metrics.to_csv(csv_b_path, index=False, encoding="utf-8")
    logger.info(f"✓ Saved Model B metrics: {csv_b_path}")
    
    # JSON with detailed results and summary
    summary = {
        "evaluation_metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "judge_model": judge_model,
            "language": language,
            "total_evaluated": len(results),
            "total_failed": failed_count,
            "total_attempted": len(results) + failed_count,
            "success_rate": f"{100 * len(results) / (len(results) + failed_count):.1f}%"
            if (len(results) + failed_count) > 0
            else "0%",
            "is_incomplete": is_incomplete,
            "incomplete_reason": "API rate/token limit exceeded" if is_incomplete else None,
        },
        "model_a_stats": {
            "usefulness": {
                "mean": float(df_results["a_usefulness"].mean()),
                "median": float(df_results["a_usefulness"].median()),
                "std": float(df_results["a_usefulness"].std()),
                "min": int(df_results["a_usefulness"].min()),
                "max": int(df_results["a_usefulness"].max()),
            },
            "accuracy": {
                "mean": float(df_results["a_accuracy"].mean()),
                "median": float(df_results["a_accuracy"].median()),
                "std": float(df_results["a_accuracy"].std()),
                "min": int(df_results["a_accuracy"].min()),
                "max": int(df_results["a_accuracy"].max()),
            },
            "conciseness": {
                "mean": float(df_results["a_conciseness"].mean()),
                "median": float(df_results["a_conciseness"].median()),
                "std": float(df_results["a_conciseness"].std()),
                "min": int(df_results["a_conciseness"].min()),
                "max": int(df_results["a_conciseness"].max()),
            },
        },
        "model_b_stats": {
            "usefulness": {
                "mean": float(df_results["b_usefulness"].mean()),
                "median": float(df_results["b_usefulness"].median()),
                "std": float(df_results["b_usefulness"].std()),
                "min": int(df_results["b_usefulness"].min()),
                "max": int(df_results["b_usefulness"].max()),
            },
            "accuracy": {
                "mean": float(df_results["b_accuracy"].mean()),
                "median": float(df_results["b_accuracy"].median()),
                "std": float(df_results["b_accuracy"].std()),
                "min": int(df_results["b_accuracy"].min()),
                "max": int(df_results["b_accuracy"].max()),
            },
            "conciseness": {
                "mean": float(df_results["b_conciseness"].mean()),
                "median": float(df_results["b_conciseness"].median()),
                "std": float(df_results["b_conciseness"].std()),
                "min": int(df_results["b_conciseness"].min()),
                "max": int(df_results["b_conciseness"].max()),
            },
        },
        "pairwise_comparison": {
            "model_a_wins": int((df_results["better_variant"] == "A").sum()),
            "model_b_wins": int((df_results["better_variant"] == "B").sum()),
            "model_a_win_rate": f"{100 * (df_results['better_variant'] == 'A').sum() / len(df_results):.1f}%",
            "model_b_win_rate": f"{100 * (df_results['better_variant'] == 'B').sum() / len(df_results):.1f}%",
        },
        "detailed_results": [
            {
                "query": row["pytanie"],
                "model_a": {
                    "usefulness": int(row["a_usefulness"]),
                    "accuracy": int(row["a_accuracy"]),
                    "conciseness": int(row["a_conciseness"]),
                },
                "model_b": {
                    "usefulness": int(row["b_usefulness"]),
                    "accuracy": int(row["b_accuracy"]),
                    "conciseness": int(row["b_conciseness"]),
                },
                "better_variant": row["better_variant"],
                "judge_reason": row["judge_reason"],
            }
            for _, row in df_results.iterrows()
        ],
    }
    
    json_path = output_dir / "llm_judge_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved detailed results: {json_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model A - Usefulness: μ={summary['model_a_stats']['usefulness']['mean']:.3f}")
    logger.info(f"Model A - Accuracy: μ={summary['model_a_stats']['accuracy']['mean']:.3f}")
    logger.info(f"Model A - Conciseness: μ={summary['model_a_stats']['conciseness']['mean']:.3f}")
    logger.info(f"\nModel B - Usefulness: μ={summary['model_b_stats']['usefulness']['mean']:.3f}")
    logger.info(f"Model B - Accuracy: μ={summary['model_b_stats']['accuracy']['mean']:.3f}")
    logger.info(f"Model B - Conciseness: μ={summary['model_b_stats']['conciseness']['mean']:.3f}")
    logger.info(
        f"\nModel A wins: {summary['pairwise_comparison']['model_a_wins']} "
        f"({summary['pairwise_comparison']['model_a_win_rate']})"
    )
    logger.info(
        f"Model B wins: {summary['pairwise_comparison']['model_b_wins']} "
        f"({summary['pairwise_comparison']['model_b_win_rate']})"
    )
    
    if is_incomplete:
        logger.warning("\n⚠️  EVALUATION INCOMPLETE")
        logger.warning(f"Completed: {len(results)}/{summary['evaluation_metadata']['total_attempted']}")
        logger.warning("Reason: API rate/token limit exceeded")
        logger.warning("Run script again to continue from where it stopped")
    
    logger.info("\n✓ All outputs saved to: " + str(output_dir))


    parser = argparse.ArgumentParser(
        description="Batch LLM-as-Judge evaluation for pre-computed CSV answers"
    )
    parser.add_argument(
        "--model-a",
        required=True,
        help="CSV file with Model A answers (columns: pytanie, odpowiedz_wygenerowana)",
    )
    parser.add_argument(
        "--model-b",
        required=True,
        help="CSV file with Model B answers (columns: pytanie, odpowiedz_wygenerowana)",
    )
    parser.add_argument(
        "--judge-model",
        default="anthropic/claude-opus-4.7",
        help="Judge model ID (via OpenRouter)",
    )
    parser.add_argument(
        "--language",
        default="pl",
        help="Language hint for judge (pl, en, etc)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LLM-as-Judge BATCH EVALUATOR")
    logger.info("=" * 80)

    # Load CSV files
    logger.info(f"Loading Model A: {args.model_a}")
    df_a = load_answers_csv(args.model_a)
    logger.info(f"  ✓ {len(df_a)} questions loaded")

    logger.info(f"Loading Model B: {args.model_b}")
    df_b = load_answers_csv(args.model_b)
    logger.info(f"  ✓ {len(df_b)} questions loaded")

    # Merge datasets
    merged = df_a[["pytanie", "odpowiedz_wygenerowana"]].copy()
    merged.columns = ["pytanie", "answer_a"]

    merged = merged.merge(
        df_b[["pytanie", "odpowiedz_wygenerowana"]].rename(
            columns={"odpowiedz_wygenerowana": "answer_b"}
        ),
        on="pytanie",
        how="inner",
    )

    merged = merged.dropna(subset=["answer_a", "answer_b"])

    if args.limit:
        merged = merged.head(args.limit)

    logger.info(f"Merged dataset: {len(merged)} common questions")

    # Run evaluation
    logger.info("=" * 80)
    logger.info(f"Starting evaluation with {args.judge_model}")
    logger.info("=" * 80)

    results = []
    failed_count = 0
    evaluation_interrupted = False
    total_attempted = 0

    try:
        for idx, row in merged.iterrows():
            total_attempted = idx + 1
            
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Progress: {idx + 1}/{len(merged)} evaluated | Failed: {failed_count}"
                )

            query = row["pytanie"]
            answer_a = row["answer_a"]
            answer_b = row["answer_b"]

            try:
                judge_scores = evaluate_pair(
                    query, answer_a, answer_b, args.judge_model, args.language
                )
            except TokenLimitError as e:
                logger.error(f"\n⚠️  INTERRUPTING: {str(e)}")
                evaluation_interrupted = True
                break

            if judge_scores:
                results.append(
                    {
                        "pytanie": query,
                        "a_usefulness": judge_scores["a_usefulness"],
                        "a_accuracy": judge_scores["a_accuracy"],
                        "a_conciseness": judge_scores["a_conciseness"],
                        "b_usefulness": judge_scores["b_usefulness"],
                        "b_accuracy": judge_scores["b_accuracy"],
                        "b_conciseness": judge_scores["b_conciseness"],
                        "better_variant": judge_scores["better_variant"],
                        "judge_reason": judge_scores["reason"],
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                )
            else:
                failed_count += 1
                logger.warning(f"Failed for question {idx + 1}")

            time.sleep(args.delay)

    except Exception as e:
        logger.error(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        evaluation_interrupted = True

    logger.info(f"\nEvaluation stopped after {total_attempted}/{len(merged)} questions")
    logger.info(f"  Successful: {len(results)}")
    logger.info(f"  Failed: {failed_count}")

    # Save results (whether complete or incomplete)
    if results:
        save_results(
            results=results,
            failed_count=failed_count,
            output_dir=output_dir,
            judge_model=args.judge_model,
            language=args.language,
            is_incomplete=evaluation_interrupted,
        )
    else:
        logger.error("❌ No successful evaluations - nothing to save")
        return 1

    # Return non-zero if interrupted, so GitHub Actions can detect the issue
    if evaluation_interrupted:
        logger.warning(
            "\n⚠️  Evaluation was interrupted. Partial results were saved.\n"
            "Run the script again to continue (it will re-evaluate from the beginning).\n"
            "To resume where you left off, you may need to disable re-evaluation of already-scored entries."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
