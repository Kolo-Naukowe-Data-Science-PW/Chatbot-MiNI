#!/usr/bin/env python3
"""
Quick generation metrics: query chatbot, collect answers, compute metrics.
Simpler than run_experiment.py - just text metrics without full RAG evaluation.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def query_chatbot(api_url: str, question: str, timeout: int = 60) -> str:
    """Send question to /chat endpoint, return answer."""
    try:
        response = requests.post(
            api_url,
            json={"query": question},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "")
    except Exception as e:
        logger.warning(f"Failed to query chatbot for '{question[:50]}...': {e}")
        return ""


def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    """Compute BLEU, ROUGE, METEOR metrics."""
    import nltk
    import sacrebleu
    from nltk.translate import meteor_score
    from rouge import Rouge

    # Ensure NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    metrics = {}

    # BLEU
    try:
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        metrics["bleu"] = result.score / 100
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        metrics["bleu"] = 0.0

    # ROUGE
    try:
        rouge = Rouge()
        scores = rouge.get_scores(hypotheses, references, avg=True)
        metrics["rouge1_f"] = scores["rouge1"]["f"]
        metrics["rouge2_f"] = scores["rouge2"]["f"]
        metrics["rougeL_f"] = scores["rougeL"]["f"]
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        metrics["rouge1_f"] = metrics["rouge2_f"] = metrics["rougeL_f"] = 0.0

    # METEOR
    try:
        meteor_scores = [
            meteor_score.single_meteor_score(ref, hyp)
            for hyp, ref in zip(hypotheses, references)
        ]
        metrics["meteor"] = (
            sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        )
    except Exception as e:
        logger.warning(f"METEOR computation failed: {e}")
        metrics["meteor"] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Quick generation metrics: query chatbot and compute text metrics"
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="Chatbot /chat endpoint (e.g., http://localhost:8000/chat)",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("src/evaluation/data/QA_rag.csv"),
        help="CSV with pytanie, odpowiedz columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/evaluation/results"),
        help="Output directory",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout per query (seconds)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Load reference data
    logger.info("Loading reference answers from %s", args.input_csv)
    df = pd.read_csv(args.input_csv, sep="|", engine="python")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=str.lower)

    if "pytanie" not in df.columns or "odpowiedz" not in df.columns:
        raise ValueError("CSV must have 'pytanie' and 'odpowiedz' columns")

    questions = df["pytanie"].tolist()
    references = df["odpowiedz"].tolist()

    # Query chatbot
    logger.info("Querying chatbot for %d questions", len(questions))
    generated = []
    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0:
            logger.info("  Progress: %d/%d", i + 1, len(questions))
        answer = query_chatbot(args.api_url, q, timeout=args.timeout)
        generated.append(answer)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(generated, references)

    # Save results
    results = {
        "timestamp": ts,
        "api_url": args.api_url,
        "n_questions": len(questions),
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
