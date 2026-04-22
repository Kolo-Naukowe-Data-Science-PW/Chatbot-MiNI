"""
Generate golden (reference) answers for the evaluation question set using a supermodel.

The supermodel (GPT-4o by default) answers each question from its own knowledge,
without RAG context.  These golden answers serve as reference texts for text-quality
evaluation: we compare the chatbot's answers against them using BERTScore and
embedding cosine similarity.

Usage
-----
    export PYTHONPATH=src
    python -m evaluation.generate_golden_answers               # all questions
    python -m evaluation.generate_golden_answers --limit 20   # first 20 (cheap test)
    python -m evaluation.generate_golden_answers --resume     # skip already-answered rows

Output
------
    src/evaluation/data/golden_answers.csv
    Columns: query, gold_url, golden_answer
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV = "src/evaluation/data/questions_filtered.csv"
OUTPUT_CSV = "src/evaluation/data/golden_answers.csv"
MODEL = "openai/gpt-4o"

SYSTEM_PROMPT = (
    "Jesteś ekspertem od spraw Wydziału Matematyki i Nauk Informacyjnych (MiNI) "
    "Politechniki Warszawskiej. Odpowiadasz na pytania studentów i pracowników wydziału. "
    "Odpowiadaj po polsku, zwięźle (2-5 zdań), rzeczowo i na temat. "
    "Jeśli pytanie dotyczy konkretnej procedury, wymień kluczowe kroki. "
    "Jeśli nie znasz dokładnej odpowiedzi, podaj ogólne wskazówki charakterystyczne "
    "dla polskiej uczelni publicznej."
)


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _load_eval_rows(path: str) -> list[dict[str, str]]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            norm = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            query = norm.get("pytanie") or norm.get("query") or norm.get("question", "")
            url = (
                norm.get("strona")
                or norm.get("relevant_urls")
                or norm.get("url")
                or norm.get("link")
                or ""
            )
            if query:
                rows.append({"query": query, "gold_url": url})
    return rows


def _load_already_answered(path: str) -> set[str]:
    """Return the set of queries already present in the output CSV."""
    answered: set[str] = set()
    p = Path(path)
    if not p.exists():
        return answered
    with p.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get("query") or "").strip()
            if q:
                answered.add(q)
    return answered


def _ask_supermodel(client: OpenAI, query: str) -> str:
    """Call the supermodel and return its answer, or an empty string on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Failed to get answer for '%s': %s", query[:60], exc)
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden answers using a supermodel.")
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--model", default=MODEL, help="OpenRouter model ID.")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to process.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions already present in the output CSV.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (rate-limit safety).",
    )
    args = parser.parse_args()

    rows = _load_eval_rows(args.input_csv)
    logger.info("Loaded %d questions from %s", len(rows), args.input_csv)

    already_answered: set[str] = set()
    if args.resume:
        already_answered = _load_already_answered(args.output_csv)
        logger.info("Resuming — %d questions already answered, skipping them.", len(already_answered))

    rows_to_process = [r for r in rows if r["query"] not in already_answered]
    if args.limit is not None:
        rows_to_process = rows_to_process[: args.limit]
    logger.info("Will process %d questions.", len(rows_to_process))

    if not rows_to_process:
        logger.info("Nothing to do.")
        return

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode so --resume works correctly
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    client = _get_client()

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "gold_url", "golden_answer"])
        if not file_exists:
            writer.writeheader()

        for i, row in enumerate(rows_to_process, start=1):
            answer = _ask_supermodel(client, row["query"])
            writer.writerow(
                {"query": row["query"], "gold_url": row["gold_url"], "golden_answer": answer}
            )
            f.flush()
            logger.info("[%d/%d] %s", i, len(rows_to_process), row["query"][:70])
            if args.delay > 0 and i < len(rows_to_process):
                time.sleep(args.delay)

    logger.info("Done. Saved to %s", args.output_csv)


if __name__ == "__main__":
    main()
