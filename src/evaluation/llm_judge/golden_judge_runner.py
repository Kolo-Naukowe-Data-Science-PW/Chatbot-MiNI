"""
Batch runner: for each question in golden_answers.csv, fetch the chatbot's answer via /chat,
then judge it against the chosen supermodel's golden answer.

Usage
-----
    export PYTHONPATH=src

    # Compare chatbot vs Opus golden answers, first 20 questions
    python -m evaluation.llm_judge.golden_judge_runner \\
        --judge-model anthropic/claude-opus-4.7 \\
        --golden-model opus \\
        --limit 20

    # Full run against GPT golden answers
    python -m evaluation.llm_judge.golden_judge_runner \\
        --judge-model openai/gpt-5.5 \\
        --golden-model gpt

    # Resume after interruption
    python -m evaluation.llm_judge.golden_judge_runner \\
        --judge-model anthropic/claude-opus-4.7 \\
        --resume

Output
------
    src/data/feedback/golden_judge_results.csv
    Columns: query, gold_url, golden_model_used, chatbot_answer, golden_answer,
             chatbot_usefulness, chatbot_accuracy, chatbot_completeness,
             golden_usefulness, golden_accuracy, golden_completeness,
             better, chatbot_weaknesses, golden_weaknesses, judge_model, created_at
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.evaluation.llm_judge.golden_judge import GoldenJudgeResult, judge_against_golden

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_ANSWERS_CSV = "src/evaluation/data/golden_answers.csv"
OUTPUT_CSV = "src/data/feedback/golden_judge_results.csv"
API_URL = "http://127.0.0.1:8000/chat"

GOLDEN_MODEL_COLUMN = {
    "gpt": "golden_answer_gpt",
    "opus": "golden_answer_opus",
    "gemini": "golden_answer_gemini",
}

OUTPUT_FIELDNAMES = [
    "query",
    "gold_url",
    "golden_model_used",
    "chatbot_answer",
    "golden_answer",
    "chatbot_usefulness",
    "chatbot_accuracy",
    "chatbot_completeness",
    "golden_usefulness",
    "golden_accuracy",
    "golden_completeness",
    "better",
    "chatbot_weaknesses",
    "golden_weaknesses",
    "judge_model",
    "created_at",
]


def _load_golden_rows(path: str, golden_col: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("skip_reason"):
                continue
            query = (r.get("query") or "").strip()
            golden = (r.get(golden_col) or "").strip()
            if not query or not golden:
                continue
            rows.append({
                "query": query,
                "gold_url": r.get("gold_url", ""),
                "golden_answer": golden,
            })
            if limit and len(rows) >= limit:
                break
    return rows


def _load_already_judged(path: str) -> set[str]:
    done: set[str] = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            q = (r.get("query") or "").strip()
            if q:
                done.add(q)
    return done


def _call_chatbot(api_url: str, query: str, timeout: int) -> str:
    payload = json.dumps({"query": query, "language": "pl"}).encode()
    req = Request(api_url, data=payload, method="POST",
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()).get("answer", "").strip()
    except HTTPError as exc:
        raise RuntimeError(f"Chat API HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"Chat API unreachable: {exc}") from exc


def _result_row(
    query: str,
    gold_url: str,
    golden_model_used: str,
    chatbot_answer: str,
    golden_answer: str,
    result: GoldenJudgeResult,
    judge_model: str,
) -> dict:
    return {
        "query": query,
        "gold_url": gold_url,
        "golden_model_used": golden_model_used,
        "chatbot_answer": chatbot_answer,
        "golden_answer": golden_answer,
        "chatbot_usefulness": result.chatbot.usefulness,
        "chatbot_accuracy": result.chatbot.accuracy,
        "chatbot_completeness": result.chatbot.completeness,
        "golden_usefulness": result.golden.usefulness,
        "golden_accuracy": result.golden.accuracy,
        "golden_completeness": result.golden.completeness,
        "better": result.better,
        "chatbot_weaknesses": result.chatbot_weaknesses,
        "golden_weaknesses": result.golden_weaknesses,
        "judge_model": judge_model,
        "created_at": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge chatbot answers against golden supermodel answers."
    )
    parser.add_argument("--golden-csv", default=GOLDEN_ANSWERS_CSV)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument(
        "--golden-model",
        default="opus",
        choices=list(GOLDEN_MODEL_COLUMN.keys()),
        help="Which supermodel's golden answer to use as reference (default: opus).",
    )
    parser.add_argument("--judge-model", required=True, help="OpenRouter model ID for the judge.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-judged queries.")
    parser.add_argument("--language", default="pl")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    golden_col = GOLDEN_MODEL_COLUMN[args.golden_model]
    rows = _load_golden_rows(args.golden_csv, golden_col, args.limit)
    if not rows:
        logger.error("No answerable rows found in %s", args.golden_csv)
        return 1
    logger.info("Loaded %d questions from %s", len(rows), args.golden_csv)

    already_done: set[str] = set()
    if args.resume:
        already_done = _load_already_judged(args.output_csv)
        logger.info("Resuming — skipping %d already judged.", len(already_done))
    rows = [r for r in rows if r["query"] not in already_done]
    logger.info("Will judge %d questions.", len(rows))

    if not rows:
        logger.info("Nothing to do.")
        return 0

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for i, row in enumerate(rows, start=1):
            query = row["query"]
            logger.info("[%d/%d] %s", i, len(rows), query[:70])

            try:
                chatbot_answer = _call_chatbot(args.api_url, query, args.timeout)
            except RuntimeError as exc:
                logger.error("Chatbot call failed: %s — skipping.", exc)
                continue

            try:
                result = judge_against_golden(
                    query,
                    chatbot_answer,
                    row["golden_answer"],
                    judge_model=args.judge_model,
                    language=args.language,
                    max_retries=args.max_retries,
                )
            except RuntimeError as exc:
                logger.error("Judge failed: %s — skipping.", exc)
                continue

            writer.writerow(_result_row(
                query=query,
                gold_url=row["gold_url"],
                golden_model_used=args.golden_model,
                chatbot_answer=chatbot_answer,
                golden_answer=row["golden_answer"],
                result=result,
                judge_model=args.judge_model,
            ))
            f.flush()

            if args.delay > 0 and i < len(rows):
                time.sleep(args.delay)

    logger.info("Done. Results saved to %s", args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
