"""
Generate golden (reference) answers for the evaluation question set using three supermodels.

Three frontier models (GPT, Claude Opus, Gemini) each answer every answerable question from
their own knowledge — WITHOUT RAG context. Questions requiring personal/session context
(personal schedule, personal room, "kto prowadzi MI...") are skipped.

These golden answers serve as reference texts for text-quality evaluation: we compare
the chatbot's answers against them using BERTScore and embedding cosine similarity.

Why no RAG context?  The supermodels act as an "oracle" — they represent the best answer
a world-class expert with general Polish-university knowledge could give.  Passing RAG
chunks would just make the supermodel a better version of our own pipeline, which is
less informative as a benchmark.

Usage
-----
    export PYTHONPATH=src
    python -m evaluation.generate_golden_answers               # all questions
    python -m evaluation.generate_golden_answers --limit 20   # first 20 (cheap test)
    python -m evaluation.generate_golden_answers --resume     # skip already-processed rows

Output
------
    src/evaluation/data/golden_answers.csv
    Columns: query, gold_url, golden_answer_gpt, golden_answer_opus, golden_answer_gemini, skip_reason
"""

import argparse
import csv
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV = "src/evaluation/data/questions_filtered.csv"
OUTPUT_CSV = "src/evaluation/data/golden_answers.csv"

# OpenRouter model IDs — verified available as of 2026-05
SUPERMODELS: dict[str, str] = {
    "gpt": "openai/gpt-5.5",
    "opus": "anthropic/claude-opus-4.7",
    "gemini": "google/gemini-3.1-pro-preview-customtools",
}

OUTPUT_FIELDNAMES = [
    "query",
    "gold_url",
    "golden_answer_gpt",
    "golden_answer_opus",
    "golden_answer_gemini",
    "skip_reason",
]

SYSTEM_PROMPT = (
    "Jesteś ekspertem od spraw Wydziału Matematyki i Nauk Informacyjnych (MiNI) "
    "Politechniki Warszawskiej. Odpowiadasz na pytania studentów i pracowników wydziału. "
    "Odpowiadaj po polsku, zwięźle (2-5 zdań), rzeczowo i na temat. "
    "Jeśli pytanie dotyczy konkretnej procedury, wymień kluczowe kroki. "
    "Jeśli nie znasz dokładnej odpowiedzi, podaj ogólne wskazówki charakterystyczne "
    "dla polskiej uczelni publicznej."
)

# Regex patterns flagging questions that require personal/session context.
# These are questions like "w której sali mam zajęcia?" or "kto prowadzi MI wykład?"
# where the answer depends on the specific student's timetable — which the supermodel
# has no access to.
_SESSION_PATTERNS = re.compile(
    r"""
    # first-person verb "mam" combined with personal-schedule nouns
    \bmam\s+(zajęcia|wykład|ćwiczenia|laboratorium|lektorat|sale?|grupy?|kolokwium|egzamin)\b
    |
    # "w której sali mam / są moje zajęcia" etc.
    \bw\s+której\s+sali\b
    |
    # "kto prowadzi mi X" / "kto mi prowadzi"
    \bkto\s+(prowadzi\s+mi|mi\s+prowadzi)\b
    |
    # "moje zajęcia / mój wykład / moje oceny / mój plan" — possessive + schedule noun
    \b(moje?|mój|moja)\s+(zajęcia|wykład|ćwiczenia|laboratorium|lektorat|plan|grupy?|ocen\w*|indeks|prowadzący)
    |
    # "kiedy mam zajęcia / wykład / ćwiczenia"
    \bkiedy\s+mam\s+(zajęcia|wykład|ćwiczenia|laboratorium|lektorat)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _skip_reason(query: str) -> str:
    """Return a human-readable skip reason if the question needs session context, else ''."""
    if _SESSION_PATTERNS.search(query):
        return "session_context"
    return ""


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


def _load_already_processed(path: str) -> set[str]:
    """Return the set of queries already present in the output CSV (answered OR skipped)."""
    processed: set[str] = set()
    p = Path(path)
    if not p.exists():
        return processed
    with p.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get("query") or "").strip()
            if q:
                processed.add(q)
    return processed


def _ask_model(client: OpenAI, model_id: str, query: str) -> str:
    """Call one supermodel and return its answer, or '' on failure."""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Model %s failed for '%s': %s", model_id, query[:60], exc)
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate golden answers using three supermodels (GPT, Opus, Gemini)."
    )
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max questions to process."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions already present in the output CSV.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between full question batches (rate-limit safety).",
    )
    parser.add_argument(
        "--no-skip-personal",
        action="store_true",
        help="Disable the session-context filter and attempt all questions.",
    )
    args = parser.parse_args()

    rows = _load_eval_rows(args.input_csv)
    logger.info("Loaded %d questions from %s", len(rows), args.input_csv)

    already_processed: set[str] = set()
    if args.resume:
        already_processed = _load_already_processed(args.output_csv)
        logger.info(
            "Resuming — %d questions already processed, skipping them.",
            len(already_processed),
        )

    rows_to_process = [r for r in rows if r["query"] not in already_processed]
    if args.limit is not None:
        rows_to_process = rows_to_process[: args.limit]
    logger.info("Will process %d questions.", len(rows_to_process))

    if not rows_to_process:
        logger.info("Nothing to do.")
        return

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    client = _get_client()

    skipped = 0
    answered = 0

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for i, row in enumerate(rows_to_process, start=1):
            query = row["query"]
            gold_url = row["gold_url"]

            reason = "" if args.no_skip_personal else _skip_reason(query)

            if reason:
                writer.writerow(
                    {
                        "query": query,
                        "gold_url": gold_url,
                        "golden_answer_gpt": "",
                        "golden_answer_opus": "",
                        "golden_answer_gemini": "",
                        "skip_reason": reason,
                    }
                )
                f.flush()
                skipped += 1
                logger.info(
                    "[%d/%d] SKIP (%s): %s", i, len(rows_to_process), reason, query[:70]
                )
                continue

            answers: dict[str, str] = {}
            for key, model_id in SUPERMODELS.items():
                answers[key] = _ask_model(client, model_id, query)

            writer.writerow(
                {
                    "query": query,
                    "gold_url": gold_url,
                    "golden_answer_gpt": answers["gpt"],
                    "golden_answer_opus": answers["opus"],
                    "golden_answer_gemini": answers["gemini"],
                    "skip_reason": "",
                }
            )
            f.flush()
            answered += 1
            logger.info("[%d/%d] OK: %s", i, len(rows_to_process), query[:70])

            if args.delay > 0 and i < len(rows_to_process):
                time.sleep(args.delay)

    logger.info(
        "Done. Answered: %d, Skipped: %d. Saved to %s",
        answered,
        skipped,
        args.output_csv,
    )


if __name__ == "__main__":
    main()
