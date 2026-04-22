"""
Batch runner: load evaluation questions from CSV, fetch two TestPro-style answers from ``/chat``,
then score the pair with ``judge_pair`` and write results to CSV.
"""

import argparse
import csv
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.evaluation.llm_judge.judge import JudgeResult, judge_pair

MODEL_POOL = [
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324",
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "microsoft/phi-4",
]


def _pick_variant_configs(rng: random.Random) -> tuple[dict, dict]:
    """
    Sample two ``modelConfig`` dicts (profiles A and B: model, sampling params, style instruction).
    Args:
        rng: Seeded RNG for reproducible runs (see ``--seed``).
    Returns:
        ``(config_a, config_b)`` ready for the chat API.
    """

    def pick(items: list[str]) -> str:
        """Return one random item from a non-empty list."""
        return items[rng.randrange(len(items))]

    def float_range(min_val: float, max_val: float, step: float = 0.1) -> float:
        """Random float from ``min_val`` to ``max_val`` inclusive, stepped by ``step``."""
        count = int(round((max_val - min_val) / step))
        return round(min_val + rng.randint(0, count) * step, 2)

    def int_range(min_val: int, max_val: int) -> int:
        """Random integer in ``[min_val, max_val]`` inclusive."""
        return rng.randint(min_val, max_val)

    variant_configs = {
        "A": {
            "model": pick(MODEL_POOL),
            "temperature": float_range(0.1, 0.3, 0.1),
            "top_p": float_range(0.2, 0.5, 0.1),
            "frequency_penalty": float_range(0.0, 0.2, 0.1),
            "presence_penalty": float_range(0.0, 0.2, 0.1),
            "max_tokens": int_range(150, 250),
            "styleInstruction": "Odpowiedz bardzo krotko i konkretnie. Bez owijania.",
        },
        "B": {
            "model": pick(MODEL_POOL),
            "temperature": float_range(0.4, 0.6, 0.1),
            "top_p": float_range(0.5, 0.7, 0.1),
            "frequency_penalty": float_range(0.1, 0.3, 0.1),
            "presence_penalty": float_range(0.1, 0.3, 0.1),
            "max_tokens": int_range(150, 250),
            "styleInstruction": "Odpowiedz luzno, prosto i przyjaznie.",
        },
    }
    return variant_configs["A"], variant_configs["B"]


def _read_eval_rows(input_csv: str, limit: int | None = None) -> list[dict[str, str]]:
    """
    Read evaluation CSV rows into ``query`` and optional ``gold_link`` (supports common column names).

    Args:
        input_csv: Path to CSV (UTF-8 with optional BOM).
        limit: If set, stop after this many valid rows.
    Returns:
        List of ``{"query": str, "gold_link": str}``.
    """
    rows: list[dict[str, str]] = []
    with open(input_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            normalized = {
                (k or "").strip().lower(): (v or "").strip() for k, v in r.items()
            }
            query = (
                normalized.get("pytanie")
                or normalized.get("query")
                or normalized.get("question")
            )
            if not query:
                continue
            gold_link = (
                normalized.get("strona")
                or normalized.get("relevant_urls")
                or normalized.get("url")
                or normalized.get("link")
                or ""
            )
            rows.append({"query": query, "gold_link": gold_link})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _post_chat(api_url: str, payload: dict, timeout_sec: int) -> dict:
    """
    POST JSON to the chat API and return parsed JSON body.

    Raises:
        RuntimeError: On HTTP error or network failure (includes response body when available).
    """
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        api_url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat API HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Chat API request failed: {exc}") from exc


def _call_variant(
    api_url: str,
    query: str,
    language: str,
    variant_label: str,
    model_config: dict,
    timeout_sec: int,
) -> tuple[str, list[str]]:
    """
    One ``/chat`` call in TestPro shape: mode ``testPro``, ``modelConfig``, and styled user query.

    Returns:
        ``(answer_text, sources_list)``.
    """
    payload = {
        "query": f"{model_config['styleInstruction']}\n\nPytanie uzytkownika: {query}",
        "language": language,
        "mode": "testPro",
        "variant": variant_label,
        "modelConfig": model_config,
    }
    response_json = _post_chat(api_url, payload, timeout_sec=timeout_sec)
    answer = (response_json.get("answer") or "").strip()
    sources = response_json.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    return answer, sources


def _csv_fieldnames() -> list[str]:
    """Column names for the output judge CSV (one row per evaluation question)."""
    return [
        "run_id",
        "created_at",
        "query",
        "gold_link",
        "variant_a_answer",
        "variant_b_answer",
        "variant_a_sources",
        "variant_b_sources",
        "variant_a_config",
        "variant_b_config",
        "a_usefulness",
        "a_accuracy",
        "a_conciseness",
        "b_usefulness",
        "b_accuracy",
        "b_conciseness",
        "better_variant",
        "judge_reason",
        "judge_model",
    ]


def _result_row(
    run_id: str,
    query: str,
    gold_link: str,
    answer_a: str,
    answer_b: str,
    sources_a: list[str],
    sources_b: list[str],
    config_a: dict,
    config_b: dict,
    judge_result: JudgeResult,
    judge_model: str,
) -> dict[str, str | int]:
    """Flatten one evaluation run into a CSV row dict (JSON-serialize nested lists/dicts)."""
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "query": query,
        "gold_link": gold_link,
        "variant_a_answer": answer_a,
        "variant_b_answer": answer_b,
        "variant_a_sources": json.dumps(sources_a, ensure_ascii=False),
        "variant_b_sources": json.dumps(sources_b, ensure_ascii=False),
        "variant_a_config": json.dumps(config_a, ensure_ascii=False),
        "variant_b_config": json.dumps(config_b, ensure_ascii=False),
        "a_usefulness": judge_result.variant_a.usefulness,
        "a_accuracy": judge_result.variant_a.accuracy,
        "a_conciseness": judge_result.variant_a.conciseness,
        "b_usefulness": judge_result.variant_b.usefulness,
        "b_accuracy": judge_result.variant_b.accuracy,
        "b_conciseness": judge_result.variant_b.conciseness,
        "better_variant": judge_result.better_variant,
        "judge_reason": judge_result.reason,
        "judge_model": judge_model,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for input/output paths, API URL, judge model, limits, and timeouts."""
    parser = argparse.ArgumentParser(
        description="Run offline LLM-as-a-judge evaluation for TestPro."
    )
    parser.add_argument(
        "--input-csv",
        default="src/evaluation/data/questions_with_links.csv",
        help="Evaluation CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        default="src/data/feedback/llm_judge_feedback.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/chat",
        help="Chatbot API /chat endpoint URL.",
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="OpenRouter model name for judge.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of queries."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for variant configs."
    )
    parser.add_argument(
        "--language", default="pl", help="Language code passed to /chat."
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="HTTP timeout in seconds."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Judge retries for parse/API failures.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Iterate evaluation rows: fetch two chat answers, judge with OpenRouter, append CSV rows.

    Returns:
        ``0`` on success, ``1`` if no rows were loaded from the input CSV.
    """
    args = _parse_args()
    rng = random.Random(args.seed)

    rows = _read_eval_rows(args.input_csv, limit=args.limit)
    if not rows:
        print("No evaluation queries found in input CSV.")
        return 1

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            query = row["query"]
            gold_link = row["gold_link"]
            config_a, config_b = _pick_variant_configs(rng)

            answer_a, sources_a = _call_variant(
                args.api_url, query, args.language, "A", config_a, args.timeout
            )
            answer_b, sources_b = _call_variant(
                args.api_url, query, args.language, "B", config_b, args.timeout
            )

            judge_result = judge_pair(
                query,
                answer_a,
                answer_b,
                judge_model=args.judge_model,
                language=args.language,
                max_retries=args.max_retries,
            )

            writer.writerow(
                _result_row(
                    run_id=run_id,
                    query=query,
                    gold_link=gold_link,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    sources_a=sources_a,
                    sources_b=sources_b,
                    config_a=config_a,
                    config_b=config_b,
                    judge_result=judge_result,
                    judge_model=args.judge_model,
                )
            )
            print(f"[{index}/{len(rows)}] done")

    print(f"Saved judge results to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
