"""
Batch runner: load evaluation questions from CSV, fetch two TestPro-style answers from ``/chat``,
then score the pair with ``judge_pair`` and write results to CSV.

Experiment dimension
--------------------
Set the ``EXPERIMENT_DIM`` environment variable to control what varies between variant A and B.
Only ONE dimension changes at a time so results are comparable across runs.

    EXPERIMENT_DIM=model        (default) — A and B get different models; params/persona fixed
    EXPERIMENT_DIM=temperature  — A and B get different temperature ranges; model/persona fixed
    EXPERIMENT_DIM=persona      — A and B get different style instructions; model/params fixed

You can also pin the baseline values with:
    EXPERIMENT_MODEL=openai/gpt-4o-mini   (used for temperature and persona experiments)
    EXPERIMENT_TEMP=0.2                   (used for model and persona experiments)
    EXPERIMENT_PERSONA=0                  (index 0-3 into PERSONAS list; used for model and temperature experiments)
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.evaluation.llm_judge.judge import JudgeResult, judge_pair

MODEL_POOL = [
    # mid-tier
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324",
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "microsoft/phi-4",
    # supermodels
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "google/gemini-3.1-pro-preview-customtools",
]

PERSONAS = [
    "Odpowiedz bardzo krótko i konkretnie. Bez owijania w bawełnę.",
    "Odpowiedz luzno, prosto i przyjaźnie.",
    "Odpowiedz formalnie i akademicko, pełnymi zdaniami.",
    "Podaj wyczerpującą odpowiedź z detalami i przykładami.",
]

# Valid values for EXPERIMENT_DIM
_VALID_DIMS = {"model", "temperature", "persona"}


def _pick_variant_configs(rng: random.Random) -> tuple[dict, dict]:
    """
    Build two ``modelConfig`` dicts (A and B) that differ on exactly ONE dimension.

    The dimension is controlled by the ``EXPERIMENT_DIM`` environment variable:
      - ``model``       — different models, same temperature and persona
      - ``temperature`` — different temperature ranges, same model and persona
      - ``persona``     — different style instructions, same model and temperature

    Args:
        rng: Seeded RNG for reproducible runs (see ``--seed``).
    Returns:
        ``(config_a, config_b)`` ready for the chat API.
    """
    dim = os.getenv("EXPERIMENT_DIM", "model").lower()
    if dim not in _VALID_DIMS:
        raise ValueError(
            f"EXPERIMENT_DIM='{dim}' is not valid. Choose from: {sorted(_VALID_DIMS)}"
        )

    baseline_model = os.getenv("EXPERIMENT_MODEL", "openai/gpt-4o-mini")
    baseline_temp = float(os.getenv("EXPERIMENT_TEMP", "0.2"))
    persona_idx = int(os.getenv("EXPERIMENT_PERSONA", "0"))
    if not 0 <= persona_idx < len(PERSONAS):
        raise ValueError(
            f"EXPERIMENT_PERSONA={persona_idx} out of range 0–{len(PERSONAS) - 1}"
        )
    baseline_persona = PERSONAS[persona_idx]
    max_tokens = 200

    if dim == "model":
        # Pick two *different* models from the pool
        model_a = rng.choice(MODEL_POOL)
        remaining = [m for m in MODEL_POOL if m != model_a]
        model_b = rng.choice(remaining)
        config_a = {
            "model": model_a,
            "temperature": baseline_temp,
            "max_tokens": max_tokens,
            "styleInstruction": baseline_persona,
        }
        config_b = {
            "model": model_b,
            "temperature": baseline_temp,
            "max_tokens": max_tokens,
            "styleInstruction": baseline_persona,
        }

    elif dim == "temperature":
        # A = low temperature (focused), B = high temperature (creative)
        temp_a = round(rng.choice([0.0, 0.1, 0.2, 0.3]), 1)
        temp_b = round(rng.choice([0.6, 0.7, 0.8, 0.9]), 1)
        config_a = {
            "model": baseline_model,
            "temperature": temp_a,
            "max_tokens": max_tokens,
            "styleInstruction": baseline_persona,
        }
        config_b = {
            "model": baseline_model,
            "temperature": temp_b,
            "max_tokens": max_tokens,
            "styleInstruction": baseline_persona,
        }

    else:  # persona
        # Pick two *different* personas
        persona_a = rng.choice(PERSONAS)
        remaining_personas = [p for p in PERSONAS if p != persona_a]
        persona_b = rng.choice(remaining_personas)
        config_a = {
            "model": baseline_model,
            "temperature": baseline_temp,
            "max_tokens": max_tokens,
            "styleInstruction": persona_a,
        }
        config_b = {
            "model": baseline_model,
            "temperature": baseline_temp,
            "max_tokens": max_tokens,
            "styleInstruction": persona_b,
        }

    return config_a, config_b


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
        "query": query,
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
