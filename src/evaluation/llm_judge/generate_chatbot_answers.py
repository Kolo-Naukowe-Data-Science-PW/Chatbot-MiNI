"""
generate_chatbot_answers.py — Generate chatbot answers for a grid of (model × temperature × persona) configs.

Asks the running ``/chat`` API for every (question, model, temperature, persona)
combination and writes one row per call to the output CSV.  Downstream this is
consumed by ``single_judge_runner.py`` (multiple judges score the same answer).

Workflow
--------
1. Read questions from an evaluation CSV (accepts ``query`` / ``pytanie`` /
   ``question`` column names; carries through ``gold_link`` / ``strona`` if
   present).
2. For every (question × model × temperature × persona) combination, call the
   ``/chat`` API with the explicit ``modelConfig`` and store the answer.
3. Append to output CSV; resumable — skips combinations already present.

Persona indices match ``PERSONAS`` (same order as in ``testpro_runner.py``).

Usage
-----
    python -m evaluation.llm_judge.generate_chatbot_answers \\
        --input-csv  src/evaluation/data/questions_filtered.csv \\
        --output-csv src/data/feedback/chatbot_answers.csv \\
        --api-url    http://localhost:8000/chat \\
        --models     openai/gpt-5.5,anthropic/claude-opus-4.7,google/gemini-3.1-pro-preview-customtools \\
        --temperatures 0.0,0.5,0.9 \\
        --personas   3 \\
        --limit 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Persona styles available in the chatbot — index here MUST match the order
# in src/api/api.py::_EXPERIMENT_PERSONAS and llm_judge/testpro_runner.py::PERSONAS.
PERSONAS: list[str] = [
    "Odpowiedz bardzo krótko i konkretnie. Bez owijania w bawełnę.",
    "Odpowiedz luzno, prosto i przyjaźnie.",
    "Odpowiedz formalnie i akademicko, pełnymi zdaniami.",
    "Podaj wyczerpującą odpowiedź z detalami i przykładami.",
]

OUTPUT_FIELDNAMES = [
    "answer_id",
    "created_at",
    "query",
    "gold_link",
    "language",
    "gen_model",
    "gen_temperature",
    "gen_persona_idx",
    "gen_persona",
    "answer",
    "sources",
]


# ── Input CSV ───────────────────────────────────────────────────────────────

def _read_questions(input_csv: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Read questions from CSV; tolerate common column names (PL / EN); return list of dicts."""
    rows: list[dict[str, str]] = []
    with open(input_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            normalized = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
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


# ── Chat API call ───────────────────────────────────────────────────────────

def _post_chat(api_url: str, payload: dict, timeout_sec: int) -> dict:
    """POST JSON to /chat and return parsed JSON body. Raises RuntimeError on failure."""
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        api_url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat API HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Chat API request failed: {exc}") from exc


def call_chat(
    api_url: str,
    query: str,
    *,
    model: str,
    temperature: float,
    persona: str,
    language: str = "pl",
    max_tokens: int = 2048,
    timeout_sec: int = 180,
) -> tuple[str, list[str]]:
    """One /chat call with an explicit modelConfig; returns (answer, sources_list)."""
    payload = {
        "query": query,
        "language": language,
        "mode": "generation",
        "modelConfig": {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "styleInstruction": persona,
        },
    }
    response = _post_chat(api_url, payload, timeout_sec=timeout_sec)
    answer = (response.get("answer") or "").strip()
    sources = response.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    return answer, sources


# ── Output CSV ──────────────────────────────────────────────────────────────

def make_answer_id(query: str, model: str, temperature: float, persona_idx: int) -> str:
    """Stable short id derived from the inputs — same combo always yields the same id."""
    key = f"{query}|{model}|{temperature:.2f}|{persona_idx}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _read_existing_ids(output_csv: Path) -> set[str]:
    """Return the set of answer_id values already in the output CSV (for resume)."""
    if not output_csv.exists():
        return set()
    with open(output_csv, encoding="utf-8-sig", newline="") as f:
        return {row.get("answer_id", "") for row in csv.DictReader(f) if row.get("answer_id")}


# ── Main runner ─────────────────────────────────────────────────────────────

def run(
    input_csv: Path,
    output_csv: Path,
    api_url: str,
    models: list[str],
    temperatures: list[float],
    persona_indices: list[int],
    language: str = "pl",
    limit: int | None = None,
    sleep_between: float = 0.5,
    timeout_sec: int = 180,
) -> None:
    """Generate answers for the cartesian product of (questions × models × temps × personas)."""
    for idx in persona_indices:
        if not 0 <= idx < len(PERSONAS):
            raise ValueError(f"persona index {idx} out of range 0–{len(PERSONAS) - 1}")

    questions = _read_questions(input_csv, limit=limit)
    if not questions:
        logger.warning("Input CSV %s has no usable questions — nothing to do.", input_csv)
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing_ids(output_csv)
    file_existed = output_csv.exists()

    total = len(questions) * len(models) * len(temperatures) * len(persona_indices)
    todo_count = 0
    for q in questions:
        for m in models:
            for t in temperatures:
                for pi in persona_indices:
                    if make_answer_id(q["query"], m, t, pi) not in existing:
                        todo_count += 1

    logger.info(
        "Generating %d (questions) × %d (models) × %d (temps) × %d (personas) = %d combos "
        "(%d already done, %d to do).",
        len(questions), len(models), len(temperatures), len(persona_indices),
        total, total - todo_count, todo_count,
    )

    with open(output_csv, "a", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
        if not file_existed:
            writer.writeheader()

        done = 0
        for q in questions:
            for m in models:
                for t in temperatures:
                    for pi in persona_indices:
                        aid = make_answer_id(q["query"], m, t, pi)
                        if aid in existing:
                            continue
                        try:
                            answer, sources = call_chat(
                                api_url,
                                query=q["query"],
                                model=m,
                                temperature=t,
                                persona=PERSONAS[pi],
                                language=language,
                                timeout_sec=timeout_sec,
                            )
                        except RuntimeError as exc:
                            logger.error(
                                "FAILED aid=%s q=%r model=%s t=%s p=%d — skipping. %s",
                                aid, q["query"][:60], m, t, pi, exc,
                            )
                            continue

                        if not answer:
                            logger.warning("Empty answer for aid=%s — skipping write.", aid)
                            continue

                        writer.writerow({
                            "answer_id": aid,
                            "created_at": datetime.now(UTC).isoformat(),
                            "query": q["query"],
                            "gold_link": q.get("gold_link", ""),
                            "language": language,
                            "gen_model": m,
                            "gen_temperature": t,
                            "gen_persona_idx": pi,
                            "gen_persona": PERSONAS[pi],
                            "answer": answer,
                            "sources": json.dumps(sources, ensure_ascii=False),
                        })
                        out_f.flush()
                        done += 1
                        logger.info(
                            "[%d/%d] aid=%s model=%s t=%s p=%d (%d chars)",
                            done, todo_count, aid, m, t, pi, len(answer),
                        )
                        if sleep_between > 0:
                            time.sleep(sleep_between)

    logger.info("Done. Wrote %d new answers to %s.", done, output_csv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="Evaluation questions CSV (cols: pytanie/query/question [, strona/url]).")
    parser.add_argument("--output-csv", required=True, type=Path,
                        help="Output CSV: one row per (question × model × temperature × persona).")
    parser.add_argument("--api-url", default="http://localhost:8000/chat",
                        help="Chatbot /chat endpoint (use SSH tunnel for VM API).")
    parser.add_argument("--models", required=True,
                        help="Comma-separated list of OpenRouter model ids to use as generators.")
    parser.add_argument("--temperatures", required=True,
                        help="Comma-separated list of float temperatures (e.g. 0.0,0.5,0.9).")
    parser.add_argument("--personas", required=True,
                        help=f"Comma-separated list of persona indices 0–{len(PERSONAS) - 1}.")
    parser.add_argument("--language", default="pl",
                        help="Language code passed to /chat (default: pl).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N input questions (smoke test).")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between API calls (rate-limit hint).")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Per-call /chat timeout in seconds (default: 180).")
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        parser.error("--models must contain at least one entry")

    try:
        temperatures = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    except ValueError as exc:
        parser.error(f"--temperatures must be comma-separated floats: {exc}")
    if not temperatures:
        parser.error("--temperatures must contain at least one entry")

    try:
        persona_indices = [int(p.strip()) for p in args.personas.split(",") if p.strip()]
    except ValueError as exc:
        parser.error(f"--personas must be comma-separated integers: {exc}")
    if not persona_indices:
        parser.error("--personas must contain at least one entry")

    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        api_url=args.api_url,
        models=models,
        temperatures=temperatures,
        persona_indices=persona_indices,
        language=args.language,
        limit=args.limit,
        sleep_between=args.sleep,
        timeout_sec=args.timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
