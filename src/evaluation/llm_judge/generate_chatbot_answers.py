"""
generate_chatbot_answers.py — Generate chatbot answers for a grid of (model × temperature × persona) configs.

Asks the running ``/chat`` API for every (question, model, temperature, persona)
combination and writes one row per call to the output CSV.  Downstream this is
consumed by ``single_judge_runner.py`` (multiple judges score the same answer).

Workflow
--------
1. Read questions from an evaluation CSV (accepts ``query`` / ``pytanie`` /
   ``question`` column names; carries through ``gold_link`` / ``strona`` if
   present) or JSONL (accepts ``Pytanie`` / ``Pytanie:`` / ``query`` /
   ``question`` keys).
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

    # Compact output for NotebookLM JSONL questions:
    python -m evaluation.llm_judge.generate_chatbot_answers \\
        --input-csv  src/evaluation/data/final_notebooklm_QA.jsonl \\
        --output-csv src/data/feedback/answers_A_opus.csv \\
        --api-url    http://localhost:8000/chat \\
        --models     anthropic/claude-opus-4.8 \\
        --temperatures 0.2 \\
        --personas   2 \\
        --output-format answers_and_links
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

ANSWER_ONLY_FIELDNAMES = ["pytanie", "odpowiedz_wygenerowana"]
LINKS_ONLY_FIELDNAMES = ["pytanie", "zwrocone_linki"]
ANSWERS_AND_LINKS_FIELDNAMES = [
    "pytanie",
    "odpowiedz_wygenerowana",
    "zwrocone_linki",
]

COMPACT_OUTPUT_FORMATS = {
    "simple",
    "polish",
    "answer_only",
    "links_only",
    "answers_and_links",
}

ERROR_ANSWER_TEXT = "Sorry, I encountered an error while generating the response."
ERROR_ANSWER_RETRIES = 2


# ── Input data ──────────────────────────────────────────────────────────────


def _read_csv_questions(
    input_csv: Path, limit: int | None = None
) -> list[dict[str, str]]:
    """Read questions from CSV; tolerate common column names (PL / EN)."""
    rows: list[dict[str, str]] = []
    with open(input_csv, encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        f.seek(0)
        delimiter = "|" if "|" in header else ","
        reader = csv.DictReader(f, delimiter=delimiter)
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


def _read_jsonl_questions(
    input_jsonl: Path, limit: int | None = None
) -> list[dict[str, str]]:
    """Read questions from JSONL; tolerate NotebookLM-style PL keys."""
    rows: list[dict[str, str]] = []
    with open(input_jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping invalid JSONL line %d in %s: %s",
                    line_no,
                    input_jsonl,
                    exc,
                )
                continue

            query = (
                record.get("Pytanie")
                or record.get("Pytanie:")
                or record.get("pytanie")
                or record.get("query")
                or record.get("question")
                or ""
            )
            query = str(query).strip()
            if not query:
                continue

            gold_link = (
                record.get("Źródła")
                or record.get("Zrodla")
                or record.get("gold_link")
                or record.get("strona")
                or ""
            )
            rows.append({"query": query, "gold_link": str(gold_link).strip()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _read_questions(input_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Read questions from CSV or JSONL."""
    if input_path.suffix.lower() == ".jsonl":
        return _read_jsonl_questions(input_path, limit=limit)
    return _read_csv_questions(input_path, limit=limit)


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
        return {
            row.get("answer_id", "")
            for row in csv.DictReader(f)
            if row.get("answer_id")
        }


def _read_existing_simple_questions(output_csv: Path) -> set[str]:
    """Return questions already present in compact output."""
    if not output_csv.exists():
        return set()
    with open(output_csv, encoding="utf-8-sig", newline="") as f:
        return {
            row.get("pytanie", "").strip()
            for row in csv.DictReader(f)
            if row.get("pytanie")
        }


def _compact_required_columns(output_format: str) -> list[str]:
    required = ["pytanie"]
    if output_format in ("simple", "polish", "answer_only"):
        required.append("odpowiedz_wygenerowana")
    if output_format in ("polish", "links_only", "answers_and_links"):
        required.append("zwrocone_linki")
    return required


def _compact_row_is_complete(row: dict[str, str], output_format: str) -> bool:
    generated_answer = (row.get("odpowiedz_wygenerowana") or "").strip()
    if generated_answer == ERROR_ANSWER_TEXT:
        return False
    return all(
        (row.get(column) or "").strip()
        for column in _compact_required_columns(output_format)
    )


def _prepare_compact_output_for_resume(
    output_csv: Path,
    output_format: str,
    fieldnames: list[str],
    *,
    rewrite_incomplete: bool = True,
) -> set[str]:
    """
    Keep only rows that are complete for the selected compact format.

    This lets a later answers_and_links run refill rows created earlier by
    answer_only without losing rows that already have returned links.
    """
    if not output_csv.exists():
        return set()

    existing_fieldnames, rows = _read_compact_rows_for_resume(
        output_csv,
        output_format,
        fieldnames,
    )

    complete_rows: list[dict[str, str]] = []
    complete_questions: set[str] = set()
    for row in rows:
        question = (row.get("pytanie") or "").strip()
        if not question or question in complete_questions:
            continue
        if not _compact_row_is_complete(row, output_format):
            continue
        complete_rows.append({column: row.get(column, "") for column in fieldnames})
        complete_questions.add(question)

    needs_rewrite = (
        existing_fieldnames != fieldnames
        or len(complete_rows) != len(rows)
    )
    if needs_rewrite and rewrite_incomplete:
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(complete_rows)
        logger.info(
            "Prepared %s for %s resume: kept %d complete rows, removed %d incomplete/duplicate rows.",
            output_csv,
            output_format,
            len(complete_rows),
            len(rows) - len(complete_rows),
        )
    elif needs_rewrite:
        logger.info(
            "Prepared %s for %s resume: found %d complete rows, %d rows will be regenerated.",
            output_csv,
            output_format,
            len(complete_rows),
            len(rows) - len(complete_rows),
        )

    return complete_questions


def _read_compact_rows_for_resume(
    output_csv: Path,
    output_format: str,
    fieldnames: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    with open(output_csv, encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.reader(f))

    if not raw_rows:
        return fieldnames, []

    header = raw_rows[0]
    data_rows = raw_rows[1:]
    max_width = max(
        [len(header), *(len(row) for row in data_rows)],
        default=len(header),
    )

    if (
        output_format in ("polish", "answers_and_links")
        and header == ANSWER_ONLY_FIELDNAMES
        and max_width == len(ANSWERS_AND_LINKS_FIELDNAMES)
    ):
        header = ANSWERS_AND_LINKS_FIELDNAMES.copy()

    while len(header) < max_width:
        header.append(f"extra_{len(header) + 1}")

    rows = []
    for row in data_rows:
        padded = [*row, *([""] * (len(header) - len(row)))]
        rows.append(dict(zip(header, padded, strict=False)))

    return header, rows


def _detect_existing_compact_format(output_csv: Path) -> str | None:
    """Infer compact output format from an existing CSV header."""
    if not output_csv.exists():
        return None

    with open(output_csv, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), None)

    if header == ANSWERS_AND_LINKS_FIELDNAMES:
        return "answers_and_links"
    if header == ANSWER_ONLY_FIELDNAMES:
        return "answer_only"
    if header == LINKS_ONLY_FIELDNAMES:
        return "links_only"
    return None


def _validate_compact_config(
    output_format: str,
    models: list[str],
    temperatures: list[float],
    persona_indices: list[int],
) -> None:
    if output_format not in COMPACT_OUTPUT_FORMATS:
        return
    if len(models) == 1 and len(temperatures) == 1 and len(persona_indices) == 1:
        return
    raise ValueError(
        f"--output-format {output_format} supports exactly one model, one temperature, "
        "and one persona per output CSV. Run the script once per model."
    )


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
    output_format: str = "full",
) -> None:
    """Generate answers for the cartesian product of (questions × models × temps × personas)."""
    for idx in persona_indices:
        if not 0 <= idx < len(PERSONAS):
            raise ValueError(f"persona index {idx} out of range 0–{len(PERSONAS) - 1}")

    _validate_compact_config(output_format, models, temperatures, persona_indices)

    questions = _read_questions(input_csv, limit=limit)
    if not questions:
        logger.warning(
            "Input CSV %s has no usable questions — nothing to do.", input_csv
        )
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "full":
        detected_output_format = _detect_existing_compact_format(output_csv)
        if detected_output_format:
            logger.info(
                "Detected existing compact %s CSV at %s; resuming in that format.",
                detected_output_format,
                output_csv,
            )
            output_format = detected_output_format
    _validate_compact_config(output_format, models, temperatures, persona_indices)

    fieldnames = _fieldnames_for_output_format(output_format)
    same_input_output = input_csv.resolve() == output_csv.resolve()
    existing_ids = _read_existing_ids(output_csv) if output_format == "full" else set()
    existing_questions = (
        _prepare_compact_output_for_resume(
            output_csv,
            output_format,
            fieldnames,
            rewrite_incomplete=not same_input_output,
        )
        if output_format in COMPACT_OUTPUT_FORMATS
        else set()
    )
    file_existed = output_csv.exists()

    total = len(questions) * len(models) * len(temperatures) * len(persona_indices)
    todo_count = 0
    for q in questions:
        for m in models:
            for t in temperatures:
                for pi in persona_indices:
                    if output_format in COMPACT_OUTPUT_FORMATS:
                        if q["query"] not in existing_questions:
                            todo_count += 1
                    elif make_answer_id(q["query"], m, t, pi) not in existing_ids:
                        todo_count += 1

    logger.info(
        "Generating %d (questions) × %d (models) × %d (temps) × %d (personas) = %d combos "
        "(%d already done, %d to do).",
        len(questions),
        len(models),
        len(temperatures),
        len(persona_indices),
        total,
        total - todo_count,
        todo_count,
    )

    with open(output_csv, "a", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_existed:
            writer.writeheader()

        done = 0
        for q in questions:
            for m in models:
                for t in temperatures:
                    for pi in persona_indices:
                        aid = make_answer_id(q["query"], m, t, pi)
                        if output_format in COMPACT_OUTPUT_FORMATS:
                            if q["query"] in existing_questions:
                                continue
                        elif aid in existing_ids:
                            continue
                        answer = ""
                        sources: list[str] = []
                        for attempt in range(ERROR_ANSWER_RETRIES + 1):
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
                                    aid,
                                    q["query"][:60],
                                    m,
                                    t,
                                    pi,
                                    exc,
                                )
                                break

                            if answer and answer != ERROR_ANSWER_TEXT:
                                break

                            if attempt < ERROR_ANSWER_RETRIES:
                                retry_reason = (
                                    "empty answer"
                                    if not answer
                                    else "generator error response"
                                )
                                logger.warning(
                                    "Retrying aid=%s q=%r after %s (%d/%d).",
                                    aid,
                                    q["query"][:60],
                                    retry_reason,
                                    attempt + 1,
                                    ERROR_ANSWER_RETRIES,
                                )
                                if sleep_between > 0:
                                    time.sleep(sleep_between)

                        if not answer:
                            if output_format in ("links_only", "answers_and_links"):
                                logger.warning(
                                    "Empty answer for aid=%s after retries; writing row with %d sources for retrieval metrics.",
                                    aid,
                                    len(sources),
                                )
                            else:
                                logger.warning(
                                    "Empty answer for aid=%s — skipping write.", aid
                                )
                                continue
                        if answer == ERROR_ANSWER_TEXT:
                            logger.warning(
                                "Generator returned error response for aid=%s after retries — skipping write.",
                                aid,
                            )
                            continue

                        if output_format in COMPACT_OUTPUT_FORMATS:
                            sources_with_rank = [
                                {"rank": i + 1, "url": src}
                                for i, src in enumerate(sources)
                            ]
                            row = {"pytanie": q["query"]}
                            if output_format in ("simple", "polish", "answer_only", "answers_and_links"):
                                row["odpowiedz_wygenerowana"] = answer
                            if output_format in ("polish", "links_only", "answers_and_links"):
                                row["zwrocone_linki"] = json.dumps(
                                    sources_with_rank,
                                    ensure_ascii=False,
                                )
                            writer.writerow(
                                row
                            )
                            existing_questions.add(q["query"])
                        else:
                            writer.writerow(
                                {
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
                                }
                            )
                            existing_ids.add(aid)
                        out_f.flush()
                        done += 1
                        logger.info(
                            "[%d/%d] aid=%s model=%s t=%s p=%d (%d chars)",
                            done,
                            todo_count,
                            aid,
                            m,
                            t,
                            pi,
                            len(answer),
                        )
                        if sleep_between > 0:
                            time.sleep(sleep_between)

    logger.info("Done. Wrote %d new answers to %s.", done, output_csv)
    if output_format in COMPACT_OUTPUT_FORMATS and same_input_output:
        _prepare_compact_output_for_resume(
            output_csv,
            output_format,
            fieldnames,
            rewrite_incomplete=True,
        )


def _fieldnames_for_output_format(output_format: str) -> list[str]:
    if output_format in ("simple", "answer_only"):
        return ANSWER_ONLY_FIELDNAMES
    if output_format == "links_only":
        return LINKS_ONLY_FIELDNAMES
    if output_format in ("polish", "answers_and_links"):
        return ANSWERS_AND_LINKS_FIELDNAMES
    return OUTPUT_FIELDNAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="Evaluation questions CSV (cols: pytanie/query/question [, strona/url]).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Output CSV: one row per (question × model × temperature × persona).",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/chat",
        help="Chatbot /chat endpoint (use SSH tunnel for VM API).",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of OpenRouter model ids to use as generators.",
    )
    parser.add_argument(
        "--temperatures",
        required=True,
        help="Comma-separated list of float temperatures (e.g. 0.0,0.5,0.9).",
    )
    parser.add_argument(
        "--personas",
        required=True,
        help=f"Comma-separated list of persona indices 0–{len(PERSONAS) - 1}.",
    )
    parser.add_argument(
        "--language", default="pl", help="Language code passed to /chat (default: pl)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input questions (smoke test).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls (rate-limit hint).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-call /chat timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--output-format",
        choices=[
            "full",
            "simple",
            "polish",
            "answer_only",
            "links_only",
            "answers_and_links",
        ],
        default="full",
        help=(
            "full = metadata-rich CSV; answer_only/simple = pytanie + answer; "
            "links_only = pytanie + returned links; answers_and_links/polish = "
            "pytanie + answer + returned links (default: full)."
        ),
    )
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        parser.error("--models must contain at least one entry")

    try:
        temperatures = [
            float(t.strip()) for t in args.temperatures.split(",") if t.strip()
        ]
    except ValueError as exc:
        parser.error(f"--temperatures must be comma-separated floats: {exc}")
    if not temperatures:
        parser.error("--temperatures must contain at least one entry")

    try:
        persona_indices = [
            int(p.strip()) for p in args.personas.split(",") if p.strip()
        ]
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
        output_format=args.output_format,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
