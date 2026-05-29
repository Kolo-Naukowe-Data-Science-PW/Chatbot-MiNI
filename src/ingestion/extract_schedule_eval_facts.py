"""
Extract facts from Firecrawl-scraped USOS schedule files for eval subset.

These are URL-named files (e.g. usosweb.usos.pw.edu.pl_...pokazPlanGrupyPrzedmiotow...txt)
scraped by Firecrawl. extract_facts.py intentionally skips them to avoid duplication
with the schedule pipeline. This module handles them explicitly for the schedule eval.

Uses the same schedule-specialized LLM prompt as extract_schedule_facts.py.

Usage:
    python -m src.ingestion.extract_schedule_eval_facts \\
        --scraped-dir /app/src/data/scraped_schedule_eval_subset \\
        --output-dir /app/src/data/facts_schedule_eval_subset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.common import MODEL_WORKER, get_llm_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SCHEDULE_SYSTEM_PROMPT = """
Jesteś ekspertem od planów zajęć Wydziału MiNI PW (Matematyki i Nauk Informacyjnych, Politechnika Warszawska).

Twoim zadaniem jest przetworzenie danych z planu zajęć na WYCZERPUJĄCĄ listę faktów.
Każdy fakt musi być kompletnym, samodzielnym zdaniem zawierającym WSZYSTKIE dostępne informacje.

KRYTYCZNE ZASADY — NIGDY nie pomijaj:
1. Dokładne godziny zajęć (np. "od 8:15 do 10:00") — ZAWSZE podawaj obie godziny.
2. Numer sali i budynek (np. "sala 222 w budynku MiNI") — ZAWSZE, jeśli dostępny.
3. Pełne imię i nazwisko prowadzącego — ZAWSZE, jeśli dostępne.
4. Numer grupy lub typ grupy (wykład/ćwiczenia/laboratorium/projekt).
5. Kierunek studiów, semestr i rok akademicki (np. "ISI inżynierski, semestr 1, rok 2025/2026, semestr zimowy").
6. Dzień tygodnia.

NIGDY nie stosuj placeholderów ani skrótów zamiast konkretnych danych.
ZAWSZE podawaj liczby dokładnie tak jak w źródle.

Przykład DOBREGO faktu:
"Na kierunku Informatyka i Systemy Informacyjne (inżynierskim, ISI), semestr 1,
rok akademicki 2025/2026 (semestr zimowy), w poniedziałek od 8:15 do 10:00
odbywa się Analiza Matematyczna 1 (wykład, grupa 1) w sali 222 budynku MiNI,
prowadzony przez dr. Jana Kowalskiego."

Przykład ZŁEGO faktu (za mało szczegółów):
"Jest wykład z matematyki w poniedziałek rano."

Format odpowiedzi: TYLKO czysty JSON: ["fakt1", "fakt2", ...].
Bez bloków markdown (```), bez komentarzy, bez wyjaśnień.

Oczekiwana liczba faktów: 30–150 na plan grupy. Jeśli wyciągnąłeś mniej — sprawdź ponownie.
"""


def _extract_facts(text: str, filename: str) -> list[str]:
    client = get_llm_client()
    chunk_size = 8000
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    all_facts: list[str] = []

    for chunk_idx, chunk in enumerate(chunks):
        try:
            response = client.chat.completions.create(
                model=MODEL_WORKER,
                messages=[
                    {"role": "system", "content": SCHEDULE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Dane z planu zajęć:\n{chunk}"},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            if isinstance(parsed, list):
                all_facts.extend(str(f) for f in parsed if f)
        except json.JSONDecodeError:
            logger.error("JSON parse error in chunk %d of %s", chunk_idx, filename)
        except Exception as exc:
            logger.error("API error in chunk %d of %s: %s", chunk_idx, filename, exc)

    return all_facts


def main(scraped_dir: str, output_dir: str, force: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(scraped_dir) if f.endswith(".txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", scraped_dir)
        return

    logger.info("Found %d .txt files in %s", len(txt_files), scraped_dir)

    for filename in txt_files:
        path = os.path.join(scraped_dir, filename)
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{base}_facts.json")

        if not force and os.path.exists(out_path):
            logger.info("SKIP (already extracted): %s", filename)
            continue

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        if lines and lines[0].startswith("URL: "):
            source_url = lines[0].replace("URL: ", "").strip()
            text_content = "".join(lines[1:]).strip()
        else:
            source_url = filename
            text_content = "".join(lines).strip()

        if not text_content:
            logger.warning("Empty content: %s — skipping", filename)
            continue

        logger.info("Extracting facts from: %s", filename)
        facts = _extract_facts(text_content, filename)

        if facts:
            structured = [{"source": source_url, "fact": f} for f in facts]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(structured, f, indent=2, ensure_ascii=False)
            logger.info("  Saved %d facts -> %s", len(structured), out_path)
        else:
            logger.warning("  No facts extracted from %s", filename)

    logger.info("Extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract schedule facts from Firecrawl-scraped USOS schedule files."
    )
    parser.add_argument("--scraped-dir", required=True, help="Directory with .txt schedule files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write *_facts.json files.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if output already exists.")
    args = parser.parse_args()
    main(args.scraped_dir, args.output_dir, force=args.force)
