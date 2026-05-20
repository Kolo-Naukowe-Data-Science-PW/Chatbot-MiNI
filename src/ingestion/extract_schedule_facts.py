"""
Specialized LLM fact extraction for USOS schedule pages (plany zajęć).

Unlike the generic extract_facts.py, this module uses a schedule-specific
system prompt that explicitly preserves all timetable details: times, rooms,
lecturer names, group numbers, semester codes.

Processes only schedule_*.txt files from scraped_raw/. Always re-extracts
(does not consult the progress tracker) so that re-running the schedule
pipeline always produces fresh, up-to-date facts.
"""

import json
import logging
import os

from src.ingestion.common import MODEL_WORKER, get_llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "src/data/scraped_raw"
OUTPUT_DIR = "src/data/facts"

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


def extract_schedule_facts(text: str, filename: str) -> list[str]:
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
            logger.error(f"JSON parse error in chunk {chunk_idx} of {filename}")
        except Exception as e:
            logger.error(f"API error in chunk {chunk_idx} of {filename}: {e}")

    return all_facts


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    schedule_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.startswith("schedule_") and f.endswith(".txt")
    )

    if not schedule_files:
        logger.warning(f"No schedule_*.txt files found in {INPUT_DIR}")
        return

    logger.info(f"Found {len(schedule_files)} schedule files to extract.")

    for filename in schedule_files:
        txt_path = os.path.join(INPUT_DIR, filename)
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}_facts.json")

        with open(txt_path, encoding="utf-8") as f:
            lines = f.readlines()

        if lines and lines[0].startswith("URL: "):
            source_url = lines[0].replace("URL: ", "").strip()
            text_content = "".join(lines[1:]).strip()
        else:
            source_url = filename
            text_content = "".join(lines).strip()

        if not text_content:
            logger.warning(f"Empty content: {filename} — skipping")
            continue

        logger.info(f"Extracting facts from: {filename}")
        facts = extract_schedule_facts(text_content, filename)

        if facts:
            structured = [{"source": source_url, "fact": f} for f in facts]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(structured, f, indent=2, ensure_ascii=False)
            logger.info(f"  → Saved {len(structured)} facts to {out_path}")
        else:
            logger.warning(f"  → No facts extracted from {filename}")

    logger.info("Schedule fact extraction complete.")


if __name__ == "__main__":
    main()
