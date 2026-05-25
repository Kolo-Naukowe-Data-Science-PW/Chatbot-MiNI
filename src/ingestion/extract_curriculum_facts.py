"""
Specialized LLM fact extraction for study programme documents (plany studiów).

Unlike the generic extract_facts.py, this module uses a curriculum-specific
system prompt that extracts each subject as a separate fact, preserving
semester number, ECTS points, course type (W/Ć/L/P), and exam info.

Processes only scraped_raw files whose source URL matches the plan-studiow
pattern. Output files are prefixed with 'curriculum_' in src/data/facts/
and are subsequently picked up by ingest_facts.py.
"""

import json
import logging
import os

from src.ingestion.common import MODEL_WORKER, get_llm_client
from src.ingestion.progress import is_facts_extracted, mark_facts_extracted

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = os.environ.get("SCRAPED_RAW_DIR", "src/data/scraped_raw")
OUTPUT_DIR = os.environ.get("FACTS_DIR", "src/data/facts")

_CURRICULUM_URL_PATTERNS = ["plan-studiow", "Plan-studiow"]

CURRICULUM_SYSTEM_PROMPT = """
Jesteś ekspertem od programów studiów Wydziału MiNI PW (Matematyki i Nauk Informacyjnych, Politechnika Warszawska).

Twoim zadaniem jest przetworzenie dokumentu "Plan studiów" na WYCZERPUJĄCĄ listę faktów.
Każdy fakt musi być kompletnym, samodzielnym zdaniem zawierającym WSZYSTKIE dostępne informacje.

KRYTYCZNE ZASADY — dla każdego przedmiotu utwórz osobny fakt zawierający:
1. Pełną nazwę kierunku studiów i stopień (inżynierski/magisterski/licencjacki).
2. Numer semestru (cyfra arabska, np. "semestr 5").
3. Pełną nazwę przedmiotu — DOKŁADNIE tak jak w dokumencie, bez skrótów.
4. Liczbę punktów ECTS — ZAWSZE, jeśli dostępna.
5. Typ zajęć (wykład W, ćwiczenia Ć, laboratorium L, projekt P, seminarium S).
6. Liczbę godzin poszczególnych typów zajęć (jeśli podana).
7. Formę zaliczenia (egzamin E lub zaliczenie Zal/Z).

ZASADY DOTYCZĄCE TEGO, CZEGO NIE ROBIĆ:
- NIE twórz faktów zbiorczych o łącznej liczbie ECTS semestru ani łącznej liczbie godzin semestru.
- NIE pomijaj żadnego przedmiotu z dokumentu.
- NIE łącz dwóch przedmiotów w jeden fakt.
- Każdy przedmiot = osobny, niezależny fakt.

Przykład DOBREGO faktu:
"Na kierunku Inżynieria i Analiza Danych (IAD, inżynierski) w semestrze 5 jest przedmiot
'Uczenie maszynowe 2' wart 5 ECTS, prowadzony jako wykład (2h/tyg.) i laboratorium (2h/tyg.),
kończący się egzaminem."

Przykład ZŁEGO faktu (brak szczegółów):
"W semestrze 5 jest przedmiot z uczenia maszynowego."

Przykład ZŁEGO faktu (zbiorczy):
"W semestrze 5 IAD jest łącznie 30 ECTS i 11 godzin wykładów."

Jeśli dokument zawiera wiersze podsumowań semestrów (np. „Razem", „Total", łączne ECTS/godziny) — POMIŃ je.
Ekstrakcja dotyczy wyłącznie konkretnych przedmiotów.

Format odpowiedzi: TYLKO czysty JSON: ["fakt1", "fakt2", ...].
Bez bloków markdown (```), bez komentarzy, bez wyjaśnień.

Oczekiwana liczba faktów: minimum 1 fakt na każdy przedmiot w planie studiów.
Typowy plan ma 40–70 przedmiotów — jeśli wyciągnąłeś mniej, wróć do tekstu i sprawdź ponownie.
"""


def _is_curriculum_url(url: str) -> bool:
    return any(pattern in url for pattern in _CURRICULUM_URL_PATTERNS)


def _extract_facts_from_text(text: str, filename: str) -> list[str]:
    client = get_llm_client()
    chunk_size = 8000
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
    all_facts: list[str] = []

    for chunk_idx, chunk in enumerate(chunks):
        try:
            response = client.chat.completions.create(
                model=MODEL_WORKER,
                messages=[
                    {"role": "system", "content": CURRICULUM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Dane z planu studiów:\n{chunk}"},
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

    curriculum_files: list[tuple[str, str, str]] = []
    for filename in sorted(os.listdir(INPUT_DIR)):
        if not filename.endswith(".txt"):
            continue
        txt_path = os.path.join(INPUT_DIR, filename)
        try:
            with open(txt_path, encoding="utf-8") as f:
                first_line = f.readline().strip()
        except Exception:
            continue
        if first_line.startswith("URL: "):
            url = first_line[5:].strip()
            if _is_curriculum_url(url):
                curriculum_files.append((filename, url, txt_path))

    if not curriculum_files:
        logger.warning(f"No curriculum files found in {INPUT_DIR}")
        return

    logger.info(f"Found {len(curriculum_files)} curriculum files to process.")

    for filename, source_url, txt_path in curriculum_files:
        base = os.path.splitext(filename)[0]
        out_name = f"curriculum_{base}_facts.json"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if is_facts_extracted(out_name):
            logger.info(f"SKIP (already extracted): {filename}")
            continue

        with open(txt_path, encoding="utf-8") as f:
            lines = f.readlines()
        text_content = "".join(lines[1:]).strip()

        if not text_content:
            logger.warning(f"Empty content: {filename} — skipping")
            continue

        logger.info(f"Extracting facts from: {filename}")
        facts = _extract_facts_from_text(text_content, filename)

        if facts:
            structured = [{"source": source_url, "fact": f} for f in facts]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(structured, f, indent=2, ensure_ascii=False)
            mark_facts_extracted(out_name)
            logger.info(f"  → Saved {len(structured)} facts to {out_name}")
        else:
            logger.warning(f"  → No facts extracted from {filename}")

    logger.info("Curriculum fact extraction complete.")


if __name__ == "__main__":
    main()
