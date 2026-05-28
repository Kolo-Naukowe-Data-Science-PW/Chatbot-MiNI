import json
import os

from src.ingestion.common import (
    CURRENT_VERSION,
    MODEL_WORKER,
    get_config,
    get_llm_client,
    logger,
)
from src.ingestion.progress import is_facts_extracted, mark_facts_extracted

config = get_config()

INPUT_DIR = "src/data/processed_text"
OUTPUT_DIR = os.environ.get("FACTS_DIR", "src/data/facts")
SCRAPED_RAW_DIR = os.environ.get("SCRAPED_RAW_DIR", "src/data/scraped_raw")

SYSTEM_PROMPT = """
    Jesteś inteligentnym asystentem z Wydziału MiNI PW, który pomaga wyodrębniać fakty z różnych dokumentów.
    Twoim zadaniem jest przetworzenie tekstu na WYCZERPUJĄCĄ listę faktów — wyciągnij ABSOLUTNIE WSZYSTKIE informacje zawarte w tekście.

    KRYTYCZNIE WAŻNE: NIE SELEKCJONUJ. Nie wybieraj tylko "ważnych" czy "kluczowych" informacji.
    Wyciągnij KAŻDĄ informację z tekstu — wszystkie, bez wyjątku, niezależnie od tego, czy wydają się istotne.

    ZASADY:
    1. Ignoruj nagłówki, stopki, elementy nawigacji, reklamy i menu.
    2. Wyciągnij WSZYSTKIE informacje: kto, co, gdzie, kiedy, ile, jak długo, jakie warunki — każdą, dosłownie każdą.
    3. NIE pomijaj żadnych szczegółów — nawet pozornie drobnych (numery paragrafów, terminy, warunki, wyjątki, daty, godziny, numery sal, nazwiska, liczby).
    4. Każdy fakt musi być SAMODZIELNYM, PEŁNYM zdaniem zrozumiałym bez kontekstu.
       Dobry przykład: "Egzamin dyplomowy na Wydziale MiNI PW musi odbyć się w ciągu 3 miesięcy od złożenia pracy."
       Zły przykład: "3 miesiące od złożenia pracy."
    5. Jeśli fakt dotyczy konkretnego kierunku, roku lub semestru — zawrzyj tę informację w zdaniu.
    6. Długie listy (np. lista przedmiotów, lista warunków) — każdy element to osobny fakt.
    7. Odpowiedź zwróć TYLKO jako czysty JSON: ["fakt1", "fakt2", ...]. Bez bloków kodu markdown, bez komentarzy.

    WAŻNE: Lepiej wyciągnąć za dużo faktów niż za mało. Dla typowej strony opisującej kierunek studiów
    oczekiwane jest minimum 40–80 faktów. Dla regulaminu lub procedury — minimum 50–150 faktów.
    Jeśli wyciągnęłeś mniej — prawdopodobnie coś pominąłeś. Wróć do tekstu i sprawdź ponownie.
"""


def extract_facts_list(text: str, filename: str) -> list[str]:
    """
    Decides whether to use LLM or return raw text based on config.

    With LLM, it retrieves a list of fact strings from the provided text using the LLM client.
    Each fact is expected to be a complete sentence extracted from the text.

    Parameters
    ----------
    text : str
        The raw input text from which to extract facts.
    filename : str
        The name of the file being processed (used for error logging).

    Returns
    -------
    list[str]
        A list of strings, where each string is a fact (or the whole text if LLM is disabled).
    """
    if not config["use_llm_for_facts"]:
        return [text.strip()]

    client = get_llm_client()

    try:
        chunk_size = 8000
        text_chunks = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]

        raw_facts_strings = []

        for chunk in text_chunks:
            response = client.chat.completions.create(
                model=MODEL_WORKER,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Tekst:\n{chunk}"},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")

            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    raw_facts_strings.extend(parsed)
            except json.JSONDecodeError:
                logger.error(f"Error processing: {filename}")

        return raw_facts_strings

    except Exception as e:
        logger.error(f"Error API for {filename}: {e}")
        return []


def main(force: bool = False) -> None:
    """
    Main function to extract facts from text files and save them as structured JSON.

    Processes all .txt files in the input folders, extracts facts using the LLM (if configured),
    and saves them in the output directory with associated source URLs.

    Parameters
    ----------
    force : bool
        If True, skip the progress-tracker check and re-extract all files.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_folders = [SCRAPED_RAW_DIR, "src/data/processed_text"]

    mode_info = (
        "LLM extraction" if config["use_llm_for_facts"] else "Raw text passthrough"
    )
    logger.info(f"Starting extraction. Version: {CURRENT_VERSION} | Mode: {mode_info}")

    for folder in input_folders:

        if not os.path.exists(folder):
            logger.warning(f"Input folder does not exist: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".txt")]

        for txt_file in files:
            # Skip files handled by dedicated specialized extractors
            if txt_file.startswith("schedule_"):
                logger.info(f"SKIP (handled by schedule pipeline): {txt_file}")
                continue

            base_name = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(folder, txt_file)

            with open(txt_path, encoding="utf-8") as f:
                lines = f.readlines()

            if lines and lines[0].startswith("URL: "):
                source_url = lines[0].replace("URL: ", "").strip()
                text_content = "".join(lines[1:]).strip()
            else:
                source_url = txt_file  # Fallback to filename
                text_content = "".join(lines).strip()

            # Skip USOS schedule pages regardless of filename (Firecrawl-scraped)
            if "pokazPlanGrupyPrzedmiotow" in source_url:
                logger.info(
                    f"SKIP (USOS schedule URL, handled by schedule pipeline): {txt_file}"
                )
                continue

            # Skip study programme PDFs (handled by curriculum pipeline)
            if "plan-studiow" in source_url or "Plan-studiow" in source_url:
                logger.info(
                    f"SKIP (plan studiów, handled by curriculum pipeline): {txt_file}"
                )
                continue

            # Check if there is a metadata file that should override the URL
            meta_path = os.path.join(INPUT_DIR, f"{txt_file.replace('.txt', '.json')}")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(INPUT_DIR, f"{base_name}.json")

            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    source_url = meta.get("source_url", source_url)

            facts_filename = f"{base_name}_facts.json"
            out_path = os.path.join(OUTPUT_DIR, facts_filename)

            if not force and (
                is_facts_extracted(facts_filename) or os.path.exists(out_path)
            ):
                logger.info(f"SKIP (already extracted): {txt_file}")
                if not is_facts_extracted(facts_filename):
                    mark_facts_extracted(facts_filename)
                continue

            logger.info(f"Processing: {txt_file} (Source: {source_url})")

            content_list = extract_facts_list(text_content, txt_file)

            if content_list:
                structured_output = [
                    {"source": source_url, "fact": item} for item in content_list
                ]

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(structured_output, f, indent=2, ensure_ascii=False)

                mark_facts_extracted(facts_filename)
                logger.info(f"Saved {len(structured_output)} facts → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract all files, ignoring the progress tracker.",
    )
    args = parser.parse_args()
    main(force=args.force)
