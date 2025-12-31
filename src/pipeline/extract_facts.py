import json
import os

from pipeline.common import (
    CURRENT_VERSION,
    MODEL_WORKER,
    get_config,
    get_llm_client,
    logger,
)

config = get_config()

INPUT_DIR = "src/data/processed_text"
OUTPUT_DIR = "src/data/facts"

SYSTEM_PROMPT = """
    Jesteś inteligentnym asystentem z Wydziału MiNI PW, który pomaga wyodrębniać fakty z różnych dokumentów.
    Cechujesz się szczegółowością i precyzją.
    Twoim zadaniem jest przetworzenie tekstu na listę faktów.
    1. Ignoruj nagłówki, stopki, reklamy, menu.
    2. Wyciągnij konkretne informacje: kto, co, gdzie, kiedy.
    3. Każdy fakt musi być PEŁNYM zdaniem (np. "Dziekanem Wydziału MiNI jest prof. dr hab. Grzegorz Świątek", a nie "Grzegorz Świątek").
    4. Odpowiedź zwróć TYLKO jako czysty JSON: ["fakt1", "fakt2"]. Bez bloków kodu markdown.
    Pamiętaj, że twoim celem jest uzyskanie jak największej liczby szczegółowych faktów z dostarczonego tekstu.
    Poszczególne dokumenty mogą mieć różną strukturę i styl, więc dostosuj swoje podejście odpowiednio.
    W zależności od dokumentu, liczby faktów mogą się bardzo różnić.
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
        chunk_size = 15000
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


def main() -> None:
    """
    Main function to extract facts from text files and save them as structured JSON.

    Processes all .txt files in the input folders, extracts facts using the LLM (if configured),
    and saves them in the output directory with associated source URLs.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_folders = ["src/data/scraped_raw", "src/data/processed_text"]

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
            base_name = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(folder, txt_file)

            # Logic: Look for metadata JSON in INPUT_DIR regardless of source folder
            meta_path = os.path.join(INPUT_DIR, f"{txt_file.replace('.txt', '.json')}")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(INPUT_DIR, f"{base_name}.json")

            source_url = txt_file
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    source_url = meta.get("source_url", txt_file)

            logger.info(f"Processing: {txt_file} (Source: {source_url})")

            with open(txt_path, encoding="utf-8") as f:
                text_content = f.read()

            content_list = extract_facts_list(text_content, txt_file)

            if content_list:
                structured_output = [
                    {"source": source_url, "fact": item} for item in content_list
                ]

                out_path = os.path.join(OUTPUT_DIR, f"{base_name}_facts.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(structured_output, f, indent=2, ensure_ascii=False)

                logger.info(f"Saved {len(structured_output)} items to {out_path}")


if __name__ == "__main__":
    main()
