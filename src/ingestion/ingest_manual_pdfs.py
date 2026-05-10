"""
Convert manually downloaded PDFs to scraped_raw .txt files.

Usage:
    python -m ingestion.ingest_manual_pdfs

Put PDFs in src/data/manual_pdfs/ before running.
Known filenames are mapped to their BIP PW source URLs automatically.
Unknown filenames use the filename as URL fallback.
"""

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "src/manual_pdfs"
OUTPUT_DIR = "src/data/scraped_raw"

# Map filename → canonical source URL (used in retrieval metadata).
# Hash-named files come from BIP PW storage; URL is the direct download link.
URL_MAP = {
    # Czytelne regulaminy
    "regulamin_studiow.pdf": "https://www.bip.pw.edu.pl/index.php/Sprawy-Studenckie/Regulamin-studiow-w-Politechnice-Warszawskiej2",
    "regulamin_swiadczen_2025_2026.pdf": "https://www.bss.pw.edu.pl/Stypendia/Stypendia-z-Funduszu-Stypendialnego/Stypendium-Rektora",
    "dyplom.pdf": "https://www.bip.pw.edu.pl/Dokumenty-publiczne/Dyplom-ukonczenia-studiow-w-Politechnice-Warszawskiej",
    # Plany studiów z BIP PW (sekcja "Ważne dokumenty" w links_extended)
    "6ae70089d460bc9ba71566298fa79fcf.pdf": "https://bip.pw.edu.pl/var/pw/storage/original/application/6ae70089d460bc9ba71566298fa79fcf.pdf",
    "5e62794f9cf903f4ef2e16869f704020.pdf": "https://bip.pw.edu.pl/var/pw/storage/original/application/5e62794f9cf903f4ef2e16869f704020.pdf",
    "09dde30f2523fec5088b62dbdd727a73.pdf": "https://bip.pw.edu.pl/var/pw/storage/original/application/09dde30f2523fec5088b62dbdd727a73.pdf",
    "df978333880183b4bc4906362e74a2f8.pdf": "https://bip.pw.edu.pl/var/pw/storage/original/application/df978333880183b4bc4906362e74a2f8.pdf",
    # Dokumenty BIP PW (sekcja "BIP PW" w links_extended)
    "d306a4288f0943c31b5e9cd8fcd33f73.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/d306a4288f0943c31b5e9cd8fcd33f73.pdf",
    "a7f6351019e1d70a951c7a1ac1bb0e28.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/a7f6351019e1d70a951c7a1ac1bb0e28.pdf",
    "62f1b26a2cee774b5699342e29970bbf.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/62f1b26a2cee774b5699342e29970bbf.pdf",
    "bc54edfe5cc419713f931181db92ef46.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/bc54edfe5cc419713f931181db92ef46.pdf",
    "a3de1af7fada945beb87a825dbb0c262.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/a3de1af7fada945beb87a825dbb0c262.pdf",
    "b316729d09938e94c7eb5528b0e0744d.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/b316729d09938e94c7eb5528b0e0744d.pdf",
    "7fe181576ef6f818438bc0b3df98e13f.pdf": "https://bip.pw.edu.pl/var/pw/storage/original/application/7fe181576ef6f818438bc0b3df98e13f.pdf",
    "84cf2faf2873685ac94833c869fe866f.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/84cf2faf2873685ac94833c869fe866f.pdf",
    "35ef86bbcf1cd086fee4fd97f088aa47.pdf": "https://bip.pw.edu.pl/var/pw/storage/original/application/35ef86bbcf1cd086fee4fd97f088aa47.pdf",
    "c82085781d743edb519bd365df87aded.pdf": "https://bip.pw.edu.pl/var/pw/storage/original/application/c82085781d743edb519bd365df87aded.pdf",
    "ee0f54fc508ca6edb94560f70b4dcd8b.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/ee0f54fc508ca6edb94560f70b4dcd8b.pdf",
    "390bf595ba50c2b63d98e7056c54b61f.pdf": "https://bip.pw.edu.pl/var/pw/storage/original/application/390bf595ba50c2b63d98e7056c54b61f.pdf",
    "2b1c27e1c5245605dfdf71342a8b11b9.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/2b1c27e1c5245605dfdf71342a8b11b9.pdf",
    "e047e0025a88927817f200f6ef12d364.pdf": "https://www.bip.pw.edu.pl/var/pw/storage/original/application/e047e0025a88927817f200f6ef12d364.pdf",
}


def extract_text_pypdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def extract_text_pdfminer(path: str) -> str:
    from pdfminer.high_level import extract_text

    return extract_text(path) or ""


def extract_text(path: str) -> str:
    try:
        text = extract_text_pypdf(path)
        if text.strip():
            return text
    except Exception as e:
        logger.warning(f"pypdf failed for {path}: {e} — trying pdfminer")

    try:
        return extract_text_pdfminer(path)
    except Exception as e:
        logger.error(f"pdfminer also failed for {path}: {e}")
        return ""


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdfs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        logger.warning(f"No PDFs found in {INPUT_DIR}")
        return

    for filename in pdfs:
        path = os.path.join(INPUT_DIR, filename)
        source_url = URL_MAP.get(filename, f"file://{filename}")

        logger.info(f"Processing: {filename} → {source_url}")
        text = extract_text(path)

        if not text.strip():
            logger.warning(f"No text extracted from {filename} — skipping")
            continue

        safe_name = source_url.replace("https://", "").replace("/", "_").strip("_")[:200]
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {source_url}\n\n{text}")

        logger.info(f"Saved {len(text)} chars → {out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
