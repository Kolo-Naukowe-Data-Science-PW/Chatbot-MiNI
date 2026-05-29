"""
Scrape a fixed list of URLs into a custom output directory.

Reads URLs from a text file (one per line, # comments ignored), scrapes each
with Firecrawl, and saves using the same filename convention as scraper.py.
Already-existing output files are skipped (resume-safe).

Usage:
    python -m src.ingestion.scrape_subset \\
        --urls-file src/evaluation/data/subset_urls.txt \\
        --output-dir src/data/scraped_raw_subset
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_HEADNOTE_MARKER = "![](https://ww2.mini.pw.edu.pl/wp-content/uploads/WMiNI-01.png)"
_FOOTNOTE_MARKER = "#### Zaloguj się"


def _clean(text: str) -> str:
    if _HEADNOTE_MARKER in text:
        text = text.split(_HEADNOTE_MARKER, 1)[1]
    if _FOOTNOTE_MARKER in text:
        text = text.split(_FOOTNOTE_MARKER, 1)[0]
    return text


def _safe_name(url: str) -> str:
    return url.replace("https://", "").replace("/", "_").strip("_")[:200]


def _save(url: str, text: str, output_dir: str) -> None:
    path = os.path.join(output_dir, f"{_safe_name(url)}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n\n{text}")


def scrape_subset(urls_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    urls = [
        line.strip()
        for line in Path(urls_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    total = len(urls)
    saved = skipped = 0

    for i, url in enumerate(urls, 1):
        out_path = os.path.join(output_dir, f"{_safe_name(url)}.txt")
        if os.path.exists(out_path):
            logger.info("[%d/%d] SKIP (exists): %s", i, total, url)
            skipped += 1
            continue

        # Local PDF (file://) — read from src/manual_pdfs/ baked into the image
        if url.startswith("file://"):
            pdf_name = url[7:]
            pdf_path = Path("/app/src/manual_pdfs") / pdf_name
            if not pdf_path.exists():
                pdf_path = Path("src/manual_pdfs") / pdf_name
            logger.info("[%d/%d] Local PDF: %s", i, total, pdf_path)
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(pdf_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(p.strip() for p in pages if p.strip())
                if text:
                    _save(url, text, output_dir)
                    saved += 1
                    logger.info(
                        "[%d/%d] OK (local PDF) — %d chars", i, total, len(text)
                    )
                else:
                    logger.warning("[%d/%d] Empty PDF: %s", i, total, pdf_name)
            except Exception as exc:
                logger.warning("[%d/%d] PDF failed: %s — %s", i, total, pdf_name, exc)
            continue

        logger.info("[%d/%d] Scraping: %s", i, total, url)
        try:
            result = app.scrape(
                url,
                formats=["markdown", "links"],
                only_main_content=False,
                timeout=120000,
            )
            text = _clean(result.markdown)
            _save(url, text, output_dir)
            saved += 1
            logger.info("[%d/%d] OK — %d chars", i, total, len(text))
            time.sleep(1)
        except Exception as exc:
            logger.warning("[%d/%d] FAILED: %s — %s", i, total, url, exc)

    logger.info("Done. Saved: %d  Skipped: %d  Total: %d", saved, skipped, total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape a URL list into a directory.")
    parser.add_argument(
        "--urls-file", required=True, help="Text file with URLs (one per line)."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write .txt files."
    )
    args = parser.parse_args()
    scrape_subset(args.urls_file, args.output_dir)
