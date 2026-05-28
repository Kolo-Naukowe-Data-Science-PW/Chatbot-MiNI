"""
Convert manually downloaded XLSX files to scraped_raw .txt files.

Analogous to ingest_manual_pdfs.py — reads XLSX files, extracts content
as markdown tables (preserves structure of schedules/plans), and writes
to scraped_raw/ in the same URL: <url>\n\n<text> format used by the rest
of the ingestion pipeline.

Usage:
    python -m ingestion.ingest_manual_xlsx

Put XLSX files in src/manual_xlsx/ before running.Add each filename → source URL mapping to URL_MAP below.
"""

import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "src/manual_xlsx"
OUTPUT_DIR = "src/data/scraped_raw"

_PLAN_SESJI_URL = "https://ww2.mini.pw.edu.pl/studia/plany-zajec-i-procedury/plan-sesji"

# Map filename → canonical source URL (used in retrieval metadata).
URL_MAP: dict[str, str] = {
    # ISI (Informatyka i Systemy Informacyjne) — ang. CSIS
    "CSIS_WINTER_25Z (1).xlsx": _PLAN_SESJI_URL,
    "CSIS_SUMMER_2025.xlsx": _PLAN_SESJI_URL,
    "CSIS_FALL_2025.xlsx": _PLAN_SESJI_URL,
    # DS (Data Science)
    "DS_WINTER_25Z.xlsx": _PLAN_SESJI_URL,
    "DS_SUMMER_2025.xlsx": _PLAN_SESJI_URL,
    "DS_FALL_2025.xlsx": _PLAN_SESJI_URL,
    # ISI (Informatyka i Systemy Informacyjne) — pol. INSI
    "INSI_ZIMA_25Z.xlsx": _PLAN_SESJI_URL,
    "INSI_LATO_2025.xlsx": _PLAN_SESJI_URL,
    "INSI_JESIEN_2025.xlsx": _PLAN_SESJI_URL,
    # IAD (Inżynieria i Analiza Danych)
    "IAD_ZIMA_25Z (6).xlsx": _PLAN_SESJI_URL,
    "IAD_LATO_2025 (7).xlsx": _PLAN_SESJI_URL,
    "IAD_JESIEN_2025.xlsx": _PLAN_SESJI_URL,
    # MAT/MAD (Matematyka / Matematyka i Analiza Danych)
    "MAT_MAD_ZIMA_25Z (2).xlsx": _PLAN_SESJI_URL,
    "MAT_MAD_LATO_2025 (2).xlsx": _PLAN_SESJI_URL,
    "MAT_MAD_JESIEN_2025.xlsx": _PLAN_SESJI_URL,
}


def extract_text(path: str) -> str:
    """Extract all sheets from an XLSX file as markdown tables."""
    try:
        xls = pd.ExcelFile(path)
        parts: list[str] = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=0)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue
            # Fill merged cells (NaN propagation from Excel merges)
            df = df.ffill()
            md = df.to_markdown(index=False)
            parts.append(f"## {sheet}\n\n{md}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return ""


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    xlsx_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".xlsx")]
    if not xlsx_files:
        logger.warning(f"No XLSX files found in {INPUT_DIR}")
        return

    for filename in xlsx_files:
        path = os.path.join(INPUT_DIR, filename)
        source_url = URL_MAP.get(filename, f"file://{filename}")
        if source_url.startswith("file://"):
            logger.warning(
                f"{filename} has no URL mapping — using filename as fallback URL"
            )

        logger.info(f"Processing: {filename} → {source_url}")
        text = extract_text(path)

        if not text.strip():
            logger.warning(f"No text extracted from {filename} — skipping")
            continue

        safe_name = (
            source_url.replace("https://", "").replace("/", "_").strip("_")[:200]
        )
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {source_url}\n\n{text}")

        logger.info(f"Saved {len(text)} chars → {out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
