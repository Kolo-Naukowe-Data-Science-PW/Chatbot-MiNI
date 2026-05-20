"""
Scrape USOS schedule pages (plany zajęć) directly via HTTP requests.

USOS returns server-rendered HTML when plan_format=html is set, so no
JavaScript rendering is needed. BeautifulSoup parses the timetable tables
into structured plain text suitable for LLM fact extraction.

Output: src/data/scraped_raw/schedule_<hash>.txt with URL: header.
"""

import hashlib
import logging
import os
import time
import urllib.parse

import requests
from bs4 import BeautifulSoup

from src.ingestion.links_extended import links

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "src/data/scraped_raw"
SCHEDULE_PATTERN = "pokazPlanGrupyPrzedmiotow"
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.2

_SEMESTER_LABELS = {"2025Z": "zimowy 2025/2026", "2026L": "letni 2025/2026"}
_PROGRAM_LABELS = {
    "INSIISA": "Informatyka i Systemy Informacyjne (inżynierski, ang. Computer Science and Information Systems)",
    "INSIISP": "Informatyka i Systemy Informacyjne (inżynierski)",
    "INSIMSI": "Informatyka i Systemy Informacyjne (magisterski)",
    "INSICAD": "Informatyka i Systemy Informacyjne (magisterski, CAD)",
    "MALSP": "Matematyka (licencjacki)",
    "MDLSP": "Matematyka i Analiza Danych (licencjacki)",
    "MANSP": "Matematyka (magisterski, specjalność)",
    "MDNSP": "Matematyka i Analiza Danych (magisterski)",
    "DSISP": "Inżynieria i Analiza Danych (inżynierski)",
    "DSMSA": "Data Science (magisterski)",
    "OBIERALNE": "Przedmioty obieralne",
    "ELECTIVES": "Elective courses",
}


def is_schedule_url(url: str) -> bool:
    return SCHEDULE_PATTERN in url


def _parse_group_code(grup_kod: str) -> str:
    """Turn e.g. '1120-INSIISA-S3' into a human-readable description."""
    parts = grup_kod.split("-")
    if len(parts) < 2:
        return grup_kod
    prog_key = parts[1] if len(parts) > 1 else ""
    prog_label = _PROGRAM_LABELS.get(prog_key, prog_key)
    semester_part = parts[2] if len(parts) > 2 else ""
    return f"{prog_label}, {semester_part}"


def _ensure_html_format(url: str) -> str:
    """Add plan_format=html to USOS URL if not already present."""
    if "plan_format=" not in url:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}plan_format=html&plan_colorScheme=default"
    return url


def _extract_url_params(url: str) -> dict[str, str]:
    parsed = urllib.parse.urlparse(url)
    return dict(urllib.parse.parse_qsl(parsed.query))


def _parse_schedule_html(html: str, url: str) -> str:
    """Convert USOS HTML timetable into structured plain text."""
    params = _extract_url_params(url)
    grup_kod = params.get("grupa_kod", "")
    cdyd_kod = params.get("cdyd_kod", "")
    sem_label = _SEMESTER_LABELS.get(cdyd_kod, cdyd_kod)
    prog_desc = _parse_group_code(grup_kod)

    soup = BeautifulSoup(html, "html.parser")

    lines = [
        f"PLAN ZAJĘĆ — {prog_desc}",
        f"Kod grupy: {grup_kod}",
        f"Semestr: {sem_label} ({cdyd_kod})",
        "",
    ]

    # Look for the timetable table (USOS typically uses class 'plan' or just the largest table)
    tables = soup.find_all("table")
    if not tables:
        body_text = soup.get_text(separator="\n", strip=True)
        lines.append(body_text)
        return "\n".join(lines)

    for table_idx, table in enumerate(tables):
        rows = table.find_all("tr")
        if not rows:
            continue

        # Collect header row if present
        header_cells = rows[0].find_all(["th", "td"])
        if header_cells:
            header_texts = [c.get_text(separator=" ", strip=True) for c in header_cells]
            if any(t for t in header_texts):
                lines.append("| " + " | ".join(header_texts) + " |")
                lines.append("|" + "---|" * len(header_texts))

        # Data rows
        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            cell_texts = []
            for cell in cells:
                # Preserve newlines within a cell as semicolons
                cell_text = cell.get_text(separator="; ", strip=True)
                cell_text = " ".join(cell_text.split())  # collapse extra whitespace
                cell_texts.append(cell_text)
            if any(t for t in cell_texts):
                lines.append("| " + " | ".join(cell_texts) + " |")

        if table_idx < len(tables) - 1:
            lines.append("")

    return "\n".join(lines)


def _url_to_filename(url: str) -> str:
    params = _extract_url_params(url)
    grup_kod = params.get("grupa_kod", "unknown").replace("-", "_")
    cdyd = params.get("cdyd_kod", "")
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"schedule_{grup_kod}_{cdyd}_{url_hash}.txt"


def scrape_schedules() -> int:
    """Scrape all schedule URLs from links_extended and save to scraped_raw/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    schedule_urls = [u for u in links if is_schedule_url(u)]
    total = len(schedule_urls)
    logger.info(f"Found {total} schedule URLs to scrape.")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; MiNIonek-bot/1.0)"})

    saved = 0
    for i, url in enumerate(schedule_urls, start=1):
        fetch_url = _ensure_html_format(url)
        logger.info(f"[{i}/{total}] Fetching: {url}")
        try:
            resp = session.get(fetch_url, timeout=REQUEST_TIMEOUT)
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                logger.warning(f"[{i}/{total}] HTTP {resp.status_code}: {url}")
                continue
            text = _parse_schedule_html(resp.text, url)
            if len(text.strip()) < 100:
                logger.warning(
                    f"[{i}/{total}] Very short content ({len(text)} chars): {url}"
                )
                continue
            filename = _url_to_filename(url)
            out_path = os.path.join(OUTPUT_DIR, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"URL: {url}\n\n{text}")
            saved += 1
            logger.info(f"[{i}/{total}] OK — {len(text)} chars → {filename}")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.warning(f"[{i}/{total}] FAILED: {url} — {e}")

    logger.info(f"Schedule scraping done. Saved: {saved}/{total}")
    return saved


if __name__ == "__main__":
    scrape_schedules()
