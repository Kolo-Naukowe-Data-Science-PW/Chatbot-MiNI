"""
Pipeline progress tracker.

Persists to src/data/pipeline_progress.json (inside Docker volume, survives restarts).
Delete that file to start completely fresh.

Keys tracked:
  scraped          – URLs successfully scraped (scraper.py)
  facts_extracted  – _facts.json filenames written (extract_facts.py)
  ingested         – _facts.json filenames upserted into Qdrant (ingest_facts.py)
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

PROGRESS_FILE = "src/data/pipeline_progress.json"
_EMPTY: dict = {"scraped": [], "facts_extracted": [], "ingested": []}


def _load() -> dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                data = json.load(f)
                for key in _EMPTY:
                    data.setdefault(key, [])
                return data
        except Exception as e:
            logger.warning(f"Could not read progress file, starting fresh: {e}")
    return {k: list(v) for k, v in _EMPTY.items()}


def _save(progress: dict) -> None:
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PROGRESS_FILE)


# ── read ────────────────────────────────────────────────────────────────────


def is_scraped(url: str) -> bool:
    return url in _load()["scraped"]


def is_facts_extracted(facts_filename: str) -> bool:
    return facts_filename in _load()["facts_extracted"]


def is_ingested(facts_filename: str) -> bool:
    return facts_filename in _load()["ingested"]


def ingested_count() -> int:
    return len(_load()["ingested"])


def get_status() -> dict:
    p = _load()
    return {stage: len(items) for stage, items in p.items()}


# ── write ────────────────────────────────────────────────────────────────────


def mark_scraped(url: str) -> None:
    p = _load()
    if url not in p["scraped"]:
        p["scraped"].append(url)
    _save(p)


def mark_facts_extracted(facts_filename: str) -> None:
    p = _load()
    if facts_filename not in p["facts_extracted"]:
        p["facts_extracted"].append(facts_filename)
    _save(p)


def mark_ingested(facts_filename: str) -> None:
    p = _load()
    if facts_filename not in p["ingested"]:
        p["ingested"].append(facts_filename)
    _save(p)


def clear_progress_for_prefix(prefix: str, stages: list[str] | None = None) -> None:
    """Remove all entries starting with prefix from specified (or all) pipeline stages."""
    p = _load()
    for key in stages or list(p.keys()):
        if key in p:
            p[key] = [x for x in p[key] if not str(x).startswith(prefix)]
    _save(p)
