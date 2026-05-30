"""
Prepare the schedule-eval workspace from pytania_finall.jsonl.

Uses EXISTING schedule_*_facts.json and schedule_*.txt files already on the
Docker volume — no LLM re-extraction needed.

For each question in pytania_finall.jsonl, the source filename encodes the
URL of a Firecrawl-scraped schedule page.  From that URL we extract
(grupa_kod, cdyd_kod) and find all matching schedule_* files produced by
scrape_schedule.py (which are higher-quality: custom HTML parser vs Firecrawl).

Steps:
  1. Parse pytania_finall.jsonl → unique (grupa_kod, cdyd_kod) pairs.
  2. Find all matching schedule_*_facts.json in --facts-dir → copy to
     --facts-subset-dir.
  3. Find all matching schedule_*.txt in --scraped-dir → copy to
     --scraped-subset-dir.
  4. Build file_url_mapping from the copied .txt files and merge into
     src/evaluation/data/file_url_mapping.json.
  5. Write --workspace-dir/generated_subset.jsonl with one record per
     question; Źródła = stem of the first matching .txt file for that group.

Usage:
    python -m src.evaluation.prepare_schedule_eval \\
        --facts-dir       /app/src/data/facts \\
        --scraped-dir     /app/src/data/scraped_raw \\
        --facts-subset-dir  /app/src/data/facts_schedule_eval_subset \\
        --scraped-subset-dir /app/src/data/scraped_schedule_eval_subset \\
        --workspace-dir   /app/src/data/schedule_eval_workspace
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_FILE = PROJECT_ROOT / "src" / "evaluation" / "data" / "pytania_finall.jsonl"
FILE_URL_MAP_PATH = PROJECT_ROOT / "src" / "evaluation" / "data" / "file_url_mapping.json"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_source_filename(zrodla: str) -> str | None:
    """Extract raw filename from Źródła field like 'Dokument "filename.txt".'"""
    m = re.search(r'"([^"]+\.txt)"', zrodla)
    if m:
        return m.group(1)
    return None


def _extract_group_params(encoded_fname: str) -> tuple[str, str] | None:
    """
    Extract (grupa_kod, cdyd_kod) from a Firecrawl-style encoded filename.

    E.g. "usosweb...%3F_action=...&grupa_kod=1120-DSISP-S1&cdyd_kod=2025Z..."
    → ("1120-DSISP-S1", "2025Z")
    """
    decoded = unquote(encoded_fname)
    if "?" not in decoded:
        return None
    query_str = decoded.split("?", 1)[1]
    if query_str.endswith(".txt"):
        query_str = query_str[:-4]
    params = parse_qs(query_str)
    grp = params.get("grupa_kod", [None])[0]
    cdyd = params.get("cdyd_kod", [None])[0]
    if grp and cdyd:
        return grp, cdyd
    return None


def _schedule_prefix(grupa_kod: str, cdyd_kod: str) -> str:
    """Return the filename prefix used by scrape_schedule.py for this group."""
    safe_grp = grupa_kod.replace("-", "_")
    return f"schedule_{safe_grp}_{cdyd_kod}_"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    facts_dir: str,
    scraped_dir: str,
    facts_subset_dir: str,
    scraped_subset_dir: str,
    workspace_dir: str,
) -> None:
    for d in (facts_subset_dir, scraped_subset_dir, workspace_dir):
        os.makedirs(d, exist_ok=True)

    # ── load questions ────────────────────────────────────────────────────────
    questions: list[dict] = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info("Loaded %d questions from %s", len(questions), QUESTIONS_FILE)

    # ── map each question → (grupa_kod, cdyd_kod) ─────────────────────────────
    question_groups: list[tuple[dict, str, str]] = []
    skipped = 0
    for rec in questions:
        raw = (rec.get("Źródła") or "").strip()
        fname = _parse_source_filename(raw)
        if not fname:
            logger.warning("Could not parse source filename from: %s", raw[:80])
            skipped += 1
            continue
        params = _extract_group_params(fname)
        if not params:
            logger.warning("Could not extract group params from: %s", fname[:80])
            skipped += 1
            continue
        question_groups.append((rec, params[0], params[1]))

    unique_groups: set[tuple[str, str]] = {(g, c) for _, g, c in question_groups}
    logger.info(
        "Parsed %d/%d questions → %d unique (grupa_kod, cdyd_kod) pairs.",
        len(question_groups), len(questions), len(unique_groups),
    )
    if skipped:
        logger.warning("Skipped %d questions with unparseable sources.", skipped)

    # ── discover all schedule_* files on the volume ───────────────────────────
    all_facts_files = [f for f in os.listdir(facts_dir) if f.startswith("schedule_") and f.endswith("_facts.json")]
    all_txt_files   = [f for f in os.listdir(scraped_dir) if f.startswith("schedule_") and f.endswith(".txt")]
    logger.info("Found %d schedule facts files, %d schedule txt files on volume.", len(all_facts_files), len(all_txt_files))

    # ── copy matching files ───────────────────────────────────────────────────
    # For each (grupa_kod, cdyd_kod) find ALL matching files (there may be 2)
    group_to_txt_stems: dict[tuple[str, str], list[str]] = {}

    for grp, cdyd in sorted(unique_groups):
        prefix = _schedule_prefix(grp, cdyd)

        # facts
        matching_facts = [f for f in all_facts_files if f.startswith(prefix)]
        for fname in matching_facts:
            src = os.path.join(facts_dir, fname)
            dst = os.path.join(facts_subset_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            logger.info("  facts: %s (%d match(es) for %s_%s)", fname, len(matching_facts), grp, cdyd)

        # txt
        matching_txt = [f for f in all_txt_files if f.startswith(prefix)]
        txt_stems: list[str] = []
        for fname in matching_txt:
            src = os.path.join(scraped_dir, fname)
            dst = os.path.join(scraped_subset_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            txt_stems.append(os.path.splitext(fname)[0])
        group_to_txt_stems[(grp, cdyd)] = sorted(txt_stems)

        if not matching_facts:
            logger.warning("  NO facts files found for prefix: %s", prefix)
        if not matching_txt:
            logger.warning("  NO txt files found for prefix: %s", prefix)

    # ── build file_url_mapping from copied .txt files ─────────────────────────
    file_url_map: dict[str, str] = {}
    for fname in os.listdir(scraped_subset_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(scraped_subset_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                first = f.readline().strip()
            if first.startswith("URL: "):
                url = first[5:].strip()
                stem = fname[:-4]
                file_url_map[stem] = url
        except Exception as exc:
            logger.warning("Could not read URL from %s: %s", fname, exc)

    logger.info("Built URL mapping for %d schedule .txt files.", len(file_url_map))

    # merge into existing file_url_mapping.json
    existing_map: dict[str, str] = {}
    if FILE_URL_MAP_PATH.exists():
        try:
            with open(FILE_URL_MAP_PATH, encoding="utf-8") as f:
                existing_map = json.load(f)
        except Exception:
            pass
    merged = {**existing_map, **file_url_map}
    with open(FILE_URL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info(
        "Updated file_url_mapping.json: %d existing + %d new = %d total",
        len(existing_map), len(file_url_map), len(merged),
    )

    # ── write generated_subset.jsonl ──────────────────────────────────────────
    out_path = os.path.join(workspace_dir, "generated_subset.jsonl")
    written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for rec, grp, cdyd in question_groups:
            stems = group_to_txt_stems.get((grp, cdyd), [])
            stem = stems[0] if stems else ""
            # Resolve stem → URL using the mapping built above
            zrodla = file_url_map.get(stem, stem)
            if not stem:
                logger.warning("No matching .txt for (%s, %s) — gold URL will be empty.", grp, cdyd)
            elif zrodla == stem:
                logger.warning("No URL found in mapping for stem %s — using stem as fallback.", stem)
            norm = {
                "Pytanie":    rec.get("Pytanie",  "").strip(),
                "Odpowiedź":  rec.get("Odpowiedź","").strip(),
                "Źródła":     zrodla,
            }
            out_f.write(json.dumps(norm, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d questions → %s", written, out_path)
    logger.info("Done. Workspace ready at %s", workspace_dir)

    # summary
    facts_copied = len(os.listdir(facts_subset_dir))
    txt_copied   = len(os.listdir(scraped_subset_dir))
    logger.info("Subset: %d facts files, %d txt files.", facts_copied, txt_copied)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare schedule-eval workspace using existing schedule_* files."
    )
    parser.add_argument("--facts-dir",        default="/app/src/data/facts",
                        help="Source dir with schedule_*_facts.json (Docker volume).")
    parser.add_argument("--scraped-dir",      default="/app/src/data/scraped_raw",
                        help="Source dir with schedule_*.txt (Docker volume).")
    parser.add_argument("--facts-subset-dir",   default="/app/src/data/facts_schedule_eval_subset",
                        help="Output dir for copied facts files.")
    parser.add_argument("--scraped-subset-dir", default="/app/src/data/scraped_schedule_eval_subset",
                        help="Output dir for copied .txt files.")
    parser.add_argument("--workspace-dir",    default="/app/src/data/schedule_eval_workspace",
                        help="Output dir for generated_subset.jsonl.")
    args = parser.parse_args()
    main(
        facts_dir=args.facts_dir,
        scraped_dir=args.scraped_dir,
        facts_subset_dir=args.facts_subset_dir,
        scraped_subset_dir=args.scraped_subset_dir,
        workspace_dir=args.workspace_dir,
    )
