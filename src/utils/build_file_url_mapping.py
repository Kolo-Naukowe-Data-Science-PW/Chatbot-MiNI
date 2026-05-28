"""
Build a filename → URL mapping from scraped TXT files stored on the VM.

Every TXT file produced by the scraper or ingest_manual_pdfs starts with:
    URL: https://...

The output is a JSON object keyed by filename without the .txt extension,
matching the format used in the ``pliki`` column of QA_rag_extended CSVs.

Usage (local or on VM):
    python -m src.evaluation.build_file_url_mapping /root/scraped_raw_backup
    python -m src.evaluation.build_file_url_mapping /root/scraped_raw_backup > src/evaluation/data/file_url_mapping.json
"""

from __future__ import annotations

import json
import os
import sys


def build_mapping(scraped_dir: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for fname in sorted(os.listdir(scraped_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(scraped_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                first_line = f.readline().strip()
            if first_line.startswith("URL: "):
                url = first_line[5:].strip()
                key = fname[:-4]  # strip .txt
                mapping[key] = url
        except Exception as e:
            print(f"Warning: could not read {fname}: {e}", file=sys.stderr)
    return mapping


if __name__ == "__main__":
    scraped_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/scraped_raw_backup"
    if not os.path.isdir(scraped_dir):
        print(f"Error: directory not found: {scraped_dir}", file=sys.stderr)
        sys.exit(1)
    mapping = build_mapping(scraped_dir)
    print(json.dumps(mapping, ensure_ascii=False, indent=2))
    print(f"Built mapping for {len(mapping)} files.", file=sys.stderr)
