"""
Coverage analysis: how many gold URLs from the benchmark are actually in Qdrant?

For each gold URL checks:
  - exact   : URL is in DB verbatim
  - parent  : direct parent of the URL is in DB (one level up)
  - child   : at least one direct child of the URL is in DB
  - missing : not found at any level

Reports aggregate statistics and saves:
  - coverage_report_<ts>.json  — full per-URL breakdown
  - coverage_report_<ts>.csv   — same, spreadsheet-friendly

Usage (from repo root):
    python -m evaluation.benchmark_checks.check_coverage
    python -m evaluation.benchmark_checks.check_coverage --input-csv src/evaluation/data/questions_with_links.csv
    python -m evaluation.benchmark_checks.check_coverage --show-missing
"""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import (  # noqa: E402
    load_gold,
    load_gold_filtered,
    normalize_url,
)


def _get_all_db_urls() -> frozenset[str]:
    """Return all unique normalized URLs stored in the Qdrant collection."""
    from src.api.retrieval import _get_qdrant_client  # noqa: PLC0415
    from src.ingestion.vector_db import COLLECTION_NAME  # noqa: PLC0415

    client = _get_qdrant_client()
    urls: set[str] = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1000,
            with_payload=["url"],
            with_vectors=False,
        )
        for p in points:
            raw = (p.payload or {}).get("url", "")
            normed = normalize_url(raw)
            if normed:
                urls.add(normed)
        if next_offset is None:
            break
        offset = next_offset
    return frozenset(urls)


def _classify(target_url: str, db_urls: frozenset[str]) -> str:
    """
    Returns coverage level for a single target URL:
      'exact'   — target is in DB verbatim
      'parent'  — direct parent of target is in DB
      'child'   — at least one direct child of target is in DB
      'missing' — not found at any level
    """
    if not target_url:
        return "missing"

    if target_url in db_urls:
        return "exact"

    t = urlsplit(target_url)
    t_path = t.path.rstrip("/")

    # Parent check (one level up)
    if "/" in t_path[1:]:
        parent_path = t_path.rsplit("/", 1)[0] or "/"
        parent = normalize_url(urlunsplit((t.scheme, t.netloc, parent_path, "", "")))
        if parent in db_urls:
            return "parent"

    # Child check (any direct child)
    prefix = t_path.rstrip("/") + "/"
    for db_url in db_urls:
        db_t = urlsplit(db_url)
        if db_t.scheme == t.scheme and db_t.netloc == t.netloc:
            db_path = db_t.path.rstrip("/")
            if db_path.startswith(prefix) and db_path[len(prefix):].count("/") == 0:
                return "child"

    return "missing"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse how many gold benchmark URLs exist in the Qdrant DB."
    )
    parser.add_argument(
        "--input-csv",
        default="src/evaluation/data/questions_with_links.csv",
        help="Benchmark CSV (default: questions_with_links.csv, auto-filters wymagany kontekst=0).",
    )
    parser.add_argument(
        "--output-dir",
        default="src/evaluation/results/coverage",
        help="Where to save the report files.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print the full list of missing URLs to stdout.",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Use all rows from the CSV, not just wymagany kontekst=0.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_csv)
    if args.all_rows or input_path.name != "questions_with_links.csv":
        gold = load_gold(str(input_path))
        source_label = str(input_path)
    else:
        gold = load_gold_filtered(str(input_path))
        source_label = f"{input_path.name} (wymagany kontekst=0)"

    print(f"Loaded {len(gold)} gold queries from {source_label}")
    print("Loading Qdrant DB URLs... (this may take a moment)")
    db_urls = _get_all_db_urls()
    print(f"Found {len(db_urls)} unique URLs in Qdrant.\n")

    # Deduplicate gold URLs (multiple questions can share the same target URL)
    unique_gold_urls: dict[str, list[str]] = {}
    for row in gold:
        unique_gold_urls.setdefault(row.target_url, []).append(row.query)

    print(f"Unique gold URLs: {len(unique_gold_urls)}")
    print(f"Total gold questions: {len(gold)}\n")

    # Classify every unique gold URL
    per_url: list[dict] = []
    for url, questions in sorted(unique_gold_urls.items()):
        level = _classify(url, db_urls)
        per_url.append({
            "gold_url": url,
            "coverage": level,
            "question_count": len(questions),
            "example_question": questions[0],
        })

    # Question-level counts (weighted by how many questions point to each URL)
    q_counts: Counter = Counter()
    for entry in per_url:
        q_counts[entry["coverage"]] += entry["question_count"]

    url_counts: Counter = Counter(e["coverage"] for e in per_url)
    total_urls = len(per_url)
    total_qs = len(gold)

    def pct(n, total):
        return f"{100 * n / total:.1f}%" if total else "n/a"

    print("=" * 55)
    print(f"{'Coverage level':<15} {'URLs':>8} {'%':>7}  {'Questions':>10} {'%':>7}")
    print("-" * 55)
    for level in ("exact", "parent", "child", "missing"):
        n_urls = url_counts[level]
        n_qs = q_counts[level]
        print(f"{level:<15} {n_urls:>8} {pct(n_urls, total_urls):>7}  {n_qs:>10} {pct(n_qs, total_qs):>7}")
    print("-" * 55)
    covered_urls = url_counts["exact"] + url_counts["parent"] + url_counts["child"]
    covered_qs   = q_counts["exact"]  + q_counts["parent"]  + q_counts["child"]
    print(f"{'COVERED (any)':<15} {covered_urls:>8} {pct(covered_urls, total_urls):>7}  {covered_qs:>10} {pct(covered_qs, total_qs):>7}")
    print(f"{'TOTAL':<15} {total_urls:>8} {'100.0%':>7}  {total_qs:>10} {'100.0%':>7}")
    print("=" * 55)

    missing_entries = [e for e in per_url if e["coverage"] == "missing"]
    if missing_entries:
        print(f"\n{len(missing_entries)} URLs not found in DB at any level.")
        if args.show_missing:
            print("\n--- Missing URLs ---")
            for e in missing_entries:
                print(f"  [{e['question_count']}q] {e['gold_url']}")
                print(f"       e.g.: {e['example_question'][:80]}")
        else:
            print("  Run with --show-missing to list them.")

    # Save reports
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    summary = {
        "source": source_label,
        "db_url_count": len(db_urls),
        "unique_gold_urls": total_urls,
        "total_questions": total_qs,
        "by_level": {
            level: {
                "urls": url_counts[level],
                "url_pct": round(100 * url_counts[level] / total_urls, 2) if total_urls else 0,
                "questions": q_counts[level],
                "question_pct": round(100 * q_counts[level] / total_qs, 2) if total_qs else 0,
            }
            for level in ("exact", "parent", "child", "missing")
        },
        "covered_url_pct": round(100 * covered_urls / total_urls, 2) if total_urls else 0,
        "covered_question_pct": round(100 * covered_qs / total_qs, 2) if total_qs else 0,
    }

    json_path = output_dir / f"coverage_report_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_url": per_url}, f, ensure_ascii=False, indent=2)
    print(f"\nFull report  -> {json_path}")

    csv_path = output_dir / f"coverage_report_{ts}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gold_url", "coverage", "question_count", "example_question"])
        writer.writeheader()
        writer.writerows(per_url)
    print(f"CSV report   -> {csv_path}")


if __name__ == "__main__":
    main()
