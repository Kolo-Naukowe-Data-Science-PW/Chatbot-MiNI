"""
Prepare a scraped-file subset for chunk-based knowledge base building.

Analyses which source files are referenced by each test set, picks the top
N most-covered sources, and:
  - Copies those .txt files to <output_dir>/scraped_subset/
  - Writes filtered test sets to <output_dir>/

Usage
-----
    # Analyse only (disk stats + source ranking, no output files):
    python -m src.evaluation.prepare_chunking_subset \\
        --scraped-dir /root/scraped_raw_backup --analyze-only

    # Build a subset of the top 50 most-referenced sources:
    python -m src.evaluation.prepare_chunking_subset \\
        --scraped-dir /root/scraped_raw_backup \\
        --output-dir src/evaluation/data/chunking_subset \\
        --max-sources 50

    # Build a subset from an explicit list of source keys (one per line):
    python -m src.evaluation.prepare_chunking_subset \\
        --scraped-dir /root/scraped_raw_backup \\
        --output-dir src/evaluation/data/chunking_subset \\
        --sources-file my_sources.txt

Filtered test set files written to <output_dir>/:
    generated_subset.jsonl   (questions referencing ONLY selected sources)
    rag_subset.csv
    human_subset.csv         (only when file_url_mapping.json is available)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "src" / "evaluation" / "data"
FILE_URL_MAP_PATH = DATA_DIR / "file_url_mapping.json"
HUMAN_PATH = DATA_DIR / "questions_with_links.csv"
RAG_PATH = DATA_DIR / "QA_rag.csv"
GENERATED_PATH = DATA_DIR / "final_notebooklm_QA.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _load_file_url_map() -> dict[str, str]:
    if not FILE_URL_MAP_PATH.exists():
        print(f"[warn] file_url_mapping.json not found at {FILE_URL_MAP_PATH}")
        print(
            "       Run: python -m src.utils.build_file_url_mapping <scraped_dir> > src/evaluation/data/file_url_mapping.json"
        )
        return {}
    with open(FILE_URL_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)


def _disk_usage_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Source extraction per test set
# ---------------------------------------------------------------------------


def _sources_generated() -> dict[int, list[str]]:
    """Return {line_index: [fname_key, ...]} for generated testset."""
    result: dict[int, list[str]] = {}
    if not GENERATED_PATH.exists():
        return result
    with open(GENERATED_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fname = (rec.get("Źródła") or "").strip()
            if fname:
                key = fname[:-4] if fname.endswith(".txt") else fname
                result[i] = [key]
    return result


def _sources_rag() -> dict[int, list[str]]:
    """Return {row_index: [fname_key, ...]} for rag testset."""
    result: dict[int, list[str]] = {}
    if not RAG_PATH.exists():
        return result
    with open(RAG_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for i, r in enumerate(reader):
            norm = {
                (k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k
            }
            pliki_str = norm.get("pliki", "")
            keys = [p.strip() for p in pliki_str.split(";") if p.strip()]
            if keys:
                result[i] = keys
    return result


def _sources_human(file_url_map: dict[str, str]) -> dict[int, list[str]]:
    """Return {row_index: [fname_key, ...]} for human testset via reverse URL lookup."""
    if not file_url_map:
        return {}
    url_to_key = {_normalize_url(url): key for key, url in file_url_map.items()}
    result: dict[int, list[str]] = {}
    if not HUMAN_PATH.exists():
        return result
    with open(HUMAN_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            norm = {
                (k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k
            }
            if norm.get("wymagany kontekst") != "0":
                continue
            url = norm.get("strona") or norm.get("url") or norm.get("link", "")
            url = _normalize_url(url.strip().rstrip("/"))
            key = url_to_key.get(url, "")
            if key:
                result[i] = [key]
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse(scraped_dir: Path) -> None:
    file_url_map = _load_file_url_map()

    gen_sources = _sources_generated()
    rag_sources = _sources_rag()
    human_sources = _sources_human(file_url_map)

    # Count references per source key
    counts: Counter = Counter()
    for sources_map in (gen_sources, rag_sources, human_sources):
        for keys in sources_map.values():
            counts.update(keys)

    all_scraped = (
        set(
            f.stem
            for f in scraped_dir.glob("*.txt")
            if not f.name.startswith("schedule_")
        )
        if scraped_dir.exists()
        else set()
    )

    print(f"\n=== Scraped directory: {scraped_dir} ===")
    if scraped_dir.exists():
        total_bytes = _disk_usage_bytes(scraped_dir)
        n_files = sum(1 for _ in scraped_dir.glob("*.txt"))
        print(f"  Files total : {n_files}  ({_fmt_bytes(total_bytes)})")
        if n_files > 0:
            avg = total_bytes / n_files
            print(f"  Avg per file: {_fmt_bytes(int(avg))}")
    else:
        print("  [directory not found]")

    print("\n=== Test set coverage ===")
    print(
        f"  generated : {len(gen_sources)} questions, {len(set(k for ks in gen_sources.values() for k in ks))} unique sources"
    )
    print(
        f"  rag       : {len(rag_sources)} questions, {len(set(k for ks in rag_sources.values() for k in ks))} unique sources"
    )
    print(
        f"  human     : {len(human_sources)} questions mapped (needs file_url_mapping.json)"
    )

    print("\n=== Top 30 sources by total question coverage ===")
    print(f"  {'rank':>4}  {'questions':>9}  source_key")
    for rank, (key, cnt) in enumerate(counts.most_common(30), 1):
        in_scraped = "✓" if key in all_scraped else "✗"
        print(f"  {rank:>4}  {cnt:>9}  {in_scraped} {key}")

    print("\n=== Cumulative question coverage by N top sources ===")
    print(f"  {'N':>4}  {'q covered':>10}  {'% of total':>10}  est. disk")
    total_q = len(gen_sources) + len(rag_sources) + len(human_sources)
    covered_per_source: dict[str, set] = defaultdict(set)
    for sources_map, label in (
        (gen_sources, "g"),
        (rag_sources, "r"),
        (human_sources, "h"),
    ):
        for idx, keys in sources_map.items():
            for key in keys:
                covered_per_source[key].add(f"{label}_{idx}")
    covered_q: set = set()
    avg_size = (total_bytes / n_files) if scraped_dir.exists() and n_files > 0 else 0
    for n, (key, _) in enumerate(counts.most_common(60), 1):
        covered_q.update(covered_per_source.get(key, set()))
        pct = 100 * len(covered_q) / total_q if total_q else 0
        est = _fmt_bytes(int(avg_size * n))
        print(f"  {n:>4}  {len(covered_q):>10}  {pct:>9.1f}%  {est}")
        if n in (10, 20, 30, 40, 50, 60):
            print()


# ---------------------------------------------------------------------------
# Subset building
# ---------------------------------------------------------------------------


def build_subset(
    scraped_dir: Path,
    output_dir: Path,
    selected_keys: set[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scraped_out = output_dir / "scraped_subset"
    scraped_out.mkdir(exist_ok=True)

    # Copy selected .txt files
    copied = 0
    for key in selected_keys:
        src = scraped_dir / f"{key}.txt"
        if src.exists():
            shutil.copy2(src, scraped_out / src.name)
            copied += 1
        else:
            print(f"[warn] not found in scraped_dir: {src.name}")
    print(f"\nCopied {copied}/{len(selected_keys)} files -> {scraped_out}")
    print(f"Subset disk usage: {_fmt_bytes(_disk_usage_bytes(scraped_out))}")

    # Write selected_sources.txt
    sources_txt = output_dir / "selected_sources.txt"
    sources_txt.write_text("\n".join(sorted(selected_keys)), encoding="utf-8")
    print(f"Source list -> {sources_txt}")

    file_url_map = _load_file_url_map()

    # Filter generated testset
    gen_sources = _sources_generated()
    filtered_gen: list[str] = []
    with open(GENERATED_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    for i, keys in gen_sources.items():
        if all(k in selected_keys for k in keys):
            filtered_gen.append(lines[i])
    out_gen = output_dir / "generated_subset.jsonl"
    out_gen.write_text("".join(filtered_gen), encoding="utf-8")
    print(f"Generated subset: {len(filtered_gen)} questions -> {out_gen}")

    # Filter rag testset
    rag_sources = _sources_rag()
    rag_rows: list[dict] = []
    with open(RAG_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        fieldnames = reader.fieldnames or []
        for r in reader:
            rag_rows.append(dict(r))
    filtered_rag = [
        rag_rows[i]
        for i, keys in rag_sources.items()
        if all(k in selected_keys for k in keys)
    ]
    out_rag = output_dir / "rag_subset.csv"
    with open(out_rag, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="|", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(filtered_rag)
    print(f"RAG subset: {len(filtered_rag)} questions -> {out_rag}")

    # Filter human testset (only if mapping available)
    if file_url_map:
        human_sources = _sources_human(file_url_map)
        human_rows: list[dict] = []
        fieldnames_h: list[str] = []
        with open(HUMAN_PATH, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames_h = reader.fieldnames or []
            for r in reader:
                human_rows.append(dict(r))
        # human_sources keys are row indices that passed wymagany kontekst==0 filter
        # we need original row indices mapping
        kontekst_indices: list[int] = []
        with open(HUMAN_PATH, encoding="utf-8-sig", newline="") as f:
            reader2 = csv.DictReader(f)
            for i, r in enumerate(reader2):
                norm = {
                    (k or "").strip().lower(): (v or "").strip()
                    for k, v in r.items()
                    if k
                }
                if norm.get("wymagany kontekst") == "0":
                    kontekst_indices.append(i)
        filtered_human = []
        for rank, orig_idx in enumerate(kontekst_indices):
            if rank in human_sources:
                keys = human_sources[rank]
                if all(k in selected_keys for k in keys):
                    filtered_human.append(human_rows[orig_idx])
        out_human = output_dir / "human_subset.csv"
        with open(out_human, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_h, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(filtered_human)
        print(f"Human subset: {len(filtered_human)} questions -> {out_human}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare scraped-file subset for chunk KB."
    )
    parser.add_argument(
        "--scraped-dir", required=True, help="Path to scraped .txt files directory."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write subset files (required unless --analyze-only).",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Select top N sources by question coverage.",
    )
    parser.add_argument(
        "--sources-file",
        default=None,
        help="Text file with explicit source keys (one per line).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Print stats without creating any output files.",
    )
    args = parser.parse_args()

    scraped_dir = Path(args.scraped_dir)

    analyse(scraped_dir)

    if args.analyze_only:
        return

    if not args.output_dir:
        print(
            "\nProvide --output-dir to build the subset (or use --analyze-only for stats only)."
        )
        sys.exit(1)

    # Determine selected sources
    if args.sources_file:
        selected_keys = set(
            Path(args.sources_file).read_text(encoding="utf-8").splitlines()
        )
        selected_keys = {k.strip() for k in selected_keys if k.strip()}
    elif args.max_sources:
        file_url_map = _load_file_url_map()
        gen_sources = _sources_generated()
        rag_sources = _sources_rag()
        human_sources = _sources_human(file_url_map)
        counts: Counter = Counter()
        for sources_map in (gen_sources, rag_sources, human_sources):
            for keys in sources_map.values():
                counts.update(keys)
        selected_keys = {key for key, _ in counts.most_common(args.max_sources)}
        print(
            f"\nSelected top {args.max_sources} sources ({len(selected_keys)} unique keys)."
        )
    else:
        print("Provide --max-sources N or --sources-file to select sources.")
        sys.exit(1)

    build_subset(scraped_dir, Path(args.output_dir), selected_keys)


if __name__ == "__main__":
    main()
