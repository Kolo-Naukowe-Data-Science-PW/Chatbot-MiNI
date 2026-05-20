"""
Compare multiple experiment summary JSONs and print a Markdown table.

Usage
-----
    python -m src.evaluation.compare_experiments src/data/experiments/

Also saves comparison_{timestamp}.csv in the same directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Metric columns to include in the comparison table (in display order).
# Only columns that exist in at least one summary will be shown.
METRIC_COLS = [
    "hit@1", "hit@5", "hit@10",
    "mrr@1", "mrr@5", "mrr@10",
    "mrrw@1", "mrrw@5", "mrrw@10",
    "ndcg@1", "ndcg@5", "ndcg@10",
    "map@1", "map@5", "map@10",
    "judge_chatbot_usefulness",
    "judge_chatbot_accuracy",
    "judge_chatbot_completeness",
]

META_COLS = ["variant", "testset", "n_questions", "runtime_seconds", "judge_model"]


def _load_summaries(experiment_dir: Path) -> list[dict]:
    """Load all *_summary.json files from *experiment_dir*, sorted by timestamp."""
    json_files = sorted(experiment_dir.glob("*_summary.json"))
    summaries = []
    for fp in json_files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            data["_file"] = fp.name
            summaries.append(data)
        except Exception as exc:
            print(f"WARNING: Could not load {fp.name}: {exc}", file=sys.stderr)
    return summaries


def _active_metric_cols(summaries: list[dict]) -> list[str]:
    """Return only metric columns that appear in at least one summary."""
    present: set[str] = set()
    for s in summaries:
        present.update(s.keys())
    return [c for c in METRIC_COLS if c in present]


def _fmt(val: object) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _build_table(summaries: list[dict], metric_cols: list[str]) -> tuple[list[str], list[list[str]]]:
    """Return (header_row, data_rows) for Markdown and CSV output."""
    header = META_COLS + metric_cols
    rows: list[list[str]] = []
    for s in summaries:
        row = [_fmt(s.get(c)) for c in META_COLS] + [_fmt(s.get(c)) for c in metric_cols]
        rows.append(row)
    return header, rows


def _print_markdown(header: list[str], rows: list[list[str]]) -> None:
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(header)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    hdr = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)) + " |"
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")


def compare(experiment_dir: Path) -> None:
    summaries = _load_summaries(experiment_dir)
    if not summaries:
        print(f"No *_summary.json files found in {experiment_dir}.")
        return

    print(f"Loaded {len(summaries)} experiment summaries.\n")

    metric_cols = _active_metric_cols(summaries)
    header, rows = _build_table(summaries, metric_cols)

    _print_markdown(header, rows)

    # Save as CSV
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    csv_path = experiment_dir / f"comparison_{ts}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nSaved comparison CSV -> {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RAG ablation experiment summaries into a Markdown/CSV table."
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="src/data/experiments",
        help="Directory containing *_summary.json files (default: src/data/experiments).",
    )
    args = parser.parse_args()
    compare(Path(args.experiment_dir))


if __name__ == "__main__":
    main()
