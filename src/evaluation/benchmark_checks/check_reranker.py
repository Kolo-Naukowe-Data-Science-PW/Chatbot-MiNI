"""
Ablation: cross-encoder reranker ON vs OFF.

Runs the full benchmark twice (with reranker then without) and prints a
side-by-side comparison table.  Results are saved to:
    <output-dir>/with_rerank/    — eval_per_query_*.csv + eval_summary_*.json
    <output-dir>/no_rerank/      — eval_per_query_*.csv + eval_summary_*.json
    <output-dir>/comparison_reranker_<ts>.json — delta table

Usage (from repo root):
    python -m evaluation.benchmark_checks.check_reranker
    python -m evaluation.benchmark_checks.check_reranker --ks 3,5,10 --output-dir src/evaluation/results/reranker_check
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import (  # noqa: E402
    evaluate_to_csv,
    load_gold,
    load_gold_filtered,
)

KEY_METRICS = [
    "hit@3", "hit@5", "hit@10",
    "mrr@3", "mrr@5", "mrr@10",
    "mrrw@3", "mrrw@10",
    "ndcg@10", "map@10",
    "recall@10",
]


def _print_comparison(a: dict, b: dict, label_a: str, label_b: str) -> dict:
    shared = [m for m in KEY_METRICS if m in a and m in b]
    col_w = max(len(m) for m in shared) + 2

    header = f"{'Metric':<{col_w}} {label_a:>14} {label_b:>14} {'Delta':>10}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")

    comparison: dict[str, dict] = {}
    for m in shared:
        va, vb = a[m], b[m]
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        flag = " <" if delta < -0.005 else (" >" if delta > 0.005 else "")
        print(f"{m:<{col_w}} {va:>14.4f} {vb:>14.4f} {sign}{delta:>9.4f}{flag}")
        comparison[m] = {"with_rerank": round(va, 6), "no_rerank": round(vb, 6), "delta": round(delta, 6)}

    print(sep)
    improved = sum(1 for v in comparison.values() if v["delta"] > 0.001)
    degraded = sum(1 for v in comparison.values() if v["delta"] < -0.001)
    print(f"\nImproved (no_rerank > with_rerank): {improved}/{len(shared)}")
    print(f"Degraded (no_rerank < with_rerank): {degraded}/{len(shared)}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: compare retrieval with vs. without cross-encoder reranking."
    )
    parser.add_argument(
        "--input-csv",
        default="src/evaluation/data/questions_with_links.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="src/evaluation/results/reranker_check",
    )
    parser.add_argument(
        "--ks",
        default="3,5,10",
        help="Rank cut-offs (default: 3,5,10).",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Apply query rewriting in both runs (keeps rewriter constant, isolates reranker).",
    )
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="Also compute coverage-corrected hit@k (cch@k).",
    )
    args = parser.parse_args()

    ks = [int(k.strip()) for k in args.ks.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_csv)
    if input_path.name == "questions_with_links.csv":
        gold = load_gold_filtered(str(input_path))
        print(f"Loaded {len(gold)} queries (wymagany kontekst=0) from {input_path.name}\n")
    else:
        gold = load_gold(str(input_path))
        print(f"Loaded {len(gold)} queries from {input_path.name}\n")

    if args.rewrite:
        print("Query rewriting ENABLED in both runs (isolating reranker effect).\n")

    dir_a = output_dir / "with_rerank"
    dir_b = output_dir / "no_rerank"
    dir_a.mkdir(exist_ok=True)
    dir_b.mkdir(exist_ok=True)

    print("=" * 60)
    print("RUN 1/2 — with cross-encoder reranking")
    print("=" * 60)
    summary_a = evaluate_to_csv(
        gold, ks, dir_a,
        use_rewrite=args.rewrite,
        metric_mode="standard",
        check_coverage=args.check_coverage,
        use_rerank=True,
    )

    print("\n" + "=" * 60)
    print("RUN 2/2 — without cross-encoder reranking (raw RRF)")
    print("=" * 60)
    summary_b = evaluate_to_csv(
        gold, ks, dir_b,
        use_rewrite=args.rewrite,
        metric_mode="standard",
        check_coverage=args.check_coverage,
        use_rerank=False,
    )

    print("\n\n========== COMPARISON: with_rerank vs no_rerank ==========")
    comparison = _print_comparison(summary_a, summary_b, "with_rerank", "no_rerank")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"comparison_reranker_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"with_rerank": summary_a, "no_rerank": summary_b, "delta": comparison},
                  f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved -> {out_path}")


if __name__ == "__main__":
    main()
