"""
Ablation: cross-encoder reranker ON vs OFF.

Calls POST /retrieval on the running API for each query (reuses the existing
Qdrant singleton, no Docker volume conflicts).

Runs both variants back-to-back and prints a side-by-side comparison table.
Results saved to:
    <output-dir>/with_rerank/ — eval_summary_*.json
    <output-dir>/no_rerank/   — eval_summary_*.json
    <output-dir>/comparison_reranker_<ts>.json

Usage (from repo root, API must be running):
    python -m evaluation.benchmark_checks.check_reranker --api-base-url http://api:8000
    python -m evaluation.benchmark_checks.check_reranker --api-base-url http://api:8000 --rewrite
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import (  # noqa: E402
    average_precision_at_k,
    hierarchical_relevance,
    hit_at_k,
    load_gold,
    load_gold_filtered,
    mrr_at_k,
    mrr_weighted_single,
    ndcg_at_k,
    normalize_url,
    recall_at_k,
)

KEY_METRICS = [
    "hit@3", "hit@5", "hit@10",
    "mrr@3", "mrr@5", "mrr@10",
    "mrrw@3", "mrrw@10",
    "ndcg@10", "map@10",
    "recall@10",
]


def _get_db_urls(api_base: str) -> set[str]:
    from urllib.request import Request, urlopen
    req = Request(api_base.rstrip("/") + "/db-urls")
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return {normalize_url(u) for u in data["urls"]}


def _call_retrieval(api_base: str, query: str, top_k: int,
                    use_rerank: bool, use_rewrite: bool) -> list[str]:
    """POST /retrieval and return ordered deduplicated URLs."""
    url = api_base.rstrip("/") + "/retrieval"
    payload = json.dumps({
        "query": query,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "use_rewrite": use_rewrite,
    }).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return [normalize_url(u) for u in data["urls"] if u]


def _evaluate(gold, ks: list[int], api_base: str,
              use_rerank: bool, use_rewrite: bool, label: str) -> dict[str, float]:
    max_k = max(ks)
    total_rel = 1
    accum: dict[str, list[float]] = {}

    for idx, row in enumerate(gold):
        urls = _call_retrieval(api_base, row.query, max_k, use_rerank, use_rewrite)
        rel_scores = [hierarchical_relevance(u, row.target_url) for u in urls]

        for k in ks:
            accum.setdefault(f"hit@{k}", []).append(hit_at_k(rel_scores, k))
            accum.setdefault(f"mrr@{k}", []).append(mrr_at_k(rel_scores, k))
            accum.setdefault(f"mrrw@{k}", []).append(mrr_weighted_single(urls[:k], row.target_url))
            accum.setdefault(f"recall@{k}", []).append(recall_at_k(rel_scores, k, total_rel))
            accum.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(rel_scores, k))
            accum.setdefault(f"map@{k}", []).append(average_precision_at_k(rel_scores, k, total_rel))

        print(f"  [{label}] [{idx + 1}/{len(gold)}] {row.query[:70]}")

    return {col: mean(vals) for col, vals in accum.items() if vals}


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
    degraded  = sum(1 for v in comparison.values() if v["delta"] < -0.001)
    print(f"\nImproved (no_rerank > with_rerank): {improved}/{len(shared)}")
    print(f"Degraded (no_rerank < with_rerank): {degraded}/{len(shared)}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: compare retrieval with vs. without cross-encoder reranking."
    )
    parser.add_argument("--api-base-url", default="http://api:8000")
    parser.add_argument("--input-csv", default="src/evaluation/data/questions_with_links.csv")
    parser.add_argument("--output-dir", default="src/evaluation/results/reranker_check")
    parser.add_argument("--ks", default="3,5,10")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Apply query rewriting in both runs (keeps rewriter constant, isolates reranker).",
    )
    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="Also evaluate on covered-only subset (cch@k): queries whose gold URL is in Qdrant.",
    )
    args = parser.parse_args()

    ks = [int(k.strip()) for k in args.ks.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_csv)
    if input_path.name == "questions_with_links.csv":
        gold = load_gold_filtered(str(input_path))
        print(f"Loaded {len(gold)} queries (wymagany kontekst=0)\n")
    else:
        gold = load_gold(str(input_path))
        print(f"Loaded {len(gold)} queries\n")

    if args.rewrite:
        print("Query rewriting ENABLED in both runs (isolating reranker effect).\n")

    print("=" * 60)
    print("RUN 1/2 — with cross-encoder reranking")
    print("=" * 60)
    summary_a = _evaluate(gold, ks, args.api_base_url, use_rerank=True, use_rewrite=args.rewrite, label="with_rerank")

    print("\n" + "=" * 60)
    print("RUN 2/2 — without cross-encoder reranking (raw RRF)")
    print("=" * 60)
    summary_b = _evaluate(gold, ks, args.api_base_url, use_rerank=False, use_rewrite=args.rewrite, label="no_rerank")

    print("\n\n========== COMPARISON: with_rerank vs no_rerank ==========")
    comparison = _print_comparison(summary_a, summary_b, "with_rerank", "no_rerank")

    cch_a: dict[str, float] = {}
    cch_b: dict[str, float] = {}
    cch_comparison: dict = {}
    if args.check_coverage:
        print("\nFetching DB URLs for coverage-corrected metrics...")
        db_urls = _get_db_urls(args.api_base_url)
        gold_covered = [r for r in gold if normalize_url(r.target_url) in db_urls]
        print(f"Coverage filter: {len(gold_covered)}/{len(gold)} queries have gold URL in DB\n")
        if gold_covered:
            print("=" * 60)
            print("CCH RUN 1/2 — with reranking (covered subset only)")
            print("=" * 60)
            cch_a = _evaluate(gold_covered, ks, args.api_base_url, use_rerank=True, use_rewrite=args.rewrite, label="cch_with_rerank")
            print("\n" + "=" * 60)
            print("CCH RUN 2/2 — without reranking (covered subset only)")
            print("=" * 60)
            cch_b = _evaluate(gold_covered, ks, args.api_base_url, use_rerank=False, use_rewrite=args.rewrite, label="cch_no_rerank")
            print("\n\n===== CCH COMPARISON: with_rerank vs no_rerank (covered only) =====")
            cch_comparison = _print_comparison(cch_a, cch_b, "cch_with_rerank", "cch_no_rerank")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"comparison_reranker_{ts}.json"
    payload: dict = {"with_rerank": summary_a, "no_rerank": summary_b, "delta": comparison}
    if args.check_coverage:
        payload["cch_with_rerank"] = cch_a
        payload["cch_no_rerank"] = cch_b
        payload["cch_delta"] = cch_comparison
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved -> {out_path}")


if __name__ == "__main__":
    main()
