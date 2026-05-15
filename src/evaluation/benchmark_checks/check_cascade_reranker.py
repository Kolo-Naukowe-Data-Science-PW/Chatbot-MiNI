"""
Ablation: single-stage reranking vs cascade reranking.

Baseline:  60 RRF candidates → English CE (mmarco-mMiniLMv2) → top 30
Cascade:  120 RRF candidates → English CE → top 60 → Polish CE (herbert-base-reranker) → top 30

Calls POST /retrieval on the running API.

Results saved to:
    <output-dir>/baseline/   — eval_summary_*.json
    <output-dir>/cascade/    — eval_summary_*.json
    <output-dir>/comparison_cascade_<ts>.json

Usage (API must be running):
    python -m evaluation.benchmark_checks.check_cascade_reranker --api-base-url http://api:8000
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
    req = Request(api_base.rstrip("/") + "/db-urls")
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return {normalize_url(u) for u in data["urls"]}


def _call_retrieval(api_base: str, query: str, top_k: int,
                    use_cascade: bool) -> list[str]:
    """POST /retrieval and return ordered deduplicated URLs."""
    url = api_base.rstrip("/") + "/retrieval"
    payload = json.dumps({
        "query": query,
        "top_k": top_k,
        "use_rerank": True,
        "use_rewrite": False,
        "use_cascade_rerank": use_cascade,
    }).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    return [normalize_url(u) for u in data["urls"] if u]


def _evaluate(gold, ks: list[int], api_base: str,
              use_cascade: bool, label: str) -> dict[str, float]:
    max_k = max(ks)
    total_rel = 1
    accum: dict[str, list[float]] = {}

    for idx, row in enumerate(gold):
        urls = _call_retrieval(api_base, row.query, max_k, use_cascade)
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
        comparison[m] = {label_a: round(va, 6), label_b: round(vb, 6), "delta": round(delta, 6)}

    print(sep)
    improved = sum(1 for v in comparison.values() if v["delta"] > 0.001)
    degraded  = sum(1 for v in comparison.values() if v["delta"] < -0.001)
    print(f"\nImproved (cascade > baseline): {improved}/{len(shared)}")
    print(f"Degraded  (cascade < baseline): {degraded}/{len(shared)}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: single-stage EN reranker vs cascade EN→PL reranker."
    )
    parser.add_argument("--api-base-url", default="http://api:8000")
    parser.add_argument("--input-csv", default="src/evaluation/data/questions_with_links.csv")
    parser.add_argument("--output-dir", default="src/evaluation/results/cascade_check")
    parser.add_argument("--ks", default="3,5,10")
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

    print("=" * 60)
    print("RUN 1/2 — baseline: 60 RRF → English CE → top 30")
    print("=" * 60)
    summary_a = _evaluate(gold, ks, args.api_base_url, use_cascade=False, label="baseline")

    print("\n" + "=" * 60)
    print("RUN 2/2 — cascade: 120 RRF → English CE → 60 → Polish CE → top 30")
    print("=" * 60)
    summary_b = _evaluate(gold, ks, args.api_base_url, use_cascade=True, label="cascade")

    print("\n\n========== COMPARISON: baseline vs cascade ==========")
    comparison = _print_comparison(summary_a, summary_b, "baseline", "cascade")

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
            print("CCH RUN 1/2 — baseline (covered subset only)")
            print("=" * 60)
            cch_a = _evaluate(gold_covered, ks, args.api_base_url, use_cascade=False, label="cch_baseline")
            print("\n" + "=" * 60)
            print("CCH RUN 2/2 — cascade (covered subset only)")
            print("=" * 60)
            cch_b = _evaluate(gold_covered, ks, args.api_base_url, use_cascade=True, label="cch_cascade")
            print("\n\n===== CCH COMPARISON: baseline vs cascade (covered only) =====")
            cch_comparison = _print_comparison(cch_a, cch_b, "cch_baseline", "cch_cascade")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"comparison_cascade_{ts}.json"
    payload: dict = {"baseline": summary_a, "cascade": summary_b, "delta": comparison}
    if args.check_coverage:
        payload["cch_baseline"] = cch_a
        payload["cch_cascade"] = cch_b
        payload["cch_delta"] = cch_comparison
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved -> {out_path}")


if __name__ == "__main__":
    main()
