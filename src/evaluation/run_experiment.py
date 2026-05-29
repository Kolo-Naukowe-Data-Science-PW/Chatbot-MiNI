"""
Main CLI runner for RAG ablation experiments.

Usage
-----
    python -m src.evaluation.run_experiment \\
        --variant baseline \\
        --testset human \\
        --judge-model openai/gpt-4o-mini \\
        --output-dir src/data/experiments \\
        --n-questions 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HUMAN_TESTSET_PATH = (
    Path(PROJECT_ROOT) / "src" / "evaluation" / "data" / "questions_with_links.csv"
)
RAG_TESTSET_PATH = Path(PROJECT_ROOT) / "src" / "evaluation" / "data" / "QA_rag.csv"
GENERATED_TESTSET_PATH = (
    Path(PROJECT_ROOT) / "src" / "evaluation" / "data" / "final_notebooklm_QA.jsonl"
)
GOLDEN_ANSWERS_PATH = (
    Path(PROJECT_ROOT) / "src" / "evaluation" / "data" / "golden_answers.csv"
)

# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------


def _load_human_testset(path: Path, n: int | None) -> list[dict]:
    """Load questions_with_links.csv, filtering wymagany kontekst == 0."""
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            norm = {
                (k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k
            }
            if norm.get("wymagany kontekst") != "0":
                continue
            query = norm.get("pytanie") or norm.get("query") or norm.get("question", "")
            url = norm.get("strona") or norm.get("url") or norm.get("link", "")
            url = url.strip().rstrip("/")
            if not query or not url:
                continue
            rows.append({"query": query, "gold_url": url})
            if n and len(rows) >= n:
                break
    logger.info("Loaded %d human-testset rows (wymagany kontekst=0).", len(rows))
    return rows


def _load_rag_testset(path: Path, n: int | None) -> list[dict]:
    """Load QA_rag.csv.  Columns: pytanie, odpowiedz, zrodla, pliki, liczba_plikow."""
    from src.ingestion.ingest_manual_pdfs import URL_MAP as _PDF_URL_MAP

    # Also load the generated file→URL mapping (built by build_file_url_mapping workflow),
    # used for scraped-page TXT references in future QA_rag_extended datasets.
    _file_url_map: dict[str, str] = {}
    _fum_path = (
        Path(PROJECT_ROOT) / "src" / "evaluation" / "data" / "file_url_mapping.json"
    )
    if _fum_path.exists():
        import json as _json

        try:
            with open(_fum_path, encoding="utf-8") as _f:
                _file_url_map = _json.load(_f)
        except Exception:
            pass

    if not path.exists():
        logger.error("RAG testset not found: %s", path)
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for r in reader:
            norm = {
                (k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k
            }
            query = norm.get("pytanie") or norm.get("query", "")
            golden = norm.get("odpowiedz", "")
            if not query:
                continue
            pliki_str = norm.get("pliki", "")
            gold_urls: list[str] = []
            for fname in pliki_str.split(";"):
                fname = fname.strip()
                if fname:
                    url = _PDF_URL_MAP.get(fname) or _file_url_map.get(fname, "")
                    if url:
                        gold_urls.append(url)
            rows.append(
                {"query": query, "golden_answer": golden, "gold_urls": gold_urls}
            )
            if n and len(rows) >= n:
                break
    with_gold = sum(1 for r in rows if r["gold_urls"])
    logger.info("Loaded %d RAG testset rows (%d with gold URLs).", len(rows), with_gold)
    return rows


def _load_generated_testset(path: Path, n: int | None) -> list[dict]:
    """Load final_notebooklm_QA.jsonl.  Fields: Pytanie, Odpowiedź, Źródła."""
    _file_url_map: dict[str, str] = {}
    _fum_path = (
        Path(PROJECT_ROOT) / "src" / "evaluation" / "data" / "file_url_mapping.json"
    )
    if _fum_path.exists():
        try:
            with open(_fum_path, encoding="utf-8") as _f:
                _file_url_map = json.load(_f)
        except Exception:
            pass

    if not path.exists():
        logger.error("Generated testset not found: %s", path)
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            query = (rec.get("Pytanie") or "").strip()
            golden = (rec.get("Odpowiedź") or "").strip()
            if not query:
                continue
            fname = (rec.get("Źródła") or "").strip()
            fname_key = fname[:-4] if fname.endswith(".txt") else fname
            gold_url = _file_url_map.get(fname_key, "")
            rows.append(
                {
                    "query": query,
                    "golden_answer": golden,
                    "gold_urls": [gold_url] if gold_url else [],
                }
            )
            if n and len(rows) >= n:
                break
    with_gold = sum(1 for r in rows if r["gold_urls"])
    logger.info(
        "Loaded %d generated testset rows (%d with gold URLs).", len(rows), with_gold
    )
    return rows


def _load_golden_map(path: Path) -> dict[str, str]:
    """Return {query -> golden_answer_opus} from golden_answers.csv."""
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            query = (r.get("query") or "").strip()
            answer = (r.get("golden_answer_opus") or "").strip()
            if query and answer:
                mapping[query] = answer
    return mapping


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    cleaned = url.strip()
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _hierarchical_relevance(retrieved: str, target: str) -> float:
    from urllib.parse import urlsplit

    if not retrieved or not target:
        return 0.0
    t = urlsplit(target)
    r = urlsplit(retrieved)
    if t.scheme != r.scheme or t.netloc != r.netloc:
        return 0.0
    tp = t.path.rstrip("/")
    rp = r.path.rstrip("/")
    if tp == rp:
        return 1.0
    if tp.startswith(rp + "/"):
        depth = tp.count("/") - rp.count("/")
        return 0.5**depth
    if rp.startswith(tp + "/"):
        depth = rp.count("/") - tp.count("/")
        return 0.5 ** (depth + 1)
    return 0.0


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _depth_diff_url(r_url: str, t_url: str) -> int | None:
    """Signed depth difference between retrieved URL and target URL on the same domain."""
    from urllib.parse import urlsplit

    t = urlsplit(_normalize_url(t_url))
    rv = urlsplit(_normalize_url(r_url))
    if t.scheme != rv.scheme or t.netloc != rv.netloc:
        return None
    tp = t.path.rstrip("/")
    rp = rv.path.rstrip("/")
    if tp == rp:
        return 0
    if tp.startswith(rp + "/"):
        return -(tp.count("/") - rp.count("/"))  # retrieved shallower than target
    if rp.startswith(tp + "/"):
        return rp.count("/") - tp.count("/")  # retrieved deeper than target
    return None


_EVAL_KS = [1, 3, 5, 10]


def _compute_retrieval_metrics(
    retrieved_urls: list[str], gold_urls: list[str]
) -> dict[str, float]:
    """Compute Hit@k, MRR@k, nDCG@k, MRRw@k, MAP@k for k in _EVAL_KS.

    A retrieved document counts as relevant if it matches ANY of the gold URLs
    (multi-gold evaluation: up to all source documents per question are gold).
    """
    from math import log2

    unique_urls = _unique_preserve_order(
        [_normalize_url(u) for u in retrieved_urls if u]
    )
    norm_golds = [_normalize_url(g) for g in gold_urls if g]
    rel_scores = [
        max((_hierarchical_relevance(u, g) for g in norm_golds), default=0.0)
        for u in unique_urls
    ]

    THRESHOLD = 0.5
    results: dict[str, float] = {}

    for k in _EVAL_KS:
        topk = rel_scores[:k]
        # Hit@k
        results[f"hit@{k}"] = 1.0 if any(s >= THRESHOLD for s in topk) else 0.0

        # MRR@k
        mrr = 0.0
        for rank, s in enumerate(topk, start=1):
            if s >= THRESHOLD:
                mrr = 1.0 / rank
                break
        results[f"mrr@{k}"] = mrr

        # nDCG@k
        dcg = sum((2**rel - 1) / log2(r + 1) for r, rel in enumerate(topk, start=1))
        ideal = sorted(topk, reverse=True)
        idcg = sum((2**rel - 1) / log2(r + 1) for r, rel in enumerate(ideal, start=1))
        results[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

        # MRRw@k (simplified with alpha=0.8, beta=0.4)
        from urllib.parse import urlsplit

        def _depth_diff(r_url: str, t_url: str) -> int | None:
            t = urlsplit(_normalize_url(t_url))
            rv = urlsplit(_normalize_url(r_url))
            if t.scheme != rv.scheme or t.netloc != rv.netloc:
                return None
            tp = t.path.rstrip("/")
            rp = rv.path.rstrip("/")
            if tp == rp:
                return 0
            if tp.startswith(rp + "/"):
                return -(tp.count("/") - rp.count("/"))
            if rp.startswith(tp + "/"):
                return rp.count("/") - tp.count("/")
            return None

        mrrw = 0.0
        for rank, url in enumerate(unique_urls[:k], start=1):
            best_s = 0.0
            for g in norm_golds:
                d = _depth_diff(url, g)
                if d is None:
                    continue
                w = 1.0 if d == 0 else (0.8**d if d > 0 else 0.4 ** abs(d))
                best_s = max(best_s, w / rank)
            if best_s > mrrw:
                mrrw = best_s
        results[f"mrrw@{k}"] = mrrw

        # MAP@k
        hits, ap = 0, 0.0
        for i, s in enumerate(topk, start=1):
            if s >= THRESHOLD:
                hits += 1
                ap += hits / i
        results[f"map@{k}"] = ap / 1  # total_rel=1

    return results


def _compute_adaptive_metrics(
    retrieved_urls: list[str], gold_urls: list[str]
) -> dict[str, float]:
    """Same metrics as above but evaluated at k = actual number of retrieved docs.

    A retrieved document counts as relevant if it matches ANY of the gold URLs
    (multi-gold evaluation: up to all source documents per question are gold).
    """
    from math import log2

    unique_urls = _unique_preserve_order(
        [_normalize_url(u) for u in retrieved_urls if u]
    )
    k = len(unique_urls)
    _zero: dict[str, float] = {
        "adaptive_hit": 0.0,
        "adaptive_mrr": 0.0,
        "adaptive_ndcg": 0.0,
        "adaptive_mrrw": 0.0,
        "adaptive_map": 0.0,
        "adaptive_precision": 0.0,
        "adaptive_recall": 0.0,
        "adaptive_k": 0.0,
    }
    if k == 0:
        return _zero

    norm_golds = [_normalize_url(g) for g in gold_urls if g]
    rel_scores = [
        max((_hierarchical_relevance(u, g) for g in norm_golds), default=0.0)
        for u in unique_urls
    ]
    THRESHOLD = 0.5

    hit = 1.0 if any(s >= THRESHOLD for s in rel_scores) else 0.0

    mrr = 0.0
    for rank, s in enumerate(rel_scores, start=1):
        if s >= THRESHOLD:
            mrr = 1.0 / rank
            break

    dcg = sum((2**r - 1) / log2(i + 1) for i, r in enumerate(rel_scores, start=1))
    ideal = sorted(rel_scores, reverse=True)
    idcg = sum((2**r - 1) / log2(i + 1) for i, r in enumerate(ideal, start=1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    mrrw = 0.0
    for rank, url in enumerate(unique_urls, start=1):
        best_s = 0.0
        for g in norm_golds:
            d = _depth_diff_url(url, g)
            if d is None:
                continue
            w = 1.0 if d == 0 else (0.8 ** abs(d) if d > 0 else 0.4 ** abs(d))
            best_s = max(best_s, w / rank)
        if best_s > mrrw:
            mrrw = best_s

    n_rel, ap = 0, 0.0
    for i, s in enumerate(rel_scores, start=1):
        if s >= THRESHOLD:
            n_rel += 1
            ap += n_rel / i

    return {
        "adaptive_hit": hit,
        "adaptive_mrr": mrr,
        "adaptive_ndcg": ndcg,
        "adaptive_mrrw": mrrw,
        "adaptive_map": ap,
        "adaptive_precision": n_rel / k,
        "adaptive_recall": hit,  # single-relevant-doc setup: recall == hit
        "adaptive_k": float(k),
    }


def _get_pr_points(
    retrieved_urls: list[str], gold_urls: list[str]
) -> list[tuple[float, float]]:
    """Return (precision@i, recall@i) for i=1..n_retrieved (single-relevant-doc)."""
    unique_urls = _unique_preserve_order(
        [_normalize_url(u) for u in retrieved_urls if u]
    )
    norm_golds = [_normalize_url(g) for g in gold_urls if g]
    THRESHOLD = 0.5

    points: list[tuple[float, float]] = []
    n_rel = 0
    for i, url in enumerate(unique_urls, start=1):
        if (
            max((_hierarchical_relevance(url, g) for g in norm_golds), default=0.0)
            >= THRESHOLD
        ):
            n_rel += 1
        points.append((n_rel / i, float(n_rel > 0)))
    return points


def _plot_pr_curve(
    all_pr_points: list[list[tuple[float, float]]],
    output_path: Path,
    title: str,
) -> None:
    """Save an interpolated mean precision-recall curve as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not available — skipping P-R curve.")
        return

    non_empty = [pts for pts in all_pr_points if pts]
    if not non_empty:
        return

    recall_levels = np.linspace(0.0, 1.0, 11)
    interpolated: list[list[float]] = []
    for pts in non_empty:
        row = [
            max((p for p, r in pts if r >= thr), default=0.0) for thr in recall_levels
        ]
        interpolated.append(row)

    mean_p = np.mean(interpolated, axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        recall_levels, mean_p, marker="o", linewidth=2, color="#1f77b4", label=title
    )
    ax.fill_between(recall_levels, mean_p, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated Precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved P-R curve -> %s", output_path)


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------


def _generate_answer(query: str, chunks: list[dict]) -> str:
    """Call the LLM with retrieved context and return the generated answer."""
    try:
        from src.api.main import query_llm
        from src.api.prompt_builder import build_messages

        text_chunks = [c.get("text_chunk", "") for c in chunks if c.get("text_chunk")]
        if not text_chunks:
            text_chunks = ["(brak kontekstu)"]
        messages = build_messages(query, text_chunks)
        return query_llm(messages, model_config={"model": "openai/gpt-4o-mini"})
    except Exception as exc:
        logger.warning("LLM answer generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    variant: str,
    testset: str,
    judge_model: str | None,
    output_dir: Path,
    n_questions: int | None,
    subset_dir: Path | None = None,
) -> dict:
    """
    Run the full experiment for one *variant* on one *testset*.

    Returns the summary dict that was saved as JSON.
    """
    from src.evaluation.pipeline_config import ALL_VARIANTS
    from src.evaluation.retrieval_runner import retrieve

    if variant not in ALL_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {list(ALL_VARIANTS)}"
        )

    config = ALL_VARIANTS[variant]
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    logger.info(
        "=== Experiment: variant=%s  testset=%s  judge=%s ===",
        variant,
        testset,
        judge_model,
    )

    # Resolve test-set paths (subset_dir overrides defaults)
    human_path = (subset_dir / "human_subset.csv") if subset_dir else HUMAN_TESTSET_PATH
    rag_path = (subset_dir / "rag_subset.csv") if subset_dir else RAG_TESTSET_PATH
    generated_path = (
        (subset_dir / "generated_subset.jsonl")
        if subset_dir
        else GENERATED_TESTSET_PATH
    )
    if subset_dir:
        logger.info("Subset mode: loading test sets from %s", subset_dir)

    # Load test questions
    if testset == "human":
        questions = _load_human_testset(human_path, n_questions)
        has_gold_urls = True
    elif testset == "rag":
        questions = _load_rag_testset(rag_path, n_questions)
        has_gold_urls = True
    elif testset == "generated":
        questions = _load_generated_testset(generated_path, n_questions)
        has_gold_urls = True
    else:
        raise ValueError(
            f"Unknown testset '{testset}'. Use 'human', 'rag', or 'generated'."
        )

    if not questions:
        raise RuntimeError(f"No questions loaded from testset '{testset}'.")

    # Load golden answers for LLM judge
    golden_map: dict[str, str] = {}
    if judge_model:
        if testset == "human":
            golden_map = _load_golden_map(GOLDEN_ANSWERS_PATH)
            logger.info("Loaded %d golden answers for judge.", len(golden_map))
        elif testset in ("rag", "generated"):
            golden_map = {r["query"]: r.get("golden_answer", "") for r in questions}

    # Per-query accumulators
    per_query_rows: list[dict] = []
    retrieval_metrics_accum: dict[str, list[float]] = {}
    adaptive_metrics_accum: dict[str, list[float]] = {}
    judge_scores_accum: dict[str, list[float]] = {}
    pr_data_per_query: list[list[tuple[float, float]]] = []

    t_start = time.time()

    for idx, row in enumerate(questions):
        query = row["query"]
        # Unify gold URL handling: human testset has single gold_url, rag has list gold_urls.
        if testset == "human":
            _gu = row.get("gold_url", "")
            gold_urls_row: list[str] = [_gu] if _gu else []
        else:
            gold_urls_row = row.get("gold_urls", [])
        gold_url_display = gold_urls_row[0] if gold_urls_row else ""
        logger.info("[%d/%d] %s", idx + 1, len(questions), query[:70])

        # 1. Retrieve
        try:
            chunks = retrieve(query, config)
        except Exception as exc:
            logger.error("Retrieval failed for query '%s': %s", query[:50], exc)
            chunks = []

        retrieved_urls = [c.get("source_url", "") for c in chunks]

        # 2. Generate answer
        generated_answer = _generate_answer(query, chunks)

        # 3. Build per-query row
        csv_row: dict = {
            "query": query,
            "gold_url": gold_url_display,
            "gold_urls_count": len(gold_urls_row),
            "generated_answer": generated_answer,
            "retrieved_urls": ";".join(u for u in retrieved_urls if u),
            "n_retrieved": len(chunks),
        }

        # 4. Retrieval metrics (only when gold URLs available)
        if has_gold_urls and gold_urls_row:
            ret_metrics = _compute_retrieval_metrics(retrieved_urls, gold_urls_row)
            csv_row.update(ret_metrics)
            for k, v in ret_metrics.items():
                retrieval_metrics_accum.setdefault(k, []).append(v)
            # Adaptive metrics (evaluated at actual retrieved k)
            adap = _compute_adaptive_metrics(retrieved_urls, gold_urls_row)
            csv_row.update(adap)
            for k, v in adap.items():
                adaptive_metrics_accum.setdefault(k, []).append(v)
            # P-R points for interpolated curve
            pr_data_per_query.append(_get_pr_points(retrieved_urls, gold_urls_row))

        # 5. LLM judge
        if judge_model:
            golden_answer = golden_map.get(query, "")
            if golden_answer and generated_answer:
                try:
                    from src.evaluation.llm_judge.golden_judge import (
                        judge_against_golden,
                    )

                    result = judge_against_golden(
                        query,
                        generated_answer,
                        golden_answer,
                        judge_model=judge_model,
                        language="pl",
                    )
                    judge_row = {
                        "judge_chatbot_usefulness": result.chatbot.usefulness,
                        "judge_chatbot_accuracy": result.chatbot.accuracy,
                        "judge_chatbot_completeness": result.chatbot.completeness,
                        "judge_golden_usefulness": result.golden.usefulness,
                        "judge_golden_accuracy": result.golden.accuracy,
                        "judge_golden_completeness": result.golden.completeness,
                        "judge_better": result.better,
                        "judge_chatbot_weaknesses": result.chatbot_weaknesses,
                        "judge_golden_weaknesses": result.golden_weaknesses,
                    }
                    csv_row.update(judge_row)
                    for k in [
                        "judge_chatbot_usefulness",
                        "judge_chatbot_accuracy",
                        "judge_chatbot_completeness",
                    ]:
                        judge_scores_accum.setdefault(k, []).append(csv_row[k])
                except Exception as exc:
                    logger.warning("Judge failed for query '%s': %s", query[:50], exc)
            else:
                if not golden_answer:
                    logger.debug("No golden answer found for query: %s", query[:50])

        per_query_rows.append(csv_row)

    runtime = time.time() - t_start

    # --- P-R curve ---
    if pr_data_per_query:
        pr_path = output_dir / f"{variant}_{testset}_{ts}_pr_curve.png"
        _plot_pr_curve(
            pr_data_per_query,
            pr_path,
            title=f"Precision-Recall: {variant} / {testset} (n={len(pr_data_per_query)})",
        )

    # --- Save per-query CSV ---
    per_query_path = output_dir / f"{variant}_{testset}_{ts}_per_query.csv"
    if per_query_rows:
        # Union of all keys (some rows may have fewer columns if judge was skipped)
        all_keys: list[str] = []
        seen_keys: set[str] = set()
        for r in per_query_rows:
            for k in r:
                if k not in seen_keys:
                    all_keys.append(k)
                    seen_keys.add(k)
        with open(per_query_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_query_rows)
        logger.info("Saved per-query results -> %s", per_query_path)

    # --- Build summary ---
    def avg(lst: list[float]) -> float | None:
        return round(mean(lst), 6) if lst else None

    n_with_gold = sum(1 for r in per_query_rows if r.get("gold_urls_count", 0) > 0)
    max_gold = max((r.get("gold_urls_count", 0) for r in per_query_rows), default=0)
    if testset == "rag":
        gold_eval_note = (
            f"multi-gold via PDF_URL_MAP+file_url_mapping (up to {max_gold} gold URLs/question, "
            f"{n_with_gold}/{len(per_query_rows)} questions evaluated)"
        )
    elif testset == "generated":
        gold_eval_note = f"single-gold via file_url_mapping ({n_with_gold}/{len(per_query_rows)} questions with URL)"
    else:
        gold_eval_note = "single-gold (human-labeled URL)"

    summary: dict = {
        "variant": variant,
        "testset": testset,
        "gold_evaluation": gold_eval_note,
        "judge_model": judge_model,
        "timestamp": ts,
        "n_questions": len(questions),
        "runtime_seconds": round(runtime, 2),
        "config": {
            "retrieval_mode": config.retrieval_mode,
            "use_query_rewrite": config.use_query_rewrite,
            "use_rerank": config.use_rerank,
            "rerank_steps": config.rerank_steps,
            "content_type": config.content_type,
            "n_retrieve": config.n_retrieve,
            "n_final": config.n_final,
            "two_stage": config.two_stage,
            "description": config.description,
        },
    }

    for metric, vals in retrieval_metrics_accum.items():
        summary[metric] = avg(vals)

    for metric, vals in adaptive_metrics_accum.items():
        summary[metric] = avg(vals)

    for metric, vals in judge_scores_accum.items():
        summary[metric] = avg(vals)

    summary_path = output_dir / f"{variant}_{testset}_{ts}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved summary JSON -> %s", summary_path)

    # Print summary
    print("\n=== Summary ===")
    for k, v in summary.items():
        if k == "config":
            continue
        print(f"  {k:40s}: {v}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG ablation experiment for a single pipeline variant."
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Pipeline variant name from pipeline_config.ALL_VARIANTS (e.g. baseline).",
    )
    parser.add_argument(
        "--testset",
        required=True,
        choices=["human", "rag", "generated"],
        help="'human' = questions_with_links.csv, 'rag' = QA_rag.csv, 'generated' = final_notebooklm_QA.jsonl.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="OpenRouter model ID for LLM judge (e.g. openai/gpt-4o-mini). "
        "If omitted, LLM judging is skipped.",
    )
    parser.add_argument(
        "--output-dir",
        default="src/data/experiments",
        help="Directory where per-query CSV and summary JSON are saved.",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=0,
        help="Maximum number of questions to evaluate (0 = all).",
    )
    parser.add_argument(
        "--subset-dir",
        default=None,
        help="Path to subset_workspace directory (contains *_subset.csv/.jsonl). "
        "When set, test sets are loaded from there instead of the default data dir.",
    )
    args = parser.parse_args()

    n = args.n_questions if args.n_questions > 0 else None
    output_dir = Path(args.output_dir)
    subset_dir = Path(args.subset_dir) if args.subset_dir else None

    if args.variant == "all":
        from src.evaluation.pipeline_config import ALL_VARIANTS

        for v in ALL_VARIANTS:
            run_experiment(v, args.testset, args.judge_model, output_dir, n, subset_dir)
    else:
        run_experiment(
            args.variant, args.testset, args.judge_model, output_dir, n, subset_dir
        )


if __name__ == "__main__":
    main()
