"""
Compute retrieval metrics from generated answer CSV files.

This is the lightweight path for files produced by
evaluation.llm_judge.generate_chatbot_answers with columns:
    pytanie, odpowiedz_wygenerowana, zwrocone_linki

Unlike benchmark.py, this module does not call the API, does not import
matplotlib, and handles NotebookLM reference links stored as scraped .txt
filenames, for example:
    ww2.mini.pw.edu.pl_wp-content_uploads_doc.pdf.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from math import log2
from pathlib import Path
from statistics import mean
from urllib.parse import urlsplit, urlunsplit

RELEVANCE_THRESHOLD = 0.25
MRRW_ALPHA = 0.8
MRRW_BETA = 0.4


@dataclass(frozen=True)
class ReferenceRow:
    query: str
    target_id: str
    category: str = ""
    reference_answer: str = ""


@dataclass(frozen=True)
class AnswerRow:
    answer: str
    links: list[str]


def normalize_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    normalized_path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc.lower(), normalized_path, "", ""))


def source_id(value: str) -> str:
    """
    Convert either a URL or a scraped .txt filename into a comparable id.

    The scraper writes files as:
        url.replace("https://", "").replace("/", "_").strip("_")[:200] + ".txt"
    Generated answers store real URLs. Comparing the same safe id avoids false
    misses caused by reference rows that contain scraped filenames instead of URLs.
    """
    cleaned = (value or "").strip().strip('"').strip("'").rstrip(".")
    if not cleaned:
        return ""
    txt_match = re.search(r'"([^"]+\.txt)"', cleaned) or re.search(
        r"(\S+\.txt)", cleaned
    )
    if txt_match:
        cleaned = txt_match.group(1)
    elif cleaned.startswith("Dokument "):
        cleaned = (
            cleaned.removeprefix("Dokument ")
            .strip()
            .strip('"')
            .strip("'")
            .rstrip(".")
        )
    if cleaned.endswith(".txt"):
        cleaned = cleaned[:-4]

    parsed = urlsplit(cleaned)
    if parsed.scheme and parsed.netloc:
        normalized = normalize_url(cleaned)
        if normalized.startswith("https://"):
            normalized = normalized.removeprefix("https://")
        elif normalized.startswith("http://"):
            normalized = normalized.removeprefix("http://")
        return normalized.replace("/", "_").strip("_")[:200]

    return cleaned.strip("_")[:200]


def reconstructed_url_from_source_id(identifier: str) -> str:
    """
    Best-effort URL reconstruction for hierarchical partial credit.

    Exact matching is done by source_id(); this reconstruction is only used to
    award ancestor/descendant credit when the reference is a scraped filename.
    """
    sid = source_id(identifier)
    if not sid or "://" in sid:
        return normalize_url(sid)

    known_hosts = (
        "ww2.mini.pw.edu.pl",
        "ww4.mini.pw.edu.pl",
        "usosweb.usos.pw.edu.pl",
        "www.bip.pw.edu.pl",
        "www.bss.pw.edu.pl",
        "www.pw.edu.pl",
    )
    for known_host in known_hosts:
        prefix = f"{known_host}_"
        if sid == known_host:
            return f"https://{known_host}"
        if sid.startswith(prefix):
            path = sid[len(prefix) :].replace("_", "/")
            return normalize_url(f"https://{known_host}/{path}")
    return normalize_url(f"https://{sid.replace('_', '/')}")


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = source_id(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def hierarchical_relevance(retrieved_url: str, target_url: str) -> float:
    if not retrieved_url or not target_url:
        return 0.0

    t = urlsplit(target_url)
    r = urlsplit(retrieved_url)
    if t.scheme != r.scheme or t.netloc != r.netloc:
        return 0.0

    t_path = t.path.rstrip("/")
    r_path = r.path.rstrip("/")
    if t_path == r_path:
        return 1.0
    if t_path.startswith(r_path + "/"):
        depth = t_path.count("/") - r_path.count("/")
        return 0.75**depth
    if r_path.startswith(t_path + "/"):
        depth = r_path.count("/") - t_path.count("/")
        return 0.75 ** (depth + 1)
    return 0.0


def relevance_score(retrieved: str, target: str) -> float:
    if source_id(retrieved) == source_id(target):
        return 1.0
    return hierarchical_relevance(
        normalize_url(retrieved),
        reconstructed_url_from_source_id(target),
    )


def hit_at_k(rel_scores: list[float], k: int) -> float:
    return 1.0 if any(score >= RELEVANCE_THRESHOLD for score in rel_scores[:k]) else 0.0


def mrr_at_k(rel_scores: list[float], k: int) -> float:
    for rank, score in enumerate(rel_scores[:k], start=1):
        if score >= RELEVANCE_THRESHOLD:
            return 1.0 / rank
    return 0.0


def recall_at_k(rel_scores: list[float], k: int, total_rel: int = 1) -> float:
    if total_rel == 0:
        return 0.0
    hits = sum(1 for score in rel_scores[:k] if score >= RELEVANCE_THRESHOLD)
    return hits / total_rel


def precision_at_k(rel_scores: list[float], k: int) -> float:
    topk = rel_scores[:k]
    if not topk:
        return 0.0
    hits = sum(1 for score in topk if score >= RELEVANCE_THRESHOLD)
    return hits / len(topk)


def f_measure_at_k(rel_scores: list[float], k: int, total_rel: int = 1) -> float:
    p = precision_at_k(rel_scores, k)
    r = recall_at_k(rel_scores, k, total_rel)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def dcg_at_k(rel_scores: list[float], k: int) -> float:
    return sum(
        (2**rel - 1) / log2(rank + 1)
        for rank, rel in enumerate(rel_scores[:k], start=1)
    )


def ndcg_at_k(rel_scores: list[float], k: int) -> float:
    ideal = dcg_at_k(sorted(rel_scores, reverse=True), k)
    return 0.0 if ideal == 0.0 else dcg_at_k(rel_scores, k) / ideal


def average_precision_at_k(rel_scores: list[float], k: int, total_rel: int = 1) -> float:
    if total_rel == 0:
        return 0.0
    ap, hits = 0.0, 0
    for rank, score in enumerate(rel_scores[:k], start=1):
        if score >= RELEVANCE_THRESHOLD:
            hits += 1
            ap += hits / rank
    return ap / total_rel


def r_precision(rel_scores: list[float], total_rel: int = 1) -> float:
    if total_rel == 0:
        return 0.0
    hits = sum(1 for score in rel_scores[:total_rel] if score >= RELEVANCE_THRESHOLD)
    return hits / total_rel


def url_depth_difference(retrieved: str, target: str) -> int | None:
    retrieved_url = normalize_url(retrieved)
    target_url = reconstructed_url_from_source_id(target)
    if source_id(retrieved) == source_id(target):
        return 0

    r = urlsplit(retrieved_url)
    t = urlsplit(target_url)
    if r.scheme != t.scheme or r.netloc != t.netloc:
        return None

    r_parts = [part for part in r.path.strip("/").split("/") if part]
    t_parts = [part for part in t.path.strip("/").split("/") if part]
    min_len = min(len(r_parts), len(t_parts))
    if r_parts[:min_len] != t_parts[:min_len]:
        return None
    return len(r_parts) - len(t_parts)


def mrr_weighted_single(retrieved_urls: list[str], target: str) -> float:
    for rank, url in enumerate(retrieved_urls, start=1):
        depth_diff = url_depth_difference(url, target)
        if depth_diff is None:
            continue
        if depth_diff == 0:
            weight = 1.0
        elif depth_diff > 0:
            weight = MRRW_ALPHA**depth_diff
        else:
            weight = MRRW_BETA ** abs(depth_diff)
        return weight / rank
    return 0.0


def read_reference(path: Path) -> list[ReferenceRow]:
    if path.suffix.lower() == ".jsonl":
        rows: list[ReferenceRow] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                query = (
                    raw.get("Pytanie")
                    or raw.get("Pytanie:")
                    or raw.get("pytanie")
                    or raw.get("query")
                    or raw.get("question")
                    or ""
                )
                target = (
                    raw.get("Źródła")
                    or raw.get("Zrodla")
                    or raw.get("link")
                    or raw.get("strona")
                    or ""
                )
                if query and target:
                    rows.append(
                        ReferenceRow(
                            query=str(query).strip(),
                            target_id=str(target).split("|")[0].strip(),
                            category=str(raw.get("Kategoria") or raw.get("kategoria") or "").strip(),
                            reference_answer=str(raw.get("Odpowiedź") or raw.get("odpowiedz") or "").strip(),
                        )
                    )
        return rows

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[ReferenceRow] = []
        for raw in reader:
            norm = {(key or "").strip().lower(): (value or "").strip() for key, value in raw.items()}
            query = norm.get("pytanie") or norm.get("query") or norm.get("question") or ""
            target = norm.get("link") or norm.get("strona") or norm.get("relevant_urls") or norm.get("url") or ""
            if query and target:
                rows.append(
                    ReferenceRow(
                        query=query,
                        target_id=target.split("|")[0].strip(),
                        category=norm.get("kategoria", ""),
                        reference_answer=norm.get("odpowiedz", ""),
                    )
                )
    return rows


def parse_links(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(";") if item.strip()]

    if not isinstance(data, list):
        return []
    links: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = str(item.get("url") or "").strip()
        else:
            url = str(item).strip()
        if url:
            links.append(url)
    return links


def read_answers(path: Path) -> dict[str, AnswerRow]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        answers: dict[str, AnswerRow] = {}
        for raw in reader:
            norm = {(key or "").strip().lower(): (value or "").strip() for key, value in raw.items()}
            query = norm.get("pytanie") or norm.get("query") or norm.get("question") or ""
            if not query:
                continue
            answer = norm.get("odpowiedz_wygenerowana") or norm.get("answer") or ""
            links = parse_links(norm.get("zwrocone_linki") or norm.get("sources") or "")
            answers[query] = AnswerRow(answer=answer, links=links)
    return answers


def metric_columns_for_mode(ks: list[int], metric_mode: str) -> list[str]:
    columns: list[str] = []
    if metric_mode in ("standard", "both"):
        for k in ks:
            columns.extend(
                [
                    f"hit@{k}",
                    f"mrr@{k}",
                    f"mrrw@{k}",
                    f"recall@{k}",
                    f"precision@{k}",
                    f"f1@{k}",
                    f"ndcg@{k}",
                    f"map@{k}",
                ]
            )
        columns.append("r_prec")
    if metric_mode in ("adaptive", "both"):
        columns.extend(
            [
                "hit_adaptive",
                "mrr_adaptive",
                "mrrw_adaptive",
                "recall_adaptive",
                "precision_adaptive",
                "f1_adaptive",
                "ndcg_adaptive",
                "map_adaptive",
                "r_prec_adaptive",
            ]
        )
    return columns


def evaluate_file(
    answers_csv: Path,
    reference_rows: list[ReferenceRow],
    output_dir: Path,
    ks: list[int],
    metric_mode: str,
) -> dict[str, float | int | str]:
    answers = read_answers(answers_csv)
    per_rows: list[dict[str, str | float | int]] = []
    accum: dict[str, list[float]] = {}

    for ref in reference_rows:
        answer_row = answers.get(ref.query)
        raw_links = answer_row.links if answer_row else []
        links = unique_preserve_order(raw_links)
        rel_scores = [relevance_score(link, ref.target_id) for link in links]

        out_row: dict[str, str | float | int] = {
            "pytanie": ref.query,
            "kategoria": ref.category,
            "odpowiedz_wygenerowana": answer_row.answer if answer_row else "",
            "gold_link": ref.target_id,
            "gold_source_id": source_id(ref.target_id),
            "retrieved_links": ";".join(links),
            "retrieved_source_ids": ";".join(source_id(link) for link in links),
            "retrieved_count": len(links),
            "missing_answer": 0 if answer_row else 1,
        }

        if metric_mode in ("standard", "both"):
            for k in ks:
                metrics = {
                    f"hit@{k}": hit_at_k(rel_scores, k),
                    f"mrr@{k}": mrr_at_k(rel_scores, k),
                    f"mrrw@{k}": mrr_weighted_single(links[:k], ref.target_id),
                    f"recall@{k}": recall_at_k(rel_scores, k),
                    f"precision@{k}": precision_at_k(rel_scores, k),
                    f"f1@{k}": f_measure_at_k(rel_scores, k),
                    f"ndcg@{k}": ndcg_at_k(rel_scores, k),
                    f"map@{k}": average_precision_at_k(rel_scores, k),
                }
                out_row.update(metrics)
                for key, value in metrics.items():
                    accum.setdefault(key, []).append(value)
            r_prec_value = r_precision(rel_scores)
            out_row["r_prec"] = r_prec_value
            accum.setdefault("r_prec", []).append(r_prec_value)

        if metric_mode in ("adaptive", "both"):
            adaptive_k = len(links)
            metrics = {
                "hit_adaptive": hit_at_k(rel_scores, adaptive_k),
                "mrr_adaptive": mrr_at_k(rel_scores, adaptive_k),
                "mrrw_adaptive": mrr_weighted_single(links[:adaptive_k], ref.target_id),
                "recall_adaptive": recall_at_k(rel_scores, adaptive_k),
                "precision_adaptive": precision_at_k(rel_scores, adaptive_k),
                "f1_adaptive": f_measure_at_k(rel_scores, adaptive_k),
                "ndcg_adaptive": ndcg_at_k(rel_scores, adaptive_k),
                "map_adaptive": average_precision_at_k(rel_scores, adaptive_k),
                "r_prec_adaptive": r_precision(rel_scores),
            }
            out_row.update(metrics)
            for key, value in metrics.items():
                accum.setdefault(key, []).append(value)

        per_rows.append(out_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_columns = metric_columns_for_mode(ks, metric_mode)
    per_query_path = output_dir / f"retrieval_metrics_per_query_{answers_csv.stem}.csv"
    fieldnames = [
        "pytanie",
        "kategoria",
        "odpowiedz_wygenerowana",
        "gold_link",
        "gold_source_id",
        "retrieved_links",
        "retrieved_source_ids",
        "retrieved_count",
        "missing_answer",
        *metric_columns,
    ]
    with per_query_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_rows)

    summary: dict[str, float | int | str] = {
        "answers_csv": str(answers_csv),
        "reference_rows": len(reference_rows),
        "answer_rows": len(answers),
        "missing_answers": sum(1 for row in per_rows if row["missing_answer"]),
        "mean_retrieved_count": mean([float(row["retrieved_count"]) for row in per_rows]) if per_rows else 0.0,
        "per_query_csv": str(per_query_path),
    }
    for key in metric_columns:
        values = accum.get(key, [])
        summary[key] = mean(values) if values else 0.0

    summary_csv = output_dir / f"retrieval_metrics_summary_{answers_csv.stem}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    summary_json = output_dir / f"retrieval_metrics_summary_{answers_csv.stem}.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def resolve_answer_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob(pattern) if p.is_file())
    raise FileNotFoundError(f"Answers path not found: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-csv", required=True, type=Path, help="Answer CSV file or directory with answer CSV files.")
    parser.add_argument("--reference-csv", required=True, type=Path, help="Reference CSV with pytanie and link columns.")
    parser.add_argument("--output-dir", default=Path("src/evaluation/results/retrieval_metrics"), type=Path)
    parser.add_argument("--ks", default="3,5,7,10", help="Comma-separated rank cut-offs.")
    parser.add_argument("--metric-mode", choices=["standard", "adaptive", "both"], default="both")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern used when --answers-csv is a directory.")
    args = parser.parse_args(argv)

    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]
    if not ks:
        parser.error("--ks must contain at least one integer")

    reference_rows = read_reference(args.reference_csv)
    if not reference_rows:
        parser.error(f"No usable reference rows found in {args.reference_csv}")

    answer_files = resolve_answer_files(args.answers_csv, args.pattern)
    if not answer_files:
        parser.error(f"No answer CSV files found in {args.answers_csv} matching {args.pattern}")

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%dT%H%M%S")
    summaries = [
        evaluate_file(path, reference_rows, run_dir, ks, args.metric_mode)
        for path in answer_files
    ]

    combined_path = run_dir / "retrieval_metrics_summary_all.csv"
    combined_fields = sorted({key for summary in summaries for key in summary})
    with combined_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=combined_fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Reference rows: {len(reference_rows)}")
    print(f"Answer files: {len(answer_files)}")
    print(f"Output dir: {run_dir}")
    for summary in summaries:
        print(
            f"{Path(str(summary['answers_csv'])).name}: "
            f"answers={summary['answer_rows']} missing={summary['missing_answers']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
