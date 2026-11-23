"""
Agreement tests over returned links from two generated-answer CSV files.

Inputs are answer CSVs with at least:
    pytanie, zwrocone_linki

Goodman-Kruskal gamma is computed from the rank of the gold/correct link in
each model's returned list. If gold links are not present in the answer CSVs,
pass --gold-csv (default: src/evaluation/data/questions_with_links.csv).

This implementation uses an *adaptive* k: instead of truncating to a fixed
top-k, each model's full list of (unique) returned links for the query is
used as Z_A(q) / Z_B(q). k is therefore however many links the model
actually returned for that particular query.

Kappa is computed over the adaptive returned-link sets for each query:
    U(q) = Z_A(q) ∪ Z_B(q)

Contingency table for kappa (per the paper):

             Z^(A)         U \\ Z^(A)
Z^(B)             a                b
U\\Z^(B)          c                d

  a = |Z_A ∩ Z_B|          (both retrieved it)
  b = |Z_B \\ Z_A|          (B retrieved, A did not)
  c = |Z_A \\ Z_B|          (A retrieved, B did not)
  d = |U \\ (Z_A ∪ Z_B)| = 0

Since U is defined as the union of the two retrieved sets, d is always 0,
which gives:
    p_o = a / |U|
    p_e = (a^2 + a*c + a*b + 2*b*c) / |U|^2
    kappa = (p_o - p_e) / (1 - p_e)

"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from urllib.parse import urlsplit, urlunsplit

DEFAULT_GOLD_CSV = Path("src/evaluation/data/questions_with_links.csv")
GOLD_COLUMNS = (
    "gold_link",
    "gold_links",
    "strona",
    "strona ",
    "link",
    "links",
    "relevant_urls",
    "url",
    "zrodla",
    "źródła",
)


@dataclass(frozen=True)
class AnswerRow:
    links: list[str]
    gold_links: list[str]


@dataclass(frozen=True)
class GammaResult:
    n_queries: int
    n_pairs: int
    concordant: int
    discordant: int
    ties: int
    gamma: float | None
    rank_classes: list[str]
    contingency_counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class KappaResult:
    n_queries: int
    kappa_mean: float | None
    kappa_min: float | None
    kappa_max: float | None
    observed_agreement_mean: float | None
    expected_agreement_mean: float | None
    aggregate_counts: dict[str, int]


# ---------------------------------------------------------------------------
# URL / source normalisation
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    cleaned = (url or "").strip().strip('"').strip("'")
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    normalized_path = parts.path.rstrip("/") or parts.path
    return urlunsplit((parts.scheme, parts.netloc.lower(), normalized_path, "", ""))


def source_id(value: str) -> str:
    cleaned = (value or "").strip().strip('"').strip("'").rstrip(".")
    if not cleaned:
        return ""
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


def unique_source_ids(links: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for link in links:
        sid = source_id(link)
        if sid and sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def split_gold_links(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    for separator in ("|", ";", "\n"):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def parse_returned_links(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [part.strip() for part in text.split(";") if part.strip()]

    if not isinstance(data, list):
        return []

    links: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = str(item.get("url") or item.get("link") or "").strip()
        else:
            url = str(item).strip()
        if url:
            links.append(url)
    return links


def normalized_row(raw: dict[str, str]) -> dict[str, str]:
    return {
        (key or "").strip().lower(): (value or "").strip() for key, value in raw.items()
    }


def extract_query(row: dict[str, str]) -> str:
    return (
        row.get("pytanie")
        or row.get("query")
        or row.get("question")
        or row.get("pytanie:")
        or ""
    ).strip()


def extract_gold_links(row: dict[str, str]) -> list[str]:
    for column in GOLD_COLUMNS:
        value = row.get(column.strip().lower())
        if value:
            return split_gold_links(value)
    return []


def read_answers(path: Path) -> dict[str, AnswerRow]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows: dict[str, AnswerRow] = {}
        for raw in reader:
            row = normalized_row(raw)
            query = extract_query(row)
            if not query:
                continue
            links = parse_returned_links(
                row.get("zwrocone_linki") or row.get("sources") or ""
            )
            rows[query] = AnswerRow(links=links, gold_links=extract_gold_links(row))
    return rows


def read_gold(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        gold: dict[str, list[str]] = {}
        for raw in reader:
            row = normalized_row(raw)
            query = extract_query(row)
            links = extract_gold_links(row)
            if query and links:
                gold[query] = links
    return gold


HIT_COLUMNS = (
    "hit",
    "hit@k",
    "hit_at_k",
    "hit_rate",
    "hit_adaptive",
    "hitw_adaptive",
)


def extract_hit(row: dict[str, str]) -> float | None:
    for column in HIT_COLUMNS:
        value = row.get(column.strip().lower())
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def read_hit_metrics(path: Path) -> dict[str, float]:
    """Read per-query retrieval metrics CSV and return {query: Hit value}."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        result: dict[str, float] = {}
        for raw in reader:
            row = normalized_row(raw)
            query = extract_query(row)
            if not query:
                continue
            hit = extract_hit(row)
            if hit is not None:
                result[query] = hit
    return result


def filter_queries_by_hit(
    queries: list[str],
    hits_a: dict[str, float],
    hits_b: dict[str, float],
) -> list[str]:
    """Keep only queries where Hit == 1 for both model A and model B."""
    return [q for q in queries if hits_a.get(q) == 1 and hits_b.get(q) == 1]


def compute_link_agreement_without_inf(
    queries: list[str],
    answers_a: dict[str, AnswerRow],
    answers_b: dict[str, AnswerRow],
    gold: dict[str, list[str]],
    hits_a: dict[str, float],
    hits_b: dict[str, float],
) -> tuple[GammaResult, KappaResult, list[dict[str, object]], list[str]]:
    """Compute gamma/kappa agreement restricted to queries where Hit == 1
    for both models (read from two per-query retrieval-metrics CSVs).

    Restricting to Hit == 1 for both models guarantees the gold link was
    found in each model's returned list, so rank_A and rank_B are never
    INF_RANK -- the "inf" class is removed from the gamma contingency table.

    Returns (gamma_result, kappa_result, per_query_rows, filtered_queries).
    """
    filtered_queries = filter_queries_by_hit(queries, hits_a, hits_b)

    per_query: list[dict[str, object]] = []
    rank_pairs: list[tuple[int, int]] = []

    for query in filtered_queries:
        row_a = answers_a[query]
        row_b = answers_b[query]
        gold_links = gold_for_query(query, row_a, row_b, gold)

        rank_a = rank_of_gold_link(row_a.links, gold_links)
        rank_b = rank_of_gold_link(row_b.links, gold_links)
        rank_pairs.append((rank_a, rank_b))

        set_a = set(unique_source_ids(row_a.links))
        set_b = set(unique_source_ids(row_b.links))
        relevant = set(unique_source_ids(gold_links))

        kappa_val, p_o, p_e, counts = kappa_for_sets(set_a, set_b)
        per_query.append(
            {
                "pytanie": query,
                "rank_A": "inf" if rank_a == INF_RANK else rank_a,
                "rank_B": "inf" if rank_b == INF_RANK else rank_b,
                "gold_links_count": len(relevant),
                "links_A_count": len(set_a),
                "links_B_count": len(set_b),
                "kappa": kappa_val,
                "observed_agreement": p_o,
                "expected_agreement": p_e,
                **counts,
            }
        )

    gamma_result = compute_gamma(rank_pairs)
    kappa_result = compute_kappa(per_query)
    return gamma_result, kappa_result, per_query, filtered_queries


def gold_for_query(
    query: str,
    a: AnswerRow,
    b: AnswerRow,
    gold: dict[str, list[str]],
) -> list[str]:
    return gold.get(query) or a.gold_links or b.gold_links or []


# ---------------------------------------------------------------------------
# Rank helper
# ---------------------------------------------------------------------------

# Sentinel rank used for "gold link not found among the returned links".
# A model's adaptive k (its number of returned links) varies per query, so a
# fixed top_k+1 cannot be used as the "not retrieved" marker. INF_RANK is
# chosen large enough to never collide with a real (adaptive) rank.
INF_RANK = 1_000_000


def rank_of_gold_link(links: list[str], gold_links: list[str]) -> int:
    """Return 1-based rank of the first gold link found in `links`.

    The full (adaptive) list of unique returned links is searched -- k is
    however many links were actually returned for this query, not a fixed
    top-k cutoff.

    Returns INF_RANK (treated as ∞) if none of the gold links appears
    among the returned links.
    """
    gold_ids = set(unique_source_ids(gold_links))
    if not gold_ids:
        return INF_RANK
    for rank, sid in enumerate(unique_source_ids(links), start=1):
        if sid in gold_ids:
            return rank
    return INF_RANK


# ---------------------------------------------------------------------------
# Goodman-Kruskal gamma
# ---------------------------------------------------------------------------


def compute_gamma(rank_pairs: list[tuple[int, int]]) -> GammaResult:
    """Compute Goodman-Kruskal gamma from (rank_A, rank_B) pairs.

    gamma = (C - D) / (C + D)

    Concordant pair (i, j):  (a_i - a_j)(b_i - b_j) > 0
    Discordant pair  (i, j):  (a_i - a_j)(b_i - b_j) < 0
    Tied pair        (i, j):  product == 0
    """
    concordant = 0
    discordant = 0
    ties = 0

    for i in range(len(rank_pairs)):
        a_i, b_i = rank_pairs[i]
        for j in range(i + 1, len(rank_pairs)):
            a_j, b_j = rank_pairs[j]
            product = (a_i - a_j) * (b_i - b_j)
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                ties += 1

    denom = concordant + discordant
    gamma = None if denom == 0 else (concordant - discordant) / denom

    n = len(rank_pairs)
    max_rank = max(
        (r for pair in rank_pairs for r in pair if r != INF_RANK),
        default=0,
    )
    classes = [str(i) for i in range(1, max_rank + 1)] + ["inf"]
    counts: dict[str, dict[str, int]] = {
        row_label: {col_label: 0 for col_label in classes} for row_label in classes
    }
    for a_rank, b_rank in rank_pairs:
        a_label = "inf" if a_rank == INF_RANK else str(a_rank)
        b_label = "inf" if b_rank == INF_RANK else str(b_rank)
        counts[a_label][b_label] += 1

    return GammaResult(
        n_queries=n,
        n_pairs=n * (n - 1) // 2,
        concordant=concordant,
        discordant=discordant,
        ties=ties,
        gamma=gamma,
        rank_classes=classes,
        contingency_counts=counts,
    )


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def kappa_for_sets(
    set_a: set[str],
    set_b: set[str],
) -> tuple[float, float, float, dict[str, int]]:
    """Compute the kappa coefficient of agreement for a single query.

    U(q) = Z_A(q) ∪ Z_B(q), where Z_A(q)/Z_B(q) are the (adaptive) sets of
    unique links returned by model A/B for this query.

    Contingency table (rows = Z_B membership, cols = Z_A membership):

             Z_A           U \\ Z_A
    Z_B       a               b
    U\\Z_B    c               d

      a = |Z_A ∩ Z_B|          (both retrieved)
      b = |Z_B \\ Z_A|          (B retrieved, A did not)
      c = |Z_A \\ Z_B|          (A retrieved, B did not)
      d = |U \\ (Z_A ∪ Z_B)| = 0   (since U is defined as Z_A ∪ Z_B)

    p_o = (a + d) / |U| = a / |U|
    p_e = (a^2 + a*c + a*b + 2*b*c) / |U|^2
    κ   = (p_o - p_e) / (1 - p_e)
    """
    universe = set_a | set_b
    n = len(universe)
    if n == 0:
        counts = {"a": 0, "b": 0, "c": 0, "d": 0, "universe": 0}
        return 1.0, 1.0, 1.0, counts

    a = len(set_a & set_b)  # both retrieved
    b = len(set_b - set_a)  # B retrieved, A did not  (row Z_B, col U\Z_A)
    c = len(set_a - set_b)  # A retrieved, B did not  (row U\Z_B, col Z_A)
    d = 0  # neither retrieved -- always 0 since U = Z_A ∪ Z_B

    p_o = a / n
    p_e = (a * a + a * c + a * b + 2 * b * c) / (n * n)

    if p_e == 1.0:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return kappa, p_o, p_e, {"a": a, "b": b, "c": c, "d": d, "universe": n}


def compute_kappa(rows: list[dict[str, object]]) -> KappaResult:
    kappas = [float(row["kappa"]) for row in rows]
    observed = [float(row["observed_agreement"]) for row in rows]
    expected = [float(row["expected_agreement"]) for row in rows]
    aggregate: dict[str, int] = {"a": 0, "b": 0, "c": 0, "d": 0, "universe": 0}
    for row in rows:
        for key in aggregate:
            aggregate[key] += int(row[key])

    return KappaResult(
        n_queries=len(rows),
        kappa_mean=mean(kappas) if kappas else None,
        kappa_min=min(kappas) if kappas else None,
        kappa_max=max(kappas) if kappas else None,
        observed_agreement_mean=mean(observed) if observed else None,
        expected_agreement_mean=mean(expected) if expected else None,
        aggregate_counts=aggregate,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.6f}"


def write_per_query(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "pytanie",
        "rank_A",
        "rank_B",
        "gold_links_count",
        "links_A_count",
        "links_B_count",
        "kappa",
        "observed_agreement",
        "expected_agreement",
        "a",  # both retrieved by A and B
        "b",  # retrieved by B only
        "c",  # retrieved by A only
        "d",  # retrieved by neither
        "universe",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def result_to_jsonable(result: object) -> object:
    if hasattr(result, "__dataclass_fields__"):
        return {
            key: result_to_jsonable(getattr(result, key))
            for key in result.__dataclass_fields__
        }
    if isinstance(result, dict):
        return {key: result_to_jsonable(value) for key, value in result.items()}
    if isinstance(result, list):
        return [result_to_jsonable(value) for value in result]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Goodman-Kruskal gamma and kappa agreement over returned links.",
    )
    parser.add_argument(
        "model_a_csv", type=Path, help="Generated answers CSV for model A."
    )
    parser.add_argument(
        "model_b_csv", type=Path, help="Generated answers CSV for model B."
    )
    parser.add_argument("--model-a-name", default="Model A")
    parser.add_argument("--model-b-name", default="Model B")
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=DEFAULT_GOLD_CSV,
        help=(
            "CSV containing gold links per query. Used for gamma ranks and for "
            "the relevant-document part of kappa's universe. "
            f"Default: {DEFAULT_GOLD_CSV}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/evaluation/results/link_agreement"),
    )
    parser.add_argument(
        "--metrics-a-csv",
        type=Path,
        default=None,
        help=(
            "Per-query retrieval-metrics CSV for model A (must contain a "
            "'Hit' column plus 'pytanie'/'query'). If given together with "
            "--metrics-b-csv, an additional 'without inf' agreement report "
            "is computed over queries where Hit == 1 for both models."
        ),
    )
    parser.add_argument(
        "--metrics-b-csv",
        type=Path,
        default=None,
        help="Per-query retrieval-metrics CSV for model B (see --metrics-a-csv).",
    )
    args = parser.parse_args()

    for label, path in [("Model A", args.model_a_csv), ("Model B", args.model_b_csv)]:
        if not path.exists():
            print(f"ERROR: {label} CSV not found: {path}", file=sys.stderr)
            sys.exit(1)

    answers_a = read_answers(args.model_a_csv)
    answers_b = read_answers(args.model_b_csv)
    gold = read_gold(args.gold_csv)
    common_queries = [q for q in answers_a if q in answers_b]

    if not common_queries:
        print(
            "ERROR: No common queries found between the two CSV files.", file=sys.stderr
        )
        sys.exit(1)

    per_query: list[dict[str, object]] = []
    rank_pairs: list[tuple[int, int]] = []
    queries_with_gold = 0

    for query in common_queries:
        row_a = answers_a[query]
        row_b = answers_b[query]
        gold_links = gold_for_query(query, row_a, row_b, gold)
        if gold_links:
            queries_with_gold += 1

        # Ranks for Goodman-Kruskal gamma (adaptive k = number of links
        # actually returned per model for this query)
        rank_a = rank_of_gold_link(row_a.links, gold_links)
        rank_b = rank_of_gold_link(row_b.links, gold_links)
        rank_pairs.append((rank_a, rank_b))

        # Adaptive source-id sets for kappa: U(q) = Z_A(q) ∪ Z_B(q)
        set_a = set(unique_source_ids(row_a.links))
        set_b = set(unique_source_ids(row_b.links))
        relevant = set(unique_source_ids(gold_links))

        kappa_val, p_o, p_e, counts = kappa_for_sets(set_a, set_b)
        per_query.append(
            {
                "pytanie": query,
                "rank_A": "inf" if rank_a == INF_RANK else rank_a,
                "rank_B": "inf" if rank_b == INF_RANK else rank_b,
                "gold_links_count": len(relevant),
                "links_A_count": len(set_a),
                "links_B_count": len(set_b),
                "kappa": kappa_val,
                "observed_agreement": p_o,
                "expected_agreement": p_e,
                **counts,
            }
        )

    gamma_result = compute_gamma(rank_pairs)
    kappa_result = compute_kappa(per_query)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"link_agreement_{ts}.json"
    out_csv = args.output_dir / f"link_agreement_per_query_{ts}.csv"
    out_md = args.output_dir / f"link_agreement_report_{ts}.md"

    payload = {
        "timestamp": ts,
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "model_a_csv": str(args.model_a_csv),
        "model_b_csv": str(args.model_b_csv),
        "gold_csv": str(args.gold_csv) if args.gold_csv else None,
        "k": "adaptive (per-query, per-model count of returned links)",
        "n_model_a_queries": len(answers_a),
        "n_model_b_queries": len(answers_b),
        "n_common_queries": len(common_queries),
        "queries_with_gold_links": queries_with_gold,
        "goodman_kruskal_gamma": result_to_jsonable(gamma_result),
        "kappa": result_to_jsonable(kappa_result),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_per_query(out_csv, per_query)

    # Contingency table rows for the markdown report
    classes = gamma_result.rank_classes
    header = "| rank_A \\ rank_B | " + " | ".join(classes) + " |"
    separator = "|" + "---|" * (len(classes) + 1)
    table_rows = [header, separator]
    for row_label in classes:
        cells = " | ".join(
            str(gamma_result.contingency_counts[row_label][col_label])
            for col_label in classes
        )
        table_rows.append(f"| {row_label} | {cells} |")
    contingency_md = "\n".join(table_rows)

    report_lines = [
        "# Link Agreement Report",
        "",
        f"Model A: `{args.model_a_name}` (`{args.model_a_csv}`)",
        f"Model B: `{args.model_b_name}` (`{args.model_b_csv}`)",
        f"Gold CSV: `{args.gold_csv}`",
        "k: adaptive (per-query, per-model number of returned links)",
        "",
        "## Dataset",
        "",
        f"- Model A queries: `{len(answers_a)}`",
        f"- Model B queries: `{len(answers_b)}`",
        f"- Common queries: `{len(common_queries)}`",
        f"- Queries with gold links: `{queries_with_gold}`",
        "",
        "## Goodman-Kruskal Gamma",
        "",
        f"- **Gamma**: `{format_float(gamma_result.gamma)}`",
        f"- Concordant pairs C: `{gamma_result.concordant}`",
        f"- Discordant pairs D: `{gamma_result.discordant}`",
        f"- Tied pairs: `{gamma_result.ties}`",
        f"- Total comparable pairs: `{gamma_result.n_pairs}`",
        "",
        "### Contingency table (rank_A rows × rank_B cols)",
        "",
        contingency_md,
        "",
        "## Kappa",
        "",
        f"- **Mean kappa**: `{format_float(kappa_result.kappa_mean)}`",
        f"- Min kappa: `{format_float(kappa_result.kappa_min)}`",
        f"- Max kappa: `{format_float(kappa_result.kappa_max)}`",
        f"- Mean observed agreement p_o: `{format_float(kappa_result.observed_agreement_mean)}`",
        f"- Mean expected agreement p_e: `{format_float(kappa_result.expected_agreement_mean)}`",
        "",
        "### Aggregate contingency counts",
        "",
        "| cell | meaning | count |",
        "|------|---------|-------|",
        f"| a | both A and B retrieved | `{kappa_result.aggregate_counts['a']}` |",
        f"| b | B retrieved, A did not | `{kappa_result.aggregate_counts['b']}` |",
        f"| c | A retrieved, B did not | `{kappa_result.aggregate_counts['c']}` |",
        f"| d | neither retrieved | `{kappa_result.aggregate_counts['d']}` |",
        f"| U | total universe size | `{kappa_result.aggregate_counts['universe']}` |",
        "",
        "## Outputs",
        "",
        f"- JSON: `{out_json.name}`",
        f"- Per-query CSV: `{out_csv.name}`",
        f"- Report: `{out_md.name}`",
    ]
    report_text = "\n".join(report_lines) + "\n"
    out_md.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Results saved -> {args.output_dir}")

    # -----------------------------------------------------------------
    # Optional "without inf" agreement: restrict to queries where Hit == 1
    # for both models (requires per-query retrieval-metrics CSVs).
    # -----------------------------------------------------------------
    if args.metrics_a_csv and args.metrics_b_csv:
        for label, path in [
            ("Model A metrics", args.metrics_a_csv),
            ("Model B metrics", args.metrics_b_csv),
        ]:
            if not path.exists():
                print(f"ERROR: {label} CSV not found: {path}", file=sys.stderr)
                sys.exit(1)

        hits_a = read_hit_metrics(args.metrics_a_csv)
        hits_b = read_hit_metrics(args.metrics_b_csv)

        gamma_ni, kappa_ni, per_query_ni, filtered_queries = (
            compute_link_agreement_without_inf(
                common_queries, answers_a, answers_b, gold, hits_a, hits_b
            )
        )

        out_json_ni = args.output_dir / f"link_agreement_no_inf_{ts}.json"
        out_csv_ni = args.output_dir / f"link_agreement_no_inf_per_query_{ts}.csv"

        payload_ni = {
            "timestamp": ts,
            "model_a_name": args.model_a_name,
            "model_b_name": args.model_b_name,
            "model_a_csv": str(args.model_a_csv),
            "model_b_csv": str(args.model_b_csv),
            "metrics_a_csv": str(args.metrics_a_csv),
            "metrics_b_csv": str(args.metrics_b_csv),
            "gold_csv": str(args.gold_csv) if args.gold_csv else None,
            "filter": "Hit == 1 for both Model A and Model B",
            "n_common_queries": len(common_queries),
            "n_queries_no_inf": len(filtered_queries),
            "goodman_kruskal_gamma": result_to_jsonable(gamma_ni),
            "kappa": result_to_jsonable(kappa_ni),
        }
        with out_json_ni.open("w", encoding="utf-8") as f:
            json.dump(payload_ni, f, indent=2, ensure_ascii=False)

        write_per_query(out_csv_ni, per_query_ni)

        print(
            f"\n[no-inf] Queries with Hit==1 for both models: "
            f"{len(filtered_queries)} / {len(common_queries)}"
        )
        print(f"[no-inf] Gamma: {format_float(gamma_ni.gamma)}")
        print(f"[no-inf] Mean kappa: {format_float(kappa_ni.kappa_mean)}")
        print(f"[no-inf] Results saved -> {out_json_ni}, {out_csv_ni}")


if __name__ == "__main__":
    main()
