import csv
from dataclasses import dataclass
from statistics import mean

from src.api.retrieval import get_top_k_chunks

# logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


@dataclass
class EvalRow:
    query: str
    relevant_urls: set[str]


def load_gold(path: str) -> list[EvalRow]:
    def normalize_row(raw: dict[str, str | None]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in raw.items():
            if not key:
                continue
            normalized[key.strip().lower()] = (value or "").strip()
        return normalized

    rows: list[EvalRow] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = normalize_row(r)
            query = row.get("query") or row.get("pytanie") or row.get("question", "")
            rel_raw = (
                row.get("relevant_urls")
                or row.get("strona")
                or row.get("url")
                or row.get("link")
                or ""
            )
            if not query or not rel_raw:
                continue
            relevant = {u.strip() for u in rel_raw.split("|") if u.strip()}
            rows.append(EvalRow(query=query, relevant_urls=relevant))
    return rows


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    topk = retrieved[:k]
    return 1.0 if any(url in relevant for url in topk) else 0.0


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, url in enumerate(retrieved[:k], start=1):
        if url in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(gold: list[EvalRow], k: int) -> tuple[float, float]:
    hits = []
    mrrs = []

    for index, row in enumerate(gold):
        retrieved_chunks = get_top_k_chunks(row.query, top_k=k)
        retrieved_urls = [c.get("source_url", "").strip() for c in retrieved_chunks]
        if index < 5:
            print(f"Sample {index + 1}: {row.query}")
            print(
                f" Retrieved {len(retrieved_urls)} chunks from {len(set(retrieved_urls))} unique URLs:"
            )
            # Show unique URLs with their frequency
            from collections import Counter

            url_counts = Counter(retrieved_urls)
            for url, count in url_counts.items():
                print(f"   {url} (x{count})")
            print("-" * 20)
        hits.append(hit_at_k(retrieved_urls, row.relevant_urls, k))
        mrrs.append(mrr_at_k(retrieved_urls, row.relevant_urls, k))

    return mean(hits) if hits else 0.0, mean(mrrs) if mrrs else 0.0


def main() -> None:
    gold_path = "src/evaluation/data/questions_filtered.csv"
    ks = [1, 3, 5, 10]

    gold = load_gold(gold_path)
    print(f"Loaded {len(gold)} evaluation queries")

    for k in ks:
        hit_k, mrr_k = evaluate(gold, k)
        print(f"Hit@{k}: {hit_k:.4f}")
        print(f"MRR@{k}: {mrr_k:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    main()
