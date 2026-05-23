"""
Text-generation evaluation metrics for MiNIonek.

Computes BLEU, ROUGE-{1,2,L,W,S}, METEOR and BERTScore (P/R/F1)
by comparing generated answers against reference answers in QA_rag.csv.

Usage
-----
python -m evaluation.text_metrics \
    --generated-csv  <path/to/generated_answers.csv> \
    --output-dir     src/evaluation/results \
    [--reference-csv src/evaluation/data/QA_rag.csv] \
    [--lang          pl] \
    [--bertscore-model bert-base-multilingual-cased]

Expected columns in --generated-csv:
    pytanie     — question text (used to join with reference)
    odpowiedz   — generated answer

Expected columns in --reference-csv:
    pytanie     — question text
    odpowiedz   — reference / gold answer
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — these heavy packages are only needed at runtime
# ---------------------------------------------------------------------------

def _bleu_score(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    import sacrebleu  # type: ignore
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "bleu": result.score / 100,
        "bleu_1": result.precisions[0] / 100,
        "bleu_2": result.precisions[1] / 100,
        "bleu_3": result.precisions[2] / 100,
        "bleu_4": result.precisions[3] / 100,
    }


def _rouge_scores(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    from rouge import Rouge  # type: ignore
    rouge = Rouge()
    # rouge expects non-empty strings
    pairs = [(h, r) for h, r in zip(hypotheses, references) if h.strip() and r.strip()]
    if not pairs:
        return {}
    hyps, refs = zip(*pairs)
    scores = rouge.get_scores(list(hyps), list(refs), avg=True)
    flat: dict[str, float] = {}
    for key, sub in scores.items():
        clean = key.replace("-", "_")
        for metric, val in sub.items():
            flat[f"{clean}_{metric}"] = val
    return flat


def _meteor_score(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    import nltk  # type: ignore
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    from nltk.translate.meteor_score import meteor_score  # type: ignore
    scores = [
        meteor_score([ref.split()], hyp.split())
        for hyp, ref in zip(hypotheses, references)
        if hyp.strip() and ref.strip()
    ]
    return {"meteor": sum(scores) / len(scores) if scores else 0.0}


def _bertscore(
    hypotheses: list[str],
    references: list[str],
    lang: str,
    model_type: str,
) -> dict[str, float]:
    from bert_score import score as bs_score  # type: ignore
    P, R, F1 = bs_score(
        hypotheses, references, lang=lang, model_type=model_type, verbose=False
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


# ---------------------------------------------------------------------------
# Per-row ROUGE helper (for the per-question CSV)
# ---------------------------------------------------------------------------

def _rouge_per_row(hyp: str, ref: str) -> dict[str, float]:
    if not hyp.strip() or not ref.strip():
        return {}
    try:
        from rouge import Rouge  # type: ignore
        rouge = Rouge()
        scores = rouge.get_scores(hyp, ref, avg=True)
        flat: dict[str, float] = {}
        for key, sub in scores.items():
            clean = key.replace("-", "_")
            for metric, val in sub.items():
                flat[f"{clean}_{metric}"] = val
        return flat
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    generated_csv: Path,
    reference_csv: Path,
    output_dir: Path,
    lang: str,
    bertscore_model: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    logger.info("Loading generated answers from %s", generated_csv)
    gen_df = pd.read_csv(generated_csv, sep=None, engine="python")
    gen_df.columns = gen_df.columns.str.strip()

    logger.info("Loading reference answers from %s", reference_csv)
    ref_df = pd.read_csv(reference_csv, sep=None, engine="python")
    ref_df.columns = ref_df.columns.str.strip()

    # Normalise column names to lowercase
    gen_df = gen_df.rename(columns=str.lower)
    ref_df = ref_df.rename(columns=str.lower)

    if "pytanie" not in gen_df.columns or "odpowiedz" not in gen_df.columns:
        raise ValueError("--generated-csv must have 'pytanie' and 'odpowiedz' columns")
    if "pytanie" not in ref_df.columns or "odpowiedz" not in ref_df.columns:
        raise ValueError("--reference-csv must have 'pytanie' and 'odpowiedz' columns")

    merged = gen_df[["pytanie", "odpowiedz"]].merge(
        ref_df[["pytanie", "odpowiedz"]].rename(columns={"odpowiedz": "odpowiedz_ref"}),
        on="pytanie",
        how="inner",
    )

    if merged.empty:
        raise ValueError("No matching questions found between generated and reference CSVs")

    logger.info("Matched %d questions", len(merged))

    hypotheses = merged["odpowiedz"].fillna("").astype(str).tolist()
    references = merged["odpowiedz_ref"].fillna("").astype(str).tolist()

    results: dict[str, float] = {}

    logger.info("Computing BLEU …")
    results.update(_bleu_score(hypotheses, references))

    logger.info("Computing ROUGE …")
    results.update(_rouge_scores(hypotheses, references))

    logger.info("Computing METEOR …")
    results.update(_meteor_score(hypotheses, references))

    logger.info("Computing BERTScore (model=%s) …", bertscore_model)
    results.update(_bertscore(hypotheses, references, lang, bertscore_model))

    # ------------------------------------------------------------------
    # Per-question CSV
    # ------------------------------------------------------------------
    per_q_rows = []
    for _, row in merged.iterrows():
        entry: dict = {"pytanie": row["pytanie"], "odpowiedz": row["odpowiedz"], "odpowiedz_ref": row["odpowiedz_ref"]}
        entry.update(_rouge_per_row(str(row["odpowiedz"]), str(row["odpowiedz_ref"])))
        per_q_rows.append(entry)

    per_q_df = pd.DataFrame(per_q_rows)
    per_q_path = output_dir / f"text_metrics_per_query_{ts}.csv"
    per_q_df.to_csv(per_q_path, index=False, encoding="utf-8")
    logger.info("Per-query results → %s", per_q_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_path = output_dir / f"text_metrics_summary_{ts}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"n_questions": len(merged), **results}, f, ensure_ascii=False, indent=2)
    logger.info("Summary → %s", summary_path)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    md_lines = [
        "# Text Metrics Evaluation",
        "",
        f"**Questions evaluated:** {len(merged)}  ",
        f"**Generated answers:** `{generated_csv}`  ",
        f"**Reference:** `{reference_csv}`  ",
        f"**Timestamp:** {ts}",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for k, v in sorted(results.items()):
        md_lines.append(f"| {k} | {v:.4f} |")

    md_path = output_dir / f"text_metrics_report_{ts}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Markdown report → %s", md_path)

    # Print summary to stdout
    print("\n=== Text Metrics Summary ===")
    for k, v in sorted(results.items()):
        print(f"  {k:<35} {v:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_reference() -> Path:
    here = Path(__file__).parent
    return here / "data" / "QA_rag.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated answers with NLP metrics")
    parser.add_argument("--generated-csv", required=True, type=Path, help="CSV with generated answers (pytanie, odpowiedz)")
    parser.add_argument("--reference-csv", type=Path, default=_default_reference(), help="Reference CSV (default: QA_rag.csv)")
    parser.add_argument("--output-dir", type=Path, default=Path("src/evaluation/results"), help="Where to save results")
    parser.add_argument("--lang", default="pl", help="Language code for BERTScore (default: pl)")
    parser.add_argument("--bertscore-model", default="allegro/herbert-base-cased", help="HuggingFace model for BERTScore")
    args = parser.parse_args()

    evaluate(
        generated_csv=args.generated_csv,
        reference_csv=args.reference_csv,
        output_dir=args.output_dir,
        lang=args.lang,
        bertscore_model=args.bertscore_model,
    )
