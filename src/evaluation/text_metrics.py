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
    pytanie                  — question text (used to join with reference)
    odpowiedz_wygenerowana   — generated answer

Expected columns in --reference-csv:
    pytanie     — question text
    odpowiedz   — reference / gold answer

Reference files can also be JSONL with NotebookLM-style keys:
    Pytanie
    Odpowiedź
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


QUESTION_COL = "pytanie"
GENERATED_COL = "odpowiedz_wygenerowana"
REFERENCE_COL = "odpowiedz"
REFERENCE_JOIN_COL = "odpowiedz_ref"


# ---------------------------------------------------------------------------
# Lazy imports — these heavy packages are only needed at runtime
# ---------------------------------------------------------------------------


def _normalise_column_name(column: str) -> str:
    return (
        column.strip()
        .lower()
        .replace("ź", "z")
        .replace("ż", "z")
        .replace("ó", "o")
        .replace("ą", "a")
        .replace("ć", "c")
        .replace("ę", "e")
        .replace("ł", "l")
        .replace("ń", "n")
        .replace("ś", "s")
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        with path.open("r", encoding="utf-8-sig") as f:
            header = f.readline()
        sep = "|" if "|" in header else ","
        df = pd.read_csv(path, sep=sep, engine="python")

    df.columns = [_normalise_column_name(str(col)) for col in df.columns]
    return df


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


def _rouge_scores(
    hypotheses: list[str],
    references: list[str | list[str]],
    *,
    beta: float = 1.0,
    rouge_w_alpha: float = 2.0,
    rouge_s_d: int | None = None,
    use_jackknife: bool = False,
) -> dict[str, float]:
    """
    Computes ROUGE-N (n=1,2), ROUGE-L, ROUGE-W and ROUGE-S scores,
    consistent with the LaTeX definitions.

    Args:
        hypotheses:  list of candidate strings.
        references:  list of reference strings **or** list of lists of
                     reference strings (multiple references per candidate).
        beta:        F-measure parameter (β=1 → equal weight; β→∞ → recall).
        rouge_w_alpha: exponent α for ROUGE-W weighting f(k)=k^α (α>1).
        rouge_s_d:   maximum skip distance for ROUGE-S (None = unlimited).
        use_jackknife: if True and multiple references are given, apply the
                       jackknifing procedure from the original paper.
    Returns:
        Flat dict with keys like  rouge_1_r, rouge_l_f, rouge_w_f, …
    """
    import re
    from collections import Counter

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    # ---------- ROUGE-N ------------------------------------------------ #

    def _rouge_n_single(
        hyp_tok: list[str], ref_tok: list[str], n: int
    ) -> dict[str, float]:
        ref_ng = _ngrams(ref_tok, n)
        hyp_ng = _ngrams(hyp_tok, n)
        match = sum((hyp_ng & ref_ng).values())  # count_match
        denom_r = sum(ref_ng.values())
        denom_p = sum(hyp_ng.values())
        r = match / denom_r if denom_r else 0.0
        p = match / denom_p if denom_p else 0.0
        f = _f(r, p)
        return {"r": r, "p": p, "f": f}

    def _rouge_n_best(
        hyp_tok: list[str], refs_tok: list[list[str]], n: int
    ) -> dict[str, float]:
        """Pick r* = argmax ROUGE-N(c, r)  (multi-reference, eq. in paper)."""
        return max(
            (_rouge_n_single(hyp_tok, r, n) for r in refs_tok),
            key=lambda d: d["r"],
        )

    def _rouge_n_jackknife(
        hyp_tok: list[str], refs_tok: list[list[str]], n: int
    ) -> dict[str, float]:
        """Jackknife estimate (eq. in paper): average over leave-one-out best scores."""
        M = len(refs_tok)
        if M == 1:
            return _rouge_n_single(hyp_tok, refs_tok[0], n)
        scores = []
        for i in range(M):
            remaining = refs_tok[:i] + refs_tok[i + 1 :]
            scores.append(_rouge_n_best(hyp_tok, remaining, n))
        return {k: sum(s[k] for s in scores) / M for k in ("r", "p", "f")}

    # ---------- LCS helpers -------------------------------------------- #

    def _lcs_len(a: list[str], b: list[str]) -> int:
        """Standard DP LCS length."""
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]

    def _lcs_indices(a: list[str], b: list[str]) -> set[int]:
        """
        Returns the set of indices in *a* that belong to LCS(a, b).
        Used for LCS_∪ in ROUGE-L_sum.
        """
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # backtrack
        idx: set[int] = set()
        i, j = m, n
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                idx.add(i - 1)  # 0-based index in a
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return idx

    # ---------- ROUGE-L ------------------------------------------------ #

    def _rouge_l_single(hyp_tok: list[str], ref_tok: list[str]) -> dict[str, float]:
        lcs = _lcs_len(ref_tok, hyp_tok)
        r = lcs / len(ref_tok) if ref_tok else 0.0
        p = lcs / len(hyp_tok) if hyp_tok else 0.0
        return {"r": r, "p": p, "f": _f(r, p)}

    # ---------- ROUGE-W ------------------------------------------------ #

    def _wlcs(ref: list[str], hyp: list[str], alpha: float) -> float:
        """
        WLCS with f(k)=k^alpha.
        Returns c(m,n) as defined in the LaTeX.
        """
        m, n = len(ref), len(hyp)
        # w[i][j] = length of consecutive match ending at (i,j)
        w = [[0] * (n + 1) for _ in range(m + 1)]
        c = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    w[i][j] = w[i - 1][j - 1] + 1
                    wk = w[i][j]
                    wk1 = w[i - 1][j - 1]
                    c[i][j] = c[i - 1][j - 1] + wk**alpha - wk1**alpha
                else:
                    w[i][j] = 0
                    c[i][j] = max(c[i - 1][j], c[i][j - 1])
        return c[m][n]

    def _rouge_w_single(
        hyp_tok: list[str], ref_tok: list[str], alpha: float
    ) -> dict[str, float]:
        m, n = len(ref_tok), len(hyp_tok)
        if m == 0 or n == 0:
            return {"r": 0.0, "p": 0.0, "f": 0.0}
        score = _wlcs(ref_tok, hyp_tok, alpha)
        fm = m**alpha
        fn = n**alpha
        r = (score / fm) ** (1.0 / alpha)
        p = (score / fn) ** (1.0 / alpha)
        return {"r": r, "p": p, "f": _f(r, p)}

    # ---------- ROUGE-S ------------------------------------------------ #

    def _skip2_set(tokens: list[str], d: int | None) -> Counter:
        """
        Returns Counter of skip-bigram pairs (w_i, w_j), i<j,
        optionally constrained to j-i <= d.
        """
        pairs: Counter = Counter()
        n = len(tokens)
        for i in range(n):
            j_max = (i + d) if d is not None else (n - 1)
            j_max = min(j_max, n - 1)
            for j in range(i + 1, j_max + 1):
                pairs[(tokens[i], tokens[j])] += 1
        return pairs

    def _rouge_s_single(
        hyp_tok: list[str], ref_tok: list[str], d: int | None
    ) -> dict[str, float]:
        """
        ROUGE-S: skip-bigram matching.

        For sequences with <2 tokens, skip-bigrams cannot be formed,
        so return zero scores explicitly.
        """
        if len(ref_tok) < 2 or len(hyp_tok) < 2:
            return {"r": 0.0, "p": 0.0, "f": 0.0}
        ref_s = _skip2_set(ref_tok, d)
        hyp_s = _skip2_set(hyp_tok, d)
        match = sum((ref_s & hyp_s).values())
        denom_r = sum(ref_s.values())
        denom_p = sum(hyp_s.values())
        r = match / denom_r if denom_r else 0.0
        p = match / denom_p if denom_p else 0.0
        return {"r": r, "p": p, "f": _f(r, p)}

    # ---------- F-measure ---------------------------------------------- #

    def _f(r: float, p: float) -> float:
        denom = r + beta**2 * p
        return (1 + beta**2) * r * p / denom if denom else 0.0

    # ------------------------------------------------------------------ #
    # normalise references to list[list[str]]                             #
    # ------------------------------------------------------------------ #

    refs_list: list[list[list[str]]] = (
        []
    )  # refs_list[i] = list of tokenised refs for hyp i
    hyps_tok: list[list[str]] = []

    for h, r in zip(hypotheses, references, strict=False):
        h_tok = _tokenize(h)
        if not h_tok:
            continue
        if isinstance(r, str):
            r_toks = [_tokenize(r)]
        else:
            r_toks = [_tokenize(ri) for ri in r]
        r_toks = [t for t in r_toks if t]  # drop empty references
        if not r_toks:
            continue
        hyps_tok.append(h_tok)
        refs_list.append(r_toks)

    if not hyps_tok:
        return {}

    # ------------------------------------------------------------------ #
    # accumulate per-sentence scores                                      #
    # ------------------------------------------------------------------ #

    choose_fn = _rouge_n_jackknife if use_jackknife else _rouge_n_best

    agg: dict[str, list[float]] = {
        k: []
        for k in (
            "rouge_1_r",
            "rouge_1_p",
            "rouge_1_f",
            "rouge_2_r",
            "rouge_2_p",
            "rouge_2_f",
            "rouge_l_r",
            "rouge_l_p",
            "rouge_l_f",
            "rouge_w_r",
            "rouge_w_p",
            "rouge_w_f",
            "rouge_s_r",
            "rouge_s_p",
            "rouge_s_f",
        )
    }

    for hyp_tok, refs_tok in zip(hyps_tok, refs_list, strict=False):

        for n, prefix in ((1, "rouge_1"), (2, "rouge_2")):
            s = choose_fn(hyp_tok, refs_tok, n)
            for k in ("r", "p", "f"):
                agg[f"{prefix}_{k}"].append(s[k])

        # ROUGE-L: best reference (same selection rule as ROUGE-N multi)
        sl = max(
            (_rouge_l_single(hyp_tok, r) for r in refs_tok),
            key=lambda d: d["r"],
        )
        for k in ("r", "p", "f"):
            agg[f"rouge_l_{k}"].append(sl[k])

        # ROUGE-W: best reference
        sw = max(
            (_rouge_w_single(hyp_tok, r, rouge_w_alpha) for r in refs_tok),
            key=lambda d: d["r"],
        )
        for k in ("r", "p", "f"):
            agg[f"rouge_w_{k}"].append(sw[k])

        # ROUGE-S: best reference
        ss = max(
            (_rouge_s_single(hyp_tok, r, rouge_s_d) for r in refs_tok),
            key=lambda d: d["r"],
        )
        for k in ("r", "p", "f"):
            agg[f"rouge_s_{k}"].append(ss[k])

    # ------------------------------------------------------------------ #
    # average over corpus                                                  #
    # ------------------------------------------------------------------ #

    return {key: sum(vals) / len(vals) for key, vals in agg.items() if vals}


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
    from nltk.tokenize import word_tokenize  # type: ignore
    from nltk.translate.meteor_score import meteor_score  # type: ignore

    scores = [
        meteor_score([word_tokenize(ref)], word_tokenize(hyp))
        for hyp, ref in zip(hypotheses, references, strict=False)
        if hyp.strip() and ref.strip()
    ]
    return {"meteor": sum(scores) / len(scores) if scores else 0.0}


def _bertscore(
    hypotheses: list[str],
    references: list[str],
    lang: str,
    model_type: str,
    num_layers: int | None,
) -> dict[str, float]:
    from bert_score import BERTScorer  # type: ignore

    results: dict[str, float] = {}

    variants = [
        ("base", False, False),
        ("idf", True, False),
        ("rescaled", False, True),
        ("full", True, True),
    ]

    for suffix, idf, rescale in variants:
        scorer_kwargs = {
            "lang": lang,
            "model_type": model_type,
            "idf": idf,
            "rescale_with_baseline": rescale,
        }
        if num_layers is not None:
            scorer_kwargs["num_layers"] = num_layers

        try:
            scorer = BERTScorer(**scorer_kwargs)
        except (KeyError, ValueError) as exc:
            if rescale:
                logger.warning(
                    "BERTScore variant '%s' skipped for model=%s: %s",
                    suffix,
                    model_type,
                    exc,
                )
                continue
            raise

        try:
            P, R, F1 = scorer.score(hypotheses, references, verbose=False)
        except Exception as exc:
            logger.warning(
                "BERTScore variant '%s' skipped for model=%s: %s",
                suffix,
                model_type,
                exc,
            )
            continue

        results[f"bertscore_precision_{suffix}"] = P.mean().item()
        results[f"bertscore_recall_{suffix}"] = R.mean().item()
        results[f"bertscore_f1_{suffix}"] = F1.mean().item()

    if not results:
        logger.warning("All BERTScore variants failed for model=%s", model_type)

    return results


# ---------------------------------------------------------------------------
# Per-row ROUGE helper (for the per-question CSV)
# ---------------------------------------------------------------------------


def _rouge_per_row(hyp: str, ref: str) -> dict[str, float]:
    if not hyp.strip() or not ref.strip():
        return {}
    try:
        return _rouge_scores([hyp], [ref])
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
    bertscore_num_layers: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")

    logger.info("Loading generated answers from %s", generated_csv)
    gen_df = _read_table(generated_csv)

    logger.info("Loading reference answers from %s", reference_csv)
    ref_df = _read_table(reference_csv)

    if QUESTION_COL not in gen_df.columns or GENERATED_COL not in gen_df.columns:
        raise ValueError("--generated-csv must have 'pytanie' and 'odpowiedz_wygenerowana' columns")
    if QUESTION_COL not in ref_df.columns or REFERENCE_COL not in ref_df.columns:
        raise ValueError("--reference-csv must have 'pytanie' and 'odpowiedz' columns")

    gen_eval_df = gen_df[[QUESTION_COL, GENERATED_COL]].dropna(subset=[QUESTION_COL])
    ref_eval_df = ref_df[[QUESTION_COL, REFERENCE_COL]].dropna(subset=[QUESTION_COL])
    ref_eval_df = ref_eval_df.drop_duplicates(subset=[QUESTION_COL], keep="first")

    merged = gen_eval_df.merge(
        ref_eval_df.rename(columns={REFERENCE_COL: REFERENCE_JOIN_COL}),
        on=QUESTION_COL,
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No matching questions found between generated and reference CSVs"
        )

    skipped = len(gen_eval_df) - len(merged)
    logger.info(
        "Matched %d/%d generated questions against %d reference questions; skipped %d without reference",
        len(merged),
        len(gen_eval_df),
        len(ref_eval_df),
        skipped,
    )

    hypotheses = merged[GENERATED_COL].fillna("").astype(str).tolist()
    references = merged[REFERENCE_JOIN_COL].fillna("").astype(str).tolist()

    results: dict[str, float] = {}

    logger.info("Computing BLEU …")
    results.update(_bleu_score(hypotheses, references))

    logger.info("Computing ROUGE …")
    results.update(_rouge_scores(hypotheses, references))

    logger.info("Computing METEOR …")
    results.update(_meteor_score(hypotheses, references))

    try:
        logger.info("Computing BERTScore (model=%s) …", bertscore_model)
        results.update(
            _bertscore(
                hypotheses,
                references,
                lang,
                bertscore_model,
                bertscore_num_layers,
            )
        )
    except ModuleNotFoundError:
        logger.warning("BERTScore skipped (bert_score not installed)")
    except Exception as exc:
        logger.warning("BERTScore skipped: %s", exc)

    # ------------------------------------------------------------------
    # Per-question CSV
    # ------------------------------------------------------------------
    per_q_rows = []
    for _, row in merged.iterrows():
        entry: dict = {
            QUESTION_COL: row[QUESTION_COL],
            GENERATED_COL: row[GENERATED_COL],
            REFERENCE_JOIN_COL: row[REFERENCE_JOIN_COL],
        }
        entry.update(_rouge_per_row(str(row[GENERATED_COL]), str(row[REFERENCE_JOIN_COL])))
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
        json.dump(
            {
                "n_questions": len(merged),
                "n_generated_questions": len(gen_eval_df),
                "n_reference_questions": len(ref_eval_df),
                "n_skipped_without_reference": skipped,
                **results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Summary → %s", summary_path)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    md_lines = [
        "# Text Metrics Evaluation",
        "",
        f"**Questions evaluated:** {len(merged)}  ",
        f"**Generated questions:** {len(gen_eval_df)}  ",
        f"**Reference questions:** {len(ref_eval_df)}  ",
        f"**Skipped without reference:** {skipped}  ",
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
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers with NLP metrics"
    )
    parser.add_argument(
        "--generated-csv",
        required=True,
        type=Path,
        help="CSV with generated answers (pytanie, odpowiedz_wygenerowana)",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=_default_reference(),
        help="Reference CSV (default: QA_rag.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/evaluation/results"),
        help="Where to save results",
    )
    parser.add_argument(
        "--lang", default="pl", help="Language code for BERTScore (default: pl)"
    )
    parser.add_argument(
        "--bertscore-model",
        default="bert-base-multilingual-cased",
        help="HuggingFace model for BERTScore",
    )
    parser.add_argument(
        "--bertscore-num-layers",
        type=int,
        default=12,
        help=(
            "Transformer layer count for BERTScore custom models "
            "(default: 12 for bert-base-multilingual-cased)"
        ),
    )
    args = parser.parse_args()

    evaluate(
        generated_csv=args.generated_csv,
        reference_csv=args.reference_csv,
        output_dir=args.output_dir,
        lang=args.lang,
        bertscore_model=args.bertscore_model,
        bertscore_num_layers=args.bertscore_num_layers,
    )
