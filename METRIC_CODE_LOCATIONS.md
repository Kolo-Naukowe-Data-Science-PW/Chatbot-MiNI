# Code Implementation Details & Locations

**Quick Navigation to Metric Implementations**

---

## Retrieval Metrics (src/evaluation/benchmark.py)

### 1. Hierarchical Relevance
**Location:** [benchmark.py:L148-L181](src/evaluation/benchmark.py#L148-L181)

```python
def hierarchical_relevance(retrieved_url: str, target_url: str) -> float:
    """
    Returns a graded relevance score for a retrieved URL relative to the
    single ground-truth target URL, based on the URL path hierarchy:

        Exact match                          -> 1.00
        Direct parent (1 level above)        -> 0.50
        Grandparent (2 levels above)         -> 0.25
        Great-grandparent (3 levels above)   -> 0.125  etc.
        Direct child (1 level below)         -> 0.25
        Grandchild (2 levels below)          -> 0.125  etc.
        Unrelated path or different origin   -> 0.00
    """
    # Implementation: ~35 lines
    # - Parse URLs with urlsplit()
    # - Check origin (scheme/netloc)
    # - Normalize paths
    # - Count depth and apply exponential decay
```

**Test Location:** [test_retrieval_metrics.py::TestHierarchicalRelevance](src/evaluation/tests/test_retrieval_metrics.py#L290-L335)

**Constants Used:**
- `RELEVANCE_THRESHOLD = 0.5` (used by binary metrics, not hier_rel itself)

---

### 2. nDCG@k (Normalized Discounted Cumulative Gain)
**Location:** [benchmark.py:L414-L432](src/evaluation/benchmark.py#L414-L432)

**Helper Functions:**
- `dcg_at_k()` [L414-L418]: Computes DCG via exponential gains & log discounts
- `ideal_dcg_at_k()` [L421-L423]: Computes ideal DCG on sorted scores
- `ndcg_at_k()` [L426-L432]: Normalized ratio

```python
def dcg_at_k(rel_scores: list[float], k: int) -> float:
    """DCG@k using graded relevance: gain = 2^rel - 1."""
    return sum(
        (2 ** rel - 1) / log2(rank + 1)
        for rank, rel in enumerate(rel_scores[:k], start=1)
    )

def ideal_dcg_at_k(rel_scores: list[float], k: int) -> float:
    """IDCG@k: DCG achieved by the ideal (descending) ordering."""
    return dcg_at_k(sorted(rel_scores, reverse=True), k)

def ndcg_at_k(rel_scores: list[float], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k."""
    idcg = ideal_dcg_at_k(rel_scores, k)
    return 0.0 if idcg == 0.0 else dcg_at_k(rel_scores, k) / idcg
```

**Test Location:** [test_retrieval_metrics.py::TestDCGAndNDCG](src/evaluation/tests/test_retrieval_metrics.py)

**Dependencies:**
- `from math import log2`

---

### 3. MRRw (Weighted Mean Reciprocal Rank)

#### Helper: URL Depth Difference
**Location:** [benchmark.py:L225-L255](src/evaluation/benchmark.py#L225-L255)

```python
def _url_depth_difference(retrieved_url: str, target_url: str) -> int | None:
    """
    Return the signed depth difference d between retrieved_url and target_url,
    or None if the two URLs are not on the same path (unrelated).

    d > 0  → retrieved is deeper (more specific) than target by d levels
    d = 0  → exact path match
    d < 0  → retrieved is shallower (more general) than target by |d| levels
    """
    # Returns: int or None
```

#### Helper: MRR Weight Function
**Location:** [benchmark.py:L258-L269](src/evaluation/benchmark.py#L258-L269)

```python
def _mrr_weight(d: int, alpha: float = MRRW_ALPHA, beta: float = MRRW_BETA) -> float:
    """
    Accuracy weight w(d) from Metryka_chatbot.pdf:
        w(0)   = 1.0
        w(d>0) = alpha^d   (deeper is penalised less than shallower)
        w(d<0) = beta^|d|  (shallower is penalised more)
    """
    if d == 0:
        return 1.0
    if d > 0:
        return alpha ** d
    return beta ** abs(d)
```

#### Per-Query MRRw
**Location:** [benchmark.py:L272-L290](src/evaluation/benchmark.py#L272-L290)

```python
def mrr_weighted_single(
    retrieved_urls: list[str],
    target_url: str,
    alpha: float = MRRW_ALPHA,
    beta: float = MRRW_BETA,
) -> float:
    """
    Compute the MRRw score S for a single query (Metryka_chatbot.pdf, eq. 3).

    S = max over all returned links i of  w(d_i) / r_i

    where r_i is the 1-based rank and d_i is the depth difference.
    """
    best = 0.0
    for rank, url in enumerate(retrieved_urls, start=1):
        d = _url_depth_difference(url, target_url)
        if d is None:
            continue
        s = _mrr_weight(d, alpha, beta) / rank
        if s > best:
            best = s
    return best
```

#### Corpus-Level MRRw
**Location:** [benchmark.py:L293-L320](src/evaluation/benchmark.py#L293-L320)

```python
def mrr_weighted(
    gold: list["EvalRow"],
    k: int,
    alpha: float = MRRW_ALPHA,
    beta: float = MRRW_BETA,
) -> float:
    """
    Compute MRRw over the full evaluation set (Metryka_chatbot.pdf, eq. 4).

    MRRw = (1/N) * Σ S_j
    """
    scores = []
    for row in gold:
        retrieved_chunks = _get_top_k_chunks(row.query, top_k=k)
        urls = unique_preserve_order([
            normalize_url(c.get("source_url", ""))
            for c in retrieved_chunks
            if c.get("source_url", "")
        ])
        scores.append(mrr_weighted_single(urls[:k], row.target_url, alpha, beta))
    return mean(scores) if scores else 0.0
```

**Constants:**
```python
MRRW_ALPHA: float = 0.8  # Weight for URLs deeper than target
MRRW_BETA: float = 0.4   # Weight for URLs shallower than target
```

**Test Location:** [test_retrieval_metrics.py](src/evaluation/tests/test_retrieval_metrics.py) (cross-validation implied)

---

## Text Generation Metrics (src/evaluation/text_metrics.py)

### 4. ROUGE-W (Weighted LCS)
**Location:** [text_metrics.py:L66-L124](src/evaluation/text_metrics.py#L66-L124)

**Entry Point:**
```python
def _rouge_scores(
    hypotheses: list[str],
    references: list[str | list[str]],
    *,
    beta: float = 1.0,
    rouge_w_alpha: float = 2.0,  # KEY PARAMETER FOR ROUGE-W
    rouge_s_d: int | None = None,
    use_jackknife: bool = False,
) -> dict[str, float]:
    """Computes ROUGE-N, ROUGE-L, ROUGE-W and ROUGE-S scores."""
```

**Weighted LCS Helper (L95-105):**
```python
def _wlcs(ref: list[str], hyp: list[str], alpha: float) -> float:
    """
    WLCS with f(k)=k^alpha.
    Returns c(m,n) as defined in the LaTeX.
    """
    m, n = len(ref), len(hyp)
    # w[i][j] = length of consecutive match ending at (i,j)
    # c[i][j] = accumulated weighted score
    w = [[0]*(n+1) for _ in range(m+1)]
    c = [[0.0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                w[i][j] = w[i-1][j-1] + 1
                wk   = w[i][j]
                wk1  = w[i-1][j-1]
                c[i][j] = c[i-1][j-1] + wk**alpha - wk1**alpha
            else:
                w[i][j] = 0
                c[i][j] = max(c[i-1][j], c[i][j-1])
    return c[m][n]
```

**ROUGE-W Single Pair (L108-116):**
```python
def _rouge_w_single(hyp_tok: list[str], ref_tok: list[str], alpha: float) -> dict[str, float]:
    m, n = len(ref_tok), len(hyp_tok)
    if m == 0 or n == 0:
        return {"r": 0.0, "p": 0.0, "f": 0.0}
    score = _wlcs(ref_tok, hyp_tok, alpha)
    fm    = m ** alpha
    fn    = n ** alpha
    r = (score / fm) ** (1.0 / alpha)
    p = (score / fn) ** (1.0 / alpha)
    return {"r": r, "p": p, "f": _f(r, p)}
```

**Corpus Aggregation (L163-175):**
```python
# ROUGE-W: best reference
sw = max(
    (_rouge_w_single(hyp_tok, r, rouge_w_alpha) for r in refs_tok),
    key=lambda d: d["r"],
)
for k in ("r", "p", "f"):
    agg[f"rouge_w_{k}"].append(sw[k])

# ... [continues for all ROUGE variants]

return {key: sum(vals) / len(vals) for key, vals in agg.items() if vals}
```

**Test Location:** [test_text_metrics_comprehensive.py::TestROUGEMetrics](src/evaluation/tests/test_text_metrics_comprehensive.py)

**Key Parameters:**
- `rouge_w_alpha = 2.0` – Weighting exponent (α > 1)
- `beta = 1.0` – F-measure weight (equal R/P if 1.0)

---

### 5. METEOR
**Location:** [text_metrics.py:L127-L145](src/evaluation/text_metrics.py#L127-L145)

```python
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
        for hyp, ref in zip(hypotheses, references)
        if hyp.strip() and ref.strip()
    ]
    return {"meteor": sum(scores) / len(scores) if scores else 0.0}
```

**Dependencies:**
- `import nltk`
- `from nltk.tokenize import word_tokenize`
- `from nltk.translate.meteor_score import meteor_score`

**Test Location:** [test_text_metrics_comprehensive.py::TestMETEORMetric](src/evaluation/tests/test_text_metrics_comprehensive.py)

---

### 6. BERT-Score
**Location:** [text_metrics.py:L148-L176](src/evaluation/text_metrics.py#L148-L176)

```python
def _bertscore(
    hypotheses: list[str],
    references: list[str],
    lang: str,
    model_type: str,
) -> dict[str, float]:
    from bert_score import BERTScorer  # type: ignore

    results: dict[str, float] = {}

    variants = [
        ("base",     False, False),
        ("idf",      True,  False),
        ("rescaled", False, True),
        ("full",     True,  True),
    ]

    for suffix, idf, rescale in variants:
        scorer = BERTScorer(
            lang=lang,
            model_type=model_type,
            idf=idf,
            rescale_with_baseline=rescale,
        )
        P, R, F1 = scorer.score(hypotheses, references, verbose=False)
        results[f"bertscore_precision_{suffix}"] = P.mean().item()
        results[f"bertscore_recall_{suffix}"]    = R.mean().item()
        results[f"bertscore_f1_{suffix}"]        = F1.mean().item()

    return results
```

**Dependencies:**
- `from bert_score import BERTScorer`

**Test Location:** [test_text_metrics_comprehensive.py::TestBERTScore](src/evaluation/tests/test_text_metrics_comprehensive.py)

**Variants Generated:**
- `bertscore_precision_base`, `bertscore_recall_base`, `bertscore_f1_base`
- `bertscore_precision_idf`, `bertscore_recall_idf`, `bertscore_f1_idf`
- `bertscore_precision_rescaled`, `bertscore_recall_rescaled`, `bertscore_f1_rescaled`
- `bertscore_precision_full`, `bertscore_recall_full`, `bertscore_f1_full`

---

## Integration Points

### Retrieval Evaluation Loop
**Location:** [benchmark.py:L538-L600](src/evaluation/benchmark.py#L538-L600)

```python
def evaluate(
        gold: list[EvalRow],
        k: int,
        use_rewrite: bool = False,
) -> tuple[MetricsAtK, list[tuple[list[float], list[float]]]]:
    """Evaluates the retriever on the gold set for a given rank cut-off k."""
    
    # ... initialization ...
    
    for index, row in enumerate(gold):
        retrieval_q = _rewrite_query(row.query) if use_rewrite else row.query
        retrieved_chunks = _get_top_k_chunks(retrieval_q, top_k=k)
        
        # Extract and normalize URLs
        raw_urls = [normalize_url(c.get("source_url", "")) for c in retrieved_chunks]
        unique_urls = unique_preserve_order(raw_urls)
        
        # ► COMPUTE RUN HERE: hierarchical_relevance()
        rel_scores = [
            hierarchical_relevance(url, row.target_url)
            for url in unique_urls
        ]
        
        # ► COMPUTE MRRw SEPARATELY
        mrrws.append(mrr_weighted_single(unique_urls[:k], row.target_url))
        
        # ► COMPUTE nDCG@k
        hits.append(hit_at_k(rel_scores, k))
        mrrs.append(mrr_at_k(rel_scores, k))
        recalls.append(recall_at_k(rel_scores, k, total_rel))
        precisions.append(precision_at_k(rel_scores, k))
        f1s.append(f_measure_at_k(rel_scores, k, total_rel))
        ndcgs.append(ndcg_at_k(rel_scores, k))  # ← nDCG CALL
        aps.append(average_precision_at_k(rel_scores, k, total_rel))
        r_precs.append(r_precision(rel_scores, total_rel))
        
    # ... average across queries ...
```

### Text Evaluation Pipeline
**Location:** [text_metrics.py:L289-L350](src/evaluation/text_metrics.py#L289-L350)

```python
def evaluate(
    generated_csv: Path,
    reference_csv: Path,
    output_dir: Path,
    lang: str,
    bertscore_model: str,
) -> None:
    # ... load CSVs ...
    
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
    
    # ... save results ...
```

---

## Constants & Configuration

**Retrieval Constants:** [benchmark.py:L31-L52](src/evaluation/benchmark.py#L31-L52)
```python
RELEVANCE_THRESHOLD: float = 0.5
MRRW_ALPHA: float = 0.8
MRRW_BETA: float = 0.4
```

**EvalRow Dataclass:** [benchmark.py:L59-62](src/evaluation/benchmark.py#L59-L62)
```python
@dataclass
class EvalRow:
    query: str
    target_url: str  # single ground-truth URL per question
```

**MetricsAtK Container:** [benchmark.py:L513-529](src/evaluation/benchmark.py#L513-L529)
```python
@dataclass
class MetricsAtK:
    k: int
    hit: float
    mrr: float
    mrr_weighted: float     # MRRw (Metryka_chatbot.pdf)
    recall: float
    precision: float
    f1: float
    ndcg: float
    map_score: float        # MAP@k
    r_prec: float           # R-Precision
```

---

## File Organization Summary

```
src/evaluation/
├── benchmark.py                    ← Retrieval metrics
│   ├── hierarchical_relevance()    [L148-181]
│   ├── ndcg_at_k()                 [L426-432]
│   ├── mrr_weighted_single()       [L272-290]
│   ├── mrr_weighted()              [L293-320]
│   └── evaluate()                  [L538-600]
│
├── text_metrics.py                 ← Text generation metrics
│   ├── _rouge_scores()             [L66-199]  (includes ROUGE-W)
│   ├── _meteor_score()             [L127-145]
│   ├── _bertscore()                [L148-176]
│   └── evaluate()                  [L289-350]
│
└── tests/
    ├── test_retrieval_metrics.py
    │   └── TestHierarchicalRelevance, TestDCGAndNDCG, etc.
    │
    └── test_text_metrics_comprehensive.py
        └── TestROUGEMetrics, TestMETEORMetric, TestBERTScore, etc.
```

---

## Running Individual Metrics

### Via Python Interpreter
```python
from src.evaluation.benchmark import hierarchical_relevance, ndcg_at_k

# Hierarchical Relevance
score = hierarchical_relevance(
    "https://example.com/path/to/page",
    "https://example.com/path"
)
print(f"Rel: {score}")  # Output: 0.5

# nDCG@k
rel_scores = [1.0, 0.5, 0.25, 0.125, 0.0]
ndcg = ndcg_at_k(rel_scores, k=3)
print(f"nDCG@3: {ndcg}")
```

> Via Text Metrics
```python
from src.evaluation.text_metrics import _rouge_scores, _meteor_score, _bertscore

# ROUGE-W
result = _rouge_scores(
    ["The cat sat"],
    ["A cat was sitting"],
    rouge_w_alpha=2.0
)
print(f"ROUGE-W R/P/F: {result['rouge_w_r']}, {result['rouge_w_p']}, {result['rouge_w_f']}")

# METEOR
result = _meteor_score(["The quick brown fox"], ["A fast brown fox"])
print(f"METEOR: {result['meteor']}")

# BERT-Score
result = _bertscore(["Kot siedział"], ["Kot był"], lang="pl", model_type="bert-base-multilingual-cased")
print(f"BERT F1 (base): {result['bertscore_f1_base']}")
```

---

