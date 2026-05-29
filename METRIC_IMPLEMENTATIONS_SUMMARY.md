# Metric Implementations Summary
**Analysis Date:** 2026-05-28
**Scope:** src/evaluation/benchmark.py, src/evaluation/text_metrics.py

---

## Overview
This document provides a structured analysis of all implemented evaluation metrics, organized by type (retrieval vs. text generation). For each metric, current implementations, return types, and parameter specifications are documented.

---

# PART I: RETRIEVAL METRICS (src/evaluation/benchmark.py)

## 1. Hierarchical Relevance
**Location:** [benchmark.py](src/evaluation/benchmark.py#L148)

### Implementation Details
```python
def hierarchical_relevance(retrieved_url: str, target_url: str) -> float
```

### Current Formula/Implementation
- **Exact match:** 1.0
- **Direct parent (1 level above):** 0.5
- **Grandparent (2+ levels above):** 0.5^depth (exponential decay)
- **Direct child (1 level below):** 0.25 (stricter than parent)
- **Grandchild (2+ levels below):** 0.5^(depth+1) (exponential decay with +1 penalty)
- **Unrelated path or different origin:** 0.0

### Return Type
- **Type:** `float`
- **Range:** [0.0, 1.0]

### Parameter Names & Ranges
| Parameter | Type | Purpose | Valid Range |
|-----------|------|---------|-------------|
| `retrieved_url` | str | URL retrieved by system | Any valid/invalid URL string |
| `target_url` | str | Ground-truth target URL | Any valid/invalid URL string |

### Algorithm Steps
1. Parse both URLs using `urlsplit()`
2. Check if origins differ (scheme/netloc) → return 0.0
3. Normalize paths by stripping trailing slashes
4. Determine ancestor/descendant relationship via path prefix matching
5. Apply exponential scoring: depth_score = 0.5^depth or 0.5^(depth+1) for children

### Key Implementation Notes
- **Empty URLs** → 0.0
- **URL normalization:** Strips query strings and fragments
- **Asymmetric scoring:** Children penalized more heavily than parents (reflects source reliability)
- **Decay function:** `0.5^depth` for ancestors, `0.5^(depth+1)` for descendants

---

## 2. nDCG@k (Normalized Discounted Cumulative Gain)
**Location:** [benchmark.py](src/evaluation/benchmark.py#L430-L432)

### Implementation Details
```python
def ndcg_at_k(rel_scores: list[float], k: int) -> float
```

### Current Formula/Implementation
$$\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

where:
- **DCG@k** = $\sum_{rank=1}^{k} \frac{2^{rel_i} - 1}{\log_2(rank + 1)}$
- **rel_i** = graded relevance score (from hierarchical_relevance)
- **IDCG@k** = ideal DCG (sorted by descending relevance)

### Return Type
- **Type:** `float`
- **Range:** [0.0, 1.0]

### Parameter Names & Ranges
| Parameter | Type | Purpose | Valid Range |
|-----------|------|---------|-------------|
| `rel_scores` | list[float] | Relevance scores for retrieved URLs | [0, ∞), but by design [0, 1] |
| `k` | int | Rank cut-off | 1 to |rel_scores| |

### Algorithm Steps (via helper functions)
1. **DCG calculation** [L414-418]:
   - Iterate through top-k items
   - Apply gain function: `2^rel - 1`
   - Apply discount: `1 / log₂(rank + 1)` (rank starts at 1)
   - Sum all discounted gains
2. **IDCG calculation** [L421-423]:
   - Sort rel_scores in descending order
   - Apply same DCG formula to ideal ordering
   - Use all sorted values (not just top-k)
3. **Normalize** [L426-427]:
   - If IDCG = 0 → return 0.0
   - Else → DCG / IDCG

### Key Implementation Notes
- **Zero-IDCG handling:** When all rel_scores ≤ 0 or list is empty → nDCG = 0.0
- **Gain function:** Exponential (2^x) emphasizes larger relevance differences
- **Discount:** Log-based, allows some rank flexibility
- **No threshold applied:** Operates on raw graded relevance scores

---

## 3. MRRw (Weighted Mean Reciprocal Rank)
**Location:** [benchmark.py](src/evaluation/benchmark.py#L194-L290)

### Implementation Details
```python
def mrr_weighted_single(
    retrieved_urls: list[str],
    target_url: str,
    alpha: float = MRRW_ALPHA,
    beta: float = MRRW_BETA,
) -> float

def mrr_weighted(
    gold: list[EvalRow],
    k: int,
    alpha: float = MRRW_ALPHA,
    beta: float = MRRW_BETA,
) -> float
```

### Current Formula/Implementation
**Per query (MRRw_single):**
$$S = \max_{i \in \text{retrieved}} \frac{w(d_i)}{r_i}$$

where:
- $d_i$ = signed depth difference (URL position relative to target)
- $w(d)$ = accuracy weight:
  - $w(0) = 1.0$ (exact path match)
  - $w(d > 0) = \alpha^d$ (deeper than target, penalized less)
  - $w(d < 0) = \beta^{|d|}$ (shallower than target, penalized more)
- $r_i$ = 1-based rank position

**Per corpus (MRRw):**
$$\text{MRRw} = \frac{1}{N} \sum_{j=1}^{N} S_j$$

where N = number of queries

### Default Parameters
- **α (MRRW_ALPHA):** 0.8 (depth penalty weight)
- **β (MRRW_BETA):** 0.4 (shallowness penalty weight)

### Return Type
- **Type:** `float` (single query), `float` (corpus average)
- **Range:** [0.0, 1.0]

### Parameter Names & Ranges
| Parameter | Type | Purpose | Valid Range | Default |
|-----------|------|---------|-------------|---------|
| `retrieved_urls` | list[str] | URLs returned by retriever | Any list | N/A |
| `target_url` | str | Ground-truth target URL | Valid URL string | N/A |
| `alpha` | float | Decay weight for deeper results | 0 < α < 1 | 0.8 |
| `beta` | float | Decay weight for shallower results | 0 < β < α | 0.4 |
| `gold` | list[EvalRow] | Full evaluation set | Query+URL pairs | N/A |
| `k` | int | Rank cut-off | ≥ 1 | N/A |

### Helper Function: `_url_depth_difference()`
**Returns** signed integer or None:
- **d > 0:** retrieved is deeper (more specific) than target
- **d = 0:** exact path match
- **d < 0:** retrieved is shallower (more general) than target
- **None:** URLs on different branches (unrelated)

### Key Implementation Notes
- **Reference:** Metryka_chatbot.pdf (B. Gawlik, 2026-03-13)
- **Asymmetric weighting:** Distinguishes deepness (less penalty) from shallowness (more penalty)
- **Unrelated URLs:** Skipped (depth_difference returns None)
- **Zero scores allowed:** If no URL matches on same path → MRRw = 0.0
- **Corpus-level averaging:** Simple mean across all queries

---

## 4. Binary & Standard Retrieval Metrics (Summary Table)

| Metric | Function | Return | Formula | Notes |
|--------|----------|--------|---------|-------|
| Hit@k | `hit_at_k(rel_scores, k, threshold)` | float ∈ [0,1] | 1 if ∃rel_i ≥ threshold else 0 | Binary metric |
| MRR@k | `mrr_at_k(rel_scores, k, threshold)` | float ∈ [0,1] | 1/rank of first relevant | Early rank reward |
| Recall@k | `recall_at_k(rel_scores, k, total_rel, threshold)` | float ∈ [0,1] | hits / total_rel | Coverage metric |
| Precision@k | `precision_at_k(rel_scores, k, threshold)` | float ∈ [0,1] | hits / k | Quality metric |
| F1@k | `f_measure_at_k(rel_scores, k, total_rel, beta, threshold)` | float ∈ [0,1] | Harmonic mean P & R | β=1 default |
| AP@k | `average_precision_at_k(rel_scores, k, total_rel, threshold)` | float ∈ [0,1] | Σ Precision@i / total_rel | Rank-aware |
| R-Precision | `r_precision(rel_scores, total_rel, threshold)` | float ∈ [0,1] | Precision@total_rel | Rank-cutoff independent |

### Common Parameters
- `threshold (RELEVANCE_THRESHOLD)`: Default 0.5 — Minimum score to count as "relevant"
- `total_rel`: Always 1 per query in this system (single gold URL per question)

### Metric Computation Flow
In `evaluate()` function [L538-600]:
1. Retrieve top-k chunks using `_get_top_k_chunks()`
2. Normalize URLs and deduplicate
3. **Compute hierarchical_relevance scores** → list of floats [0,1]
4. **Compute MRRw score separately** (depth-aware, on raw URLs)
5. Apply all standard metrics to rel_scores + MRRw
6. Average across all queries (N = size of gold set)

---

---

# PART II: TEXT GENERATION METRICS (src/evaluation/text_metrics.py)

## 1. ROUGE-W (Weighted Longest Common Subsequence)
**Location:** [text_metrics.py](src/evaluation/text_metrics.py#L66-L124)

### Implementation Details
```python
def _rouge_scores(
    hypotheses: list[str],
    references: list[str | list[str]],
    beta: float = 1.0,
    rouge_w_alpha: float = 2.0,
    rouge_s_d: int | None = None,
    use_jackknife: bool = False,
) -> dict[str, float]
```

### Current Formula/Implementation (ROUGE-W subsection)

#### Weighted LCS (WLCS) Calculation
$$\text{WLCS}(ref, hyp, \alpha) = c(m, n)$$

where dynamic programming table tracks:
- `w[i][j]` = length of consecutive match ending at position (i,j)
- `c[i][j]` = accumulated weighted score
- Weight function: $f(k) = k^\alpha$ (k = consecutive match length)

#### Precision & Recall
$$R = \left(\frac{\text{WLCS}}{m^\alpha}\right)^{1/\alpha}$$
$$P = \left(\frac{\text{WLCS}}{n^\alpha}\right)^{1/\alpha}$$

where m = reference length, n = hypothesis length

#### F-Measure
$$F = \frac{(1 + \beta^2) \cdot R \cdot P}{\beta^2 \cdot R + P}$$

### Return Type
- **Type:** `dict[str, float]`
- **Keys:** `rouge_w_r`, `rouge_w_p`, `rouge_w_f` (for each hypothesis-reference pair)
- **Range:** [0.0, 1.0]

### Parameter Names & Ranges
| Parameter | Type | Purpose | Valid Range | Default |
|-----------|------|---------|-------------|---------|
| `hypotheses` | list[str] | Generated answers | Any list of strings | Required |
| `references` | list[str/list[str]] | Reference answers | Single or multiple refs per hyp | Required |
| `rouge_w_alpha` (α) | float | Weight exponent | α > 1 | 2.0 |
| `beta` (β) | float | F-measure weight | β > 0 | 1.0 |
| `use_jackknife` | bool | Multi-ref strategy | True/False | False |

### Key Implementation Notes
- **Alpha parameter:** Controls how much consecutive matches are rewarded
  - α = 2.0: Length-2 sequences weighted as 2² = 4, length-3 as 3² = 9
- **Multi-reference handling:**
  - Without jackknife: Pick best reference by recall
  - With jackknife: Leave-one-out average for robustness
- **Empty text handling:** Returns {...: 0.0} for empty hypothesis or reference
- **Tokenization:** Space-separated word tokens via regex `\b\w+\b`

### Usage Example
```python
gen_answers = ["The cat sat on the mat"]
ref_answers = ["A cat was sitting on a mat"]
result = _rouge_scores(gen_answers, ref_answers, rouge_w_alpha=2.0)
# Returns: {'rouge_w_r': 0.X, 'rouge_w_p': 0.Y, 'rouge_w_f': 0.Z, ...}
```

---

## 2. METEOR (Metric for Evaluation of Translation with Explicit Ordering)
**Location:** [text_metrics.py](src/evaluation/text_metrics.py#L127-L145)

### Implementation Details
```python
def _meteor_score(
    hypotheses: list[str],
    references: list[str]
) -> dict[str, float]
```

### Current Formula/Implementation

**NLTK METEOR Score** (leverages external library)
$$\text{METEOR} = F_{\text{align}} \times (1 - \text{Penalty})$$

where:
- **F_align** = Harmonic mean of unigram precision/recall
- **Penalty** = $0.5 \times \left(\frac{\text{chunks}}{|matches|}\right)^3$
  - Penalties for fragmented matches (many small sequences)

**Features included:**
- Exact word matches
- Lemmatized word matches (via WordNet)
- Synonym matches (via WordNet)
- Word order preservation

### Return Type
- **Type:** `dict[str, float]`
- **Keys:** `"meteor"`
- **Range:** [0.0, 1.0]

### Parameter Names & Ranges
| Parameter | Type | Purpose | Valid Range | Default |
|-----------|------|---------|-------------|---------|
| `hypotheses` | list[str] | Generated answers | Any list of strings | Required |
| `references` | list[str] | Gold standard answers | Any list of strings | Required |

### Dependencies & Setup
- **Library:** NLTK (Natural Language Toolkit)
- **Auto-download on first run:**
  - `punkt_tab` (tokenizer)
  - `wordnet` (lemmatization data)
- **Language:** Configurable, defaults to English pattern matching

### Key Implementation Notes
- **Corpus-level metric:** Computed per (hypothesis, reference) pair, then averaged
- **External library reliance:** Uses `nltk.translate.meteor_score.meteor_score()`
- **Token-level matching:** Works on word tokens (pre-tokenized by `word_tokenize()`)
- **Empty text handling:** Skips empty strings before averaging
- **Output:** Single aggregated score (not P/R/F breakout)

### Usage Example
```python
gen = ["The quick brown fox"]
ref = ["A fast brown fox"]
result = _meteor_score(gen, ref)
# Returns: {'meteor': 0.8}
```

---

## 3. BERTScore (Contextual Embedding-Based Similarity)
**Location:** [text_metrics.py](src/evaluation/text_metrics.py#L148-L176)

### Implementation Details
```python
def _bertscore(
    hypotheses: list[str],
    references: list[str],
    lang: str,
    model_type: str,
) -> dict[str, float]
```

### Current Formula/Implementation

**Embedding-Based Token Matching:**
$$P_{\text{BERT}} = \frac{1}{m} \sum_{i=1}^{m} \max_{j} \cos(e_{\text{hyp}}^{i}, e_{\text{ref}}^{j})$$

$$R_{\text{BERT}} = \frac{1}{n} \sum_{j=1}^{n} \max_{i} \cos(e_{\text{ref}}^{j}, e_{\text{hyp}}^{i})$$

$$F_{\text{BERT}} = \frac{2PR}{P+R}$$

where:
- $e_{\text{hyp}}, e_{\text{ref}}$ = Contextual token embeddings
- $\cos$ = Cosine similarity
- m = hypothesis length, n = reference length

**Variants (4 scoring modes):**
1. **base** (no IDF, no rescale)
   - Keys: `bertscore_precision_base`, `bertscore_recall_base`, `bertscore_f1_base`
2. **idf** (inverse document frequency weighting)
   - Keys: `bertscore_precision_idf`, `bertscore_recall_idf`, `bertscore_f1_idf`
3. **rescaled** (baseline rescaling, no IDF)
   - Keys: `bertscore_precision_rescaled`, `bertscore_recall_rescaled`, `bertscore_f1_rescaled`
4. **full** (IDF + rescaling combined)
   - Keys: `bertscore_precision_full`, `bertscore_recall_full`, `bertscore_f1_full`

### Return Type
- **Type:** `dict[str, float]`
- **Keys:** 12 total — 3 metrics × 4 variants
- **Range:** [0.0, 1.0]

### Parameter Names & Ranges
| Parameter | Type | Purpose | Valid Range | Default |
|-----------|------|---------|-------------|---------|
| `hypotheses` | list[str] | Generated answers | Any list of strings | Required |
| `references` | list[str] | Reference answers | Any list of strings | Required |
| `lang` | str | Language code | 'pl', 'en', etc. | Required |
| `model_type` | str | BERT model identifier | 'bert-base-multilingual-cased', etc. | Required |

### Scoring Variants Explained
| Variant | IDF | Baseline Rescale | Use Case |
|---------|-----|------------------|----------|
| base | ✗ | ✗ | Raw contextual similarity |
| idf | ✓ | ✗ | Weights important tokens higher |
| rescaled | ✗ | ✓ | Normalizes against human baseline |
| full | ✓ | ✓ | Best practice: importance + calibration |

### Key Implementation Notes
- **External library:** `bert_score` package
- **Model loading:** First use loads embedder (can be slow)
- **Corpus aggregation:** Mean across all hypothesis-reference pairs
- **IDF computation:** Based on corpus frequencies in first batch
- **Baseline rescaling:** Normalizes against pre-computed human pairwise scores
- **Output:** Tensor aggregation via `.mean().item()`

### Usage Example
```python
gen = ["Kot siedział na macie"]  # Polish
ref = ["Kot był na macie"]
result = _bertscore(gen, ref, lang="pl", model_type="bert-base-multilingual-cased")
# Returns: {
#   'bertscore_precision_base': 0.92,
#   'bertscore_recall_base': 0.90,
#   'bertscore_f1_base': 0.91,
#   ... (8 more variants)
# }
```

---

## 4. ROUGE-N Family (Summary Table)
**Part of `_rouge_scores()` function**

| Variant | Base Unit | Formula | Keys | Notes |
|---------|-----------|---------|------|-------|
| ROUGE-1 | Unigrams (single words) | N-gram matching | rouge_1_r/p/f | Word overlap |
| ROUGE-2 | Bigrams (2-grams) | N-gram matching | rouge_2_r/p/f | Phrase overlap |
| ROUGE-L | Longest Common Subsequence | LCS length | rouge_l_r/p/f | Word order aware |
| ROUGE-S | Skip-bigrams | Word pairs with gap | rouge_s_r/p/f | Flexible pairing |

### Common Parameters for All ROUGE Variants
- `beta`: F-measure weight (default 1.0 = equal recall/precision)
- `use_jackknife`: Multi-reference robustness (default False)

---

## 5. BLEU (Bilingual Evaluation Understudy)
**Location:** [text_metrics.py](src/evaluation/text_metrics.py#L30-L39)

### Implementation Details
```python
def _bleu_score(
    hypotheses: list[str],
    references: list[str]
) -> dict[str, float]
```

### Current Formula/Implementation
$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{4} w_n \log(p_n)\right)$$

where:
- **BP** = Brevity Penalty = $e^{(1 - r/c)}$ if c < r else 1.0
- **p_n** = Precision of n-grams (n=1..4)
- **w_n** = Weights (default uniform 0.25)
- **r** = Reference length, **c** = Candidate length

### Return Type
- **Type:** `dict[str, float]`
- **Keys:** `bleu`, `bleu_1`, `bleu_2`, `bleu_3`, `bleu_4`
- **Range:** [0.0, 1.0]

### Key Implementation Notes
- **External library:** `sacrebleu` (standard BLEU implementation)
- **Output:** Normalized to [0,1] (divides by 100)
- **Corpus-level:** Averaged across all hypothesis-reference pairs
- **No threshold:** Works on raw n-gram overlaps

---

---

# PART III: SUMMARY TABLES & COMPARISONS

## Return Type Summary

### Retrieval Metrics (benchmark.py)

| Metric | Return Type | Cardinality | Notes |
|--------|------------|------------|-------|
| hierarchical_relevance | float | Single | Immediate score |
| ndcg_at_k | float | Single | Scalar metric |
| mrr_weighted_single | float | Single | Per-query result |
| mrr_weighted | float | Single | Corpus average |
| hit_at_k, mrr_at_k, recall_at_k, etc. | float | Single | All return scalars |

### Text Metrics (text_metrics.py)

| Function | Return Type | Keys in Dict | Aggregation |
|----------|------------|--------------|-------------|
| _bleu_score | dict[str, float] | 5 keys | Corpus mean |
| _rouge_scores | dict[str, float] | 15 keys | Corpus mean |
| _meteor_score | dict[str, float] | 1 key | Corpus mean |
| _bertscore | dict[str, float] | 12 keys | Tensor mean |

---

## Parameter Conventions

### Retrieval Metrics
- **rel_scores:** List of graded relevance floats [0,1], typically from hierarchical_relevance()
- **k:** Rank cutoff (top-k results)
- **threshold:** Binarization boundary (default RELEVANCE_THRESHOLD=0.5)
- **total_rel:** Always 1 (single gold URL per query)

### Text Metrics
- **hypotheses/candidates:** Generated text strings
- **references:** Gold standard text (single or list)
- **lang:** Language code for BERT/METEOR models
- **alpha/beta:** Weighting exponents for ROUGE-W and F-measures

---

## Current vs. Thesis Requirements

### Implemented Metrics ✓
- **Hierarchical Relevance** — URL-depth decay with asymmetric child penalties
- **nDCG@k** — Standard DCG normalization with exponential gains
- **ROUGE-W** — Weighted LCS with configurable exponent (α=2.0)
- **METEOR** — Lemmatization + word order penalty via NLTK
- **BERTScore** — 4 variants (base, idf, rescaled, full)

### Not Mentioned in Request but Implemented
- Binary metrics: Hit@k, MRR@k, Precision@k, Recall@k
- Ranking metrics: AP@k, R-Precision
- ROUGE variants: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-S
- BLEU-score

### Integration Point
- **Main evaluation loop:** [benchmark.py:evaluate()](src/evaluation/benchmark.py#L538-L600)
- **Text evaluation:** [text_metrics.py:evaluate()](text_metrics.py#L289-L350)

---

## Test Coverage

### Retrieval Metrics Tests
- **File:** `src/evaluation/tests/test_retrieval_metrics.py`
- **Coverage:** hierarchical_relevance, nDCG, MRR, precision, recall, etc.
- **Status:** Pytest suite available

### Text Metrics Tests
- **File:** `src/evaluation/tests/test_text_metrics_comprehensive.py`
- **Coverage:** BLEU, ROUGE (all variants), METEOR,BERTScore
- **Status:** Pytest suite available

---
