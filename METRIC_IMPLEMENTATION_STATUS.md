# Metric Implementation Status Report

**Analysis Date:** 2026-05-28
**Focus:** nDCG, hierarchical_relevance, ROUGE-W, METEOR, BERT-Score

---

## Executive Summary

All five requested metrics are **fully implemented and tested**:

| Metric | Type | Status | Location | Return Type | Thesis Alignment |
|--------|------|--------|----------|-------------|------------------|
| **nDCG@k** | Retrieval | ✓ Complete | benchmark.py:426-432 | float | Standard definition |
| **hierarchical_relevance** | Retrieval | ✓ Complete | benchmark.py:148-181 | float | Custom (URL hierarchy) |
| **ROUGE-W** | Text | ✓ Complete | text_metrics.py:66-124 | dict[str, float] | Weighted LCS per standard |
| **METEOR** | Text | ✓ Complete | text_metrics.py:127-145 | dict[str, float] | NLTK standard |
| **BERT-Score** | Text | ✓ Complete | text_metrics.py:148-176 | dict[str, float] | 4 variants (lib standard) |

---

## 1. nDCG@k (Normalized Discounted Cumulative Gain)

### Current Implementation
- **Function:** `ndcg_at_k(rel_scores: list[float], k: int) → float`
- **Location:** [src/evaluation/benchmark.py](src/evaluation/benchmark.py#L426-L432)

### Formula ✓
$$\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

**DCG Component:**
$$\text{DCG@k} = \sum_{rank=1}^{k} \frac{2^{rel_i} - 1}{\log_2(rank + 1)}$$

**IDCG Component:**
- Computed on ideal (descending sorted) relevance scores
- Same formula as DCG, applied to sorted values

### Return Type ✓
- **Type:** `float` (scalar)
- **Range:** [0.0, 1.0]
- **Interpretation:** Higher is better; 1.0 = perfect ranking

**Exception Cases:**
- IDCG = 0 (all rel_scores ≤ 0) → nDCG returns 0.0
- Empty list → nDCG returns 0.0

### Implementation Notes
- **Gain function:** Exponential `2^rel - 1` emphasizes relevance differences
- **Discount function:** Logarithmic `log₂(rank + 1)` allows flexibility in top-k
- **Threshold:** None — operates on full graded scores [0,1]
- **Dependency:** Uses `hierarchical_relevance()` for rel_scores

### Parameter Specification ✓
| Parameter | Type | Range | Default | Purpose |
|-----------|------|-------|---------|---------|
| `rel_scores` | list[float] | [0, ∞), by design [0, 1] | Required | Graded relevance per URL |
| `k` | int | 1 to len(rel_scores) | Required | Rank cut-off |

### Test Coverage ✓
- **Test File:** `src/evaluation/tests/test_retrieval_metrics.py`
- **Test Class:** `TestDCGAndNDCG`
- **Tests:**
  - Perfect ranking (nDCG = 1.0)
  - Worst ranking (nDCG = 0.0)
  - Mixed ranking (0 < nDCG < 1)
  - Edge cases (empty, k > length)

### Alignment with Thesis ✓
**Status:** Matches standard IR definition
- ✓ Exponential gain function
- ✓ Logarithmic discount
- ✓ Self-normalized (divided by ideal)
- ✓ Graded relevance scores [0,1]

---

## 2. Hierarchical Relevance

### Current Implementation
- **Function:** `hierarchical_relevance(retrieved_url: str, target_url: str) → float`
- **Location:** [src/evaluation/benchmark.py](src/evaluation/benchmark.py#L148-L181)

### Formula ✓
**URL-based hierarchy scoring:**

```
retrieved == target        → 1.0 (exact match)

target = URL/a/b
retrieved = URL/a         → 0.5 (parent, 1 level above)
retrieved = URL/          → 0.25 (grandparent, 2 levels up)

target = URL/a
retrieved = URL/a/b       → 0.25 (child, 1 level below)
retrieved = URL/a/b/c     → 0.125 (grandchild, 2 levels down)

retrieved ≠ target && different_origin  → 0.0 (unrelated)
```

**Mathematical Formulation:**
$$rel(r, t) = \begin{cases}
1.0 & \text{if } path(r) = path(t) \\
0.5^d & \text{if } path(t) \text{ is ancestor} \; (d \text{ levels up}) \\
0.5^{d+1} & \text{if } path(t) \text{ is descendant} \; (d \text{ levels down}) \\
0.0 & \text{otherwise}
\end{cases}$$

### Return Type ✓
- **Type:** `float` (scalar)
- **Range:** [0.0, 1.0]
- **Precision:** Float (not rounded)

### Implementation Notes
- **Normalization:** URLs stripped of query strings & fragments
- **Asymmetric Scoring:**
  - Children penalized MORE than parents (0.5^(d+1) vs 0.5^d)
  - Philosophy: Parent page more reliable than child page for general query
- **Empty URL Handling:** Empty or None URLs → 0.0
- **Origin Checking:** Scheme + netloc must match; different domains never relevant

### Parameter Specification ✓
| Parameter | Type | Purpose | Valid Range | Notes |
|-----------|------|---------|-------------|-------|
| `retrieved_url` | str | Returned by retriever | Any string (auto-validated) | Can be invalid URL |
| `target_url` | str | Ground-truth URL | Any string (auto-validated) | Can be invalid URL |

### Depth Calculation Algorithm
1. Parse both URLs with `urlsplit()`
2. Extract normalized paths (strip trailing `/`)
3. Count path segments: `path.count("/")`
4. Determine ancestor/descendant via string prefix matching
5. Apply exponential decay based on depth

### Test Coverage ✓
- **Test File:** `src/evaluation/tests/test_retrieval_metrics.py`
- **Test Class:** `TestHierarchicalRelevance`
- **Tests:**
  - Exact match: `href="../../intro/index.html"` vs same → 1.0
  - Parent (1↑): `href="../../intro"` vs `href="../../intro/index.html"` → 0.5
  - Grandparent (2↑): → 0.25
  - Child (1↓): → 0.25
  - Grandchild (2↓): → 0.125
  - Different domain → 0.0
  - Empty URL → 0.0

### Alignment with Thesis ✓
**Status:** Custom metric, defined for this RAG system
- ✓ URL hierarchy-aware
- ✓ Asymmetric depth reward
- ✓ Exponential decay
- ✓ Tests comprehensive

---

## 3. ROUGE-W (Weighted Longest Common Subsequence)

### Current Implementation
- **Function:** `_rouge_scores(..., rouge_w_alpha=2.0, ...) → dict[str, float]`
- **Location:** [src/evaluation/text_metrics.py](src/evaluation/text_metrics.py#L66-L124)
- **Helper:** `_wlcs(ref, hyp, alpha)` computes weighted LCS

### Formula ✓

**Weighted LCS (WLCS) Definition:**

Using dynamic programming:
- `w[i][j]` = length of consecutive match ending at (i, j)
- `c[i][j]` = accumulated weighted score
- Weight function: $f(k) = k^\alpha$

$$c[i][j] = \begin{cases}
c[i-1][j-1] + (w[i][j])^\alpha - (w[i-1][j-1])^\alpha & \text{if } ref[i]=hyp[j] \\
\max(c[i-1][j], c[i][j-1]) & \text{otherwise}
\end{cases}$$

**Precision & Recall:**
$$R = \left(\frac{\text{WLCS}(ref, hyp)}{|ref|^\alpha}\right)^{1/\alpha}$$

$$P = \left(\frac{\text{WLCS}(hyp, ref)}{|hyp|^\alpha}\right)^{1/\alpha}$$

**F-Measure:**
$$F = \frac{(1+\beta^2) \cdot R \cdot P}{\beta^2 \cdot R + P}$$

### Return Type ✓
- **Type:** `dict[str, float]`
- **Keys:** `rouge_w_r`, `rouge_w_p`, `rouge_w_f`
- **Range:** Each value [0.0, 1.0]
- **Additional:** Returns within larger dict including other ROUGE variants

### Parameter Specification ✓
| Parameter | Type | Default | Range | Purpose |
|-----------|------|---------|-------|---------|
| `hypotheses` | list[str] | Required | Any strings | Generated answers |
| `references` | list[str \| list[str]] | Required | Any strings | Reference answers (1 or M per hyp) |
| `rouge_w_alpha` | float | 2.0 | > 1 | Weighting exponent |
| `beta` | float | 1.0 | > 0 | F-measure weight (1.0 = equal R/P) |
| `use_jackknife` | bool | False | T/F | Multi-reference strategy |

### Multi-Reference Handling ✓
- **Single reference:** Direct scoring
- **Multiple references:**
  - Without jackknife: Pick reference with highest recall
  - With jackknife: Leave-one-out averaging for robustness

### Implementation Details
- **Tokenization:** `re.findall(r'\b\w+\b', text.lower())` — word boundaries
- **Empty text:** Returns `{...: 0.0}` for empty hypothesis or reference
- **Corpus aggregation:** Returns mean across all (hyp, ref) pairs
- **Alpha parameter:**
  - α=2.0 (default): f(1)=1, f(2)=4, f(3)=9 — quadratic reward for consecutive matches
  - Higher α → more reward for long matches

### Test Coverage ✓
- **Test File:** `src/evaluation/tests/test_text_metrics_comprehensive.py`
- **Test Class:** `TestROUGEMetrics`
- **Coverage:** Basic LCS, weighted LCS, multi-reference

### Alignment with Thesis ✓
**Status:** Standard ROUGE-W per Lin & Och 2004
- ✓ WLCS dynamic programming
- ✓ Configurable weighting exponent (α)
- ✓ Standard precision/recall/F-measure
- ✓ Multi-reference support

---

## 4. METEOR (Metric for Evaluation of Translation with Explicit Ordering)

### Current Implementation
- **Function:** `_meteor_score(hypotheses: list[str], references: list[str]) → dict[str, float]`
- **Location:** [src/evaluation/text_metrics.py](src/evaluation/text_metrics.py#L127-L145)
- **Library:** NLTK (`nltk.translate.meteor_score`)

### Formula ✓
**METEOR computation (via NLTK):**

$$\text{METEOR} = F_{\text{align}} \times \left(1 - \text{Penalty}\right)$$

where:
- **F_align:** Harmonic mean of unigram precision and recall (standard F-score)
- **Penalty:** Fragmentation penalty = $0.5 \times \left(\frac{\text{num_chunks}}{|matches|}\right)^3$

**Matching types included:**
1. Exact word matches
2. Lemmatized matches (WordNet)
3. Synonym matches (WordNet)
4. Word order preservation

The penalty decreases the score for fragmented matches (many small sequences rather than few large ones).

### Return Type ✓
- **Type:** `dict[str, float]`
- **Keys:** `"meteor"`
- **Range:** [0.0, 1.0]
- **Structure:** Single key (unlike ROUGE which returns 15 keys)

### Parameter Specification ✓
| Parameter | Type | Default | Purpose | Valid Range |
|-----------|------|---------|---------|-------------|
| `hypotheses` | list[str] | Required | Generated answers | Any list of strings |
| `references` | list[str] | Required | Reference answers | Any list of strings (1:1 pairings) |

### Language & Dependencies
- **Primary Library:** NLTK (Natural Language Toolkit)
- **Auto-downloads on first run:**
  - `punkt_tab` — sentence tokenizer
  - `wordnet` — lemmatization/synonym data
- **Language Support:** Default English; needs configuration for Polish (current system uses English-based NLTK)
- **Import:** `nltk.translate.meteor_score.meteor_score()`

### Implementation Details
- **Tokenization:** `word_tokenize()` per NLTK
- **Corpus aggregation:** Computes METEOR per (hyp, ref) pair, returns mean
- **Empty handling:** Skips empty hypothesis/reference pairs
- **Output:** Single aggregated score (not per-component breakdown)

### Test Coverage ✓
- **Test File:** `src/evaluation/tests/test_text_metrics_comprehensive.py`
- **Test Class:** `TestMETEORMetric`
- **Tests:** Basic functionality, edge cases

### Alignment with Thesis ✓
**Status:** Standard METEOR per Lavie & Denkowski 2009
- ✓ F-score with lemmatization/synonymy
- ✓ Fragmentation penalty
- ✓ Word order awareness via NLTK
- ✓ Corpus-level aggregation

**Note:** Current implementation uses English-centric NLTK patterns. For Polish-specific METEOR, would need Polish wordnet + stemmer.

---

## 5. BERT-Score (Contextual Embedding-Based Similarity)

### Current Implementation
- **Function:** `_bertscore(hypotheses, references, lang, model_type) → dict[str, float]`
- **Location:** [src/evaluation/text_metrics.py](src/evaluation/text_metrics.py#L148-L176)
- **Library:** `bert_score` (external package)

### Formula ✓

**Token-Level Embedding Similarity:**

$$P_{\text{BERT}} = \frac{1}{m} \sum_{i=1}^{m} \max_{j=1}^{n} \cos(e_{\text{ref}}^i, e_{\text{hyp}}^j)$$

$$R_{\text{BERT}} = \frac{1}{n} \sum_{j=1}^{n} \max_{i=1}^{m} \cos(e_{\text{ref}}^i, e_{\text{hyp}}^j)$$

$$F_{\text{BERT}} = \frac{2 P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

where:
- $e_{\text{ref}}^i, e_{\text{hyp}}^j$ = contextual token embeddings from BERT
- $\cos$ = cosine similarity in embedding space
- m = hypothesis length, n = reference length

### Return Type ✓
- **Type:** `dict[str, float]`
- **Keys:** 12 total (4 variants × 3 metrics)
- **Range:** [0.0, 1.0] per metric

### Scoring Variants ✓

**4 Independent Computations:**

1. **base** (no IDF, no rescale)
   - Keys: `bertscore_precision_base`, `bertscore_recall_base`, `bertscore_f1_base`
   - Use case: Raw contextual similarity

2. **idf** (Inverse Document Frequency weighting)
   - Keys: `bertscore_precision_idf`, `bertscore_recall_idf`, `bertscore_f1_idf`
   - Weights important/rare tokens higher
   - Use case: Focus on content words vs. stop words

3. **rescaled** (Baseline rescaling, no IDF)
   - Keys: `bertscore_precision_rescaled`, `bertscore_recall_rescaled`, `bertscore_f1_rescaled`
   - Normalizes against pre-computed human baseline
   - Use case: Calibration to human judgment

4. **full** (IDF + Baseline Rescaling combined)
   - Keys: `bertscore_precision_full`, `bertscore_recall_full`, `bertscore_f1_full`
   - Combines importance weighting + calibration
   - Use case: Best practice for robust scoring

### Parameter Specification ✓
| Parameter | Type | Purpose | Valid Range | Example |
|-----------|------|---------|-------------|---------|
| `hypotheses` | list[str] | Generated answers | Any strings | `["Kot siedział na macie"]` |
| `references` | list[str] | Reference answers | Any strings | `["Kot był na macie"]` |
| `lang` | str | Language code | 'pl', 'en', 'fr', etc. | `'pl'` |
| `model_type` | str | BERT model ID | Valid HuggingFace model | `'bert-base-multilingual-cased'` |

### Model Requirements ✓
- **BERT Model Loading:** First call loads embedder (can be slow ~2-5s)
- **HuggingFace Models Supported:**
  - `bert-base-multilingual-cased` (Polish support ✓)
  - `bert-base-uncased` (English)
  - Custom fine-tuned models
- **Device:** Auto-detects GPU if available; falls back to CPU

### Implementation Details
- **Loop:** Creates 4 independent BERTScorer instances
- **Tokenization:** BERT's wordpiece tokenizer (built-in)
- **Aggregation:**
  - Per (hyp, ref) pair → precision/recall/F1 tensors
  - Corpus: `.mean().item()` to get scalar float
- **IDF Mode:** Computes token frequencies from reference corpus
- **Rescaling:** Uses pre-computed baseline from BERTScore publication

### Test Coverage ✓
- **Test File:** `src/evaluation/tests/test_text_metrics_comprehensive.py`
- **Test Class:** `TestBERTScore`
- **Tests:** Model loading, output shape, variant independence

### Alignment with Thesis ✓
**Status:** BERTScore per Zhang et al. 2020
- ✓ Contextual embeddings (BERT)
- ✓ Cosine similarity matching
- ✓ 4 independent variants
- ✓ Multi-language support (via multilingual-BERT)
- ✓ IDF weighting option
- ✓ Baseline rescaling option

---

---

# Comparative Analysis

## Retrieval vs. Text Metrics

| Aspect | Retrieval | Text |
|--------|-----------|------|
| **Input** | URLs + targets | Text documents + references |
| **Scoring** | Graded relevance [0,1] | Embedding similarity / n-gram overlap |
| **Cardinality** | Single float per metric | Dict of floats (multiple variants) |
| **Normalization** | Self-normalized (nDCG) | Corpus mean (all text metrics) |
| **Interpretability** | High (URL hierarchy explicit) | Medium (embeddings opaque) |

## nDCG vs. ROUGE-W

| Feature | nDCG@k | ROUGE-W |
|---------|--------|---------|
| **Domain** | Information Retrieval | NLP Generation |
| **Unit** | URLs/documents | Words/tokens |
| **Ranking** | Yes (rank-aware via log discount) | No (best score wins) |
| **Graded** | Yes (exponential gains) | Yes (weighted LCS) |
| **Multi-ref** | N/A (single gold URL) | Yes (multi-ref support) |
| **Threshold** | None (graded scores) | None (graded scores) |
| **Return** | float [0,1] | dict w/ r/p/f [0,1] |

## METEOR vs. BERT-Score

| Feature | METEOR | BERTScore |
|---------|--------|-----------|
| **Model** | Hand-crafted features | Neural embeddings |
| **Matching** | Exact + lemma + synonym | Continuous similarity |
| **Language Agnostic** | No (WordNet-based) | Yes (multilingual BERT) |
| **Speed** | Fast (no GPU needed) | Slow (large model) |
| **Variants** | 1 (single score) | 4 (base/idf/rescaled/full) |
| **Interpretability** | Higher (explicit features) | Lower (black-box embeddings) |
| **Thesis Alignment** | ✓ Standard METEOR | ✓ Zhang et al. 2020 |

---

## Implementation Completeness Matrix

| Metric | Implemented | Tested | Documented | Used in Evaluation | Alignment |
|--------|-------------|--------|------------|-------------------|-----------|
| nDCG@k | ✓ | ✓ | ✓ | ✓ | ✓ Thesis |
| hierarchical_relevance | ✓ | ✓ | ✓ | ✓ | ✓ Custom |
| ROUGE-W | ✓ | ✓ | ✓ | ✓ | ✓ Standard |
| METEOR | ✓ | ✓ | ✓ | ✓ | ✓ Standard |
| BERT-Score | ✓ | ✓ | ✓ | ✓ | ✓ 4 variants |

---

## Missing Implementations (If Any)

**None.** All five requested metrics are fully implemented.

**Optional enhancements:**
- Polish-specific METEOR (via Polish WordNet)
- Custom BERT models fine-tuned on Polish text
- Ablation studies for ROUGE-W alpha parameter
- MRRw integration with text metrics pipeline

---
