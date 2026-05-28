# Metric Quick Reference

## At-a-Glance Comparison

### RETRIEVAL METRICS (benchmark.py)

#### Hierarchical Relevance
```
hierarchical_relevance(retrieved_url: str, target_url: str) → float
├─ Exact match: 1.0
├─ Parent (1↑): 0.5
├─ Children (1↓): 0.25 (0.5²)
├─ Grandchild (2↓): 0.125 (0.5³)
└─ Different origin: 0.0
```

#### nDCG@k
```
ndcg_at_k(rel_scores: list[float], k: int) → float [0,1]
├─ DCG = Σ (2^rel - 1) / log₂(rank+1)
├─ IDCG = DCG of ideal ordering
└─ nDCG = DCG / IDCG
```

#### MRRw (Weighted MRR)
```
mrr_weighted_single(
  urls: list[str],
  target: str,
  alpha=0.8,
  beta=0.4
) → float [0,1]

mrr_weighted(
  gold: list[EvalRow],
  k: int,
  alpha=0.8,
  beta=0.4
) → float [0,1]

weight(d) = {
  d=0:  1.0
  d>0:  0.8^d  (deeper = less penalty)
  d<0:  0.4^|d| (shallower = more penalty)
}
formula: S = max(w(d_i) / rank_i) over all URLs
```

#### Standard Metrics (Summary)
```
hit_at_k()         → float [0,1]  (binary: hit or miss?)
mrr_at_k()         → float [0,1]  (reciprocal of first match)
precision_at_k()   → float [0,1]  (hits / k)
recall_at_k()      → float [0,1]  (hits / total_rel)
f_measure_at_k()   → float [0,1]  (harmonic mean)
average_precision_at_k() → float [0,1]  (rank-aware)
r_precision()      → float [0,1]  (precision@total_rel)
```

---

### TEXT GENERATION METRICS (text_metrics.py)

#### ROUGE-W (Weighted LCS)
```
_rouge_scores(
  hypotheses: list[str],
  references: list[str],
  beta=1.0,
  rouge_w_alpha=2.0,  ← KEY PARAM
  rouge_s_d=None,
  use_jackknife=False
) → dict[str, float]

Returns: {
  'rouge_w_r': float,  # Recall
  'rouge_w_p': float,  # Precision
  'rouge_w_f': float,  # F-measure
  ... + ROUGE-1/2/L/S variants
}

WLCS with f(k) = k^alpha (default alpha=2.0)
Precision = (WLCS / n^alpha)^(1/alpha)
Recall = (WLCS / m^alpha)^(1/alpha)
```

#### METEOR
```
_meteor_score(
  hypotheses: list[str],
  references: list[str]
) → dict[str, float]

Returns: {'meteor': float [0,1]}

Features:
├─ Exact matches
├─ Lemmatized matches (WordNet)
├─ Synonym matches (WordNet)
├─ Word order penalty
└─ Fragmentation penalty: 0.5*(chunks/matches)³
```

#### BERTScore (4 Variants)
```
_bertscore(
  hypotheses: list[str],
  references: list[str],
  lang: str,           ← e.g., 'pl', 'en'
  model_type: str      ← e.g., 'bert-base-multilingual-cased'
) → dict[str, float]

Returns (12 keys total):
[base | idf | rescaled | full] × [precision | recall | f1]

Keys: {
  'bertscore_precision_base',
  'bertscore_recall_base',
  'bertscore_f1_base',
  ... (idf, rescaled, full variants)
}

Formula:
P = (1/n) Σ max_j cos(e_hyp_i, e_ref_j)
R = (1/m) Σ max_i cos(e_ref_j, e_hyp_i)
F = 2PR/(P+R)
```

#### BLEU
```
_bleu_score(
  hypotheses: list[str],
  references: list[str]
) → dict[str, float]

Returns: {
  'bleu': float,      # Overall
  'bleu_1': float,    # Unigram
  'bleu_2': float,    # Bigram
  'bleu_3': float,    # Trigram
  'bleu_4': float,    # 4-gram
}

BLEU = BP × exp(Σ w_n * log(p_n))
```

#### ROUGE Family (All Variants)
```
_rouge_scores(...) returns:

ROUGE-1 (unigrams):
  'rouge_1_r', 'rouge_1_p', 'rouge_1_f'

ROUGE-2 (bigrams):
  'rouge_2_r', 'rouge_2_p', 'rouge_2_f'

ROUGE-L (LCS):
  'rouge_l_r', 'rouge_l_p', 'rouge_l_f'

ROUGE-W (Weighted LCS):
  'rouge_w_r', 'rouge_w_p', 'rouge_w_f'

ROUGE-S (Skip-bigrams):
  'rouge_s_r', 'rouge_s_p', 'rouge_s_f'
```

---

## Parameter Reference

### Common Retrieval Parameters
| Name | Type | Range | Default | Purpose |
|------|------|-------|---------|---------|
| k | int | ≥1 | — | Rank cut-off |
| threshold | float | [0,1] | 0.5 | Relevance binarization |
| total_rel | int | ≥1 | 1 | Gold set size |

### Common Text Parameters
| Name | Type | Range | Default | Purpose |
|------|------|-------|---------|---------|
| beta | float | >0 | 1.0 | F-measure weight |
| alpha | float | >1 | 2.0 (ROUGE-W) | Weighting exponent |
| rouge_s_d | int | ≥1 or None | None | Skip distance |
| use_jackknife | bool | T/F | False | Leave-one-out refs |
| lang | str | 'pl','en',... | — | BERT/METEOR language |
| model_type | str | valid model | — | BERT model ID |

---

## Return Type Matrix

```
METRIC                    RETURN TYPE              KEYS/CARDINALITY
====================================================================
Retrieval Metrics:
hierarchical_relevance    float [0,1]              1 value
ndcg_at_k                 float [0,1]              1 value
mrr_weighted_single       float [0,1]              1 value
mrr_weighted              float [0,1]              1 value (avg)
hit_at_k, etc.            float [0,1]              1 value each

Text Metrics:
_bleu_score               dict[str, float]         5 keys
_rouge_scores             dict[str, float]         15 keys
_meteor_score             dict[str, float]         1 key
_bertscore                dict[str, float]         12 keys
```

---

## Integration Points

### Retrieval Evaluation
**File:** `src/evaluation/benchmark.py:evaluate()`
1. Load gold set (queries + URLs)
2. For each query:
   - Retrieve top-k chunks
   - Extract URLs, normalize, deduplicate
   - **Compute hierarchical_relevance** → rel_scores
   - Apply hit_at_k, mrr_at_k, ndcg_at_k, etc.
   - **Compute mrr_weighted_single** separately
3. Average across all queries

### Text Evaluation
**File:** `src/evaluation/text_metrics.py:evaluate()`
1. Load generated and reference answers
2. Match by question
3. Compute all text metrics (**BLEU, ROUGE, METEOR, BERTScore**)
4. Output per-question CSV + summary JSON

---

## File Locations & Dependencies

| Metric | File | Imports |
|--------|------|---------|
| hierarchical_relevance | benchmark.py | urllib, math |
| nDCG@k | benchmark.py | math.log2 |
| MRRw | benchmark.py | statistics.mean |
| ROUGE-W | text_metrics.py | collections.Counter, re |
| METEOR | text_metrics.py | nltk.translate, nltk.download |
| BERTScore | text_metrics.py | bert_score.BERTScorer |
| BLEU | text_metrics.py | sacrebleu |

---

## Key Equations Summary

### Hierarchical Relevance
```
rel = {
  1.0              if retrieved == target
  0.5^d            if target ancestor (d levels up)
  0.5^(d+1)        if target descendant (d levels down)
  0.0              otherwise
}
```

### nDCG
```
DCG = Σ(1..k) (2^rel_i - 1) / log₂(rank_i + 1)
IDCG = DCG(sorted_desc)
nDCG = DCG / IDCG ∈ [0, 1]
```

### MRRw
```
w(d) = {
  1.0        if d = 0 (exact)
  α^d        if d > 0 (deeper)
  β^|d|      if d < 0 (shallower)
}
S_q = max_i(w(d_i) / rank_i)
MRRw = (1/N) Σ S_q
```

### ROUGE-W
```
WLCS = c(m,n) via DP with f(k) = k^α
R = (WLCS / m^α)^(1/α)
P = (WLCS / n^α)^(1/α)
F = (1+β²) RP / (β²R + P)
```

### METEOR
```
F_align = (2 · Precision · Recall) / (Precision + Recall)
Penalty = 0.5 * (num_chunks / num_matches)³
METEOR = F_align × (1 - Penalty)
```

### BERTScore
```
P = (1/m) Σ max_j cos(e_hyp_i, e_ref_j)
R = (1/n) Σ max_i cos(e_ref_j, e_hyp_i)
F = 2PR / (P+R)
```

---

## Testing & Validation

### Test Files Available
```
src/evaluation/tests/test_retrieval_metrics.py
├─ TestHierarchicalRelevance
├─ TestDCGAndNDCG
├─ TestMRRAtK
└─ ... (other retrieval metrics)

src/evaluation/tests/test_text_metrics_comprehensive.py
├─ TestROUGEMetrics
├─ TestMETEORMetric
├─ TestBERTScore
└─ TestBLEUScore
```

### Running Tests
```bash
# Retrieval metrics
pytest src/evaluation/tests/test_retrieval_metrics.py -v

# Text metrics
pytest src/evaluation/tests/test_text_metrics_comprehensive.py -v

# All evaluation tests
pytest src/evaluation/tests/ -v
```

---
