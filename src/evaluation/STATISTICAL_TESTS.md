# Testy Statystyczne — Porównanie Modeli A vs B

Moduł `statistical_comparison.py` implementuje trzy statystyczne testy porównujące dwa warianty modelu RAG na tej samej zbiorze pytań.

## Testy

### 1. **Wilcoxon Signed-Rank Test**

Test nieparametryczny badający hipotezę $H_0: \mathbb{E}[d_q] = 0$, gdzie $d_q = \text{score}_A(q) - \text{score}_B(q)$.

**Kiedy użyć:**
- Gdy metryki retrieval są ograniczone (0–1) i mogą mieć asymetryczny rozkład
- Nie spełniono założenia normalności wymaganego przez $t$-test
- Chcemy porównać dwa warianty na **tych samych** pytan'ach (paired test)

**Wynik:**
- `p-value ≤ 0.05` → **statystycznie istotna różnica** (zaznaczamy `***`)
- `p-value > 0.05` → brak istotnej różnicy (`n.s.`)

### 2. **Goodman–Kruskal γ (Gamma) Coefficient**

Miara asocjacji ordinalnej między rankingami przydzielonymi przez dwa modele.

$$\hat{\gamma} = \frac{C - D}{C + D} \in [-1, 1]$$

Gdzie:
- $C$ = liczba par konkordantnych (oba rankingi rosną lub oba maleją)
- $D$ = liczba par dyskordantnych (jedno rośnie, drugie maleje)

**Interpretacja:**
- $\gamma \approx +1$ → modele się mocno zgadzają na temat trudnych pytań
- $\gamma \approx 0$ → brak korelacji ordinalnej
- $\gamma \approx -1$ → modele mają przeciwne opinie na temat trudności

### 3. **Kappa Coefficient**

Mierzy **stabilność** retrievera: czy powtórzone uruchomienia zwracają te same dokumenty.

$$\bar{\kappa} = \frac{1}{|Q|\binom{n}{2}}\sum_{q \in Q}\sum_{1 \le i < j \le n}\kappa_{i,j}(q)$$

**Interpretacja:**
- $\bar{\kappa} \approx 1$ → retriever jest deterministyczny (zawsze te same wyniki)
- $\bar{\kappa} \approx 0$ → wyniki są losowe między uruchomieniami

---

## Użycie

### Krok 1: Wygeneruj dwa benchmarki (dla obu modeli)

```bash
# Model A
export PYTHONPATH=src
python -m evaluation.benchmark \
  --api-url http://localhost:8000/chat \
  --output-dir src/evaluation/data/model_a

# Model B (mogą się różnić: init. parametry, inne modele LLM, itp.)
python -m evaluation.benchmark \
  --api-url http://localhost:8000/chat \
  --output-dir src/evaluation/data/model_b
```

Z każdego benchmarku otrzymasz pliki:
- `eval_per_query_<timestamp>.csv` — metryki dla każdego pytania
- `eval_summary_<timestamp>.csv` — średnie

### Krok 2a: Uruchom testy (Fixed k)

```bash
python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/data/model_a/eval_per_query_20260515T100000.csv \
  --model_b_csv src/evaluation/data/model_b/eval_per_query_20260515T110000.csv \
  --metrics "mrr@10,hit@10,ndcg@10,map@10" \
  --alpha 0.05 \
  --output-dir src/evaluation/data
```

**Dostępne opcje:**
- `--model_a_csv` (required) — ścieżka do eval_per_query CSV dla Model A
- `--model_b_csv` (required) — ścieżka do eval_per_query CSV dla Model B
- `--metrics` — lista metryk oddzielonych przecinkami (default: `mrr@10,hit@10,ndcg@10`)
  - Dostępne: `hit@k`, `mrr@k`, `mrrw@k`, `recall@k`, `precision@k`, `f1@k`, `ndcg@k`, `map@k`, `r_prec`
  - Dla każdego $k$: `@1`, `@3`, `@5`, `@10`, `@20`, `@30`
- `--alpha` — poziom istotności (default 0.05)
- `--output-dir` — katalog na wyniki (default `src/evaluation/data`)

### Krok 2b: Uruchom testy (Adaptive k) ⭐

Adaptive metryki są **automatycznie generowane przez benchmark.py** i zapisywane do CSV jako kolumny `hit_adaptive` i `mrr_adaptive`.

**Opcja 1: Jeśli CSV z nowego benchmark.py** (ma już `hit_adaptive`, `mrr_adaptive`)

```bash
python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/data/model_a/eval_per_query_20260515T100000.csv \
  --model_b_csv src/evaluation/data/model_b/eval_per_query_20260515T110000.csv \
  --metrics "hit_adaptive,mrr_adaptive" \
  --output-dir src/evaluation/data
```

**Opcja 2: Jeśli CSV ze starego benchmark.py** (bez adaptive k) — oblicz je na bieżąco

```bash
python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/data/model_a/eval_per_query_20260515T100000.csv \
  --model_b_csv src/evaluation/data/model_b/eval_per_query_20260515T110000.csv \
  --use-adaptive-k \
  --output-dir src/evaluation/data
```

**Co robi adaptive k?**

Zamiast obliczać `hit@10`, `mrr@10` dla wszystkich queryów (nawet jeśli retriever zwrócił tylko 5 linków), adaptacyjne k oblicza metryki **dla rzeczywistej liczby zwróconych linków**:

Każdy query ma własne k = liczba faktycznie zwróconych linków dla tego pytania.

**Przykład:**
```
Query A: zwrócono 3 linki  → hit_adaptive = czy gold jest w 3 linkach?
Query B: zwrócono 5 linków → hit_adaptive = czy gold jest w 5 linkach?
Query C: zwrócono 2 linki  → hit_adaptive = czy gold jest w 2 linkach?
```

**Korzyści adaptive k:**
- ✅ Unikasz artefaktów (jak `hit@10 = hit@5 = hit@3` gdy max jest 5)
- ✅ Każdy query ma uczciwą ocenę na podstawie tego co zwrócił retriever
- ✅ Godniej porównywać warianty modelu na tych samych danych



### Krok 3: Interpretuj wyniki

Wynik w terminalu:
```
======================================================================
Metric: mrr@10
======================================================================
  Queries:              50
  Model A (mean):       0.823400
  Model B (mean):       0.795600
  Difference (A - B):   +0.027800
  Difference (std):     0.087123

  Wilcoxon Signed-Rank Test (α=0.05, two-tailed):
    Test statistic:     312.0000
    p-value:            0.0312 ***

  Goodman–Kruskal γ Coefficient:
    γ:                  +0.642
    Concordant pairs:   987
    Discordant pairs:   543
```

**Interpretacja:**
- Model A ma średnio o ~2.78 wysokszy MRR@10
- Verschiedenheit jest **istotna statystycznie** ($p = 0.0312 < 0.05$)
- $\gamma = 0.642$ → modele dość dobrze się zgadzają na temat których pytań są trudne

Dla adaptive k:
```
======================================================================
Metric: mrr_adaptive
======================================================================
  Queries:              50
  Model A (mean):       0.712400
  Model B (mean):       0.658900
  Difference (A - B):   +0.053500   ← Większa różnica niż przy fixed k
  Difference (std):     0.142100

  Wilcoxon Signed-Rank Test (α=0.05, two-tailed):
    Test statistic:     267.0000
    p-value:            0.0087 ***   ← Bardziej istotna różnica

  Goodman–Kruskal γ Coefficient:
    γ:                  +0.558
    Concordant pairs:   812
    Discordant pairs:   638
```

**Wynikowy JSON** (`statistical_comparison_<timestamp>.json`):
```json
{
  "timestamp": "20260515T120000",
  "model_a_csv": "src/evaluation/data/model_a/eval_per_query_...csv",
  "model_b_csv": "src/evaluation/data/model_b/eval_per_query_...csv",
  "alpha": 0.05,
  "use_adaptive_k": false,
  "results": [
    {
      "metric": "mrr@10",
      "n_queries": 50,
      "mean_A": 0.8234,
      "mean_B": 0.7956,
      "mean_diff": 0.0278,
      "std_diff": 0.0871,
      "wilcoxon_statistic": 312.0,
      "wilcoxon_pvalue": 0.0312,
      "wilcoxon_significant": true,
      "gamma_coeff": 0.642,
      "gamma_concordant": 987,
      "gamma_discordant": 543
    }
  ]
}
```

---

## Krok 4: Analiza wyników

### Adaptive k — kiedy użyć?

Jeśli w CSV widzisz:
```
Query A: 3 linki
Query B: 5 linków
Query C: 2 linki
```

Nie mają sensu fixed metryki jak `hit@10` (bo max jest 5 linków).
Użyj `--use-adaptive-k`, otrzymasz metryki dla rzeczywistych $k$.

---

## Przykład praktyczny

### Scenariusz 1: Porównanie Fixed k (zwykłe benchmarki)

```bash
# Terminal 1: Model A (temperatura 0.2)
export PYTHONPATH=src
export EXPERIMENT_DIM=temperature
export EXPERIMENT_TEMP=0.2
uvicorn api.api:app --port 8000

# Terminal 2: Model B (temperatura 0.8)
export PYTHONPATH=src
export EXPERIMENT_DIM=temperature
export EXPERIMENT_TEMP=0.8
uvicorn api.api:app --port 8001

# Terminal 3: Benchmark dla A
python -m evaluation.benchmark \
  --api-url http://localhost:8000/chat \
  --input-csv src/evaluation/data/questions_filtered.csv \
  --output-dir src/evaluation/data/temp_0.2

# Terminal 4: Benchmark dla B
python -m evaluation.benchmark \
  --api-url http://localhost:8001/chat \
  --input-csv src/evaluation/data/questions_filtered.csv \
  --output-dir src/evaluation/data/temp_0.8

# Porównanie
python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/data/temp_0.2/eval_per_query_*.csv \
  --model_b_csv src/evaluation/data/temp_0.8/eval_per_query_*.csv \
  --metrics "mrr@5,hit@5,ndcg@5"
```

### Scenariusz 2: Porównanie Adaptive k (liczba linków się różni)

```bash
# Gdy wiemy, że liczba zwróconych linków różni się między queryami
python -m evaluation.statistical_comparison \
  --model_a_csv src/evaluation/data/temp_0.2/eval_per_query_*.csv \
  --model_b_csv src/evaluation/data/temp_0.8/eval_per_query_*.csv \
  --use-adaptive-k
```

Wynik pokażeotechnique hit po każdego pytania na podstawie rzeczywistej liczby linków:
- Query 1: zwrócono 3 linki → hit_adaptive obliczone dla k=3
- Query 2: zwrócono 5 linków → hit_adaptive obliczone dla k=5
- itp.

---

## Kod wewnętrzny

### Obliczanie rangów

Dla każdego pytania, ranga to **pozycja złotego URL'a** w liście `chatbot_links`:

```python
def extract_rank_from_sources(sources_str: str) -> int:
    """Zwraca 1-indexed pozycję; 999 jeśli nie znaleziono."""
    urls = [u.strip() for u in sources_str.split(";")]
    return len(urls) if urls else 999
```

### Concordant vs Discordant

Dla każdej pary pytań $(q_i, q_j)$:

```
Concordant:
  (rank_A[i] < rank_A[j] AND rank_B[i] < rank_B[j])  OR
  (rank_A[i] > rank_A[j] AND rank_B[i] > rank_B[j])

Discordant:
  (rank_A[i] < rank_A[j] AND rank_B[i] > rank_B[j])  OR
  (rank_A[i] > rank_A[j] AND rank_B[i] < rank_B[j])
```

---

## Referencje

- Wilcoxon test: `scipy.stats.wilcoxon` (docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
- Goodman-Kruskal gamma: Goodman & Kruskal (1954), "Measures of association for cross classifications"
- Cohen's kappa: Cohen (1960), "A coefficient of agreement for nominal scales"
