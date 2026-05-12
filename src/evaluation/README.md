# Evaluation

All evaluation scripts, metrics, and benchmark data for MiNIonek.

## Contents

| File / Folder | Purpose |
|---|---|
| `benchmark.py` | Main benchmark: runs retrieval on the gold eval set, computes Hit@k, MRR@k, nDCG@k, MAP@k, Precision-Recall curves. **Use this.** |
| `benchmark_v1.py` | Legacy simple benchmark (basic Hit@k / MRR, no hierarchical scoring). |
| `metrics.py` | BERTScore class for text-level answer quality evaluation. |
| `prepare_data.py` | CSV loading utilities used by the BERTScore pipeline. |
| `eval_with_playwright.py` | Playwright macro: automates asking questions through the live chatbot UI and collecting answers to a TSV. Requires a running deployment. |
| `llm_judge/judge.py` | LLM-as-a-judge: rates two chatbot answers on usefulness / accuracy / conciseness (1–5 scales) and picks the better variant. |
| `llm_judge/testpro_runner.py` | Batch runner: reads eval CSV, calls `/chat` twice per question with random model configs, runs judge, writes results to CSV. |
| `data/` | Evaluation datasets (see below). |

## Evaluation Data (`data/`)

| File | Description |
|---|---|
| `questions.csv` | Raw questions from student survey (3 columns of free-text answers) |
| `questions_cat.csv` | Questions with topic category labels |
| `questions_filtered.csv` | Filtered eval set: `query` + gold `strona` URL — **used by benchmark.py** |
| `questions_with_links.csv` | Full eval set with source links — used by testpro_runner |
| `test.csv` | Small fixture for unit tests |

## Running the main benchmark

The benchmark requires a running Qdrant instance with ingested data. Start the API first, or just run the retrieval module directly.

```bash
# From repo root
export PYTHONPATH=src
python -m evaluation.benchmark
```

Outputs:
- Metrics table in stdout (Hit@k, MRR@k, nDCG@k, MAP@k, R-Precision for k = 3, 5, 7, 10)
- `pr_curves.png` — mean interpolated Precision-Recall curves for each k
- `metrics_summary.png` — grouped bar chart of all metrics

## Running LLM-as-a-judge batch evaluation

Requires the `/chat` API to be running locally.

```bash
export PYTHONPATH=src
python -m evaluation.llm_judge.testpro_runner \
  --judge-model openai/gpt-4o \
  --input-csv src/evaluation/data/questions_with_links.csv \
  --output-csv src/data/feedback/llm_judge_results.csv \
  --limit 50
```

## Generating golden answers (3 supermodels)

Requires `OPENROUTER_API_KEY` set in environment.  
Qdrant / running API **not needed** — supermodele odpowiadają bez RAG.

```bash
export PYTHONPATH=src

# Test na 10 pytaniach (tani sprawdzian)
python -m evaluation.generate_golden_answers --limit 10

# Wszystkie pytania
python -m evaluation.generate_golden_answers

# Wznów po przerwaniu (pomija już zapisane wiersze)
python -m evaluation.generate_golden_answers --resume

# Wymuś odpowiedź nawet na pytaniach sesja-specyficznych
python -m evaluation.generate_golden_answers --no-skip-personal
```

Output: `src/evaluation/data/golden_answers.csv`

| Kolumna | Opis |
|---|---|
| `query` | Pytanie ze zbioru testowego |
| `gold_url` | Złoty URL (z `questions_filtered.csv`) |
| `golden_answer_gpt` | Odpowiedź `openai/gpt-5.5` |
| `golden_answer_opus` | Odpowiedź `anthropic/claude-opus-4.7` |
| `golden_answer_gemini` | Odpowiedź `google/gemini-3.1-pro-preview-customtools` |
| `skip_reason` | `session_context` jeśli pytanie wymaga planu osobistego, inaczej puste |

Pytania z `skip_reason=session_context` są zapisywane z pustymi odpowiedziami —
nie pomijane całkowicie, żeby CSV miał kompletną listę pytań.

---

## Kontrola eksperymentu A/B (EXPERIMENT_DIM)

`testpro_runner.py` porównuje dwa warianty odpowiedzi (A vs B).  
Zmienna `EXPERIMENT_DIM` decyduje, co różni A od B — **tylko jedna rzecz na raz**.

| Wartość | Co się losuje | Co jest stałe |
|---|---|---|
| `model` **(domyślne)** | Dwa różne modele z `MODEL_POOL` | temperatura, persona |
| `temperature` | Niska temp (0.0–0.3) vs wysoka (0.6–0.9) | model, persona |
| `persona` | Dwa różne style odpowiedzi | model, temperatura |

```bash
# Eksperyment: który model odpowiada lepiej?
EXPERIMENT_DIM=model python -m evaluation.llm_judge.testpro_runner \
  --judge-model anthropic/claude-opus-4.7 --limit 50

# Eksperyment: czy temperatura ma znaczenie?
EXPERIMENT_DIM=temperature \
EXPERIMENT_MODEL=openai/gpt-4o-mini \
python -m evaluation.llm_judge.testpro_runner \
  --judge-model anthropic/claude-opus-4.7 --limit 50

# Eksperyment: która persona (styl) działa lepiej?
EXPERIMENT_DIM=persona \
EXPERIMENT_MODEL=openai/gpt-4o-mini \
EXPERIMENT_TEMP=0.2 \
python -m evaluation.llm_judge.testpro_runner \
  --judge-model anthropic/claude-opus-4.7 --limit 50

# Eksperyment: który model działa lepiej, przy formalnej personie (indeks 2)?
EXPERIMENT_DIM=model \
EXPERIMENT_TEMP=0.2 \
EXPERIMENT_PERSONA=2 \
python -m evaluation.llm_judge.testpro_runner \
  --judge-model anthropic/claude-opus-4.7 --limit 50
```

Dodatkowe zmienne baseline (używane gdy dany wymiar jest stały):

| Zmienna | Domyślna wartość | Kiedy używana |
|---|---|---|
| `EXPERIMENT_MODEL` | `openai/gpt-4o-mini` | przy `temperature` i `persona` |
| `EXPERIMENT_TEMP` | `0.2` | przy `model` i `persona` |
| `EXPERIMENT_PERSONA` | `0` (indeks w liście `PERSONAS`) | przy `model` i `temperature` |

Dostępne persony (indeksy 0–3):

| Indeks | Styl |
|---|---|
| `0` | Krótko i konkretnie |
| `1` | Luzno i przyjaźnie |
| `2` | Formalnie i akademicko |
| `3` | Wyczerpująco z detalami |

---

## LLM-as-a-judge vs golden answers

Porównuje odpowiedź chatbota z odpowiedzią supermodelu (`golden_answers.csv`).
Wymaga uruchomionego API (`/chat`) oraz wygenerowanych golden answers.

```bash
export PYTHONPATH=src

# Chatbot vs odpowiedzi Opus (domyślne), pierwsze 20 pytań
python -m evaluation.llm_judge.golden_judge_runner \
  --judge-model anthropic/claude-opus-4.7 \
  --limit 20

# Chatbot vs odpowiedzi GPT, pełny przebieg
python -m evaluation.llm_judge.golden_judge_runner \
  --judge-model openai/gpt-5.5 \
  --golden-model gpt

# Wznów po przerwaniu
python -m evaluation.llm_judge.golden_judge_runner \
  --judge-model anthropic/claude-opus-4.7 \
  --resume
```

Output: `src/data/feedback/golden_judge_results.csv`

| Kolumna | Opis |
|---|---|
| `chatbot_answer` | Odpowiedź naszego chatbota (z RAG) |
| `golden_answer` | Odpowiedź supermodelu (bez RAG) |
| `chatbot_*/golden_*` | Oceny 1–5: usefulness, accuracy, completeness |
| `better` | `chatbot` lub `golden` — która odpowiedź lepsza |
| `chatbot_weaknesses` | Jedno zdanie o słabościach odpowiedzi chatbota |
| `golden_weaknesses` | Jedno zdanie o słabościach odpowiedzi supermodelu |

**Ważna asymetria w promptcie sędziego**: golden answer może mówić *„nie mam dostępu do aktualnego planu, ale generalnie..."* — to jest poprawne zachowanie modelu bez RAG i NIE jest karane. Sędzia ocenia chatbota za to, czy dobrze wykorzystał swój kontekst RAG.

---

## Retrieval metrics explained

All metrics use **hierarchical URL relevance** — a retrieved URL that is a parent or child of the gold URL gets partial credit (not just exact match):

| Relationship | Score |
|---|---|
| Exact match | 1.00 |
| Parent URL (1 level up) | 0.50 |
| Grandparent (2 levels up) | 0.25 |
| Child URL (1 level down) | 0.25 |
| Grandchild (2 levels down) | 0.125 |
| Unrelated | 0.00 |

`RELEVANCE_THRESHOLD = 0.5` (configurable) determines what counts as a "hit" in binary metrics.
