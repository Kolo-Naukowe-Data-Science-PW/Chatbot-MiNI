# LLM-as-Judge Batch Evaluator

Narzędzie do porównania dwóch modeli chatbota za pomocą LLM-as-Judge, które ocenia odpowiedzi na skali 1-5.

## Cechy

- ✓ Porównuje pre-computed odpowiedzi z CSV (bez dodatkowych API calls dla generowania odpowiedzi)
- ✓ Zwraca metryki kompatybilne ze `statistical_comparison.py`
- ✓ Generuje szczegółowe raporty JSON z wynikami
- ✓ Uruchamialny przez GitHub Actions workflow
- ✓ Obsługa CLI i environment variables

## Output

Skrypt generuje 3 pliki:

### 1. `llm_judge_metrics_model_a.csv`
Metryki modelu A per query (kompatybilne z `statistical_comparison.py`):
```
pytanie,usefulness,accuracy,conciseness
"Co powinienem zrobić...",4,5,3
```

### 2. `llm_judge_metrics_model_b.csv`
Metryki modelu B w tej samej strukturze.

### 3. `llm_judge_results.json`
Szczegółowe wyniki z podsumowaniem:
```json
{
  "evaluation_metadata": {
    "timestamp": "2026-06-10T...",
    "judge_model": "anthropic/claude-opus-4.7",
    "total_evaluated": 217,
    "success_rate": "98.5%"
  },
  "model_a_stats": {
    "usefulness": {"mean": 4.2, "median": 4, "std": 0.8, ...},
    "accuracy": {...},
    "conciseness": {...}
  },
  "model_b_stats": {...},
  "pairwise_comparison": {
    "model_a_wins": 105,
    "model_b_wins": 112,
    "model_a_win_rate": "48.4%"
  },
  "detailed_results": [
    {
      "query": "...",
      "model_a": {"usefulness": 4, "accuracy": 5, "conciseness": 3},
      "model_b": {"usefulness": 3, "accuracy": 4, "conciseness": 4},
      "better_variant": "A",
      "judge_reason": "..."
    }
  ]
}
```

## Użycie Lokalne

### Wymagania
```bash
pip install pandas numpy scipy python-dotenv openai
```

### Setup API key
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### Uruchomienie
```bash
python -m src.evaluation.llm_judge.batch_evaluator \
  --model-a src/evaluation/data/stat_test/200/answers_google_gemini-3.5-flash_t0.2_p2_200.csv \
  --model-b src/evaluation/data/stat_test/200/answers_openai_gpt-oss-120b_free_t0.2_p2_200.csv \
  --judge-model anthropic/claude-opus-4.7 \
  --language pl \
  --output-dir ./results
```

### Parametry

| Parametr | Default | Opis |
|----------|---------|------|
| `--model-a` | ✓ wymagane | CSV Model A (kolumny: pytanie, odpowiedz_wygenerowana) |
| `--model-b` | ✓ wymagane | CSV Model B |
| `--judge-model` | anthropic/claude-opus-4.7 | Model do judgingu (via OpenRouter) |
| `--language` | pl | Język dla judge prompt |
| `--output-dir` | . | Katalog dla wyników |
| `--limit` | None | Limit pytań do evaluacji (dla testów) |
| `--delay` | 0.5 | Delay między API calls (s) |

## GitHub Actions Workflow

### Setup

1. Dodaj `OPENROUTER_API_KEY` jako secret w GitHub:
   - Repo Settings → Secrets and variables → Actions → New repository secret
   - Name: `OPENROUTER_API_KEY`
   - Value: Twój klucz OpenRouter

2. Workflow jest zainstalowany w `.github/workflows/llm_judge_evaluation.yml`

### Uruchomienie

```bash
# Otwórz Actions tab w GitHub
# Kliknij "LLM-as-Judge Batch Evaluation"
# Kliknij "Run workflow"
# Podaj parametry (domyślne już ustawione)
# Czekaj ~30-60 minut
# Download artifacts gdy skończy
```

Lub z CLI:
```bash
gh workflow run llm_judge_evaluation.yml \
  -f model_a_csv="src/evaluation/data/stat_test/200/answers_google_gemini-3.5-flash_t0.2_p2_200.csv" \
  -f model_b_csv="src/evaluation/data/stat_test/200/answers_openai_gpt-oss-120b_free_t0.2_p2_200.csv" \
  -f judge_model="anthropic/claude-opus-4.7" \
  -f language="pl"
```

## Integracja z `statistical_comparison.py`

Wygenerowane CSV można użyć bezpośrednio:

```bash
python -m src.evaluation.statistical_comparison \
  --file1 llm_judge_metrics_model_a.csv \
  --file2 llm_judge_metrics_model_b.csv \
  --metric-col "usefulness" \
  --output-dir ./comparison_results
```

Lub porównanie wielu metryk:

```bash
for metric in usefulness accuracy conciseness; do
  python -m src.evaluation.statistical_comparison \
    --file1 llm_judge_metrics_model_a.csv \
    --file2 llm_judge_metrics_model_b.csv \
    --metric-col "$metric" \
    --output-dir "./comparison_$metric"
done
```

## Ograniczenia i Notatki

- **Rate limiting**: Podawaj `--delay` wyższy na production (~1s rekomendowane)
- **Koszty API**: Każde porównanie to 1 API call → ~$0.01 za 20 pytań z Claude Opus
- **Timeout**: Workflow ma limit 120 minut; dla >1000 pytań możesz potrzebować zwiękkszenia
- **Determinizm**: Judge z `temperature=0.0` powinien być determinystyczny, ale mogą biti marginalne różnice

## Troubleshooting

### Błąd: `OPENROUTER_API_KEY not set`
```bash
# Lokalnie
export OPENROUTER_API_KEY="sk-..."

# GitHub: dodaj secret do repo
```

### Błąd: `ModuleNotFoundError: src.evaluation`
```bash
# Z głównego katalogu projektu:
python -m src.evaluation.llm_judge.batch_evaluator ...

# Lub dodaj do PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src.evaluation.llm_judge.batch_evaluator ...
```

### CSV load error: kolumny nie znalezione
Script automatycznie szuka wspólnych nazw kolumn (`pytanie`, `odpowiedz_wygenerowana`).
Jeśli masz inne nazwy, zmień je w CSV lub zmodyfikuj `load_answers_csv()` w `batch_evaluator.py`.

## Licencja

Part of Chatbot-MiNI project. See LICENSE.
