# Lokalny setup chatbota (Windows)

Instrukcja krok po kroku — od zera do działającego chatbota z interfejsem React.

**Szacowany czas:** ~30–45 minut (plus pobieranie modelu bge-m3 ~570 MB i indeksowanie).

---

## Wymagania wstępne

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) lub Anaconda (zainstalowane i dostępne w terminalu)
- [Node.js 18+](https://nodejs.org/) (sprawdź: `node -v`)
- Klucze API:
  - **OPENROUTER_API_KEY** — [openrouter.ai](https://openrouter.ai) → Keys (darmowe konto wystarczy)
  - **FIRECRAWL_API_KEY** — [firecrawl.dev](https://firecrawl.dev) → Dashboard (500 stron free/mies.)

---

## Krok 1 — Środowisko Conda

Oba pliki `.yml` w repozytorium to dumpy osobistych środowisk konkretnych osób (Python 3.13, linux-specific itp.) — **nie używaj ich**. Zamiast tego stwórz nowe środowisko ręcznie:

```powershell
# PowerShell — uruchom z dowolnego miejsca
conda create -n chatbot_mini python=3.11 -y
conda activate chatbot_mini
```

Następnie zainstaluj PyTorch CPU (wersja specyficzna dla Windows) i resztę zależności:

```powershell
# PyTorch CPU dla Windows — koniecznie tą komendą, nie pip install torch wprost
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Wszystkie pozostałe zależności z pliku linuxowego (pip-only, conda deps nie potrzebujemy)
pip install sentence-transformers>=3.0.1 transformers>=4.41.0 scikit-learn numpy pandas
pip install firecrawl-py huggingface_hub>=0.23.0 langchain-huggingface>=0.1.7
pip install fastapi uvicorn python-dotenv openai
pip install langchain langchain-community langchain-text-splitters
pip install "qdrant-client[fastembed]" fastembed
pip install pypdf python-docx pdfminer.six requests beautifulsoup4 networkx
pip install bert-score facebook-scraper
```

Zainstaluj sam projekt (potrzebne żeby pytest i uvicorn znalazły moduły):

```powershell
# Z katalogu repo (tam gdzie jest pyproject.toml)
cd "C:\Users\basia\OneDrive\Pulpit\Chatbot-MiNI-1"
pip install -e ".[dev]"
```

---

## Krok 2 — Plik `.env`

Stwórz plik `.env` w katalogu głównym repozytorium:

```env
OPENROUTER_API_KEY=sk-or-twój-klucz-tutaj
FIRECRAWL_API_KEY=fc-twój-klucz-tutaj
PIPELINE_VERSION=2
MODEL_NAME=openai/gpt-4o-mini
QDRANT_DIR=src/data/qdrant_db
```

> **PIPELINE_VERSION=2** — scrape ~35 skurowanych URL-i (dziekanat, regulaminy, plany studiów, kierunki), LLM-based fact extraction. Wystarczy do testów i jest szybki (~10–20 min). Wersja 3/4 scrapuje setki stron i trwa kilka godzin.

---

## Krok 3 — Vite proxy (jednorazowa zmiana dla local dev)

Vite jest skonfigurowany do pracy w Dockerze (`http://api:8000`). Na lokalnej maszynie musisz zmienić hosta na `localhost`.

Otwórz `src/frontend/vite.config.js` i zmień **jedną linię**:

```js
// PRZED (Docker):
target: "http://api:8000",

// PO (local dev):
target: "http://localhost:8000",
```

> ⚠️ Nie commituj tej zmiany — jest tylko do lokalnych testów. Możesz też użyć `git stash` przed commitem.

---

## Krok 4 — Pipeline ingestion (budowanie bazy wiedzy)

Wszystkie komendy uruchamiaj z **katalogu głównego repozytorium** w PowerShell z aktywnym środowiskiem `chatbot_mini`.

```powershell
# Ustaw PYTHONPATH (wymagane dla każdej sesji PowerShell)
$env:PYTHONPATH = "src"

# Krok 4a — Scraping (pobiera ~35 stron MiNI PW)
# Zajmuje ~5 minut, wymaga FIRECRAWL_API_KEY
python -m ingestion.scraper
# → zapisuje pliki do: src/data/scraped_raw/

# Krok 4b — Opis plików XLSX/DOCX (opcjonalny dla v2, potrzebny od v3)
# python -m ingestion.describe_files  # pomiń dla PIPELINE_VERSION=2

# Krok 4c — Ekstrakcja faktów przez LLM
# Zajmuje ~5–10 minut, wymaga OPENROUTER_API_KEY
python -m ingestion.extract_facts
# → zapisuje pliki JSON do: src/data/facts/

# Krok 4d — Embedding i załadowanie do Qdrant
# Pobiera model BAAI/bge-m3 (~570 MB) przy pierwszym uruchomieniu
# Zajmuje ~5–15 minut
python -m ingestion.ingest_facts
# → tworzy bazę wektorową w: src/data/qdrant_db/
```

Po zakończeniu `ingest_facts` powinieneś zobaczyć logi w stylu:
```
INFO: Ingested X facts into Qdrant collection 'mini_facts'
```

---

## Krok 5 — Uruchomienie API

W nowym oknie PowerShell (z aktywnym `chatbot_mini`):

```powershell
cd "C:\Users\basia\OneDrive\Pulpit\Chatbot-MiNI-1"
$env:PYTHONPATH = "src"
uvicorn api.api:app --reload --port 8000
```

Sprawdź czy działa: otwórz [http://localhost:8000/docs](http://localhost:8000/docs) — powinna pojawić się Swagger UI z endpointami `/chat`, `/chat/stream`, `/feedback`.

---

## Krok 6 — Uruchomienie frontendu React

W kolejnym nowym oknie PowerShell:

```powershell
cd "C:\Users\basia\OneDrive\Pulpit\Chatbot-MiNI-1\src\frontend"
npm install        # tylko przy pierwszym uruchomieniu
npm run dev
```

Otwórz [http://localhost:8501](http://localhost:8501) — chatbot powinien działać.

---

## Krok 7 — Weryfikacja end-to-end

1. Wybierz rolę (np. Student I roku)
2. Wybierz kierunek i semestr
3. Zadaj pytanie po polsku, np. *„Jakie są godziny pracy dziekanatu?"*
4. W trybie **Użytkowa** odpowiedź powinna pojawić się stopniowo (streaming)
5. Sprawdź logi w oknie z uvicorn — powinny pokazać retrieval_query i model użyty do odpowiedzi

---

## Uruchamianie testów

```powershell
cd "C:\Users\basia\OneDrive\Pulpit\Chatbot-MiNI-1"
$env:PYTHONPATH = "src"   # nie jest potrzebny gdy masz pip install -e .
pytest
```

Testy nie potrzebują działającego Qdrant ani kluczy API — wszystko jest mockowane.

---

## Typowe problemy

| Problem | Rozwiązanie |
|---------|-------------|
| `ModuleNotFoundError: No module named 'ingestion'` | Sprawdź czy `$env:PYTHONPATH = "src"` jest ustawione, lub czy masz `pip install -e .` |
| `ModuleNotFoundError: No module named 'src'` | Uruchamiasz komendy z innego katalogu niż root repozytorium |
| Scraper zwraca błąd 401 | Sprawdź FIRECRAWL_API_KEY w `.env` |
| LLM zwraca błąd 401 | Sprawdź OPENROUTER_API_KEY w `.env` |
| Qdrant nie startuje | Usuń `src/data/qdrant_db/` i uruchom `ingest_facts` ponownie |
| Frontend nie może połączyć się z API | Sprawdź czy `target: "http://localhost:8000"` jest w `vite.config.js` |
| `torch` nie importuje się | Przeinstaluj: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Testy failują na `bert-score` / `torch` | Zainstaluj: `pip install bert-score torch --index-url https://download.pytorch.org/whl/cpu` |

---

## Struktura danych (co powstaje po ingestion)

```
src/data/
├── scraped_raw/          # Surowe teksty ze stron (po scraper)
├── facts/                # Fakty wyekstrahowane przez LLM, JSON (po extract_facts)
└── qdrant_db/            # Lokalna baza wektorowa Qdrant (po ingest_facts)
```

Wszystkie te katalogi są w `.gitignore` — nie wchodzą do repozytorium.
