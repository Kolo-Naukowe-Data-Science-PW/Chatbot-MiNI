# What's already implemented

Status on 2026-04-23. Everything listed here is in `new_pipeline` branch and works.

---

## Core RAG system

### Data pipeline (`src/ingestion/`)
- **Web scraping** via Firecrawl API — supports HTML, PDF, XLSX, DOCX
- **Facebook scraping** (`scraper.py`) — `scrape_facebook_page("wrsminipw")` uses `facebook-scraper` library; each post stored as `[Facebook WRS MiNI | YYYY-MM-DD]\n{text}`; graceful fallback if library missing or blocked; integrated for v4 *(added 2026-04-23)*
- **Four pipeline versions** (set via `PIPELINE_VERSION` env var):
  - v1 — 15 hand-picked URLs, file-as-chunk, no LLM
  - v2 — 15 hand-picked URLs, LLM-extracted atomic facts, 1 fact = 1 chunk
  - v3 — full MiNI website, all file types, fact-based chunking
  - v4 — all sources (MiNI + external), fact-based chunking
- **XLSX/DOCX text extraction** (`describe_files.py`)
- **LLM-based fact extraction** (`extract_facts.py`) — each scraped document → list of atomic facts
- **Curated URL list** (`links_extended.py`) for v3/v4

### Vector database (`src/ingestion/`)
- **Qdrant** with hybrid indexing: dense vectors + sparse BM25 vectors
- **Dense embeddings**: `BAAI/bge-m3` — multilingual, 1024-dim, state-of-art *(updated 2026-04-23)*
- **Sparse embeddings**: `Qdrant/bm25` — language-agnostic keyword matching
- Collection auto-creation and reset on re-ingestion

### Retrieval (`src/api/retrieval.py`)
- **Hybrid search**: dense (cosine similarity) + sparse (BM25), fused with **Reciprocal Rank Fusion** (RRF) in Qdrant
- Returns top-K chunks with source URL metadata

### Query processing (`src/api/`)
- **Query rewriting** (`query_rewriter.py`) — LLM rewrites the user's vague question into a keyword-rich retrieval query before hitting Qdrant; falls back to original on failure *(added 2026-04-23)*
- **Multi-language support** (`translator.py`) — user can ask in PL/EN/UA; query is translated to PL before retrieval, answer translated back
- **Prompt builder** (`prompt_builder.py`) — system message with static FAQ + retrieved context + conversation history; **per-role tone hints** injected when `user_type` is set *(updated 2026-04-23)*

### API (`src/api/api.py`)
- **FastAPI** `/chat` endpoint — full RAG pipeline, returns `answer`, `sources`, `retrieval_query`
- **`/chat/stream`** SSE endpoint — same pipeline with `StreamingResponse`; tokens streamed as `data: {token}\n\n`; final `data: {"event":"done","sources":[...]}\n\n` *(added 2026-04-23)*
- **`/feedback`** endpoint — persists ratings and model config to `model_feedback.csv`
- CORS configured for local development
- Thread-safe CSV feedback writer

### LLM
- **OpenRouter** as provider — supports model switching per request via `modelConfig`
- Default pool: GPT-4o-mini, Gemini 2.5 Flash, Llama 3.1 8B, DeepSeek, Mistral, Phi-4, Qwen
- Configurable: temperature, top_p, frequency_penalty, presence_penalty, max_tokens
- **`query_llm_stream()`** generator (`src/api/main.py`) — same params as `query_llm`, yields text delta chunks via `stream=True`; error yields Polish error message *(added 2026-04-23)*

---

## Evaluation (`src/evaluation/`)

### Retrieval benchmarking (`benchmark.py`)
- Full eval loop over gold question set (`data/questions_filtered.csv`, 322 questions)
- **Hierarchical URL relevance scoring** — exact match = 1.0, parent URL = 0.5, grandparent = 0.25, child = 0.25, etc.
- **Metrics at rank cut-off k** (k = 3, 5, 7, 10):
  - Hit@k, MRR@k, Recall@k, Precision@k, F1@k, nDCG@k, MAP@k, R-Precision
- **Precision-Recall curve plots** (mean interpolated + raw vs interpolated)
- **Grouped bar chart** of all metrics across k values

### LLM-as-a-judge (`llm_judge/`)
- **`judge.py`** — rates two chatbot answers on 1-5 scales: usefulness, accuracy, conciseness; picks better variant; returns structured `JudgeResult`
- **`testpro_runner.py`** — batch runner: reads eval CSV, calls `/chat` twice per question with random model/sampling configs (A/B), runs judge, writes full results CSV

### Text metrics (`metrics.py`)
- **BERTScore** wrapper for Polish (`lang="pl"`) — F1, Precision, Recall between candidate and reference answers

### MRRw metric *(added 2026-04-23)*
- `_url_depth_difference()`, `_mrr_weight()`, `mrr_weighted_single()`, `mrr_weighted()` in `benchmark.py`
- Parameters α=0.8, β=0.4 — deeper (more specific) links penalised less than shallower ones
- Included in benchmark output table and bar chart

### Golden answers generator (`generate_golden_answers.py`) *(added 2026-04-23)*
- Calls GPT-4o via OpenRouter for each eval question, without RAG context
- `--resume` flag (append mode, skips already-answered), `--limit N`, `--delay`
- Saves `data/golden_answers.csv` (query, gold_url, golden_answer)

### Weighted user feedback (`weighted_feedback.py`) *(added 2026-04-23)*
- Reads `model_feedback.csv`, computes weighted means per user role
- Weights: admin=10, phd=5, master=3, student_senior=2, student_junior=1
- Breakdown: overall / by_user_type / by_language / by_model
- CLI: `--feedback-csv`, `--output-csv`

### Evaluation data (`data/`)
- `questions_filtered.csv` — 322 eval questions with gold URLs (used by benchmark)
- `questions_with_links.csv` — full set used by testpro_runner
- `questions_cat.csv` — questions with topic category labels
- `questions.csv` — raw student survey responses

---

## Frontend (`src/frontend/`)

### React/Vite
- Three testing modes: **Production** (thumbs up/down), **Test** (A/B comparison), **Test Pro** (5-star scales on multiple criteria)
- Language selection (PL/EN/UA) at conversation start
- **Role selection screen** — user picks one of 5 roles before chatting; PhD/admin skip field/semester selection *(added 2026-04-23)*
- `user_type` sent with every `/chat` and `/feedback` request *(added 2026-04-23)*
- **Streaming in production mode** — `fetch` + `ReadableStream` against `/api/chat/stream`; tokens appended in-place to a placeholder bubble; typing-dots animation while waiting for first token; `canRate` set to `true` on `done` event *(added 2026-04-23)*
- Test / Test Pro modes still use regular `fetch` (A/B parallel calls)
- Model config passed with each request (supports A/B testing)
- Feedback submission to `/feedback`

---

## Infrastructure

- **Docker Compose** — 5 services: scraper, ingest, api, frontend, test health checks
- **GitHub Actions** — manual deploy (`deploy.yml`) and manual ingest (`ingest.yml`) to self-hosted faculty VM runner
- **Pre-commit hooks** — Black, Ruff, isort

---

## Documentation

- **`README.md`** — full project description, Mermaid architecture diagram, repo structure, local setup instructions, pipeline version table, conventions *(updated 2026-04-23)*
- **`src/evaluation/README.md`** — evaluation scripts guide, metrics explanation, run instructions *(added 2026-04-23)*
- **`docs/PLAN.md`** — implementation plan for remaining tasks *(added 2026-04-23)*
