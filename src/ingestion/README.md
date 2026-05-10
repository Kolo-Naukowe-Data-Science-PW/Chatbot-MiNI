# Ingestion Pipeline

Builds the Qdrant vector knowledge base from web pages and manually downloaded PDFs.

## Pipeline steps

```
scraper.py            →  src/data/scraped_raw/       (raw .txt files, one per URL)
ingest_manual_pdfs.py →  src/data/scraped_raw/       (PDFs from src/manual_pdfs/ → .txt)
describe_files.py     →  src/data/processed_text/    (XLSX/DOCX → .txt)
extract_facts.py      →  src/data/facts/             (LLM → atomic facts JSON)
ingest_facts.py       →  src/data/qdrant_db/         (embeddings → Qdrant)
```

Progress is tracked in `src/data/pipeline_progress.json` — all steps are resumable.
Delete that file + `src/data/qdrant_db/` to start completely fresh.

## Modules

| File | Purpose |
|---|---|
| `common.py` | Pipeline version config, LLM client (OpenRouter), shared logger |
| `scraper.py` | Firecrawl-based web scraper; saves each page immediately, skips already-scraped URLs |
| `describe_files.py` | Extracts plain text from XLSX and DOCX files |
| `extract_facts.py` | Sends text to LLM and extracts ALL atomic facts (no selection); saves as JSON |
| `ingest_facts.py` | Embeds facts (BAAI/bge-m3) and upserts into Qdrant in batches of 10 files |
| `ingest_manual_pdfs.py` | Converts PDFs from `src/manual_pdfs/` to `.txt` in `scraped_raw/` |
| `progress.py` | Tracks which URLs/files are scraped, extracted, and ingested across runs |
| `links_curated.py` | ~15 hand-picked URLs (used for pipeline versions 1–2) |
| `links_extended.py` | Full URL list for MiNI PW website (used for pipeline version 3+) |
| `embedder.py` | HuggingFace embedding wrapper (BAAI/bge-m3, dense vectors) |
| `vector_db.py` | Qdrant collection setup, hybrid indexing (dense + sparse BM25), read/write helpers |

## Pipeline versions

Set via `PIPELINE_VERSION` env var in `.env` or `docker-compose.yml`.

| Version | URLs | LLM fact extraction |
|---------|------|---------------------|
| 1 | `links_curated` (~15 URLs) | No — whole file as one chunk |
| 2 | `links_curated` (~15 URLs) | Yes |
| 3 | `links_extended` (full MiNI website) | Yes |
| 4 | `links_extended` + Facebook WRS MiNI | Yes |

## Manual PDFs (`src/manual_pdfs/`)

BIP PW pages are behind a cookiewall and cannot be scraped automatically.
PDFs must be downloaded manually and placed in `src/manual_pdfs/`.
The filename → source URL mapping is defined in `ingest_manual_pdfs.py` (`URL_MAP`).

See `DEPLOYMENT.md` for the full list of PDFs to download.

## Running locally

```bash
export PYTHONPATH=src

python -m ingestion.scraper
python -m ingestion.ingest_manual_pdfs
python -m ingestion.describe_files
python -m ingestion.extract_facts
python -m ingestion.ingest_facts
```
