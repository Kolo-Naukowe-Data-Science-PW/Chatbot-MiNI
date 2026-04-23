# Data Pipeline Documentation

## Overview

The `src/pipeline` module implements a comprehensive data ingestion pipeline for the Chatbot-MiNI application. It handles web scraping, document processing, fact extraction, and vector database ingestion to build a knowledge base for the RAG (Retrieval-Augmented Generation) system.

## Architecture

The pipeline consists of several sequential stages:

1. **Scraping** (`scraper.py`): Collects raw data from web sources
2. **File Processing** (`describe_files.py`): Extracts text from complex file formats (XLSX, DOCX)
3. **Fact Extraction** (`extract_facts.py`): Uses LLM to extract structured facts from text
4. **Ingestion** (`ingest_facts.py`): Generates embeddings and stores facts in vector database

## Pipeline Versions

The pipeline supports multiple versions with different capabilities, controlled by the `PIPELINE_VERSION` environment variable:

| Version | Complex Files | LLM Facts | Chunking Strategy | Scope |
|---------|---------------|-----------|-------------------|-------|
| 1 | ❌ | ❌ | file_as_chunk | Limited URLs (15) |
| 2 | ❌ | ✅ | fact_based | Limited URLs (15) |
| 3 | ✅ | ✅ | fact_based | Full MiNI website |
| 4 | ✅ | ✅ | fact_based | All sources + PDFs |

## Key Components

### common.py

Central configuration module providing:
- **Version Management**: Pipeline version control via `PIPELINE_VERSION` env var
- **LLM Configuration**: OpenRouter API client setup with `MODEL_WORKER` model
- **Pipeline Config**: Feature flags for each version

### scraper.py

Web scraping component using Firecrawl API:
- **Data Collection**: Scrapes MiNI PW website content
- **Content Cleaning**: Removes headers, footers, and navigation elements
- **URL Management**: Version-dependent URL lists from `links_extended.py`
- **Output**: Raw text files with URLs and content

**Key Functions:**
- `scrap_data()`: Main scraping orchestration
- `clean_headnote()` / `clean_footnote()`: Content sanitization
- `ScrapedPage`: Data structure for scraped content

### describe_files.py

Processes complex document formats:
- **XLSX Support**: Extracts tabular data from Excel files
- **DOCX Support**: Extracts text from Word documents
- **Metadata Generation**: Creates JSON metadata files with source URLs
- **Output**: Cleaned text files ready for fact extraction

### extract_facts.py

LLM-powered fact extraction:
- **Prompt Engineering**: Specialized prompts for fact extraction in Polish
- **Chunking**: Handles large documents by splitting into manageable chunks
- **JSON Output**: Structured facts with source attribution
- **Fallback**: Raw text passthrough when LLM is disabled

**System Prompt Features:**
- Ignores headers, footers, menus
- Extracts complete sentences as facts
- Focuses on who, what, where, when information
- Returns clean JSON arrays

### ingest_facts.py

Vector database ingestion:
- **Embedding Generation**: Uses FastEmbed for text vectorization
- **Qdrant Integration**: Stores facts with metadata in vector database
- **Batch Processing**: Handles large volumes of facts efficiently
- **Collection Reset**: Ensures clean database state

### links_extended.py

Comprehensive URL registry for MiNI PW website:
- **Organized Categories**: URLs grouped by faculty sections
- **Document Links**: Includes PDFs, regulations, study plans
- **News Articles**: Recent announcements and updates
- **Complete Coverage**: All major faculty information sources

## Data Flow

```
URLs (links_extended.py)
    ↓
Firecrawl Scraping (scraper.py)
    ↓
Raw Text Files (scraped_raw/)
    ↓
File Processing (describe_files.py)
    ↓
Cleaned Text (processed_text/)
    ↓
LLM Fact Extraction (extract_facts.py)
    ↓
Structured Facts JSON (facts/)
    ↓
Embedding Generation (ingest_facts.py)
    ↓
Vector Database (Qdrant)
```

## Configuration

### Environment Variables

- `PIPELINE_VERSION`: Controls pipeline capabilities (1-4)
- `MODEL_NAME`: LLM model for fact extraction (default: openai/gpt-4o-mini)
- `OPENROUTER_API_KEY`: Required for LLM API access
- `FIRECRAWL_API_KEY`: Required for web scraping
- `QDRANT_DIR`: Vector database path (optional)

### Directory Structure

```
src/data/
├── scraped_raw/      # Raw scraped content
├── processed_text/   # Text from complex files
├── complex_files/    # XLSX/DOCX originals
├── facts/           # Extracted facts JSON
└── qdrant_db/       # Vector database
```

## Usage

### Running Individual Stages

```bash
# Scrape web content
python -m src.pipeline.scraper

# Process complex files
python -m src.pipeline.describe_files

# Extract facts
python -m src.pipeline.extract_facts

# Ingest to vector database
python -m src.pipeline.ingest_facts
```

### Full Pipeline Execution

The pipeline stages should be run in sequence. Each stage depends on the output of the previous one.

## Dependencies

- **firecrawl-py**: Web scraping API client
- **openai**: LLM API client (OpenRouter)
- **pandas**: Excel file processing
- **python-docx**: Word document processing
- **qdrant-client**: Vector database client
- **fastembed**: Embedding generation

## Testing

Unit tests are located in `tests/`:
- `scraper_test.py`: Scraping functionality
- `common_test.py`: Configuration and utilities
- `describe_files_test.py`: File processing
- `extract_facts_test.py`: Fact extraction
- `ingest_facts_test.py`: Database ingestion

## Error Handling

- **API Failures**: Graceful degradation with logging
- **File Processing**: Skips corrupted files with warnings
- **Network Issues**: Retry logic for scraping operations
- **LLM Errors**: Fallback to raw text when extraction fails

## Performance Considerations

- **Token Limits**: Fact extraction chunks large documents
- **Rate Limiting**: Built-in delays between scraping requests
- **Batch Processing**: Efficient embedding generation for large datasets
- **Memory Management**: Streaming processing for large files

## Security

- **API Keys**: Environment variable configuration
- **Input Validation**: Sanitization of scraped content
- **File Access**: Controlled directory permissions
- **Logging**: Sensitive data exclusion from logs

## Maintenance

- **Version Control**: Increment `PIPELINE_VERSION` for new features
- **URL Updates**: Regularly review and update `links_extended.py`
- **Model Updates**: Monitor LLM performance and costs
- **Database Cleanup**: Periodic vector database optimization</content>
<parameter name="filePath">/home/tymon/coding/Chatbot-MiNI/src/pipeline/DOCUMENTATION.md
