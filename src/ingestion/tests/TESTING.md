# Testing

Tests are run with `pytest` and measured with `pytest-cov`.

## Running tests locally

```bash
# Install dev dependencies (once)
pip install -e .[dev]

# Run all tests with coverage report
pytest
```

This runs `pytest` according to the config in `pyproject.toml`:
- discovers all `*_test.py` files under `src/`
- uses `--import-mode=importlib` so modules are resolved relative to `src/` (no `src.` prefix needed in imports)
- prints a coverage summary to terminal

## Coverage report columns

| Column | Meaning |
|--------|---------|
| **Name** | File path of the module being measured |
| **Stmts** | Total executable lines (blank lines, comments, docstrings excluded) |
| **Miss** | Lines never executed during the test run |
| **Cover** | `(Stmts - Miss) / Stmts` — percentage of code exercised |
| **Missing** | Exact line numbers not hit by any test |

## Test structure

```
src/
├── ingestion/tests/
│   ├── scraper_test.py         # clean_headnote, clean_footnote, scrap_data, main()
│   ├── common_test.py          # placeholder
│   ├── describe_files_test.py  # placeholder
│   ├── extract_facts_test.py   # placeholder
│   └── ingest_facts_test.py    # placeholder
├── api/tests/
│   ├── api_test.py             # placeholder
│   └── main_test.py            # placeholder
└── evaluation/tests/
    └── evaluation_test.py      # read_data_from_csv, Metrics (BERTScore)
```

## CI

Tests do **not** run automatically. Trigger them manually: GitHub Actions → **Run Tests** → `workflow_dispatch`.
