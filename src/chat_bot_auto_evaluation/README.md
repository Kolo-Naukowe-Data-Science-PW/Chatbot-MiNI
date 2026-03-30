This folder contains scripts and modules for automatic evaluation of chatbot performance, focusing on NLP metrics and data preprocessing for the Chatbot-MiNI project.

---

```
chat_bot_auto_evaluation/
│
├── evaluate.py                   <-- main evaluation script (currently placeholder)
├── llm_judge/
│   ├── judge.py                  <-- LLM-as-a-judge module
│   └── testpro_runner.py         <-- batch runner for TestPro-style A/B evaluation
├── metrics.py                    <-- metrics class with BERTScore implementation
├── prepare_data.py               <-- data loading and preprocessing utilities
├── README.md                     <-- this documentation file
│
└── tests/                        <-- unit tests and test data
    ├── evaluation_test.py        <-- comprehensive tests for all functions
    ├── test.csv                  <-- sample CSV data for testing

```

---

## Module Overview

The `chat_bot_auto_evaluation` module provides automated evaluation capabilities for chatbot responses using state-of-the-art NLP metrics. It's designed to assess the quality of generated responses against ground-truth references, particularly optimized for Polish language evaluation.

## Dependencies

Required packages (add to `pyproject.toml`):
- `pandas` - data manipulation
- `numpy` - numerical operations
- `torch` - PyTorch for deep learning
- `bert-score` - BERTScore metric implementation
- `openai` - OpenRouter-compatible client for LLM judge calls
- `pydantic` - schema validation for judge responses

## Core Functions

### prepare_data.py

**`read_data_from_csv(file_path, column_names)`**
- Reads specified columns from CSV file
- Returns pandas DataFrame or Exception on error
- Validates file existence and CSV format

**`convert_data_frame_to_string(data_frame)`**
- Converts DataFrame to NumPy string array
- Requires exactly 2 columns (candidates, references)
- Returns np.ndarray or Exception

**`get_data(file_path, column_names)`**
- High-level function combining reading and conversion
- One-liner for loading evaluation data
- Returns ready-to-use numpy array

### metrics.py

**`Metrics` class**
- Encapsulates evaluation metrics
- Constructor takes candidates and references lists

**`bert_score()`**
- Calculates BERTScore using Polish BERT model
- Returns (Precision, Recall, F1) as torch tensors
- Uses semantic similarity instead of lexical matching

**`perplexity()`**
- Placeholder for future fluency metric
- Will measure text naturalness using language models

### evaluate.py

Currently a placeholder with import comment. Future main orchestration script for running complete evaluation pipelines.

### llm_judge/judge.py

**`judge_pair(query, answer_a, answer_b, ...)`**
- Calls an OpenRouter model as an impartial judge
- Scores A and B on usefulness, accuracy, and conciseness (1-5)
- Returns validated result with chosen better variant (`A` or `B`)

### llm_judge/testpro_runner.py

**`main()`**
- Reads evaluation queries from CSV
- Calls `/chat` twice per query (variant A and B, TestPro mode)
- Runs LLM judge on both answers and saves results to output CSV

## Usage Examples

### Basic Evaluation Workflow

```python
from chat_bot_auto_evaluation.prepare_data import get_data
from chat_bot_auto_evaluation.metrics import Metrics

# Load data
data = get_data("evaluation.csv", ["candidate", "reference"])
candidates = data[:, 0].tolist()
references = data[:, 1].tolist()

# Evaluate
evaluator = Metrics(candidates, references)
P, R, F1 = evaluator.bert_score()

print(f"Average F1: {F1.mean().item():.4f}")
```

## Testing

Run tests with from root:
```bash
python -m pytest
```

Tests cover data loading, conversion, metrics calculation, and error handling.
