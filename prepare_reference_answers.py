#!/usr/bin/env python3
"""
Przygotować referencyjne odpowiedzi z final_notebooklm_QA.jsonl
filtrowane do pytań które są w answers_google_gemini-3.5-flash_t0.2_p2.csv
"""

import json

import pandas as pd

# Wczytaj pytania z answers CSV
answers_df = pd.read_csv(
    "src/evaluation/data/answers_google_gemini-3.5-flash_t0.2_p2.csv"
)
generated_questions = set(answers_df["pytanie"].unique())
print(f"✓ Pytania w answers CSV: {len(generated_questions)}")

# Wczytaj JSONL z referencyjnymi odpowiedziami
reference_data = []
with open("src/evaluation/data/final_notebooklm_QA.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            reference_data.append(json.loads(line))

print(f"✓ Wszystkie pytania w JSONL: {len(reference_data)}")

# Filtruj do pytań które są w answers CSV
filtered_reference = [
    item for item in reference_data if item["Pytanie"] in generated_questions
]

print(f"✓ Filtrowane pytania (wspólne): {len(filtered_reference)}")
print(
    f"  Pytania z answers ale nie w JSONL: {len(generated_questions) - len(filtered_reference)}"
)

# Konwertuj do DataFrame z kolumnami: pytanie, odpowiedz
ref_df = pd.DataFrame(
    [
        {"pytanie": item["Pytanie"], "odpowiedz": item["Odpowiedź"]}
        for item in filtered_reference
    ]
)

# Zapisz do pliku CSV dla text_metrics
output_path = "src/evaluation/data/final_notebooklm_QA_reference.csv"
ref_df.to_csv(output_path, index=False, encoding="utf-8", sep="|")
print(f"\n✓ Zapisano: {output_path}")
print(f"  Wierszy: {len(ref_df)}")

print("\n--- Następnie uruchom text_metrics ---")
print("python -m evaluation.text_metrics \\")
print(
    "  --generated-csv src/evaluation/data/answers_google_gemini-3.5-flash_t0.2_p2.csv \\"
)
print(f"  --reference-csv {output_path} \\")
print("  --output-dir src/evaluation/results")
