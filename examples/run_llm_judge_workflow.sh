#!/bin/bash
# Example: Complete workflow for LLM-as-Judge evaluation + statistical comparison

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "LLM-as-Judge Evaluation + Statistical Comparison Workflow"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Paths
MODEL_A_CSV="src/evaluation/data/stat_test/200/answers_google_gemini-3.5-flash_t0.2_p2_200.csv"
MODEL_B_CSV="src/evaluation/data/stat_test/200/answers_openai_gpt-oss-120b_free_t0.2_p2_200.csv"
OUTPUT_DIR="./llm_judge_results_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Model A: $MODEL_A_CSV"
echo "  Model B: $MODEL_B_CSV"
echo "  Output:  $OUTPUT_DIR"
echo ""

# Check if CSV files exist
if [ ! -f "$MODEL_A_CSV" ]; then
    echo "ERROR: Model A CSV not found: $MODEL_A_CSV"
    exit 1
fi

if [ ! -f "$MODEL_B_CSV" ]; then
    echo "ERROR: Model B CSV not found: $MODEL_B_CSV"
    exit 1
fi

# Check API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable not set"
    echo "Export it: export OPENROUTER_API_KEY='sk-...'"
    exit 1
fi

echo "✓ All prerequisites met"
echo ""

# Step 1: Run LLM-as-Judge Evaluation
echo "════════════════════════════════════════════════════════════════"
echo "STEP 1: Running LLM-as-Judge Evaluation"
echo "════════════════════════════════════════════════════════════════"
echo ""

python -m src.evaluation.llm_judge.batch_evaluator \
    --model-a "$MODEL_A_CSV" \
    --model-b "$MODEL_B_CSV" \
    --judge-model "anthropic/claude-opus-4.7" \
    --language "pl" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✓ LLM-as-Judge evaluation complete"
echo "  Output files in: $OUTPUT_DIR"
echo ""

# Step 2: Display summary
echo "════════════════════════════════════════════════════════════════"
echo "STEP 2: Summary from JSON Results"
echo "════════════════════════════════════════════════════════════════"
echo ""

python << 'EOF'
import json
import sys
from pathlib import Path

output_dir = "$OUTPUT_DIR"
json_file = Path(output_dir) / "llm_judge_results.json"

if not json_file.exists():
    print(f"ERROR: results file not found: {json_file}")
    sys.exit(1)

with open(json_file) as f:
    results = json.load(f)

print("📊 Evaluation Summary:")
print(f"  Total evaluated: {results['evaluation_metadata']['total_evaluated']}")
print(f"  Success rate: {results['evaluation_metadata']['success_rate']}")
print()

print("Model A Statistics:")
for metric in ["usefulness", "accuracy", "conciseness"]:
    stats = results['model_a_stats'][metric]
    print(f"  {metric}: μ={stats['mean']:.3f}±{stats['std']:.3f} (median={stats['median']})")
print()

print("Model B Statistics:")
for metric in ["usefulness", "accuracy", "conciseness"]:
    stats = results['model_b_stats'][metric]
    print(f"  {metric}: μ={stats['mean']:.3f}±{stats['std']:.3f} (median={stats['median']})")
print()

comp = results['pairwise_comparison']
print("Pairwise Comparison:")
print(f"  Model A wins: {comp['model_a_wins']} ({comp['model_a_win_rate']})")
print(f"  Model B wins: {comp['model_b_wins']} ({comp['model_b_win_rate']})")
EOF

echo ""

# Step 3: Run statistical comparison for each metric
echo "════════════════════════════════════════════════════════════════"
echo "STEP 3: Statistical Comparison (if statistical_comparison.py available)"
echo "════════════════════════════════════════════════════════════════"
echo ""

if command -v python -m src.evaluation.statistical_comparison &> /dev/null; then
    for metric in usefulness accuracy conciseness; do
        echo "Running statistical comparison for: $metric"
        comparison_dir="${OUTPUT_DIR}/comparison_${metric}"
        mkdir -p "$comparison_dir"
        
        python -m src.evaluation.statistical_comparison \
            --file1 "${OUTPUT_DIR}/llm_judge_metrics_model_a.csv" \
            --file2 "${OUTPUT_DIR}/llm_judge_metrics_model_b.csv" \
            --metric-col "$metric" \
            --output-dir "$comparison_dir" || echo "  Note: statistical_comparison.py not available or failed"
    done
else
    echo "Note: statistical_comparison.py not available in this environment"
    echo "You can run it manually later:"
    echo ""
    echo "  python -m src.evaluation.statistical_comparison \\"
    echo "    --file1 ${OUTPUT_DIR}/llm_judge_metrics_model_a.csv \\"
    echo "    --file2 ${OUTPUT_DIR}/llm_judge_metrics_model_b.csv \\"
    echo "    --metric-col usefulness \\"
    echo "    --output-dir ${OUTPUT_DIR}/comparison_usefulness"
    echo ""
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ WORKFLOW COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/ | grep -E "csv|json"
echo ""
echo "Next steps:"
echo "  1. Review llm_judge_results.json for detailed results"
echo "  2. Use CSV files with statistical_comparison.py"
echo "  3. Archive results: zip -r ${OUTPUT_DIR}.zip ${OUTPUT_DIR}/"
echo ""
