# Run this script after parsing with LLMs.
# Usage: bash cleanse.sh [OUTPUT_DIR]
#   OUTPUT_DIR defaults to data/v1_llm_gemini-3.1-pro-preview

OUTPUT_DIR="${1:-data/v1_llm_gemini-3.1-pro-preview}"

python parser/cleanse_scripts/cleanse_weird_segmentation.py --data-dir "$OUTPUT_DIR"

# Check non-existent labels (LLMs disobeyed constrained decoding)
python parser/cleanse_scripts/cleanse_nonexisting_labels.py --data-dir "$OUTPUT_DIR"

# Modify segmentation around <think> and </think> tags
python parser/cleanse_scripts/cleanse_think_metatags.py --data-dir "$OUTPUT_DIR"

# Modify common errors
# 1. self-verification
python parser/cleanse_scripts/cleanse_common_errors.py --pattern 1 --data-dir "$OUTPUT_DIR"
# 2. decomposition
python parser/cleanse_scripts/cleanse_common_errors.py --pattern 2 --data-dir "$OUTPUT_DIR"

# Reorder nodes and edges
python parser/cleanse_scripts/cleanse_reorder_nodes.py --data-dir "$OUTPUT_DIR"

# Extract final answer
python parser/cleanse_scripts/update_final_answer.py --data-dir "$OUTPUT_DIR"