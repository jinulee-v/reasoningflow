# Run this script after parsing with LLMs.

python cleanse_weird_segmentation.py

# Check non-existent labels (LLMs disobeyed constrained decoding)
python parser/cleanse_scripts/cleanse_nonexisting_labels.py

# Modify segmentation around <think> and </think> tags
python parser/cleanse_scripts/clenase_think_metatags.py

# Modify common errors
# 1. self-verification
python parser/cleanse_scripts/cleanse_common_errors.py --pattern 1
# 2. decomposition
python parser/cleanse_scripts/cleanse_common_errors.py --pattern 2

# Reorder nodes and edges
python parser/cleanse_scripts/cleanse_reorder_nodes.py