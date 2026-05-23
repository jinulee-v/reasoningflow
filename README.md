# ReasoningFlow

Repository for annotating reasoningflow.

## Setup Python Dependencies

```bash
cd web
pip install -r requirements.txt
```

## Start web annotation tool

```bash
./start_server.sh
# localhost:5000 will point to the annotation tool.
# localhost:5001 will point to the annotation guide. You can also check the current main branch's guide at:
# jinulee-v.github.io/reasoningflow
```

## Perform automatic annotation

1. To use VertexAI, add the GCP configuration JSON file in `.env` file. To use other frameworks like Gemini studio, OpenAI, or DeepInfra, (1) open parser/llm_labeler.py and import appropriate package; (2) add the corresponding env key (GOOGLE_API_KEY, OPENAI_API_KEY, DEEPINFRA_API_KEY) to the `.env` file.

```
# .env
GOOGLE_APPLICATION_CREDENTIALS="gcp-config.json"
```


2. Run this code:

```bash
python parser/llm_labeler.py --model gemini-3-flash-preview --raw_data data/v1_raw_data --output_dir OUTPUT_DIR # e.g., data/v1_llm_gemini-3-flash-preview
```

3. After running everything, run this script for cleansing the data:

```bash
./cleanse.sh OUTPUT_DIR # from above script
```