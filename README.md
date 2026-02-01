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

1. Enter your Google AI Studio API key in `.env` file for Gemini.

```
# .env
GOOGLE_API_KEY=AIzaS...NQ
```


2. Run this code:

```bash
python parser/llm_labeler.py # Ongoing updates
```