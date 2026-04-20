import json
import time
import threading
from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import HttpOptions
import dotenv
import os
import json_repair

dotenv.load_dotenv()

client = genai.Client(
    vertexai=True,
    project=os.environ.get("PROJECT_ID"),
    location=os.environ.get("REGION"),
    # http_options=HttpOptions(
    #     api_version="v1",
    #     headers={
    #         "X-Vertex-AI-LLM-Shared-Request-Type": "flex"
    #     },
    #     # timeout = 600000  # Timeout in milliseconds
    # )
)
LLM_MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-3.1-pro-preview")

in_token = 0
out_token = 0
price = 0
_token_lock = threading.Lock()

prompt_token_rate = {
    "gemini-3-pro-preview": 2.0/1000000,
    "gemini-3.1-pro-preview": 2.0/1000000,
    "gemini-3-flash-preview": 0.50/1000000,
}
candidates_token_rate = {
    "gemini-3-pro-preview": 12.0/1000000,
    "gemini-3.1-pro-preview": 12.0/1000000,
    "gemini-3-flash-preview": 3.00/1000000,
}

_RETRY_DELAYS = [5, 20, 80, 320]

def call_llm(prompt: str, schema=None, llm_model_name=None, **args):
    thinking_level = args.get("thinking_level", "minimal")
    llm_model_name = llm_model_name or LLM_MODEL_NAME
    if "pro" in llm_model_name and thinking_level == "minimal":
        # Gemini-3-Pro does not support minimal, so we use Gemini-3-flash
        llm_model_name = LLM_MODEL_NAME.replace("3-pro", "3-flash").replace("3.1-pro", "3-flash")

    config = {
        "response_mime_type": "application/json",
        "response_json_schema": schema.model_json_schema(),
        "temperature": 0.0,
        "thinking_config": {
            "include_thoughts": False,
            "thinking_level": thinking_level
        }
    } if schema is not None else {"temperature": 0.0}

    for attempt, delay in enumerate([None] + _RETRY_DELAYS):
        if delay is not None:
            print(f"Error. Retrying in {delay}s (attempt {attempt}/{len(_RETRY_DELAYS)})...")
            time.sleep(delay)
        try:
            response = client.models.generate_content(
                model=llm_model_name,
                contents=prompt,
                config=config,
            )
            break
        except ClientError as e:
            if e.code in [429] and attempt < len(_RETRY_DELAYS):
                continue
            raise
        except ServerError as e:
            if e.code in [503] and attempt < len(_RETRY_DELAYS):
                continue
            raise
    with _token_lock:
        global in_token, out_token, price
        in_token += response.usage_metadata.prompt_token_count
        out_token += response.usage_metadata.candidates_token_count
        price += response.usage_metadata.prompt_token_count * prompt_token_rate.get(LLM_MODEL_NAME, 0) + response.usage_metadata.candidates_token_count * candidates_token_rate.get(LLM_MODEL_NAME, 0)

    if response.text is None:
        raise ValueError("Response text is None")

    response_text = response.text.split("```json")[-1].split("```")[0]
    response_text = response_text.strip()
    response_text = json_repair.repair_json(response_text)
    # print(response_text)

    if schema is not None:
        return json.loads(response_text)
    else:
        return response_text

def get_metadata():
    return {
        "in_token": in_token,
        "out_token": out_token,
        "price": price
    }
