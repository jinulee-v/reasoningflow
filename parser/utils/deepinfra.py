import json
import time
import threading
import json_repair
import dotenv
import os
from openai import OpenAI, RateLimitError, APIStatusError

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai",
)

_SYSTEM_PROMPT = (
    "You are an annotator creating the ReasoningFlow dataset. "
    "Read the provided annotation guide carefully, and make sure to respond "
    "correctly and concisely based on these annotation guides."
)

in_token = 0
out_token = 0
price = 0
_token_lock = threading.Lock()

input_token_rate = {
    "deepseek-ai/DeepSeek-V4-Flash": 0.14/1000000,
    "google/gemma-4-31B-it": 0.13/1000000,
    "Qwen/Qwen3.5-35B-A3B": 0.20/1000000,
}
output_token_rate = {
    "deepseek-ai/DeepSeek-V4-Flash": 0.28/1000000,
    "google/gemma-4-31B-it": 0.38/1000000,
    "Qwen/Qwen3.5-35B-A3B": 0.95/1000000,
}

_RETRY_DELAYS = [5, 20, 80, 320]


def call_llm(prompt: str, llm_model_name: str, schema=None, **args):
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    kwargs = {
        "model": llm_model_name,
        "messages": messages,
        "temperature": 0.0,
    }

    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }

    for attempt, delay in enumerate([None] + _RETRY_DELAYS):
        if delay is not None:
            print(f"Error. Retrying in {delay}s (attempt {attempt}/{len(_RETRY_DELAYS)})...")
            time.sleep(delay)
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except RateLimitError:
            if attempt < len(_RETRY_DELAYS):
                continue
            raise
        except APIStatusError as e:
            if e.status_code in [503] and attempt < len(_RETRY_DELAYS):
                continue
            raise

    with _token_lock:
        global in_token, out_token, price
        in_token += response.usage.prompt_tokens
        out_token += response.usage.completion_tokens
        price += (
            response.usage.prompt_tokens * input_token_rate.get(llm_model_name, 0)
            + response.usage.completion_tokens * output_token_rate.get(llm_model_name, 0)
        )

    response_text = response.choices[0].message.content
    if response_text is None:
        raise ValueError("Response text is None")

    response_text = response_text.split("```json")[-1].split("```")[0].strip()
    response_text = json_repair.repair_json(response_text)

    if schema is not None:
        return json.loads(response_text)
    else:
        return response_text


def get_metadata():
    return {
        "in_token": in_token,
        "out_token": out_token,
        "price": price,
    }
