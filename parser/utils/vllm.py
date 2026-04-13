import json
import threading
import json_repair
import dotenv
import os

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

dotenv.load_dotenv()

LLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3.5-9B")
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", 0.6))

_SYSTEM_PROMPT = (
    "You are an annotator creating the ReasoningFlow dataset. "
    "Read the provided annotation guide carefully, and make sure to respond "
    "correctly and concisely based on these annotation guides."
)

_llm = LLM(model=LLM_MODEL_NAME, gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION)
_lock = threading.Lock()  # LLM.chat() is not thread-safe

in_token = 0
out_token = 0
_token_lock = threading.Lock()

def call_llm(prompt: str, schema=None, **args):
    global in_token, out_token

    structured_outputs = (
        StructuredOutputsParams(json=schema.model_json_schema())
        if schema is not None
        else None
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4096,
        structured_outputs=structured_outputs,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    with _lock:
        outputs = _llm.chat(messages=messages, sampling_params=sampling_params)

    output = outputs[0]
    response_text = output.outputs[0].text

    with _token_lock:
        in_token += len(output.prompt_token_ids)
        out_token += len(output.outputs[0].token_ids)

    response_text = response_text.split("```json")[-1].split("```")[0].strip()
    response_text = json_repair.repair_json(response_text)
    print(response_text)

    if schema is not None:
        return json.loads(response_text)
    else:
        return response_text


def get_metadata():
    return {
        "in_token": in_token,
        "out_token": out_token,
        "price": 0,  # local inference, no cost
    }
