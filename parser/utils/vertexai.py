import json
from google import genai
import dotenv
import os
import json_repair

dotenv.load_dotenv()

client = genai.Client(
    vertexai=True,
    project=os.environ.get("PROJECT_ID"),
    location=os.environ.get("REGION"),
)
LLM_MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")

in_token = 0
out_token = 0
price = 0

prompt_token_rate = {
    "gemini-3-pro-preview": 2.0/1000000,
    "gemini-3-flash-preview": 0.50/1000000,
}
candidates_token_rate = {
    "gemini-3-pro-preview": 12.0/1000000,
    "gemini-3-flash-preview": 3.00/1000000,
}

def call_llm(prompt: str, schema=None, **args):
    thinking_level = args.get("thinking_level", "minimal")
    if "gemini-3-pro" in LLM_MODEL_NAME and thinking_level == "minimal":
        # Gemini-3-Pro does not support minimal, so we use Gemini-3-flash
        llm_model_name = LLM_MODEL_NAME.replace("gemini-3-pro", "gemini-3-flash")
    else:
        llm_model_name = LLM_MODEL_NAME
    response = client.models.generate_content(
        model=llm_model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema.model_json_schema(),
            "temperature": 0.0,
            "thinking_config": {
                "include_thoughts": False,
                "thinking_level": thinking_level
            }
        } if schema is not None else {"temperature": 0.0}
    )
    global in_token, out_token, price
    in_token += response.usage_metadata.prompt_token_count
    out_token += response.usage_metadata.candidates_token_count
    price += response.usage_metadata.prompt_token_count * prompt_token_rate.get(LLM_MODEL_NAME, 0) + response.usage_metadata.candidates_token_count * candidates_token_rate.get(LLM_MODEL_NAME, 0)

    response_text = response.text.split("```json")[-1].split("```")[0]
    response_text = response_text.strip()
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
        "price": price
    }
