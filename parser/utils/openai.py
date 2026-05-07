from openai import OpenAI
import dotenv
import os
dotenv.load_dotenv()
client = OpenAI()
_SYSTEM_PROMPT = "You are an annotator creating the ReasoningFlow dataset. Read the provided annotation guide carefully, and make sure to respond correctly and concisely based on these annotation guides. Reasoning"

in_token = 0
out_token = 0
price = 0

input_token_rate = {
    "gpt-5.1-2025-11-13": 1.25/1000000,
    "gpt-5-mini-2025-08-07": 0.25/1000000,
}
output_token_rate = {
    "gpt-5.1-2025-11-13": 10.00/1000000,
    "gpt-5-mini-2025-08-07": 2.00/1000000,
}

def call_llm(prompt: str, llm_model_name: str, schema=None, thinking_level: str = None): # OpenAI API
    global in_token, out_token, price
    if thinking_level == "minimal":
        llm_model_name = "gpt-5-mini-2025-08-07"
        thinking_level = "low"

    if schema is not None:
        response = client.responses.parse(
            model=llm_model_name,
            reasoning={"effort": thinking_level} if thinking_level is not None else None,
            input=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            text_format=schema,
        )
        
        in_token += response.usage.input_tokens
        out_token += response.usage.output_tokens
        price += response.usage.input_tokens * input_token_rate[llm_model_name] + response.usage.output_tokens * output_token_rate[llm_model_name]

        return response.output_parsed.model_dump(mode='json')
    else:
        response = client.responses.create(
            model=llm_model_name,
            reasoning={"effort": thinking_level} if thinking_level is not None else None,
            input=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )
        in_token += response.usage.input_tokens
        out_token += response.usage.output_tokens
        price += response.usage.input_tokens * input_token_rate[llm_model_name] + response.usage.output_tokens * output_token_rate[llm_model_name]

        # print(response.output_text)
        return response.output_text

def get_metadata():
    return {
        "in_token": in_token,
        "out_token": out_token,
        "price": price
    }