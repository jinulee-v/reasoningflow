from openai import OpenAI
import dotenv
dotenv.load_dotenv()
client = OpenAI()
LLM_MODEL_NAME = "gpt-5-mini"
_SYSTEM_PROMPT = "You are an annotator creating the ReasoningFlow dataset. Read the provided annotation guide carefully, and make sure to respond correctly and concisely based on these annotation guides."

in_token = 0
out_token = 0
price = 0
def call_llm(prompt: str, schema=None): # OpenAI API
    global in_token, out_token, price

    if schema is not None:
        response = client.responses.parse(
            model=LLM_MODEL_NAME,
            input=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            text_format=schema,
        )
        
        print(response.output_text)
        in_token += response.usage.input_tokens
        out_token += response.usage.output_tokens
        price += response.usage.input_tokens * 0.25/1000000 + response.usage.output_tokens * 2.00/1000000

        return response.output_parsed.model_dump(mode='json')
    else:
        response = client.responses.create(
            model=LLM_MODEL_NAME,
            input=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )
        in_token += response.usage.input_tokens
        out_token += response.usage.output_tokens
        price += response.usage.input_tokens * 0.25/1000000 + response.usage.output_tokens * 2.00/1000000

        print(response.output_text)
        return response.output_text

def get_metadata():
    return {
        "in_token": in_token,
        "out_token": out_token,
        "price": price
    }