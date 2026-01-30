import json
from google import genai
import dotenv
import os
import json_repair
dotenv.load_dotenv()
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

in_token = 0
out_token = 0
price = 0
def call_llm(prompt: str, schema=None): # Gemini API
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema.model_json_schema(),
            "temperature": 0,
        } if schema is not None else {"temperature": 0}
    )
    global in_token, out_token, price
    in_token += response.usage_metadata.prompt_token_count
    out_token += response.usage_metadata.candidates_token_count
    price += response.usage_metadata.prompt_token_count * 0.3/1000000 + response.usage_metadata.candidates_token_count * 2.5/1000000
    
    # print(response.text); exit()
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