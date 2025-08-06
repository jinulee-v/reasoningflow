import os
import json
import random
import time
from typing import List, Dict, Any
import requests
from datasets import load_dataset, get_dataset_config_names
import multiprocessing

# export DEEPINFRA_API_KEY=your_api_key_here

class DeepInfraClient:
    """Client for interacting with DeepInfra API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepinfra.com/v1/openai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, model: str, prompt: str, max_tokens: int = None) -> str:
        """Generate response from specified model"""
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=1000
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling {model}: {e}")
            return f"Error: Failed to get response from {model}"
        except (KeyError, IndexError) as e:
            print(f"Error parsing response from {model}: {e}")
            return f"Error: Invalid response format from {model}"

def process_sample(args):
    model, client, data = args
    response = client.generate_response(model, data["question"])
    # Add delay to avoid rate limiting
    time.sleep(1)
    return {
        "question_id": data["question_id"],
        "question": data["question"],
        "correct_answer": data["correct_answer"],
        "response": response
    }

def main():
    """Main function to run the MMLU response generation"""
    
    # Get API key from environment variable
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        print("Error: Please set DEEPINFRA_API_KEY environment variable")
        print("You can get an API key from: https://deepinfra.com/")
        return
    client = DeepInfraClient(api_key)
    
    models = [
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/QwQ-32B",
        # "Qwen/Qwen2.5-32B-Instruct"
    ]
    dataset_names = [
        "jinulee-v/aime2024",
        "jinulee-v/gpqa-diamond",
        "jinulee-v/argkp"
    ]
    
    for model in models:
        for dataset_name in dataset_names:
            print(f"\n--- {model} - {dataset_name} ---")
            dataset = []
            model_short = model.split("/")[-1]
            dataset_name_short = dataset_name.split("/")[-1]
            with open(f"v1_data_backup/{model_short}_{dataset_name_short}_responses.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line))

            # model_responses = []
            # with multiprocessing.Pool(processes=min(16, multiprocessing.cpu_count())) as pool:
            #     model_responses = pool.map(
            #         process_sample,
            #         [(model, client, data) for data in dataset]
            #     )
            model_responses = []
            for d in dataset:
                if "Error: Failed to get response from" in d["response"]:
                    new_d = process_sample((model, client, d))
                else:
                    new_d = d
                model_responses.append(new_d)

            model_short = model.split("/")[-1]
            dataset_name_short = dataset_name.split("/")[-1]
            with open(f"v1_data/{model_short}_{dataset_name_short}_responses.jsonl", 'w', encoding='utf-8') as f:
                for response in model_responses:
                    f.write(json.dumps(response, ensure_ascii=False) + "\n")
            print(f"Responses saved to v1_data/{model_short}_{dataset_name_short}_responses.jsonl")


if __name__ == "__main__":
    main()