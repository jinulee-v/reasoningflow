import os
import json
import random
import time
from typing import List, Dict, Any
import requests
from datasets import load_dataset, get_dataset_config_names
import multiprocessing

# export DEEPINFRA_API_KEY=your_api_key_here

class VLLMClient:
    """Client for interacting with VLLM"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000/v1/chat/completions"  # Assuming VLLM is running locally
        self.headers = {
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
                timeout=300
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
    (i, sample), mmlu_processor, model = args
    prompt = mmlu_processor.format_question(sample)
    response = mmlu_processor.client.generate_response(model, prompt)
    # Add delay to avoid rate limiting
    time.sleep(1)
    return {
        "question_index": i,
        "subject": sample["subject"],
        "question": sample["question"],
        "correct_answer": sample["choices"][sample["answer"]],
        "response": response
    }


class MMLUProcessor:
    """Process MMLU dataset and generate responses"""
    
    def __init__(self):
        self.client = VLLMClient()
        self.models = [
            # "deepseek-ai/DeepSeek-R1",
            # "deepseek-ai/DeepSeek-V3",
            # "Qwen/QwQ-32B",
            "Qwen/Qwen2.5-32B"
        ]
    
    def load_mmlu_samples(self) -> List[Dict[str, Any]]:
        """Load random samples from MMLU validation set"""
        try:
            # Load the MMLU dataset
            samples = []
            for subject in get_dataset_config_names("tasksource/mmlu"):
                dataset = load_dataset("tasksource/mmlu", subject, split="dev")
                
                for sample in dataset:
                    samples.append({
                        "subject": subject,
                        "question": sample["question"],
                        "choices": sample["choices"],
                        "answer": sample["answer"]
                    })
            
            return samples
            
        except Exception as e:
            print(f"Error loading MMLU dataset: {e}")
            return []
    
    def format_question(self, sample: Dict[str, Any]) -> str:
        """Format MMLU question for the models"""
        question = sample["question"]
        
        prompt = question
        
        return prompt
    
    def generate_responses(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate responses from all models for given samples"""
        results = {
            "samples": samples,
            "responses": {}
        }
        
        for model in self.models:
            print(f"\n--- Processing with {model} ---")
            model_responses = []

            with multiprocessing.Pool(processes=min(16, multiprocessing.cpu_count())) as pool:
                model_responses = pool.map(process_sample, list(zip(enumerate(samples), [self]*len(samples), [model]*len(samples))))

            results["responses"][model] = model_responses
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = "mmlu_responses_qwen2.5_32b.json"):
        """Save results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the results"""
        print("\n" + "="*80)
        print("MMLU RESPONSE GENERATION SUMMARY")
        print("="*80)
        
        for i, sample in enumerate(results["samples"], 1):
            print(f"\n--- Question {i} ---")
            print(f"Subject: {sample['subject']}")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Correct Answer: {sample['choices'][sample['answer']]}")
            
            for model in self.models:
                response = results["responses"][model][i-1]["response"]
                # Extract just the first line or first 100 chars for summary
                summary = response.split('\n')[0][:100] + "..." if len(response) > 100 else response
                print(f"{model}: {summary}")

def main():
    """Main function to run the MMLU response generation"""
    
    # Initialize processor
    processor = MMLUProcessor()
    
    print("Loading MMLU validation samples...")
    samples = processor.load_mmlu_samples()
    
    if not samples:
        print("Failed to load MMLU samples")
        return
    
    # Debug
    # samples = samples[:5]
    print(f"Loaded {len(samples)} samples from MMLU validation set")
    
    # Generate responses
    print("\nGenerating responses from all models...")
    results = processor.generate_responses(samples)
    
    # Save results
    processor.save_results(results)
    
    # Print summary
    processor.print_summary(results)
    
    print(f"\nComplete! Generated responses from {len(processor.models)} models for {len(samples)} questions.")

if __name__ == "__main__":
    main()