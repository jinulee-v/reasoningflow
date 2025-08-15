import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

# from gemini import call_llm, get_metadata
from trapi import call_llm, get_metadata

from pydantic import BaseModel
from typing import List
from enum import Enum
from functools import partial
import concurrent.futures


class NodeLabel(Enum):
    planning = "planning"
    fact = "fact"
    reasoning = "reasoning"
    restatement = "restatement"
    assumption = "assumption"
    example = "example"
    reflection = "reflection"
    conclusion = "conclusion"

class NodeResponse(BaseModel, extra="forbid"):
    text: str
    label: NodeLabel
class NodeResponseList(BaseModel, extra="forbid"):
    responses: list[NodeResponse]

class EdgeLabel(Enum):
    REASON_PREMISE_CONCLUSION = "reason:premise-conclusion"
    REASON_PLAN_STEP = "reason:plan-step"
    REASON_CONCEPT_EXAMPLE = "reason:concept-example"
    REASON_FACT_DETAIL = "reason:fact-detail"
    REASON_STMT_RESTATEMENT = "reason:stmt-restatement"
    REASON_STMT_CORRECTION = "reason:stmt-correction"
    PLAN_FRONTIER_PLAN = "plan:frontier-plan"
    PLAN_FRONTIER_VERIFY = "plan:frontier-verify"
    PLAN_PLAN_SUBPLAN = "plan:plan-subplan"
    PLAN_PLAN_NEXTPLAN = "plan:plan-nextplan"
    PLAN_PLAN_ALTERNATIVE = "plan:plan-alternative"
    EVALUATE_SUPPORT = "evaluate:support"
    EVALUATE_REFUTE = "evaluate:refute"
    EVALUATE_UNCERTAINTY = "evaluate:uncertainty"

class EdgeResponse(BaseModel, extra="forbid"):
    from_node_id: str
    label: EdgeLabel
class EdgeResponseList(BaseModel, extra="forbid"):
    responses: list[EdgeResponse]

def format_node_prompt(datum, example=False):
    if example:
        return f"Raw text:\n{json.dumps(datum['raw_text'], ensure_ascii=False)}\n" \
               f"Output:\n{json.dumps(datum['nodes'], indent=2, ensure_ascii=False)}\n"
    else:
        return f"Raw text:\n{json.dumps(datum['raw_text'], ensure_ascii=False)}\n" \
               f"Output:\n"

def format_edge_prompt(datum, example=False):
    if example:
        return f"Previous nodes:\n{json.dumps(datum['prev_steps'], indent=2, ensure_ascii=False)}\n" \
               f"Current node:\n{json.dumps(datum['current_step'], indent=2, ensure_ascii=False)}\n" \
               f"Output:\n{json.dumps(datum['edges'], indent=2, ensure_ascii=False)}\n"
    else:
        return f"Previous nodes:\n{json.dumps(datum['prev_steps'], indent=2, ensure_ascii=False)}\n" \
               f"Current node:\n{json.dumps(datum['current_node'], indent=2, ensure_ascii=False)}\n" \
               f"Output:\n"


def annotate_edges(node_idx, nodes, EDGE_PROMPT, EDGE_EXAMPLES):
    # For parallelization
    node = nodes[node_idx]
    if node["source"] != "response":
        return []
    
    # Call the LLM to get edges
    response = call_llm(
        EDGE_PROMPT
            .replace("<<example1>>", format_edge_prompt(EDGE_EXAMPLES[node["label"]][0], example=True))
            .replace("<<example2>>", format_edge_prompt(EDGE_EXAMPLES[node["label"]][1], example=True))
            .replace("<<example3>>", format_edge_prompt(EDGE_EXAMPLES[node["label"]][2], example=True))
            .replace("<<input>>", format_edge_prompt({
                "prev_steps": [
                    {
                        "id": n["id"],
                        "text": n["text"],
                        "label": n["label"]
                    } for n in nodes[:node_idx]
                ],
                "current_node": node
            })),
        schema=EdgeResponseList
    )["responses"]
    
    return [{
        "id": f"e{i}",
        "from_node_id": edge["from_node_id"],
        "to_node_id": node["id"],
        "label": edge["label"]
    } for i, edge in enumerate(response)]

def model_id_map(file):
    if "DeepSeek-R1" in file:
        return "deepseek-ai/DeepSeek-R1"
    elif "DeepSeek-V3" in file:
        return "deepseek-ai/DeepSeek-V3"
    elif "Qwen2.5-32B-Instruct" in file:
        return "Qwen/Qwen2.5-32B-Instruct"
    elif "QwQ-32B" in file:
        return "Qwen/QwQ-32B"

def main_predict(file):
    with open("parser/node_prompt.txt", "r") as f:
        NODE_PROMPT = f.read()
    with open("parser/edge_prompt.txt", "r") as f:
        EDGE_PROMPT = f.read()
    with open("parser/node_fewshot_examples.json", "r") as f:
        NODE_EXAMPLES = json.load(f)
    with open("parser/edge_fewshot_examples.json", "r") as f:
        EDGE_EXAMPLES = json.load(f)

    # Load the data
    print("<Load data...>")
    data = []
    with open(file, "r") as f:
        raw_data = []
        for line in f:
            raw_data.append(json.loads(line))
    for i, datum in enumerate(raw_data):
        response = datum["response"]
        if "<think>" in response and "</think>" in response:
            # Deepseek-R1
            response = response.split("<think>")[-1].split("</think>")[0].strip()
        data.append({
            "doc_id": file.replace(".jsonl", f"_{i}.json").replace("v1_data_raw", "v1_data"),
            "metadata": {
                "model_id": model_id_map(file),
                "correct_answer": datum["correct_answer"],
            },
            "raw_text": {
                "question": datum["question"],
                "response": response
            },
            "nodes": [],
            "edges": []
        })
    print(f"Loaded {len(data)} data samples.")

    # test
    # data = data[:1]
    for datum in data:
        try:
            continue_flag = False
            # If data exists, skip
            if os.path.exists(f"{datum['doc_id']}"):
                print(f"Data for {datum['doc_id']} already exists, skipping.")
                continue

            print(f"Processing document: {datum['doc_id']}")
            # with open(f"{datum['doc_id']}", "w") as f:
            #     json.dump(datum, f, indent=4)
            response = datum["raw_text"]["response"]
            response = call_llm(
                NODE_PROMPT
                    .replace("<<example1>>", format_node_prompt(NODE_EXAMPLES[0], example=True))
                    .replace("<<example2>>", format_node_prompt(NODE_EXAMPLES[1], example=True))
                    .replace("<<example3>>", format_node_prompt(NODE_EXAMPLES[2], example=True))
                    .replace("<<input>>", format_node_prompt({"raw_text": response}, example=False)),
                schema=NodeResponseList
            )["responses"]

            # Reorganize nodes
            nodes = datum["nodes"]
            # 1. add "question" nodes
            last_idx = 0
            sentences = sent_tokenize(datum["raw_text"]["question"])
            sentence_indices = []
            for sent in sentences:
                try:
                    idx = datum["raw_text"]["question"].index(sent[:15], last_idx)
                except ValueError:
                    # Find idx with regex. Replace all non-alphanumeric characters with \s* wildcard.
                    try:
                        idx = datum["raw_text"]["question"].index(sent[:5], last_idx)
                    except ValueError:
                        print(f"Warning: Sentence '{sent}' not found in the question text.")
                        raise ValueError(f"Sentence '{sent}' not found in the question text.")
                last_idx = idx + len(sent)
                sentence_indices.append(idx)
            sentence_indices.append(len(datum["raw_text"]["question"]))
            
            assert len(sentence_indices) == len(sentences) + 1, "Sentence indices do not match the number of sentences."

            for i in range(len(sentences)):
                nodes.append({
                    "id": f"ctx{i}",
                    "annotation": False,
                    "start": sentence_indices[i],
                    "end": sentence_indices[i + 1],
                    "label": "context",
                    "text": datum["raw_text"]["question"][sentence_indices[i]:sentence_indices[i + 1]],
                    "source": "question"
                })

            # 2. add "response" nodes
            last_idx = 0
            sentence_indices = []
            for sent in response: # concatenating the "text" field should be the original response text
                try:
                    idx = datum["raw_text"]["response"].index(sent["text"][:10], last_idx)
                except ValueError:
                    # Find idx with regex. Replace all whitespace/tabs/newlines/... with \s* wildcard.
                    # pattern = re.escape(sent["text"])
                    # pattern = re.sub(r'[^a-zA-Z0-9]', r'.?', pattern, re.DOTALL)
                    # pattern = re.sub(r'[^a-zA-Z0-9\\]', r'\\s*', pattern, re.DOTALL)
                    # match = re.search(pattern, datum["raw_text"]["response"], re.DOTALL)
                    # if match and match.start() >= last_idx:
                    #     idx = match.start()
                    # else:
                    print(f"Warning: Sentence '{sent['text']}' not found in the response text.")
                    # raise ValueError(f"Sentence '{sent['text']}' not found in the response text.")
                    continue_flag = True; break
                last_idx = idx + len(sent["text"])
                sentence_indices.append(idx)
            if continue_flag:
                continue
            sentence_indices[0] = 0
            sentence_indices.append(len(datum["raw_text"]["response"]))

            final_conclusion = None
            for i in range(len(response)):
                if response[i]["label"] == "conclusion":
                    final_conclusion = i
            # if final_conclusion is None:                
            #     assert False, "No conclusion found in the response."
            for i in range(len(response)):
                if response[i]["label"] == "conclusion":
                    if i != final_conclusion:
                        response[i]["label"] = "reasoning"

            # assertion checks
            assert sentence_indices[0] == 0, "First sentence index should be 0."
            assert len(sentence_indices) == len(response) + 1, "Sentence indices do not match the number of sentences."
            # assert sum([int(x["label"] == "conclusion") for x in response]) == 1, f"There should be exactly one conclusion in the response but found {sum([int(x['label'] == 'conclusion') for x in response])}"

            # Append to the final nodes
            for i in range(len(response)):
                nodes.append({
                    "id": f"resp{i}",
                    "annotation": False,
                    "start": sentence_indices[i],
                    "end": sentence_indices[i + 1],
                    "label": response[i]["label"],
                    "text": datum["raw_text"]["response"][sentence_indices[i]:sentence_indices[i + 1]],
                    "source": "response"
                })

            # with open(f"{datum['doc_id']}", "w") as f:
            #     json.dump(datum, f, indent=4)

            # For each node that are from "source: response", annotate edges
            edges = datum["edges"]
            # Process nodes in parallel to generate edges
            edge_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor: # rate limit control
                # Create a partial function with fixed parameters
                process_func = partial(annotate_edges, 
                                    nodes=nodes, 
                                    EDGE_PROMPT=EDGE_PROMPT, 
                                    EDGE_EXAMPLES=EDGE_EXAMPLES)
                
                # Submit all node indices that need processing
                future_to_idx = {executor.submit(process_func, i): i for i in range(len(nodes))}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        node_edges = future.result()
                        for i, edge_data in enumerate(node_edges):
                            edge_results.append(edge_data)
                            # print(edge_data)
                    except Exception as exc:
                        print(f"Node {idx} generated an exception: {exc}")

            # Add all edges to the datum
            edges.extend(edge_results)
            # Sort edge by "to_node_id" and "from_node_id", by the number in the ID.
            def get_number(node_id):
                return 1000000 * ("resp" in node_id) + int(re.search(r'\d+', node_id).group())
            edges.sort(key=lambda x: (get_number(x["to_node_id"]), get_number(x["from_node_id"])))
            # Reset the IDs to be sequential
            for i, edge in enumerate(edges):
                edge["id"] = f"e{i}"

            with open(f"{datum['doc_id']}", "w") as f:
                json.dump(datum, f, indent=4)
        except Exception as e:
            # raise e
            print(f"Error processing document {datum['doc_id']}: {e}")
            continue
    metadata = get_metadata()
    print(f"Total input tokens: {metadata['in_token']}, output tokens: {metadata['out_token']}, price: ${metadata['price']:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Labeler for data")
    parser.add_argument("--file", type=str, required=True, help="Path to the input JSONL file")
    args = parser.parse_args()

    main_predict(args.file)
    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     response=SEGMENTATION_PROMPT,
    #     examples=data[:2],
    # )
    # print(response.text)
    # with open("scp116k_data/gemini_labeler.json", "w") as f:
    #     json.dump(response.text, f, indent=4)