import os
import json
import re
import concurrent.futures
from functools import partial
from enum import Enum
from typing import List
from collections import defaultdict
import yaml

from pydantic import BaseModel

# from utils.gemini import LLM_MODEL_NAME, call_llm, get_metadata
from utils.openai import LLM_MODEL_NAME, call_llm, get_metadata
# from utils.vertexai import LLM_MODEL_NAME, call_llm, get_metadata


# =============================================================================
# Pydantic Models and Enums
# =============================================================================

with open("schema/node_labels.yaml", 'r', encoding='utf-8') as f:
    NODE_LABELS = yaml.safe_load(f)
    remove_labels = {"context", "conclusion"}
    NODE_LABELS["nodes"] = [n for n in NODE_LABELS["nodes"] if n["name"] not in remove_labels]
with open("schema/edge_labels.yaml", 'r', encoding='utf-8') as f:
    EDGE_LABELS = yaml.safe_load(f)

# 1. Node segmentation
class SentenceList(BaseModel):
    units: List[str]
    
# 2. Node classification
node_labels = {label['name']: label['name'] for label in NODE_LABELS['nodes']}
NodeLabel = Enum(
    "NodeLabel",
    node_labels
)
# class NodeLabelResponse(BaseModel):
#     label: NodeLabel
class NodeLabelResponse(BaseModel):
    node_id: str
    label: NodeLabel
class NodeLabelResponseList(BaseModel):
    responses: List[NodeLabelResponse]

# 3. Post-hoc update of conclusion nodes
class ConclusionNodeList(BaseModel):
    conclusion_node_ids: List[str]

# 4. Edge detection + classification
edge_labels = {label['name']: label['name'] for label in EDGE_LABELS['edges']}
edge_labels.update({"none": None})
EdgeLabel = Enum(
    "EdgeLabel",
    edge_labels
)
class EdgeResponse(BaseModel):
    source_node_id: str
    label: EdgeLabel
class EdgeResponseList(BaseModel):
    responses: List[EdgeResponse]

# =============================================================================
# Label Mappings and Definitions
# =============================================================================

NODE_LABEL_TO_EDGE_LABELS = defaultdict(list)
for edge_def in EDGE_LABELS['edges']:
    for target in edge_def.get('dest', []):
        NODE_LABEL_TO_EDGE_LABELS[target].append(edge_def['name'])

EDGE_LABEL_DEFINITIONS = {}
for edge_def in EDGE_LABELS['edges']:
    definition = "## " + edge_def.get('name') + "\n\n"
    definition += edge_def.get('description', '') + "\n\n"
    for subtype in edge_def.get('subtypes', []):
        definition += f"### {subtype.get('type', '')}**\n\n"
        if subtype['description']:
            definition += f"{subtype['description']}\n\n"
        definition += "Possible source labels: " + ", ".join(subtype.get('source', [])) + "\n\n"
        for example in subtype.get('examples', []):
            for node in example.get('nodes', []):
                if not (node['type'] and node['text']):
                    continue
                if node.get('edge_to') == False:
                    definition += f"**source:** [{node['type']}]" + node.get('text', '') + "\n"
                else:
                    definition += f"**destination:** [{node['type']}]" + node.get('text', '') + "\n\n"
    
    EDGE_LABEL_DEFINITIONS[edge_def['name']] = definition

# =============================================================================
# Utility Functions
# =============================================================================

def fuzzy_find(sentence, text, start_pos=0):
    """Find sentence in text with tolerance for whitespace/punctuation differences.

    Returns (start_idx, end_idx) tuple or None if not found.
    """
    tokens = re.findall(r'[a-zA-Z0-9]+', sentence)
    if not tokens:
        return None

    escaped_tokens = [re.escape(tok) for tok in tokens]
    pattern = r'[\s\S]{0,20}?'.join(escaped_tokens)

    match = re.search(pattern, text[start_pos:])
    if match:
        return start_pos + match.start(), start_pos + match.end()
    return None


def get_node_sort_key(node_id):
    """Get sort key for node ID (context nodes first, then response nodes)."""
    try:
        return 1000000 * ("resp" in node_id) + int(re.search(r'\d+', node_id).group())
    except:
        return 100000000 # In case of unexpected node_id format


def model_id_map(file):
    """Map filename to model ID."""
    if "DeepSeek-R1" in file:
        return "deepseek-ai/DeepSeek-R1"
    elif "DeepSeek-V3" in file:
        return "deepseek-ai/DeepSeek-V3"
    elif "Qwen2.5-32B-Instruct" in file:
        return "Qwen/Qwen2.5-32B-Instruct"
    elif "QwQ-32B" in file:
        return "Qwen/QwQ-32B"
    return None


# =============================================================================
# Prompt Formatting Functions
# =============================================================================

def format_node_label_prompt(datum, example=False):
    """Format prompt for node labeling."""
    nodes_str = [json.dumps({"node_id": nodes["id"], "text": nodes["text"]}, ensure_ascii=False) for nodes in datum["nodes"]]
    nodes_str_joined = '\n'.join(nodes_str)
    return (
        f"Nodes:\n{nodes_str_joined}\n\n"
        f"Label:\n"
    )

def format_conclusion_prompt(datum, example=False):
    steps = '\n'.join(
        f"- ID: {step['id']}, Label: {step['label']}\n"
        f"  {json.dumps(step['text'], ensure_ascii=False)}"
        for step in datum['nodes']
    )
    return (
        f"Steps:\n{steps}\n"
    )

def format_edge_prompt(datum, example=False):
    """Format prompt for edge annotation."""
    prev_step_strings = '\n'.join(
        f"- ID: {step['id']}, Label: {step['label']}\n"
        f"  {json.dumps(step['text'], ensure_ascii=False)}"
        for step in datum['prev_steps']
    )
    curr_step_str = (
        f"- Label: {datum['current_node']['label']}\n"
        f"  {json.dumps(datum['current_node']['text'], ensure_ascii=False)}"
    )
    return (
        f"Previous nodes:\n{prev_step_strings}\n"
        f"Current node:\n{curr_step_str}\n"
        f"Output:\n"
    )


# =============================================================================
# Prompt and Example Loading
# =============================================================================

def _load_prompts_and_examples():
    """Load all prompt templates and few-shot examples from files."""
    prompts = {}
    examples = {}

    prompt_files = {
        "node_segmentation": "parser/prompts/1_node_segmentation_prompt.txt",
        "node_classification": "parser/prompts/2_node_classification_prompt.txt",
        "update_conclusion": "parser/prompts/3_update_conclusion_prompt.txt",
        "edge": "parser/prompts/4_edge_prompt.txt",
    }
    example_files = {
        "edge": "parser/prompts/edge_fewshot_examples.json",
    }

    for key, path in prompt_files.items():
        with open(path, "r") as f:
            prompts[key] = f.read()

    for key, path in example_files.items():
        with open(path, "r") as f:
            examples[key] = json.load(f)

    return prompts, examples


PROMPTS, EXAMPLES = _load_prompts_and_examples()

# Maximum number of previous nodes to consider for edge detection
MAX_CONTEXT_WINDOW = 10


# =============================================================================
# Core Annotation Functions
# =============================================================================

def _build_edge_definitions(node_label):
    """Build edge definition text for a given node label."""
    return "\n\n".join(
        f"{EDGE_LABEL_DEFINITIONS[edge_label]}"
        for edge_label in NODE_LABEL_TO_EDGE_LABELS[node_label]
    )


def _build_prompt_with_examples(template, examples, format_func, input_data):
    """Build a prompt by replacing example placeholders and input."""
    prompt = template
    for i, example in enumerate(examples):
        prompt = prompt.replace(
            f"<<example{i+1}>>",
            format_func(example, example=True)
        )
    prompt = prompt.replace("<<input>>", format_func(input_data, example=False))
    return prompt


def _extract_node_summary(node):
    """Extract minimal node info for edge detection context."""
    return {"id": node["id"], "text": node["text"], "label": node["label"]}


def node_segmentation(text):
    """Call LLM to perform sentence tokenization."""
    prompt = PROMPTS["node_segmentation"].replace("<<text>>", text)
    response = call_llm(prompt, schema=SentenceList)
    return response["units"]


def node_classification(nodes, question):
    """Labels all sentences using LLM."""
    input_data = {"nodes": nodes, "previous_steps": question}
    prompt = _build_prompt_with_examples(
        PROMPTS["node_classification"],
        [], # No examples for node classification
        format_node_label_prompt,
        input_data
    )
    response = call_llm(prompt, schema=NodeLabelResponseList)
    return response


def edge_detection_and_classification(node_idx, nodes):
    """Annotate edges for a single node using LLM."""
    node = nodes[node_idx]
    if node["source"] != "response":
        return []

    node_label = node["label"]
    if node_label not in NODE_LABEL_TO_EDGE_LABELS:
        print("No edges for node label:", node_label)
        return []
    context_start = 0

    # Build input data for edge detection
    input_data = {
        "prev_steps": [_extract_node_summary(n) for n in nodes[context_start:node_idx]],
        "current_node": node,
    }

    # Build prompt
    edge_definitions = _build_edge_definitions(node_label)
    prompt = PROMPTS["edge"].replace("<<edge_definitions>>", edge_definitions)
    prompt = _build_prompt_with_examples(
        prompt,
        [],
        format_edge_prompt,
        input_data
    )

    response = call_llm(prompt, schema=EdgeResponseList)

    return [
        {
            "id": f"e{i}",
            "source_node_id": edge["source_node_id"],
            "dest_node_id": node["id"],
            "label": edge["label"],
        }
        for i, edge in enumerate(response["responses"]) if edge["label"]
    ]


def tokenize_and_align(text, tokenize_func):
    """Tokenize text and find character indices for each sentence.

    Returns (sentences, sentence_indices) where sentence_indices has length
    len(sentences) + 1, with the last element being the text length.
    Returns (None, None) if alignment fails.
    """
    sentences = tokenize_func(text)
    sentence_indices = []
    last_idx = 0

    for sent in sentences:
        try:
            idx = text.index(sent[:10], max(0, last_idx - 5))
            matched_end = idx + len(sent)
        except ValueError:
            result = fuzzy_find(sent, text, max(0, last_idx - 5))
            if result is None:
                print(f"Warning: Sentence '{sent}' not found in text.")
                return None, None
            idx, matched_end = result
        last_idx = matched_end
        sentence_indices.append(idx)

    sentence_indices.append(len(text))
    return sentences, sentence_indices


# =============================================================================
# Node Creation Helpers
# =============================================================================

def _create_context_node(index, start, end, text):
    """Create a context node from question text."""
    return {
        "id": f"ctx{index}",
        "annotation": False,
        "start": start,
        "end": end,
        "label": "context",
        "text": text,
        "source": "question",
    }


def _create_response_node(index, start, end, label, text):
    """Create a response node from labeled sentence."""
    return {
        "id": f"resp{index}",
        "annotation": True,
        "start": start,
        "end": end,
        "label": label,
        "text": text,
        "source": "response",
    }


# =============================================================================
# Main Processing Function
# =============================================================================

MAX_WORKERS = 4


def main_predict(data, output_dir=""):
    """Main prediction function to annotate nodes and edges."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for datum in data:
        print(f"Processing document: {datum['doc_id']}")

        # Process question sentences into context nodes
        question_text = datum["raw_text"]["question"]
        question_sentences, indices = tokenize_and_align(question_text, node_segmentation)
        if question_sentences is None:
            continue
        indices[0] = 0

        nodes = [
            _create_context_node(i, indices[i], indices[i + 1], question_text[indices[i]:indices[i + 1]])
            for i in range(len(question_sentences))
        ]

        # Process response sentences
        response_text = datum["raw_text"]["response"]
        response_sentences, response_indices = tokenize_and_align(response_text, node_segmentation)
        if response_sentences is None:
            continue
        response_indices[0] = 0

        # Validation
        assert response_indices[0] == 0, "First sentence index should be 0."
        assert len(response_indices) == len(response_sentences) + 1, \
            "Sentence indices do not match the number of sentences."

        # Create response nodes
        nodes.extend([
            _create_response_node(
                i,
                response_indices[i],
                response_indices[i + 1],
                None, # Placeholder for label
                response_text[response_indices[i]:response_indices[i + 1]]
            )
            for i in range(len(response_indices) - 1)
        ])
        
        # annotate node labels together
        node_to_labels_list = node_classification(nodes, question=question_text)
        node_to_labels = {item['node_id']: item['label'] for item in node_to_labels_list['responses']}
        for node in nodes:
            if node['id'] in node_to_labels and node['source'] == 'response':
                node['label'] = node_to_labels[node['id']]
        
        datum["nodes"] = nodes
        
        # Post-hoc update of conclusion nodes
        conclusion_prompt = PROMPTS["update_conclusion"].replace("<<input>>", format_conclusion_prompt(datum))
        conclusion_response = call_llm(conclusion_prompt, schema=ConclusionNodeList)
        conclusion_node_ids = set(conclusion_response["conclusion_node_ids"])
        for node in datum["nodes"]:
            if node["id"] in conclusion_node_ids:
                node["label"] = "conclusion"

        # Annotate edges in parallel
        edge_func = partial(edge_detection_and_classification, nodes=nodes)
        non_context_indices = [i for i, node in enumerate(nodes) if node["label"] != "context"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            edge_lists = list(executor.map(edge_func, non_context_indices))
        print("Reach here: edge count", sum(len(edges) for edges in edge_lists))

        # Flatten and sort edges
        edge_results = [edge for edges in edge_lists for edge in edges]
        edge_results.sort(
            key=lambda e: (get_node_sort_key(e["dest_node_id"]), get_node_sort_key(e["source_node_id"]))
        )

        # Assign sequential IDs
        for i, edge in enumerate(edge_results):
            edge["id"] = f"e{i}"
        datum["edges"] = edge_results
        
        # Update annotator info
        datum["metadata"]["annotator"] = f"{LLM_MODEL_NAME}"
        datum["metadata"]["is_human_annotated"] = False

        # Save output
        output_path = os.path.join(output_dir, f"{datum['doc_id']}.json")
        with open(output_path, "w") as f:
            json.dump(datum, f, indent=4)

    metadata = get_metadata()
    print(
        f"Total input tokens: {metadata['in_token']}, "
        f"output tokens: {metadata['out_token']}, "
        f"price: ${metadata['price']:.6f}"
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Labeler for data")
    args = parser.parse_args()

    print("<Load data...>")
    data = []
    # Make directory for output if not exists
    os.makedirs(f"data/v0_llm_{LLM_MODEL_NAME}", exist_ok=True)
    for file in sorted(os.listdir("data/v0_raw_data")):
        if file.endswith(".json"):
            output_path = os.path.join(
                "data",
                f"v0_llm_{LLM_MODEL_NAME}",
                file
            )
            if os.path.exists(output_path):
                print(f"Data for {file} already exists, skipping.")
                continue

            with open(os.path.join("data/v0_raw_data", file), "r") as f:
                datum = json.load(f)
                data.append(datum)

    print(f"Loaded {len(data)} data samples.")
    main_predict(data, output_dir=f"data/v0_llm_{LLM_MODEL_NAME}")
