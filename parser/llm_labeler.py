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

from utils.gemini import call_llm, get_metadata


# =============================================================================
# Pydantic Models and Enums
# =============================================================================

with open("schema/node_labels.yaml", 'r', encoding='utf-8') as f:
    NODE_LABELS = yaml.safe_load(f)
    NODE_LABELS.pop("context", None)  # Remove 'context' label if present
with open("schema/edge_labels.yaml", 'r', encoding='utf-8') as f:
    EDGE_LABELS = yaml.safe_load(f)

# 1. Node segmentation
class SentenceList(BaseModel):
    units: List[str]
    
# 2. Node classification
NodeLabel = Enum(
    "NodeLabel",
    {label['name']: label['name'] for label in NODE_LABELS['nodes']}
)
class NodeLabelResponse(BaseModel):
    label: NodeLabel

# 3. Edge detection + classification
EdgeLabel = Enum(
    "EdgeLabel",
    {label['name']: label['name'] for label in EDGE_LABELS['edges']}
)
class EdgeResponse(BaseModel):
    from_node_id: str
    label: EdgeLabel
class EdgeResponseList(BaseModel):
    responses: List[EdgeResponse]

# =============================================================================
# Label Mappings and Definitions
# =============================================================================

NODE_LABEL_TO_EDGE_LABELS = defaultdict(list)
for edge_def in EDGE_LABELS['edges']:
    for target in edge_def.get('to', []):
        NODE_LABEL_TO_EDGE_LABELS[target].append(edge_def['name'])

EDGE_LABEL_DEFINITIONS = {}
for edge_def in EDGE_LABELS['edges']:
    definition = "### " + edge_def.get('name') + "\n"
    definition += edge_def.get('description', '') + "\n"
    for subtype in edge_def.get('subtypes', []):
        definition += f" - **{subtype.get('type', '')}**\n"
        if subtype['description']:
            definition += f"    ({subtype['description']})\n"
        
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
    return 1000000 * ("resp" in node_id) + int(re.search(r'\d+', node_id).group())


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
    # if example:
    #     return (
    #         f"Raw text:\n{json.dumps(datum['raw_text'], ensure_ascii=False)}\n"
    #         f'Label:\n{{"label": {datum["label"]}}}\n'
    #     )
    # else:
    previous_steps = '\n'.join(datum['previous_steps'])
    return (
        f"Previous steps:\n{previous_steps}\n"
        f"Text:\n{json.dumps(datum['raw_text'], ensure_ascii=False)}\n"
        f"Label:\n"
    )


def format_edge_prompt(datum, example=False):
    """Format prompt for edge annotation."""
    if example:
        return (
            f"Previous nodes:\n{json.dumps(datum['prev_steps'], indent=2, ensure_ascii=False)}\n"
            f"Current node:\n{json.dumps(datum['current_step'], indent=2, ensure_ascii=False)}\n"
            f"Output:\n{json.dumps(datum['edges'], indent=2, ensure_ascii=False)}\n"
        )
    else:
        return (
            f"Previous nodes:\n{json.dumps(datum['prev_steps'], indent=2, ensure_ascii=False)}\n"
            f"Current node:\n{json.dumps(datum['current_node'], indent=2, ensure_ascii=False)}\n"
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
        "edge": "parser/prompts/3_edge_prompt.txt",
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

# Labels that require full context (all previous nodes) for edge detection
FULL_CONTEXT_LABELS = {"planning", "assumption", "conclusion", "restatement"}

# Maximum number of previous nodes to consider for edge detection
MAX_CONTEXT_WINDOW = 10


# =============================================================================
# Core Annotation Functions
# =============================================================================

def _build_edge_definitions(node_label):
    """Build edge definition text for a given node label."""
    return "## Possible edges\n" + "\n".join(
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


def node_classification(sent, prev_steps):
    """Label a single sentence using LLM."""
    input_data = {"raw_text": sent, "previous_steps": prev_steps}
    prompt = _build_prompt_with_examples(
        PROMPTS["node_classification"],
        [], # No examples for node classification
        format_node_label_prompt,
        input_data
    )
    label = call_llm(prompt, schema=NodeLabelResponse)["label"]
    return {"text": sent, "label": label.lower()}


def edge_detection_and_classification(node_idx, nodes):
    """Annotate edges for a single node using LLM."""
    node = nodes[node_idx]
    if node["source"] != "response":
        return []

    node_label = node["label"]
    if node_label not in NODE_LABEL_TO_EDGE_LABELS:
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
        EXAMPLES["edge"].get(node_label, []),
        format_edge_prompt,
        input_data
    )

    response = call_llm(prompt, schema=EdgeResponseList)

    return [
        {
            "id": f"e{i}",
            "from_node_id": edge["from_node_id"],
            "to_node_id": node["id"],
            "label": edge["label"],
        }
        for i, edge in enumerate(response["responses"])
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
        "source": "response",
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


def _keep_only_last_conclusion(labels):
    """Ensure only the last 'conclusion' label remains; others become 'reasoning'.

    Args:
        labels: List of label dicts with 'text' and 'label' keys.

    Returns:
        Modified labels list (mutated in place).
    """
    # Find the last conclusion index
    last_conclusion_idx = None
    for i, item in enumerate(labels):
        if item["label"] == "conclusion":
            last_conclusion_idx = i

    # Default to last sentence if no conclusion found
    if last_conclusion_idx is None:
        last_conclusion_idx = len(labels) - 1

    # Convert all other conclusions to reasoning
    for i, item in enumerate(labels):
        if item["label"] == "conclusion" and i != last_conclusion_idx:
            item["label"] = "reasoning"

    return labels


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
        sentences, indices = tokenize_and_align(question_text, node_segmentation)
        if sentences is None:
            continue
        indices[0] = 0

        nodes = [
            _create_context_node(i, indices[i], indices[i + 1], question_text[indices[i]:indices[i + 1]])
            for i in range(len(sentences))
        ]

        # Process response sentences
        response_text = datum["raw_text"]["response"]
        response_sentences, response_indices = tokenize_and_align(response_text, node_segmentation)
        if response_sentences is None:
            continue
        response_indices[0] = 0

        # Build list of previous steps for each sentence (for context in classification)
        prev_steps_list = [response_sentences[:i] for i in range(len(response_sentences))]

        # Label sentences in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            labels = list(executor.map(node_classification, response_sentences, prev_steps_list))

        _keep_only_last_conclusion(labels)

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
                labels[i]["label"],
                response_text[response_indices[i]:response_indices[i + 1]]
            )
            for i in range(len(labels))
        ])
        datum["nodes"] = nodes

        # Annotate edges in parallel
        edge_func = partial(edge_detection_and_classification, nodes=nodes)
        non_context_indices = [i for i, node in enumerate(nodes) if node["label"] != "context"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            edge_lists = list(executor.map(edge_func, non_context_indices))

        # Flatten and sort edges
        edge_results = [edge for edges in edge_lists for edge in edges]
        edge_results.sort(
            key=lambda e: (get_node_sort_key(e["to_node_id"]), get_node_sort_key(e["from_node_id"]))
        )

        # Assign sequential IDs
        for i, edge in enumerate(edge_results):
            edge["id"] = f"e{i}"
        datum["edges"] = edge_results

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
    for file in sorted(os.listdir("data/v0_raw_data")):
        if file.endswith(".json"):
            output_path = os.path.join(
                "data",
                "v0_llm_Gemini-2.5-Flash",
                file
            )
            if os.path.exists(output_path):
                print(f"Data for {file} already exists, skipping.")
                continue

            with open(os.path.join("data/v0_raw_data", file), "r") as f:
                datum = json.load(f)
                data.append(datum)

    print(f"Loaded {len(data)} data samples.")
    main_predict(data, output_dir="data/v0_llm_Gemini-2.5-Flash")