import os
import json
import re
import concurrent.futures
from functools import partial
from enum import Enum
from typing import List
from collections import defaultdict
import yaml
from tqdm import tqdm
from random import shuffle

from pydantic import BaseModel

# from utils.gemini import LLM_MODEL_NAME, call_llm, get_metadata
# from utils.openai import LLM_MODEL_NAME, call_llm, get_metadata
from utils.vertexai import LLM_MODEL_NAME, call_llm, get_metadata
# from utils.vllm import LLM_MODEL_NAME, call_llm, get_metadata


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


def _dp_boundaries(text, sentences, n, k):
    """DP to find optimal split of text[0:n] into k parts, maximising
    total subsequence-match count between sentences[i] and each part.
    """
    # nxt[j][c] = smallest index >= j where text[index] == c, or n if absent
    nxt = [None] * (n + 1)
    nxt[n] = {}
    for j in range(n - 1, -1, -1):
        d = dict(nxt[j + 1])
        d[text[j]] = j
        nxt[j] = d

    def subseq_count(sent, j1, j2):
        pos, cnt = j1, 0
        for c in sent:
            p = nxt[pos].get(c, n)
            if p >= j2:
                break
            pos = p + 1
            cnt += 1
        return cnt

    NEG_INF = float('-inf')
    dp   = [[NEG_INF] * (n + 1) for _ in range(k + 1)]
    back = [[-1]      * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0

    for i in range(1, k + 1):
        lo = i              # must have used at least i characters
        hi = n - (k - i)   # leave at least 1 char per remaining sentence
        for j in range(lo, hi + 1):
            for pj in range(i - 1, j):
                if dp[i - 1][pj] == NEG_INF:
                    continue
                sc = dp[i - 1][pj] + subseq_count(sentences[i - 1], pj, j)
                if sc > dp[i][j]:
                    dp[i][j] = sc
                    back[i][j] = pj

    bounds = [0] * (k + 1)
    j = n
    for i in range(k, 0, -1):
        bounds[i] = j
        j = back[i][j]
    bounds[0] = 0
    return bounds



def optimal_alignment(text, sentences, beam_width=50, window_factor=4.0):
    """Segment *text* into exactly len(sentences) contiguous non-overlapping
    parts that best match the given (possibly imperfect) sentences.

    Strategy
    --------
    1. Greedy left-to-right pass: assign each sentence its first valid exact
       match (i.e. starting at or after the previous match's end).
    2. Turn confirmed matches into "anchor" boundaries.
    3. Between consecutive anchors, interpolate with a DP that maximises the
       total subsequence-character overlap between each sentence and its
       assigned text slice.

    Returns (segments, char_boundaries) where char_boundaries has length N+1,
    char_boundaries[0] == 0, and char_boundaries[N] == len(text).
    """
    N = len(sentences)
    if N == 0:
        return [], [0]

    # ------------------------------------------------------------------
    # Phase 1 – collect all exact match positions for each sentence
    # ------------------------------------------------------------------
    candidates = []
    for sent in sentences:
        positions = []
        start = 0
        while sent:
            idx = text.find(sent, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(sent)))
            start = idx + 1
        candidates.append(positions)

    # ------------------------------------------------------------------
    # Phase 2 – greedy left-to-right assignment
    # end_pos tracks the RAW end of the last match (not whitespace-extended)
    # so subsequent searches are not confused by extension.
    # ------------------------------------------------------------------
    matched = [None] * N   # (raw_start, raw_end) tuple or None
    end_pos = 0
    gaps_since_last_match = 0   # unmatched sentences after the last anchor
    for i in range(N):
        # Reserve at least 1 character per unmatched sentence that precedes i
        min_start = end_pos + gaps_since_last_match
        valid = [(s, e) for s, e in candidates[i] if s >= min_start]
        if valid:
            matched[i] = valid[0]
            end_pos = valid[0][1]   # raw end; extension happens in Phase 3
            gaps_since_last_match = 0
        else:
            gaps_since_last_match += 1

    # ------------------------------------------------------------------
    # Phase 3 – build anchor map  {boundary_index: char_position}
    #
    # Each matched sentence i contributes two candidate boundary positions:
    #   start_anchors[i]   = raw match start  (so sentence i has no leading ws)
    #   end_anchors[i+1]   = match end extended past trailing whitespace
    #                        (so sentence i owns its trailing ws)
    #
    # When both sources exist for the same boundary index, take the minimum.
    # This correctly handles the case where the next sentence itself begins
    # with whitespace that belongs to it (min = start of next sentence).
    # ------------------------------------------------------------------
    start_anchors = {}
    end_anchors   = {}
    for i in range(N):
        if matched[i] is not None:
            s, e = matched[i]
            e_ext = e
            while e_ext < len(text) and text[e_ext].isspace():
                e_ext += 1
            start_anchors[i]   = s
            end_anchors[i + 1] = e_ext

    anchors = {}
    for idx in set(start_anchors) | set(end_anchors):
        s = start_anchors.get(idx)
        e = end_anchors.get(idx)
        anchors[idx] = min(v for v in (s, e) if v is not None)

    # Hard constraints: first and last boundary
    anchors[0] = 0
    anchors[N] = len(text)

    # Enforce strict monotonicity (drop anchors that would invert the order)
    valid_anchors = {0: 0}
    prev_pos = 0
    for idx in range(1, N + 1):
        if idx in anchors and anchors[idx] >= prev_pos:
            valid_anchors[idx] = anchors[idx]
            prev_pos = anchors[idx]
    valid_anchors[N] = len(text)   # always override to enforce full coverage

    # ------------------------------------------------------------------
    # Phase 4 – fill boundaries; interpolate gaps
    # ------------------------------------------------------------------
    char_boundaries = [None] * (N + 1)
    sorted_keys = sorted(valid_anchors.keys())

    for ki in range(len(sorted_keys) - 1):
        li = sorted_keys[ki]
        ri = sorted_keys[ki + 1]
        lp = valid_anchors[li]
        rp = valid_anchors[ri]
        char_boundaries[li] = lp

        gap = ri - li
        if gap <= 1:
            continue

        # Split text[lp:rp] among sentences[li:ri]
        sub_text = text[lp:rp]
        sub_sents = sentences[li:ri]
        sub_n, sub_k = len(sub_text), len(sub_sents)
        sub_bounds = [0, sub_n] if sub_k == 1 else _dp_boundaries(sub_text, sub_sents, sub_n, sub_k)
        for j in range(1, gap):
            char_boundaries[li + j] = lp + sub_bounds[j]

    char_boundaries[N] = len(text)

    assert all(b is not None for b in char_boundaries), \
        "BUG: some char_boundaries entries were not filled."
    segments = [text[char_boundaries[i]:char_boundaries[i + 1]] for i in range(N)]
    return segments, char_boundaries


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
    elif "gpt-oss-120b" in file:
        return "openai/gpt-oss-120b"
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
        f"Options for source nodes:\n\n<source>\n{prev_step_strings}\n</source>\n\n"
        f"Destination node:\n\n<destination>\n{curr_step_str}\n</destination>\n\n"
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


# =============================================================================
# Core Annotation Functions
# =============================================================================

def _build_edge_definitions(node_label):
    """Build edge definition text for a given node label."""
    return "\n\n".join(
        f"{EDGE_LABEL_DEFINITIONS[edge_label]}"
        for edge_label in NODE_LABEL_TO_EDGE_LABELS[node_label]
    )


def _build_prompt_with_examples(template, examples, format_func, input_data, example_outputs=None):
    """Build a prompt by replacing example placeholders and input.

    When example_outputs is provided, builds a fewshot examples block with both
    formatted input and expected output, replacing <<fewshot_examples>> in the template.
    Otherwise, replaces indexed <<example{i}>> placeholders with formatted input only.
    """
    prompt = template

    if examples and example_outputs:
        # Build fewshot examples block with input/output pairs
        parts = []
        for i, example in enumerate(examples):
            input_str = format_func(example, example=True)
            output_str = example_outputs[i]
            parts.append(f"[[Example {i+1}]]\n\n{input_str}{output_str}\n")
        prompt = prompt.replace("<<fewshot_examples>>", "\n".join(parts) + "\n")
    elif examples:
        # Replace indexed example placeholders (<<example1>>, <<example2>>, ...)
        for i, example in enumerate(examples):
            prompt = prompt.replace(
                f"<<example{i+1}>>",
                format_func(example, example=True)
            )

    # Clean up unused fewshot placeholder
    prompt = prompt.replace("<<fewshot_examples>>", "")

    prompt = prompt.replace("<<input>>", format_func(input_data, example=False))
    return prompt


def _extract_node_summary(node):
    """Extract minimal node info for edge detection context."""
    return {"id": node["id"], "text": node["text"], "label": node["label"]}


def node_segmentation(text):
    """Call LLM to perform sentence tokenization."""
    prompt = PROMPTS["node_segmentation"].replace("<<text>>", text)
    response = call_llm(prompt, schema=SentenceList, thinking_level="minimal")
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
    response = call_llm(prompt, schema=NodeLabelResponseList, thinking_level="minimal")
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

    # Prepare fewshot examples
    raw_examples = EXAMPLES.get("edge", {}).get(node_label, [])
    examples = [
        {"prev_steps": ex["prev_steps"], "current_node": ex["current_step"]}
        for ex in raw_examples
    ]
    example_outputs = [
        json.dumps(ex["edges"], indent=4, ensure_ascii=False)
        for ex in raw_examples
    ]

    prompt = PROMPTS["edge"].replace("<<edge_definitions>>", edge_definitions)
    prompt = _build_prompt_with_examples(
        prompt,
        [], # examples,
        format_edge_prompt,
        input_data,
        example_outputs=example_outputs,
    )
    # print(prompt); exit()

    response = call_llm(prompt, schema=EdgeResponseList, thinking_level="high")

    return [
        {
            "id": f"e{i}",
            "source_node_id": edge["source_node_id"],
            "dest_node_id": node["id"],
            "label": edge["label"],
        }
        for i, edge in enumerate(response["responses"]) if edge["label"]
    ]


def _needs_resegment(text):
    """Return True if a node is long enough and paragraph-rich enough to warrant re-segmentation."""
    return len(text) > 300 or text.strip().count('\n\n') > 1 or ". " in text


def tokenize_and_align(text, tokenize_func):
    """Tokenize text and find character indices for each sentence.

    Uses globally optimal DP alignment (optimal_alignment) to segment the text
    into exactly len(sentences) contiguous parts.  Any resulting segment that is
    longer than 300 characters AND whose stripped text contains more than one
    blank line (\\n\\n) is recursively re-segmented until either the recursive
    call returns a single segment (cannot split further) or all segments satisfy
    the size/paragraph criteria.

    Returns (segments, char_boundaries) where char_boundaries has length
    len(segments) + 1, char_boundaries[0] == 0, char_boundaries[-1] == len(text).
    Returns (None, None) if alignment fails entirely.
    """
    print("tokenize_and_align(), len(text):", len(text))
    sentences = tokenize_func(text)
    segments, boundaries = optimal_alignment(text, sentences)

    if len(segments) == 1:
        # directly return if only one segment (no alignment needed)
        return segments, boundaries

    final_segments = []
    final_boundaries = [0]

    for i, seg in enumerate(segments):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg_text = text[seg_start:seg_end]

        if len(seg_text) > 300:
            # Split by double newlines and recursively process each paragraph
            para_ranges = []
            pos = 0
            for m in re.finditer(r'\n\n', seg_text):
                para_ranges.append((pos, m.end()))
                pos = m.end()
            if pos < len(seg_text):
                para_ranges.append((pos, len(seg_text)))

            sub_expanded = False
            if len(para_ranges) > 1:
                combined_segs = []
                combined_bounds = [0]
                for para_start, para_end in para_ranges:
                    para_text = seg_text[para_start:para_end]
                    if _needs_resegment(para_text):
                        sub_segs, sub_bounds = tokenize_and_align(para_text, tokenize_func)
                    else:
                        sub_segs, sub_bounds = [para_text], [0, len(para_text)]
                    if sub_segs is None or len(sub_segs) == 0:
                        combined_segs.append(para_text)
                        combined_bounds.append(para_end)
                    else:
                        for j, sub_seg in enumerate(sub_segs):
                            combined_segs.append(sub_seg)
                            combined_bounds.append(para_start + sub_bounds[j + 1])

                if len(combined_segs) > 1:
                    sub_expanded = True
                    for j, sub_seg in enumerate(combined_segs):
                        final_segments.append(sub_seg)
                        final_boundaries.append(seg_start + combined_bounds[j + 1])

            if not sub_expanded:
                # Cannot split further – keep as-is
                final_segments.append(seg_text)
                final_boundaries.append(seg_end)
        else:
            final_segments.append(seg_text)
            final_boundaries.append(seg_end)

    return final_segments, final_boundaries


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

    for datum in tqdm(data):
        try:
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
            conclusion_response = call_llm(conclusion_prompt, schema=ConclusionNodeList, thinking_level="high")
            conclusion_node_ids = set(conclusion_response["conclusion_node_ids"])
            for node in datum["nodes"]:
                if node["id"] in conclusion_node_ids:
                    node["label"] = "conclusion"
            
            print("Reach here: node count", len(nodes))

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
        except Exception as e:
            print(f"Error processing document {datum['doc_id']}. {e.__class__.__name__}: {e}")
            continue

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
    os.makedirs(f"data/v1_llm_{LLM_MODEL_NAME}", exist_ok=True)
    # for file in sorted(os.listdir("data/v1_raw_data")):
    shuffled_files = os.listdir("data/v1_raw_data")
    shuffle(shuffled_files)
    for file in shuffled_files:
        # DEBUG
        if "QwQ" not in file and "R1" not in file:
            continue
        if file.endswith(".json"):
            output_path = os.path.join(
                "data",
                f"v1_llm_{LLM_MODEL_NAME}",
                file
            )
            if os.path.exists(output_path):
                print(f"Data for {file} already exists, skipping.")
                continue

            with open(os.path.join("data/v1_raw_data", file), "r") as f:
                datum = json.load(f)
                data.append(datum)

    print(f"Loaded {len(data)} data samples.")
    main_predict(data, output_dir=f"data/v1_llm_{LLM_MODEL_NAME}")
