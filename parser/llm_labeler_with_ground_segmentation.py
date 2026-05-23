import os
import json
import concurrent.futures
from functools import partial
from random import shuffle

from tqdm import tqdm

# from utils.openai import call_llm, get_metadata
# from utils.vertexai import call_llm, get_metadata
# from utils.vllm import call_llm, get_metadata
from utils.deepinfra import call_llm, get_metadata

from llm_labeler import (
    SentenceList, NodeLabel, NodeLabelResponse, NodeLabelResponseList,
    ConclusionNodeList, EdgeLabel, EdgeResponse, EdgeResponseList,
    NODE_LABEL_TO_EDGE_LABELS, EDGE_LABEL_DEFINITIONS,
    format_node_label_prompt, format_conclusion_prompt, format_edge_prompt,
    PROMPTS, EXAMPLES,
    _build_edge_definitions, _build_prompt_with_examples, _extract_node_summary,
    get_node_sort_key, model_id_map,
    MAX_WORKERS,
)


# =============================================================================
# Core Annotation Functions (use local call_llm from deepinfra)
# =============================================================================

def node_classification(nodes, question):
    """Labels all sentences using LLM."""
    input_data = {"nodes": nodes, "previous_steps": question}
    prompt = _build_prompt_with_examples(
        PROMPTS["node_classification"],
        [],
        format_node_label_prompt,
        input_data
    )
    response = call_llm(prompt, llm_model_name=LLM_MODEL_NAME, schema=NodeLabelResponseList, thinking_level="minimal")
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

    input_data = {
        "prev_steps": [_extract_node_summary(n) for n in nodes[:node_idx]],
        "current_node": node,
    }

    edge_definitions = _build_edge_definitions(node_label)
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

    response = call_llm(prompt, llm_model_name=LLM_MODEL_NAME, schema=EdgeResponseList, thinking_level="high")

    return [
        {
            "id": f"e{i}",
            "source_node_id": edge["source_node_id"],
            "dest_node_id": node["id"],
            "label": edge["label"],
        }
        for i, edge in enumerate(response["responses"]) if edge["label"]
    ]


# =============================================================================
# Main Processing Function
# =============================================================================

def main_predict(data, ground_dir="data/v0_human_D", output_dir=""):
    """Main prediction function using ground-truth segmentation."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for datum in tqdm(data):
        try:
            print(f"Processing document: {datum['doc_id']}")

            # Load ground-truth segmentation
            ground_path = os.path.join(ground_dir, f"{datum['doc_id']}.json")
            with open(ground_path) as f:
                ground = json.load(f)

            nodes = []
            for node in ground["nodes"]:
                n = dict(node)
                if n["source"] == "response":
                    n["label"] = None  # reset; will be re-classified by LLM
                nodes.append(n)

            # Remove nodes with empty text before classification
            nodes = [node for node in nodes if node["text"].strip()]

            # Annotate node labels together
            node_to_labels_list = node_classification(nodes, question=datum["raw_text"]["question"])
            node_to_labels = {item['node_id']: item['label'] for item in node_to_labels_list['responses']}
            for node in nodes:
                if node['id'] in node_to_labels and node['source'] == 'response':
                    node['label'] = node_to_labels[node['id']]

            datum["nodes"] = nodes

            # Post-hoc update of conclusion nodes
            conclusion_prompt = PROMPTS["update_conclusion"].replace("<<input>>", format_conclusion_prompt(datum))
            conclusion_response = call_llm(conclusion_prompt, llm_model_name=LLM_MODEL_NAME, schema=ConclusionNodeList, thinking_level="high")
            conclusion_node_ids = set(conclusion_response["conclusion_node_ids"])
            for node in datum["nodes"]:
                if node["id"] in conclusion_node_ids:
                    node["label"] = "conclusion"

            print("Reach here: node count", len(nodes))

            # Annotate edges in parallel
            edge_func = partial(edge_detection_and_classification, nodes=nodes)
            non_context_indices = [i for i, node in enumerate(nodes) if node["label"] != "context"]
            # shuffle to improve load balancing (some nodes take much longer than others)
            shuffle(non_context_indices)

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                edge_lists = list(tqdm(executor.map(edge_func, non_context_indices), total=len(non_context_indices), desc="annotating edges"))
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
    parser.add_argument("--model_name", type=str, required=True, help="LLM model name to use for annotation.")
    parser.add_argument("--raw_data_dir", type=str, default="data/v0_raw_data", help="Directory containing raw input JSON files.")
    parser.add_argument("--ground_dir", type=str, default="data/v0_human_D", help="Directory containing ground-truth segmentation JSON files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory containing ground-truth segmentation JSON files.")

    args = parser.parse_args()

    LLM_MODEL_NAME = args.model_name
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = f"data/v0_llm_{LLM_MODEL_NAME}_groundseg"

    print("<Load data...>")
    data = []
    ground_dir = args.ground_dir
    os.makedirs(output_dir, exist_ok=True)
    shuffled_files = os.listdir(args.raw_data_dir)
    shuffle(shuffled_files)
    for file in shuffled_files:
        if file.endswith(".json"):
            output_path = os.path.join(output_dir, file)
            if os.path.exists(output_path):
                print(f"Data for {file} already exists, skipping.")
                continue

            with open(os.path.join(args.raw_data_dir, file), "r") as f:
                datum = json.load(f)
                data.append(datum)

    print(f"Loaded {len(data)} data samples.")
    main_predict(data, ground_dir=ground_dir, output_dir=output_dir)
