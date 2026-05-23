"""Patch edges for a single node in an annotated JSON file.

Usage:
    python parser/llm_labeler_patch_edge.py --doc data/.../foo.json --node resp73 --model_name gemini-3-flash-preview
"""

import argparse
import json

from llm_labeler import (
    LLM_MODEL_NAME,
    edge_detection_and_classification,
    get_node_sort_key,
    get_metadata,
)


def patch_edges_for_node(doc_path: str, node_id: str, llm_model_name: str) -> None:
    with open(doc_path, "r") as f:
        datum = json.load(f)

    nodes = datum["nodes"]
    node_ids = [n["id"] for n in nodes]

    if node_id not in node_ids:
        raise ValueError(f"Node '{node_id}' not found in {doc_path}. Available IDs: {node_ids}")

    node_idx = node_ids.index(node_id)
    node = nodes[node_idx]

    if node["source"] != "response":
        raise ValueError(f"Node '{node_id}' is a '{node['source']}' node, not a response node — nothing to annotate.")

    print(f"Re-annotating edges for node {node_id} (label={node['label']}) ...")
    new_edges_for_node = edge_detection_and_classification(node_idx, nodes, llm_model_name)
    print(f"  Got {len(new_edges_for_node)} edge(s) for {node_id}.")

    # Drop all existing edges whose dest is this node, keep the rest
    kept_edges = [e for e in datum.get("edges", []) if e["dest_node_id"] != node_id]

    # Merge and sort
    merged = kept_edges + new_edges_for_node
    merged.sort(
        key=lambda e: (get_node_sort_key(e["dest_node_id"]), get_node_sort_key(e["source_node_id"]))
    )

    # Re-assign sequential IDs
    for i, edge in enumerate(merged):
        edge["id"] = f"e{i}"

    datum["edges"] = merged

    with open(doc_path, "w") as f:
        json.dump(datum, f, indent=4, ensure_ascii=False)

    print(f"Saved updated file: {doc_path}")

    meta = get_metadata()
    print(
        f"Total input tokens: {meta['in_token']}, "
        f"output tokens: {meta['out_token']}, "
        f"price: ${meta['price']:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch edges for a single node in an annotated JSON file.")
    parser.add_argument("--doc", required=True, help="Path to the annotated JSON file.")
    parser.add_argument("--node", required=True, help="Node ID whose incoming edges should be re-annotated (e.g. resp73).")
    parser.add_argument("--model_name", type=str, default=LLM_MODEL_NAME, help="LLM model name to use for annotation.")
    args = parser.parse_args()

    patch_edges_for_node(args.doc, args.node, args.model_name)
