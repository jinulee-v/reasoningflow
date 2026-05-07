"""
Generate 1D PARC evaluation context diagrams for 30 randomly selected
reasoning nodes from reasoning models (DeepSeek-R1, QwQ-32B, gpt-oss-120b)
on reasoning datasets (aime2024, gpqa-diamond).

Filters: node label == "reasoning", len(context nodes) <= 7.
Outputs SVGs to plots/parc_examples/.
"""

import json
import os
import glob
import random

from generate_svg import draw_graph

# ── PARC evaluation context helpers (mirrors parc_run_evaluation.py) ──────────

def build_adjacency(edges):
    """Returns {node_id: set of predecessor node_ids}, skipping reflect/validate."""
    preds = {}
    for e in edges:
        if e["label"].startswith("reflect") or e["label"].startswith("validate"):
            continue
        preds.setdefault(e["dest_node_id"], set()).add(e["source_node_id"])
    return preds


def get_context_nodes(node_id, preds, order_index):
    """Collect predecessors within distance 2, sorted by appearance order."""
    visited = set()
    frontier = preds.get(node_id, set())
    visited.update(frontier)
    for p in list(frontier):
        for pp in preds.get(p, set()):
            visited.add(pp)
    return sorted(visited, key=lambda nid: order_index.get(nid, 0))


# ── Candidate collection ───────────────────────────────────────────────────────

REASONING_MODELS  = {"DeepSeek-R1", "QwQ-32B", "gpt-oss-120b"}
REASONING_DATASETS = {"aime2024", "gpqa-diamond"}
DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"
MAX_CONTEXT = 7
N_EXAMPLES = 30
OUTPUT_DIR = "plots/parc_examples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_files = glob.glob(os.path.join(DATA_DIR, "*.json"))

# Filter to matching model + dataset
candidate_files = []
for path in all_files:
    base = os.path.basename(path).replace(".json", "")
    parts = base.split("_")
    # filename format: {dataset}_{idx}_{model}.json
    if len(parts) < 3:
        continue
    model   = "_".join(parts[2:])
    dataset = parts[0]
    if model in REASONING_MODELS and dataset in REASONING_DATASETS:
        candidate_files.append(path)

print(f"Candidate files: {len(candidate_files)}")

# Collect all qualifying (file, node_id) pairs
candidates = []  # (path, node, context_node_ids, nodes_by_id, edges)

for path in candidate_files:
    with open(path) as f:
        data = json.load(f)

    nodes       = data["nodes"]
    edges       = data["edges"]
    nodes_by_id = {n["id"]: n for n in nodes}
    order_index = {n["id"]: i for i, n in enumerate(nodes)}
    preds       = build_adjacency(edges)

    for node in nodes:
        if node["label"] != "reasoning":
            continue
        ctx = get_context_nodes(node["id"], preds, order_index)
        if len(ctx) <= MAX_CONTEXT:
            candidates.append((path, node, ctx, nodes_by_id, edges))

print(f"Qualifying (file, node) pairs: {len(candidates)}")

# Random sample
random.seed(42)
selected = random.sample(candidates, min(N_EXAMPLES, len(candidates)))

# ── Generate diagrams ──────────────────────────────────────────────────────────

for i, (path, target_node, ctx_ids, nodes_by_id, edges) in enumerate(selected):
    included_ids = set(ctx_ids) | {target_node["id"]}

    # Build subgraph: context nodes + target, preserving original order
    subgraph_nodes = [
        nodes_by_id[nid] for nid in
        sorted(included_ids, key=lambda nid: list(nodes_by_id.keys()).index(nid))
        if nid in nodes_by_id
    ]

    # Only edges between included nodes
    subgraph_edges = [
        e for e in edges
        if e.get("source_node_id") in included_ids
        and e.get("dest_node_id") in included_ids
    ]

    subgraph = {"nodes": subgraph_nodes, "edges": subgraph_edges}

    base = os.path.basename(path).replace(".json", "")
    out_path = os.path.join(OUTPUT_DIR, f"{i:02d}_{base}_{target_node['id']}.svg")
    draw_graph(subgraph, out_path)
