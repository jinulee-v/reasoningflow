"""
Generate 1D diagrams of local verification examples.

Pattern: a "reasoning" node X has both an outgoing "plan:verify" edge and an
outgoing "validate:" edge. The diagram shows all nodes between X and the
destination of the validate: edge (inclusive), with all edges among them.

Sampling constraints:
  - Reasoning models: DeepSeek-R1, QwQ-32B, gpt-oss-120b
  - Reasoning datasets: aime2024, gpqa-diamond
  - Node count in the displayed slice <= 8
  - 30 random examples, seed 42

Outputs SVGs to plots/localverify_examples/.
"""

import json
import os
import glob
import random
from collections import defaultdict

from generate_svg import draw_graph

# ── Config ─────────────────────────────────────────────────────────────────────

REASONING_MODELS   = {"DeepSeek-R1", "QwQ-32B", "gpt-oss-120b"}
REASONING_DATASETS = {"aime2024", "gpqa-diamond"}
DATA_DIR   = "data/v1_llm_gemini-3.1-pro-preview"
# REASONING_MODELS = {"QwQ-32B-Preview"}
# REASONING_DATASETS = {"chemistry", "math", "physics"}
# DATA_DIR   = "data/v0_human_D"
MAX_NODES  = 8
N_EXAMPLES = 100
OUTPUT_DIR = "plots/localverify_examples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Candidate collection ───────────────────────────────────────────────────────

all_files = glob.glob(os.path.join(DATA_DIR, "*.json"))

candidate_files = []
for path in all_files:
    base  = os.path.basename(path).replace(".json", "")
    parts = base.split("_")
    if len(parts) < 3:
        continue
    model   = "_".join(parts[2:])
    dataset = parts[0]
    if model in REASONING_MODELS and dataset in REASONING_DATASETS:
        candidate_files.append(path)

print(f"Candidate files: {len(candidate_files)}")

# Each entry: (path, x_node, validate_dest_id, slice_nodes, all_edges)
candidates = []

for path in candidate_files:
    with open(path) as f:
        data = json.load(f)

    nodes       = data["nodes"]
    edges       = data["edges"]
    nodes_by_id = {n["id"]: n for n in nodes}
    order_index = {n["id"]: i for i, n in enumerate(nodes)}

    out_edges = defaultdict(list)
    for e in edges:
        out_edges[e["source_node_id"]].append(e)

    for node in nodes:
        if node["label"] != "reasoning":
            continue

        edges_out    = out_edges[node["id"]]
        plan_verify  = [e for e in edges_out if e["label"] == "plan:verify"]
        validate_out = [e for e in edges_out if e["label"].startswith("validate:")]

        if not plan_verify or not validate_out:
            continue

        # Use the first validate: edge's destination
        v_dest_id = validate_out[0]["dest_node_id"]
        if v_dest_id not in nodes_by_id:
            continue

        x_idx = order_index[node["id"]]
        v_idx = order_index[v_dest_id]
        if v_idx < x_idx:
            continue  # validate: points backward; skip

        # Apply MAX_NODES only to the between-X-and-validate slice
        between_nodes = [n for n in nodes if x_idx <= order_index[n["id"]] <= v_idx]
        if len(between_nodes) > MAX_NODES:
            continue

        # Display: prepend context (ctxN) nodes before the slice
        ctx_nodes   = [n for n in nodes if n["id"].startswith("ctx")]
        slice_nodes = ctx_nodes + [
            n for n in between_nodes if not n["id"].startswith("ctx")
        ]
        candidates.append((path, node, v_dest_id, slice_nodes, edges))

print(f"Qualifying (file, node) pairs: {len(candidates)}")

# ── Random sample ──────────────────────────────────────────────────────────────

random.seed(42)
selected = random.sample(candidates, min(N_EXAMPLES, len(candidates)))

# ── Generate diagrams ──────────────────────────────────────────────────────────

for i, (path, x_node, v_dest_id, slice_nodes, all_edges) in enumerate(selected):
    included_ids = {n["id"] for n in slice_nodes}

    subgraph_edges = [
        e for e in all_edges
        if e.get("source_node_id") in included_ids
        and e.get("dest_node_id") in included_ids
    ]

    subgraph = {"nodes": slice_nodes, "edges": subgraph_edges}

    base     = os.path.basename(path).replace(".json", "")
    out_path = os.path.join(OUTPUT_DIR, f"{i:02d}_{base}_{x_node['id']}.svg")
    draw_graph(subgraph, out_path)
