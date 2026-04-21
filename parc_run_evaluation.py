"""
Premise-Augmented Reasoning Chain (PARC) evaluation.
Reads annotated JSON files and evaluates each fact/reasoning node's correctness.
"""

import json
import os
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from typing import Literal

from tqdm.asyncio import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parser.utils.vertexai import call_llm

EVALUATOR = "gemini-3-flash"
INPUT_DIR = "data/v1_llm_gemini-3.1-pro-preview"
OUTPUT_DIR = "parc_results"
MAX_WORKERS = 16


class EvalResult(BaseModel):
    chain_of_thought: str
    correctness: Literal["correct", "error", "undecidable"]


PROMPT_TEMPLATE = """\
You are an expert evaluator. Given a context of preceding reasoning steps and a target node, \
determine whether the target node is correct, erroneous, or if the context is insufficient to determine.
- "error" label should be only used if the target node contains a clear mistake that contradicts (cannot be true at the same time) the context, assuming all information in context is true.
- If the context is self-contradictory (likely already containing errors), classify as "undecidable" instead.

## Context (preceding related nodes)
{context}

## Target Node to Evaluate
{target}

Respond with:
- chain_of_thought: your chain-of-thought explanation.
- correctness: "correct" if the statement is logically/factually correct given the context, \
"error" if it contains a mistake, "undecidable" if the context is insufficient.
"""


def build_adjacency(edges):
    """Returns {node_id: set of predecessor node_ids}."""
    preds = {}
    for e in edges:
        if e["label"].startswith("reflect") or e["label"].startswith("validate"):
            continue
        preds.setdefault(e["dest_node_id"], set()).add(e["source_node_id"])
    return preds


def get_context_nodes(node_id, nodes_by_id, preds, order_index):
    """Collect predecessors within distance 2, sorted by appearance order."""
    visited = set()
    frontier = preds.get(node_id, set())
    visited.update(frontier)
    for p in list(frontier):
        for pp in preds.get(p, set()):
            visited.add(pp)
    return sorted(visited, key=lambda nid: order_index.get(nid, 0))


def evaluate_node(node, context_nodes, nodes_by_id):
    context_text = "\n\n".join(
        f"[{nodes_by_id[nid]['label'].upper()}] {nodes_by_id[nid]['text']}"
        for nid in context_nodes
        if nid in nodes_by_id
    ) or "(no preceding context)"
    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        target=f"[{node['label'].upper()}] {node['text']}",
    )
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(call_llm, prompt, schema=EvalResult, thinking_level="minimal")
    try:
        result = future.result(timeout=20)
    finally:
        executor.shutdown(wait=False)
    return {
        "node_id": node["id"],
        "evaluator": EVALUATOR,
        "correctness": result["correctness"],
        "chain_of_thought": result["chain_of_thought"],
    }


def process_file(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]
    nodes_by_id = {n["id"]: n for n in nodes}
    order_index = {n["id"]: i for i, n in enumerate(nodes)}
    preds = build_adjacency(edges)
    targets = [n for n in nodes if n["label"] in ("fact", "reasoning", "restatement", "conclusion")]
    if not targets:
        return

    results = [None] * len(targets)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                evaluate_node,
                node,
                get_context_nodes(node["id"], nodes_by_id, preds, order_index),
                nodes_by_id,
            ): i
            for i, node in enumerate(targets)
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            results[futures[fut]] = fut.result()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done: {output_path}")


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "**/*.json"), recursive=True))
    if not files:
        print(f"No JSON files found in {INPUT_DIR}")
        return

    for i, input_path in enumerate(files):
        if "aime" not in input_path and "gpqa" not in input_path:
            continue
        rel = os.path.relpath(input_path, INPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, rel)
        if os.path.exists(output_path):
            print(f"Skip (exists): {output_path}")
            continue
        # if next output path exists, skip to avoid redundant work
        if i < len(files) - 1 and os.path.exists(os.path.join(OUTPUT_DIR, os.path.relpath(files[i + 1], INPUT_DIR))):
            print(f"Skip (next exists): {output_path}")
            continue
        try:
            process_file(input_path, output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    main()
