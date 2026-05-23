#!/usr/bin/env python3
"""
Merge two or more nodes in a v1_llm_* data JSON file.

Usage:
    python merge_some_nodes.py <file.json> <node_id1> <node_id2> [node_id3 ...]

Example:
    python merge_some_nodes.py data/v1_llm_gemini-3.1-pro-preview/aime2024_25_DeepSeek-R1.json resp422 resp423 resp424
"""

import json
import re
import sys
from collections import defaultdict


def prompt_label(context: str, choices: list[str]) -> str:
    print(f"\n  {context}")
    for i, c in enumerate(choices):
        print(f"    [{i}] {c}")
    while True:
        raw = input("  Choose label index: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(choices):
            return choices[int(raw)]
        print(f"  Please enter a number between 0 and {len(choices)-1}.")


def show_node_text(node: dict) -> None:
    print(f"    [{node['id']}] ({node['label']}) {repr(node['text'])}")


def merge_nodes(filepath: str, node_ids: list[str]) -> None:
    with open(filepath) as f:
        data = json.load(f)

    nodes: list[dict] = data["nodes"]
    edges: list[dict] = data["edges"]
    raw_text: str = data["raw_text"]

    # Validate all node IDs exist
    node_map = {n["id"]: n for n in nodes}
    for nid in node_ids:
        if nid not in node_map:
            sys.exit(f"Error: node '{nid}' not found in {filepath}")

    # Check that node IDs form a consecutive numeric sequence
    id_nums = []
    for nid in node_ids:
        m = re.fullmatch(r"([a-zA-Z_]+)(\d+)", nid)
        if not m:
            sys.exit(f"Error: node id '{nid}' does not match expected <prefix><number> format")
        id_nums.append((m.group(1), int(m.group(2))))
    prefixes = {p for p, _ in id_nums}
    if len(prefixes) > 1:
        sys.exit(f"Error: node IDs have mixed prefixes: {prefixes}")
    # nums = sorted(n for _, n in id_nums)
    # if nums != list(range(nums[0], nums[0] + len(nums))):
    #     sys.exit(f"Error: node IDs are not consecutive: {nums}")

    merge_set = set(node_ids)
    merge_nodes_list = sorted([node_map[nid] for nid in node_ids], key=lambda n: n["start"])

    # --- Build merged node ---
    merged_start = merge_nodes_list[0]["start"]
    merged_end = merge_nodes_list[-1]["end"]
    merged_id = merge_nodes_list[0]["id"]  # keep earliest node's ID

    # Decide label
    labels = list(dict.fromkeys(n["label"] for n in merge_nodes_list))  # unique, ordered
    if len(labels) == 1:
        merged_label = labels[0]
    else:
        print("\n  Nodes being merged:")
        for n in merge_nodes_list:
            show_node_text(n)
        merged_label = prompt_label(
            f"Merged node [{merged_id}] has conflicting labels. Choose one:",
            labels,
        )

    # All merged nodes must share the same source (question/response)
    sources = list(dict.fromkeys(n.get("source", "") for n in merge_nodes_list if n.get("source")))
    if len(sources) > 1:
        sys.exit(f"Error: cannot merge nodes from different sources: {sources}")
    merged_source = sources[0] if sources else None

    # Slice merged text from the appropriate raw_text entry
    raw_source = raw_text[merged_source] if merged_source and isinstance(raw_text, dict) else raw_text
    merged_text = raw_source[merged_start:merged_end]

    merged_annotation = any(n.get("annotation", False) for n in merge_nodes_list)

    merged_node = {
        "id": merged_id,
        "annotation": merged_annotation,
        "start": merged_start,
        "end": merged_end,
        "label": merged_label,
        "text": merged_text,
    }
    if merged_source is not None:
        merged_node["source"] = merged_source

    # --- Process edges ---
    # Separate edges into: internal (both endpoints in merge_set), external, and others
    kept_edges = []
    # external_groups: (external_id, direction) -> list of edges
    # direction: "incoming" means external -> merged, "outgoing" means merged -> external
    incoming: dict[str, list[dict]] = defaultdict(list)  # external_id -> edges
    outgoing: dict[str, list[dict]] = defaultdict(list)  # external_id -> edges

    for e in edges:
        src_in = e["source_node_id"] in merge_set
        dst_in = e["dest_node_id"] in merge_set
        if src_in and dst_in:
            # internal edge — drop
            continue
        elif src_in:
            # outgoing: merged -> external
            outgoing[e["dest_node_id"]].append(e)
        elif dst_in:
            # incoming: external -> merged
            incoming[e["source_node_id"]].append(e)
        else:
            kept_edges.append(e)

    def resolve_edge_group(group: list[dict], direction: str, external_id: str) -> dict | None:
        """From a group of parallel edges (same external node, same direction), return one."""
        unique_labels = list(dict.fromkeys(e["label"] for e in group))
        if len(unique_labels) == 1:
            chosen_label = unique_labels[0]
        else:
            ext_node = node_map.get(external_id)
            print("\n  Conflicting edges:")
            for e in group:
                orig_node = node_map.get(e["source_node_id"] if direction == "incoming" else e["dest_node_id"])
                orig_text = repr(orig_node["text"]) if orig_node else "?"
                print(f"    {e['source_node_id']} -> {e['dest_node_id']}  [{e['label']}]  text: {orig_text}")
            print(f"  External node '{external_id}': {repr(ext_node['text']) if ext_node else '?'}")
            print(f"  Merged node text: {repr(merged_text)}")
            if direction == "incoming":
                ctx = f"'{external_id}' -> merged: choose label (or 'drop'):"
            else:
                ctx = f"merged -> '{external_id}': choose label (or 'drop'):"
            choices = unique_labels + ["drop"]
            chosen = prompt_label(ctx, choices)
            if chosen == "drop":
                return None
            chosen_label = chosen

        # Build a single representative edge
        rep = group[0].copy()
        rep["label"] = chosen_label
        if direction == "incoming":
            rep["source_node_id"] = external_id
            rep["dest_node_id"] = merged_id
        else:
            rep["source_node_id"] = merged_id
            rep["dest_node_id"] = external_id
        return rep

    for ext_id, group in incoming.items():
        edge = resolve_edge_group(group, "incoming", ext_id)
        if edge:
            kept_edges.append(edge)

    for ext_id, group in outgoing.items():
        edge = resolve_edge_group(group, "outgoing", ext_id)
        if edge:
            kept_edges.append(edge)

    # Re-assign sequential edge IDs
    for i, e in enumerate(kept_edges):
        e["id"] = f"e{i}"

    # --- Rebuild nodes list ---
    new_nodes = []
    seen_merged = False
    for n in nodes:
        if n["id"] in merge_set:
            if not seen_merged:
                new_nodes.append(merged_node)
                seen_merged = True
            # else: drop the other merged nodes
        else:
            new_nodes.append(n)

    print(f"\nMerged node '{merged_id}' ({merged_label}, chars {merged_start}–{merged_end}):")
    print(f"  {merged_text}")
    print(f"\nWill keep {len(kept_edges)} edges (drop {len(edges) - len(kept_edges)}).")
    confirm = input("\nWrite to file? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    data["nodes"] = new_nodes
    data["edges"] = kept_edges

    with open(filepath, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    filepath = sys.argv[1]
    node_ids = sys.argv[2:]
    merge_nodes(filepath, node_ids)
