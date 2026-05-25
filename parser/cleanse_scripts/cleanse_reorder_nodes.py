"""
Reindex nodes/edges, and propagate changes to argumentation/prm/parc results.
"""

import json
import glob
import os
from copy import deepcopy

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"
ARG_DIR = "argumentation_results"
PRM_DIR = "prm_results"
PARC_DIR = "parc_results"


def node_sort_key(node_id: str) -> tuple:
    """ctx < resp, then numerically within each group."""
    if node_id.startswith("ctx"):
        return (0, int(node_id[3:]))
    elif node_id.startswith("resp"):
        return (1, int(node_id[4:]))
    return (2, 0)


def edge_sort_key(edge: dict) -> tuple:
    return (node_sort_key(edge["dest_node_id"]), node_sort_key(edge["source_node_id"]))


def process_data_file(path: str) -> dict | None:
    """
    Returns a remapping dict {old_node_id: new_node_id} for all renamed nodes,
    or None if no changes were needed.
    Modifies the file in place.
    """
    with open(path) as f:
        doc = json.load(f)

    nodes = doc.get("nodes", [])
    edges = doc.get("edges", [])

    empty_ids = {n["id"] for n in nodes if not n.get("text", "").strip()}
    # if not empty_ids:
    #     return None

    # Remove empty nodes and connected edges
    kept_nodes = [n for n in nodes if n["id"] not in empty_ids]
    kept_edges = [
        e for e in edges
        if e["source_node_id"] not in empty_ids and e["dest_node_id"] not in empty_ids
    ]

    # Sort edges: dest_node ascending (ctx->resp), then source_node ascending
    kept_edges = sorted(kept_edges, key=edge_sort_key)

    # Build node remapping: only resp nodes get renumbered, ctx stays
    # Separate ctx and resp nodes, preserve original relative order
    ctx_nodes = [n for n in kept_nodes if n["id"].startswith("ctx")]
    resp_nodes = [n for n in kept_nodes if n["id"].startswith("resp")]

    # Sort each group by their numeric index to maintain original order
    ctx_nodes.sort(key=lambda n: int(n["id"][3:]))
    resp_nodes.sort(key=lambda n: int(n["id"][4:]))

    node_remap: dict[str, str] = {}

    # ctx nodes: check if any were removed and reindex
    for new_idx, node in enumerate(ctx_nodes):
        new_id = f"ctx{new_idx}"
        if node["id"] != new_id:
            node_remap[node["id"]] = new_id
        node["id"] = new_id

    # resp nodes: reindex after removal
    for new_idx, node in enumerate(resp_nodes):
        new_id = f"resp{new_idx}"
        if node["id"] != new_id:
            node_remap[node["id"]] = new_id
        node["id"] = new_id

    # Apply node remapping to edges
    for edge in kept_edges:
        edge["source_node_id"] = node_remap.get(edge["source_node_id"], edge["source_node_id"])
        edge["dest_node_id"] = node_remap.get(edge["dest_node_id"], edge["dest_node_id"])

    # Reindex edge IDs
    for new_idx, edge in enumerate(kept_edges):
        edge["id"] = f"e{new_idx}"

    # Reconstruct nodes list: ctx first, then resp
    doc["nodes"] = ctx_nodes + resp_nodes
    doc["edges"] = kept_edges

    with open(path, "w") as f:
        json.dump(doc, f, indent=4, ensure_ascii=False)

    # Return mapping including identity mappings for kept nodes that weren't renamed
    # (so callers know full old->new for all nodes)
    full_remap = {}
    for n in nodes:
        if n["id"] in empty_ids:
            full_remap[n["id"]] = None  # deleted
        else:
            full_remap[n["id"]] = node_remap.get(n["id"], n["id"])

    return full_remap


def update_prm_results(path: str, remap: dict) -> None:
    """Rename resp keys in each mode dict."""
    with open(path) as f:
        doc = json.load(f)

    changed = False
    for mode_key in list(doc.keys()):
        if mode_key == "doc_id":
            continue
        mode = doc[mode_key]
        if not isinstance(mode, dict):
            continue
        new_mode = {}
        for node_id, val in mode.items():
            new_id = remap.get(node_id, node_id)
            if new_id is None:
                changed = True
                continue  # node was deleted
            if new_id != node_id:
                changed = True
            new_mode[new_id] = val
        doc[mode_key] = new_mode

    if changed:
        with open(path, "w") as f:
            json.dump(doc, f, indent=4, ensure_ascii=False)


def update_list_result(path: str, remap: dict) -> None:
    """Update node_id fields in a list-of-dicts result file."""
    with open(path) as f:
        doc = json.load(f)

    if not isinstance(doc, list):
        return

    changed = False
    new_doc = []
    for item in doc:
        node_id = item.get("node_id")
        if node_id is not None:
            new_id = remap.get(node_id, node_id)
            if new_id is None:
                changed = True
                continue  # node was deleted
            if new_id != node_id:
                item = dict(item)
                item["node_id"] = new_id
                changed = True
        new_doc.append(item)

    if changed:
        with open(path, "w") as f:
            json.dump(new_doc, f, indent=4, ensure_ascii=False)


def update_argumentation_results(path: str, remap: dict) -> None:
    """
    Argumentation results can be either a list (like parc) or a dict with
    for_planning_node_ids / against_planning_node_ids lists.
    """
    with open(path) as f:
        doc = json.load(f)

    changed = False

    if isinstance(doc, list):
        new_doc = []
        for item in doc:
            node_id = item.get("node_id")
            if node_id is not None:
                new_id = remap.get(node_id, node_id)
                if new_id is None:
                    changed = True
                    continue
                if new_id != node_id:
                    item = dict(item)
                    item["node_id"] = new_id
                    changed = True
            new_doc.append(item)
        doc = new_doc
    elif isinstance(doc, dict):
        for list_key in ("for_planning_node_ids", "against_planning_node_ids"):
            if list_key in doc:
                new_list = []
                for nid in doc[list_key]:
                    new_id = remap.get(nid, nid)
                    if new_id is None:
                        changed = True
                        continue
                    if new_id != nid:
                        changed = True
                    new_list.append(new_id)
                doc[list_key] = new_list

    if changed:
        with open(path, "w") as f:
            json.dump(doc, f, indent=4, ensure_ascii=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Reindex nodes/edges and propagate changes to result directories."
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, metavar="DIR",
        help=f"Directory of JSON files to process (default: {DATA_DIR}).",
    )
    args = parser.parse_args()

    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    total_files = 0
    total_empty = 0

    for data_path in data_files:
        basename = os.path.basename(data_path)

        remap = process_data_file(data_path)
        if remap is None:
            continue

        deleted = [k for k, v in remap.items() if v is None]
        renamed = {k: v for k, v in remap.items() if v is not None and k != v}
        total_files += 1
        total_empty += len(deleted)

        print(f"{basename}: removed {len(deleted)} node(s) {deleted}, renamed {len(renamed)}")

        # Propagate to result files
        prm_path = os.path.join(PRM_DIR, basename)
        if os.path.exists(prm_path):
            update_prm_results(prm_path, remap)
            print(f"  updated {prm_path}")

        parc_path = os.path.join(PARC_DIR, basename)
        if os.path.exists(parc_path):
            update_list_result(parc_path, remap)
            print(f"  updated {parc_path}")

        arg_path = os.path.join(ARG_DIR, basename)
        if os.path.exists(arg_path):
            update_argumentation_results(arg_path, remap)
            print(f"  updated {arg_path}")

    print(f"\nDone: {total_files} files updated, {total_empty} empty nodes removed.")


if __name__ == "__main__":
    main()
