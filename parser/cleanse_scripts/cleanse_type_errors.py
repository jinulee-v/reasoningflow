"""
Print edges that violate the edge-type schema (source/dest node-type constraints).

For each violation class (a unique (edge_label, bad_field, bad_type) triple),
up to --max-examples examples are printed.
"""

import json
import os
from collections import defaultdict

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"

R = BOLD = DIM = YELLOW = CYAN = RED = ""

# Edge schema: label -> (valid_source_types, valid_dest_types)
EDGE_SCHEMA = {
    "reason:infer":          ({"context","fact","reasoning","restatement","assumption","example","conclusion"},
                              {"reasoning","assumption","reflection","conclusion"}),
    "reason:execute":        ({"planning"},
                              {"fact","reasoning","restatement","assumption","example","conclusion"}),
    "reason:restate":        ({"context","fact","reasoning","assumption","conclusion"},
                              {"restatement"}),
    "reason:elaborate-fact": ({"fact"},
                              {"fact"}),
    "reason:exemplify":      ({"planning","fact","reasoning","restatement","assumption"},
                              {"example"}),
    "plan:proceed":          ({"context","planning","fact","reasoning","restatement","assumption","example","reflection"},
                              {"planning","assumption"}),
    "plan:verify":           ({"fact","reasoning","restatement","assumption","example"},
                              {"planning"}),
    "plan:decompose":        ({"planning","assumption"},
                              {"planning","assumption"}),
    "plan:backtrack":        ({"planning"},
                              {"planning"}),
    "reflect:positive":      ({"planning","fact","reasoning","restatement","assumption","example","conclusion"},
                              {"reflection"}),
    "reflect:uncertain":     ({"planning","fact","reasoning","restatement","assumption","example","conclusion"},
                              {"reflection"}),
    "reflect:negative":      ({"planning","fact","reasoning","restatement","assumption","example","conclusion"},
                              {"reflection"}),
    "validate:support":      ({"context","planning","fact","reasoning","assumption","conclusion"},
                              {"reasoning","conclusion"}),
    "validate:attack":       ({"context","planning","fact","reasoning","assumption","conclusion"},
                              {"reasoning","conclusion"}),
}


def _trunc(text, width=80):
    text = (text or "").replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _collect_violations(files, data_dir=DATA_DIR):
    """
    Scan all files and return a dict:
      key   -> list of violation dicts

    key is a string describing the violation class, e.g.:
      "src 'reflection' --[reason:infer]-->"
      "--[reason:restate]--> dst 'reasoning'"
      "unknown label 'foo'"
    """
    buckets = defaultdict(list)

    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            doc = json.load(f)

        nodes_by_id = {n["id"]: n for n in doc.get("nodes", [])}

        for edge in doc.get("edges", []):
            label  = edge.get("label", "")
            src_id = edge.get("source_node_id")
            dst_id = edge.get("dest_node_id")

            src_node  = nodes_by_id.get(src_id)
            dst_node  = nodes_by_id.get(dst_id)
            src_label = src_node.get("label") if src_node else None
            dst_label = dst_node.get("label") if dst_node else None

            if label not in EDGE_SCHEMA:
                key = f"unknown label '{label}'"
                buckets[key].append({
                    "file": fname, "edge": edge,
                    "src_label": src_label, "dst_label": dst_label,
                    "src_text": (src_node or {}).get("text", ""),
                    "dst_text": (dst_node or {}).get("text", ""),
                })
                continue

            valid_srcs, valid_dsts = EDGE_SCHEMA[label]

            if src_label not in valid_srcs:
                key = f"src '{src_label}' --[{label}]-->"
                buckets[key].append({
                    "file": fname, "edge": edge,
                    "src_label": src_label, "dst_label": dst_label,
                    "src_text": (src_node or {}).get("text", ""),
                    "dst_text": (dst_node or {}).get("text", ""),
                })

            if dst_label not in valid_dsts:
                key = f"--[{label}]--> dst '{dst_label}'"
                buckets[key].append({
                    "file": fname, "edge": edge,
                    "src_label": src_label, "dst_label": dst_label,
                    "src_text": (src_node or {}).get("text", ""),
                    "dst_text": (dst_node or {}).get("text", ""),
                })

    return buckets


def _print_violation(v, idx, total):
    edge = v["edge"]
    print(f"  {DIM}[{idx}/{total}] {v['file']}  edge {edge.get('id')}{R}")
    print(f"  {BOLD}src{R} [{v['src_label']}]  {DIM}{_trunc(v['src_text'])}{R}")
    print(f"  {YELLOW}──[{edge.get('label')}]──►{R}")
    print(f"  {BOLD}dst{R} [{v['dst_label']}]  {DIM}{_trunc(v['dst_text'])}{R}")


def run(max_examples: int = 15, data_dir: str = DATA_DIR):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    buckets = _collect_violations(files, data_dir=data_dir)

    total_violations = sum(len(vs) for vs in buckets.values())
    print(f"{BOLD}Total violations: {total_violations}  |  Violation classes: {len(buckets)}{R}\n")

    for key in sorted(buckets, key=lambda k: -len(buckets[k])):
        vs = buckets[key]
        shown = min(max_examples, len(vs))
        print(f"{BOLD}{CYAN}[{key}]{R}  {len(vs)} total")
        for i, v in enumerate(vs[:shown], 1):
            _print_violation(v, i, len(vs))
            print()
        if len(vs) > max_examples:
            print(f"  {DIM}... {len(vs) - max_examples} more not shown{R}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print edge type schema violations.")
    parser.add_argument(
        "--max-examples", type=int, default=15, metavar="N",
        help="Examples to show per violation class (default: 15).",
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, metavar="DIR",
        help=f"Directory of JSON files to process (default: {DATA_DIR}).",
    )
    args = parser.parse_args()
    run(max_examples=args.max_examples, data_dir=args.data_dir)
