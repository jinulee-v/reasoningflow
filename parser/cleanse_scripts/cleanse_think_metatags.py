"""
Cleanse <think> and </think> metatags mixed with actual content in reflection nodes.

When a reflection node contains a <think> or </think> tag alongside real text,
split it into two (or more) nodes: one per tag, one for the remaining text.
Edges are reattached to the text-containing node.

Nodes where the entire text is just the tag plus whitespace are left unchanged.
"""

import argparse
import json
import os
import re

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"

R = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"


def _trunc(text, width=80):
    text = (text or "").replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _new_node_id(existing_ids):
    i = 0
    while True:
        candidate = f"n_split_{i}"
        if candidate not in existing_ids:
            return candidate
        i += 1


def _find_matches(doc):
    """
    Find reflection nodes that mix a <think> or </think> tag with real text.

    Returns a list of dicts:
      {
        "node":   the original node dict,
        "pieces": list of (text, is_tag) tuples representing the split,
      }
    """
    results = []
    for node in doc.get("nodes", []):
        if node.get("label") != "reflection":
            continue
        text = node.get("text", "")

        pieces = []
        pos = 0
        for m in re.finditer(r"</?think>", text):
            before = text[pos : m.start()]
            if before:
                pieces.append((before, False))
            pieces.append((m.group(), True))
            pos = m.end()
        if pos < len(text):
            pieces.append((text[pos:], False))

        if len(pieces) <= 1:
            continue

        # Only act when there is real (non-whitespace) text outside the tags.
        has_real_content = any(not is_tag and piece.strip() for piece, is_tag in pieces)
        if not has_real_content:
            continue

        results.append({"node": node, "pieces": pieces})

    return results


def _print_match(match, idx, incoming, outgoing, fname):
    node = match["node"]
    pieces = match["pieces"]
    print(f"\n{BOLD}{CYAN}[{idx}] Node {node['id']} (reflection){R}  {DIM}{fname}{R}")
    print(f"  text:  {DIM}{_trunc(node['text'])}{R}")
    print(f"  edges: {incoming} incoming, {outgoing} outgoing")
    print(f"  → split into {len(pieces)} piece(s):")
    for piece, is_tag in pieces:
        if is_tag:
            label = f"{YELLOW}[tag]       {R}"
        elif piece.strip():
            label = f"{GREEN}[text+edges]{R}"
        else:
            label = f"{DIM}[whitespace]{R}"
        print(f"    {label}  {DIM}{_trunc(piece)}{R}")


def _find_relabel_matches(doc):
    """
    Find nodes whose text is solely <think> or </think> but whose label is not 'reflection'.
    Returns a list of node dicts.
    """
    results = []
    for node in doc.get("nodes", []):
        if node.get("text", "").strip() in ("<think>", "</think>"):
            if node.get("label") != "reflection":
                results.append(node)
    return results


def _print_relabel_match(node, idx, fname):
    print(f"\n{BOLD}{YELLOW}[{idx}] Node {node['id']} (label={node['label']!r}){R}  {DIM}{fname}{R}")
    print(f"  text:  {DIM}{_trunc(node['text'])}{R}")
    print(f"  → relabel to {GREEN}'reflection'{R}")


def _apply_relabel_matches(doc, nodes):
    for node in nodes:
        node["label"] = "reflection"


def _apply_match(doc, match):
    node = match["node"]
    pieces = match["pieces"]
    orig_id = node["id"]

    existing_ids = {n["id"] for n in doc["nodes"]}

    # Build new nodes, tracking char offsets.
    new_nodes = []
    content_node_id = None
    char_offset = node.get("start", 0)

    for piece_text, is_tag in pieces:
        new_id = _new_node_id(existing_ids)
        existing_ids.add(new_id)
        new_nodes.append({
            "id": new_id,
            "annotation": node.get("annotation", True),
            "start": char_offset,
            "end": char_offset + len(piece_text),
            "label": "reflection",
            "text": piece_text,
            "source": node.get("source", "response"),
        })
        char_offset += len(piece_text)
        if not is_tag and piece_text.strip():
            content_node_id = new_id

    if content_node_id is None:
        content_node_id = new_nodes[-1]["id"]

    # Reattach all edges that referenced the original node.
    for edge in doc["edges"]:
        if edge["source_node_id"] == orig_id:
            edge["source_node_id"] = content_node_id
        if edge["dest_node_id"] == orig_id:
            edge["dest_node_id"] = content_node_id

    # Replace the original node in-place with the new pieces.
    new_node_list = []
    for n in doc["nodes"]:
        if n["id"] == orig_id:
            new_node_list.extend(new_nodes)
        else:
            new_node_list.append(n)
    doc["nodes"] = new_node_list


def run(dry_run: bool, max_examples: int | None = None, data_dir: str = DATA_DIR):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))

    total_matches = 0
    total_files = 0
    shown = 0

    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            doc = json.load(f)

        matches = _find_matches(doc)
        relabel_matches = _find_relabel_matches(doc)
        if not matches and not relabel_matches:
            continue

        total_matches += len(matches) + len(relabel_matches)
        total_files += 1

        if dry_run:
            edges = doc.get("edges", [])
            for m in matches:
                if max_examples is not None and shown >= max_examples:
                    break
                shown += 1
                nid = m["node"]["id"]
                incoming = sum(1 for e in edges if e["dest_node_id"] == nid)
                outgoing = sum(1 for e in edges if e["source_node_id"] == nid)
                _print_match(m, idx=shown, incoming=incoming, outgoing=outgoing, fname=fname)
            for node in relabel_matches:
                if max_examples is not None and shown >= max_examples:
                    break
                shown += 1
                _print_relabel_match(node, idx=shown, fname=fname)
            if max_examples is not None and shown >= max_examples:
                break
        else:
            for m in matches:
                _apply_match(doc, m)
            _apply_relabel_matches(doc, relabel_matches)
            with open(path, "w") as f:
                json.dump(doc, f, indent=4, ensure_ascii=False)
            print(f"  {GREEN}fixed {len(matches)} split(s), {len(relabel_matches)} relabel(s) in {fname}{R}")

    print()
    if dry_run:
        print(
            f"{BOLD}Dry run:{R} found {total_matches} match(es) across "
            f"{total_files} file(s). No changes made."
        )
    else:
        print(
            f"{BOLD}Done:{R} split {total_matches} node(s) across {total_files} file(s)."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Split <think>/<think> metatags out of reflection nodes that mix them with real text."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print matched nodes without modifying any files.",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None, metavar="N",
        help="With --dry-run, stop after showing N examples.",
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, metavar="DIR",
        help=f"Directory of JSON files to process (default: {DATA_DIR}).",
    )
    args = parser.parse_args()

    if args.max_examples is not None and not args.dry_run:
        parser.error("--max-examples only makes sense with --dry-run")

    run(dry_run=args.dry_run, max_examples=args.max_examples, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
