import argparse
import json
import os
import sys
import yaml
from collections import defaultdict

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"
SCHEMA_DIR = "schema"

# ANSI colors
R = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
BG_DARK = "\033[48;5;235m"


def load_schema_labels():
    with open(os.path.join(SCHEMA_DIR, "node_labels.yaml")) as f:
        node_schema = yaml.safe_load(f)
    with open(os.path.join(SCHEMA_DIR, "edge_labels.yaml")) as f:
        edge_schema = yaml.safe_load(f)
    # Lists preserve schema order; callers that need O(1) lookup use `set()`
    valid_nodes = [n["name"] for n in node_schema["nodes"]]
    valid_edges = [e["name"] for e in edge_schema["edges"]]
    return valid_nodes, valid_edges


# ── scan ──────────────────────────────────────────────────────────────────────

def scan_data(valid_nodes, valid_edges):
    invalid_node_labels = defaultdict(list)
    invalid_edge_labels = defaultdict(list)
    node_set = set(valid_nodes)
    edge_set = set(valid_edges)
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            doc = json.load(f)
        for node in doc.get("nodes", []):
            label = node.get("label")
            if label not in node_set:
                invalid_node_labels[label].append((fname, node["id"]))
        for edge in doc.get("edges", []):
            label = edge.get("label")
            if label not in edge_set:
                invalid_edge_labels[label].append((fname, edge["id"]))
    return invalid_node_labels, invalid_edge_labels


def cmd_scan(args):
    valid_nodes, valid_edges = load_schema_labels()
    print(f"Schema node labels ({len(valid_nodes)}): {valid_nodes}")
    print(f"Schema edge labels ({len(valid_edges)}): {valid_edges}")
    print()
    invalid_node_labels, invalid_edge_labels = scan_data(valid_nodes, valid_edges)

    print("=== Invalid node labels ===")
    if invalid_node_labels:
        for label, occurrences in sorted(invalid_node_labels.items()):
            print(f"  {repr(label)}: {len(occurrences)} occurrence(s)")
            for fname, node_id in occurrences[:5]:
                print(f"    {fname}  node={node_id}")
            if len(occurrences) > 5:
                print(f"    ... and {len(occurrences) - 5} more")
    else:
        print("  (none)")

    print()
    print("=== Invalid edge labels ===")
    if invalid_edge_labels:
        for label, occurrences in sorted(invalid_edge_labels.items()):
            print(f"  {repr(label)}: {len(occurrences)} occurrence(s)")
            for fname, edge_id in occurrences[:5]:
                print(f"    {fname}  edge={edge_id}")
            if len(occurrences) > 5:
                print(f"    ... and {len(occurrences) - 5} more")
    else:
        print("  (none)")


# ── edit helpers ──────────────────────────────────────────────────────────────

def _truncate(text, width=72):
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[:width - 1] + "…"


def _node_color(label):
    colors = {
        "context": DIM, "planning": RED, "fact": YELLOW,
        "reasoning": "\033[33m", "restatement": GREEN,
        "assumption": "\033[32m", "example": CYAN,
        "reflection": "\033[34m", "conclusion": MAGENTA,
    }
    return colors.get(label, "")


def _print_node_row(node, highlight=False):
    label = node.get("label") or "???"
    text = _truncate(node.get("text", ""))
    color = _node_color(label)
    marker = f"{BOLD}>{R}" if highlight else " "
    if highlight:
        print(f"  {marker} {RED}{BOLD}[{label}]{R}  {text}")
    else:
        print(f"  {marker} {color}[{label}]{R}  {DIM}{text}{R}")


def _build_node_index(doc):
    return {n["id"]: i for i, n in enumerate(doc["nodes"])}


def _display_node_context(doc, node, node_index):
    idx = node_index[node["id"]]
    nodes = doc["nodes"]
    window = nodes[max(0, idx - 3): idx + 4]
    print()
    for n in window:
        _print_node_row(n, highlight=(n["id"] == node["id"]))
    print()


def _display_edge_context(doc, edge, node_index):
    src_id = edge.get("source_node_id")
    dst_id = edge.get("dest_node_id")
    nodes = doc["nodes"]

    def get_node(nid):
        idx = node_index.get(nid)
        return nodes[idx] if idx is not None else None

    src = get_node(src_id)
    dst = get_node(dst_id)
    print()
    if src:
        _print_node_row(src)
    print(f"  {'':>2} {YELLOW}──[{edge.get('label') or '???'}]──►{R}")
    if dst:
        _print_node_row(dst)
    print()


def _prompt_label(valid_labels, current_label, kind):
    # valid_labels is an ordered list matching schema order
    label_set = set(valid_labels)
    print(f"  Valid {kind} labels:")
    for i, lbl in enumerate(valid_labels, 1):
        print(f"    {CYAN}{i:2}{R}. {lbl}")
    print()
    print(f"  Commands: number / label name / {BOLD}s{R}kip / {BOLD}S{R}kip file / {BOLD}q{R}uit")
    while True:
        try:
            raw = input(f"  Label [{current_label or 'empty'}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if not raw:
            continue
        if raw.lower() == "q":
            return "quit"
        if raw == "s":
            return "skip"
        if raw == "S":
            return "skip_file"
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(valid_labels):
                return valid_labels[idx]
            print(f"  {RED}Out of range.{R}")
            continue
        if raw in label_set:
            return raw
        # fuzzy: prefix match (preserve schema order)
        matches = [l for l in valid_labels if l.startswith(raw)]
        if len(matches) == 1:
            confirm = input(f"  Did you mean '{matches[0]}'? [Y/n]: ").strip().lower()
            if confirm in ("", "y"):
                return matches[0]
        elif len(matches) > 1:
            print(f"  {YELLOW}Ambiguous: {matches}{R}")
            continue
        print(f"  {RED}Unknown label '{raw}'.{R}")


# ── edit command ──────────────────────────────────────────────────────────────

def _collect_issues(valid_nodes, valid_edges, fix_nodes, fix_edges, only_missing):
    """Return list of (fname, path, doc, issues_in_file) grouped by file."""
    node_set = set(valid_nodes)
    edge_set = set(valid_edges)
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    result = []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            doc = json.load(f)

        issues = []
        if fix_nodes:
            for node in doc.get("nodes", []):
                label = node.get("label")
                if only_missing:
                    if label is None:
                        issues.append(("node", node))
                else:
                    if label not in node_set:
                        issues.append(("node", node))

        if fix_edges:
            for edge in doc.get("edges", []):
                label = edge.get("label")
                if only_missing:
                    if label is None:
                        issues.append(("edge", edge))
                else:
                    if label not in edge_set:
                        issues.append(("edge", edge))

        if issues:
            result.append((fname, path, doc, issues))
    return result


def cmd_edit(args):
    valid_nodes, valid_edges = load_schema_labels()

    fix_nodes = not args.edges_only
    fix_edges = not args.nodes_only
    only_missing = args.missing_only

    file_groups = _collect_issues(valid_nodes, valid_edges, fix_nodes, fix_edges, only_missing)

    total_issues = sum(len(iss) for _, _, _, iss in file_groups)
    if total_issues == 0:
        print("No issues found.")
        return

    scope = "null/missing" if only_missing else "missing/invalid"
    print(f"{BOLD}Found {total_issues} {scope} labels across {len(file_groups)} file(s).{R}")
    print(f"  {BOLD}s{R} = skip item   {BOLD}S{R} = skip rest of file   {BOLD}q{R} = quit & save\n")

    done = skipped = 0
    for fname, path, doc, issues in file_groups:
        node_index = _build_node_index(doc)
        file_dirty = False
        skip_file = False

        for kind, obj in issues:
            if skip_file:
                skipped += 1
                continue

            current = obj.get("label")
            total_done = done + skipped
            print(f"{BG_DARK}{BOLD} [{total_done + 1}/{total_issues}]  {fname}  {kind}={obj['id']} {R}")

            if kind == "node":
                _display_node_context(doc, obj, node_index)
                choice = _prompt_label(valid_nodes, current, "node")
            else:
                _display_edge_context(doc, obj, node_index)
                choice = _prompt_label(valid_edges, current, "edge")

            if choice == "quit":
                if file_dirty:
                    _save(path, doc)
                print(f"\n{GREEN}Saved. Exiting.{R}  done={done} skipped={skipped}")
                return
            elif choice == "skip_file":
                skip_file = True
                skipped += 1
            elif choice == "skip":
                skipped += 1
            else:
                obj["label"] = choice
                file_dirty = True
                done += 1
                print(f"  {GREEN}✓ set to '{choice}'{R}\n")

        if file_dirty:
            _save(path, doc)

    print(f"\n{GREEN}Done.{R}  edited={done} skipped={skipped}")


def _save(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=4, ensure_ascii=False)
    print(f"  {DIM}saved {os.path.basename(path)}{R}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan and edit node/edge labels against schema/*.yaml"
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("scan", help="List all labels not in schema (default)")

    p_edit = sub.add_parser("edit", help="Interactively fix missing/invalid labels")
    scope = p_edit.add_mutually_exclusive_group()
    scope.add_argument("--nodes-only", action="store_true", help="Only fix node labels")
    scope.add_argument("--edges-only", action="store_true", help="Only fix edge labels")
    p_edit.add_argument(
        "--missing-only", action="store_true",
        help="Restrict to null/missing labels only (default: all out-of-schema labels)"
    )

    args = parser.parse_args()
    if args.cmd == "edit":
        cmd_edit(args)
    else:
        cmd_scan(args)


if __name__ == "__main__":
    main()
