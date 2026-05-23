"""
Cleanse common graph annotation errors.

Pattern 1: reflect-then-plan:verify
  A --reflect:??--> reflection --plan:verify--> planning

  Fix:
    - Remove:  reflection --plan:verify--> planning
    - Add:     A          --plan:verify--> planning
    - Add:     reflection --plan:proceed--> planning
"""

import argparse
import json
import os

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"

# ANSI colors
R = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"


# ── pattern detection ─────────────────────────────────────────────────────────

def _find_pattern1(doc):
    """
    Find all instances of:
      A --reflect:??--> reflection --plan:verify--> planning

    Returns list of dicts:
      {
        "reflect_edge":   edge A -> reflection,
        "planverify_edge": edge reflection -> planning,
        "A_node":         node A,
        "reflection_node": node reflection,
        "planning_node":  node planning,
      }
    """
    nodes_by_id = {n["id"]: n for n in doc.get("nodes", [])}
    edges = doc.get("edges", [])

    # Index: dest_node_id -> list of edges arriving there
    incoming = {}
    for e in edges:
        incoming.setdefault(e["dest_node_id"], []).append(e)

    results = []
    for edge in edges:
        if edge["label"] != "plan:verify":
            continue
        src = nodes_by_id.get(edge["source_node_id"])
        dst = nodes_by_id.get(edge["dest_node_id"])
        if src is None or dst is None:
            continue
        if src.get("label") != "reflection" or dst.get("label") != "planning":
            continue

        # src is the reflection node; look for a reflect:* edge pointing to it
        for in_edge in incoming.get(src["id"], []):
            if in_edge["label"].startswith("reflect:"):
                A_node = nodes_by_id.get(in_edge["source_node_id"])
                if A_node is None:
                    continue
                results.append({
                    "reflect_edge": in_edge,
                    "planverify_edge": edge,
                    "A_node": A_node,
                    "reflection_node": src,
                    "planning_node": dst,
                })
    return results


# ── display helpers ───────────────────────────────────────────────────────────

def _trunc(text, width=80):
    text = (text or "").replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _print_pattern(match, idx=None, total=None):
    header = f"[{idx}/{total}] " if idx is not None else ""
    A = match["A_node"]
    ref = match["reflection_node"]
    pln = match["planning_node"]
    re = match["reflect_edge"]

    print(f"\n{BOLD}{CYAN}{header}Pattern: reflect-then-plan:verify{R}")
    print(f"  {BOLD}A{R}          [{A['label']}] {DIM}{_trunc(A['text'])}{R}")
    print(f"  {YELLOW}──[{re['label']}]──►{R}")
    print(f"  reflection [{ref['label']}] {DIM}{_trunc(ref['text'])}{R}")
    print(f"  {YELLOW}──[plan:verify]──►{R}")
    print(f"  planning   [{pln['label']}] {DIM}{_trunc(pln['text'])}{R}")
    print()
    print(f"  {GREEN}Fix:{R}")
    print(f"    Remove:  reflection ──[plan:verify]──► planning")
    print(f"    Add:     A          ──[plan:verify]──► planning")
    print(f"    Add:     reflection ──[plan:proceed]──► planning")


# ── fix ───────────────────────────────────────────────────────────────────────

def _new_edge_id(existing_ids):
    """Generate a unique edge id not already in existing_ids."""
    i = 0
    while True:
        candidate = f"e_patch_{i}"
        if candidate not in existing_ids:
            return candidate
        i += 1


def _apply_pattern1(doc, match):
    """Apply the fix for one pattern match in-place. Returns True."""
    pve = match["planverify_edge"]
    A_node = match["A_node"]
    planning_node = match["planning_node"]
    reflection_node = match["reflection_node"]

    edges = doc["edges"]
    existing_ids = {e["id"] for e in edges}

    # Remove the reflection --plan:verify--> planning edge
    doc["edges"] = [e for e in edges if e["id"] != pve["id"]]
    existing_ids.discard(pve["id"])

    def edge_exists(src_id, dst_id, label):
        return any(
            e["source_node_id"] == src_id
            and e["dest_node_id"] == dst_id
            and e["label"] == label
            for e in doc["edges"]
        )

    # Add A --plan:verify--> planning (if not already present)
    if not edge_exists(A_node["id"], planning_node["id"], "plan:verify"):
        id1 = _new_edge_id(existing_ids)
        existing_ids.add(id1)
        doc["edges"].append({
            "id": id1,
            "source_node_id": A_node["id"],
            "dest_node_id": planning_node["id"],
            "label": "plan:verify",
        })

    # Add reflection --plan:proceed--> planning (if not already present)
    if not edge_exists(reflection_node["id"], planning_node["id"], "plan:proceed"):
        id2 = _new_edge_id(existing_ids)
        doc["edges"].append({
            "id": id2,
            "source_node_id": reflection_node["id"],
            "dest_node_id": planning_node["id"],
            "label": "plan:proceed",
        })

    return True


# ── pattern 2 ─────────────────────────────────────────────────────────────────

def _find_pattern2(doc):
    """
    Find all instances of:
      A --plan:decompose--> B
      A --plan:decompose--> C
      B --plan:proceed-->   C

    Returns list of dicts:
      {
        "decompose_AB": edge A -> B,
        "decompose_AC": edge A -> C,
        "proceed_BC":   edge B -> C,
        "A_node", "B_node", "C_node",
      }
    """
    nodes_by_id = {n["id"]: n for n in doc.get("nodes", [])}
    edges = doc.get("edges", [])

    # decompose_children[A] = list of destination node ids
    decompose_children: dict[str, list] = {}
    decompose_edge: dict[tuple, object] = {}  # (src, dst) -> edge
    proceed_edge: dict[tuple, object] = {}    # (src, dst) -> edge

    for e in edges:
        if e["label"] == "plan:decompose":
            decompose_children.setdefault(e["source_node_id"], []).append(e["dest_node_id"])
            decompose_edge[(e["source_node_id"], e["dest_node_id"])] = e
        elif e["label"] == "plan:proceed":
            proceed_edge[(e["source_node_id"], e["dest_node_id"])] = e

    results = []
    for A_id, children in decompose_children.items():
        for i, B_id in enumerate(children):
            for C_id in children[i + 1:]:
                # Check B --plan:proceed--> C
                if (B_id, C_id) in proceed_edge:
                    results.append({
                        "decompose_AB": decompose_edge[(A_id, B_id)],
                        "decompose_AC": decompose_edge[(A_id, C_id)],
                        "proceed_BC": proceed_edge[(B_id, C_id)],
                        "A_node": nodes_by_id[A_id],
                        "B_node": nodes_by_id[B_id],
                        "C_node": nodes_by_id[C_id],
                    })
                # Also check C --plan:proceed--> B
                elif (C_id, B_id) in proceed_edge:
                    results.append({
                        "decompose_AB": decompose_edge[(A_id, C_id)],
                        "decompose_AC": decompose_edge[(A_id, B_id)],
                        "proceed_BC": proceed_edge[(C_id, B_id)],
                        "A_node": nodes_by_id[A_id],
                        "B_node": nodes_by_id[C_id],
                        "C_node": nodes_by_id[B_id],
                    })
    return results


def _apply_pattern2(doc, match):
    """Remove the A --plan:decompose--> C edge."""
    ac_id = match["decompose_AC"]["id"]
    doc["edges"] = [e for e in doc["edges"] if e["id"] != ac_id]


def _print_pattern2(match, idx=None, total=None):
    header = f"[{idx}/{total}] " if idx is not None else ""
    A = match["A_node"]
    B = match["B_node"]
    C = match["C_node"]

    print(f"\n{BOLD}{CYAN}{header}Pattern 2: plan:decompose + plan:proceed sibling{R}")
    print(f"  {BOLD}A{R}  [{A['label']}] {DIM}{_trunc(A['text'])}{R}")
    print(f"  {YELLOW}├─[plan:decompose]──►{R}")
    print(f"  {BOLD}B{R}  [{B['label']}] {DIM}{_trunc(B['text'])}{R}")
    print(f"  {YELLOW}│  └─[plan:proceed]──►{R}")
    print(f"  {YELLOW}└─[plan:decompose]──►{R}")
    print(f"  {BOLD}C{R}  [{C['label']}] {DIM}{_trunc(C['text'])}{R}")


# ── pattern 3 ─────────────────────────────────────────────────────────────────

def _find_pattern3(doc):
    """
    Find all instances of:
      A --reason:infer-->   C
      B --reason:execute--> C
      B and C are adjacent in the node list

    Returns list of dicts:
      {
        "infer_AC":    edge A -> C,
        "execute_BC":  edge B -> C,
        "A_node", "B_node", "C_node",
      }
    """
    nodes = doc.get("nodes", [])
    nodes_by_id = {n["id"]: n for n in nodes}
    node_index = {n["id"]: i for i, n in enumerate(nodes)}
    edges = doc.get("edges", [])

    # Index edges by label and destination for fast lookup
    infer_by_dest: dict[str, list] = {}
    execute_by_dest: dict[str, list] = {}
    for e in edges:
        if e["label"] == "reason:infer":
            infer_by_dest.setdefault(e["dest_node_id"], []).append(e)
        elif e["label"] == "reason:execute":
            execute_by_dest.setdefault(e["dest_node_id"], []).append(e)

    results = []
    # For each C that has both an infer and an execute incoming edge
    for C_id in set(infer_by_dest) & set(execute_by_dest):
        C_idx = node_index.get(C_id)
        if C_idx is None:
            continue
        for execute_edge in execute_by_dest[C_id]:
            B_id = execute_edge["source_node_id"]
            B_idx = node_index.get(B_id)
            if B_idx is None:
                continue
            # Adjacency constraint: B must immediately precede C (canonical A->B->C order)
            if B_idx + 1 != C_idx:
                continue
            for infer_edge in infer_by_dest[C_id]:
                A_id = infer_edge["source_node_id"]
                if A_id not in nodes_by_id:
                    continue
                results.append({
                    "infer_AC": infer_edge,
                    "execute_BC": execute_edge,
                    "A_node": nodes_by_id[A_id],
                    "B_node": nodes_by_id[B_id],
                    "C_node": nodes_by_id[C_id],
                })
    return results


def _apply_pattern3(doc, match):
    """Add A --plan:proceed--> B if not already present."""
    A_id = match["A_node"]["id"]
    B_id = match["B_node"]["id"]

    existing_ids = {e["id"] for e in doc["edges"]}
    already = any(
        e["source_node_id"] == A_id and e["dest_node_id"] == B_id and e["label"] == "plan:proceed"
        for e in doc["edges"]
    )
    if not already:
        new_id = _new_edge_id(existing_ids)
        doc["edges"].append({
            "id": new_id,
            "source_node_id": A_id,
            "dest_node_id": B_id,
            "label": "plan:proceed",
        })


def _print_pattern3(match, idx=None, total=None):
    header = f"[{idx}/{total}] " if idx is not None else ""
    A = match["A_node"]
    B = match["B_node"]
    C = match["C_node"]

    print(f"\n{BOLD}{CYAN}{header}Pattern 3: reason:infer + reason:execute → same node{R}")
    print(f"  {BOLD}A{R}  [{A['label']}] {DIM}{_trunc(A['text'])}{R}")
    print(f"  {YELLOW}├─[reason:infer]─────────────────────────────►{R}")
    print(f"  {YELLOW}└─[plan:proceed (NEW)]──►{R}")
    print(f"  {BOLD}B{R}  [{B['label']}] {DIM}{_trunc(B['text'])}{R}")
    print(f"  {YELLOW}   └─[reason:execute]──►{R}")
    print(f"  {BOLD}C{R}  [{C['label']}] {DIM}{_trunc(C['text'])}{R}")


# ── main logic ────────────────────────────────────────────────────────────────

def _run_pattern(files, find_fn, print_fn, apply_fn, dry_run, max_examples):
    total_matches = 0
    total_files = 0
    shown = 0

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            doc = json.load(f)

        matches = find_fn(doc)
        if not matches:
            continue

        total_matches += len(matches)
        total_files += 1

        if dry_run:
            for m in matches:
                if max_examples is not None and shown >= max_examples:
                    break
                shown += 1
                print_fn(m, idx=shown)
                print(f"  {DIM}file: {fname}{R}")
            if max_examples is not None and shown >= max_examples:
                break
        elif apply_fn is not None:
            for m in matches:
                apply_fn(doc, m)
            with open(path, "w") as f:
                json.dump(doc, f, indent=4, ensure_ascii=False)
            print(f"  {GREEN}fixed {len(matches)} pattern(s) in {fname}{R}")

    return total_matches, total_files


def run(dry_run: bool, pattern: int, max_examples: int | None = None):
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))

    if pattern == 1:
        total, nfiles = _run_pattern(
            files, _find_pattern1, _print_pattern, _apply_pattern1,
            dry_run, max_examples,
        )
    elif pattern == 2:
        total, nfiles = _run_pattern(
            files, _find_pattern2, _print_pattern2, _apply_pattern2,
            dry_run, max_examples,
        )
    elif pattern == 3:
        total, nfiles = _run_pattern(
            files, _find_pattern3, _print_pattern3, _apply_pattern3,
            dry_run, max_examples,
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    print()
    if dry_run:
        print(f"{BOLD}Dry run (pattern {pattern}):{R} found {total} match(es) across {nfiles} file(s). No changes made.")
    else:
        print(f"{BOLD}Done (pattern {pattern}):{R} fixed {total} match(es) across {nfiles} file(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanse common graph annotation errors."
    )
    parser.add_argument(
        "--pattern", type=int, default=1, choices=[1, 2, 3], metavar="{1,2,3}",
        help="Which pattern to scan/fix (default: 1).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print matched examples without modifying any files.",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None, metavar="N",
        help="With --dry-run, stop after showing N examples.",
    )
    args = parser.parse_args()

    if args.max_examples is not None and not args.dry_run:
        parser.error("--max-examples only makes sense with --dry-run")

    run(dry_run=args.dry_run, pattern=args.pattern, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
