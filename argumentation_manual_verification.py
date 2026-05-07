"""CLI tool for manually verifying argumentation matching precision.

Samples 30 files from argumentation_results/, then for each file
presents all matched (node, argument) pairs and asks the user to
judge whether each match is valid. Measures precision across the session.

Saves results to argumentation_annotations/.
"""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("argumentation_results")
DATA_DIR = Path("data/v1_llm_gemini-3.1-pro-preview")
ANNOT_DIR = Path("argumentation_annotations")
TARGET_FILES = 30
TARGET_NODES = 50   # total nodes to evaluate across all sampled files
RANDOM_SEED = 42


def select_files_and_nodes(rng: random.Random) -> dict[Path, list]:
    """Sample TARGET_FILES files, then globally sample TARGET_NODES nodes.

    Returns a dict mapping each selected file path to the list of node dicts
    that should be evaluated.
    """
    candidates = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        nodes_with_args = [
            n for n in data.get("nodes", [])
            if n.get("argument_list")
        ]
        if nodes_with_args:
            candidates.append((path, nodes_with_args))

    print(f"Found {len(candidates)} files with at least one matched argument.")

    selected_files = (
        candidates if len(candidates) <= TARGET_FILES
        else rng.sample(candidates, TARGET_FILES)
    )
    selected_files.sort(key=lambda x: x[0])

    # Build global pool of (file_path, node) and sample TARGET_NODES from it
    pool = [(path, n) for path, nodes in selected_files for n in nodes]
    sample_size = min(TARGET_NODES, len(pool))
    sampled = rng.sample(pool, sample_size)

    # Group back by file, preserving file order
    file_order = [path for path, _ in selected_files]
    nodes_by_file: dict[Path, list] = {path: [] for path, _ in selected_files}
    for path, node in sampled:
        nodes_by_file[path].append(node)

    # Sort nodes within each file by node_id for consistency
    for path in nodes_by_file:
        nodes_by_file[path].sort(key=lambda n: n["node_id"])

    # Remove files with no sampled nodes
    result = {p: ns for p, ns in nodes_by_file.items() if ns}
    print(f"Selected {len(result)} file(s), {sample_size} node(s) total.\n")
    return result


def prompt_ysn(prompt: str) -> str:
    """Return 'yes', 'somewhat', or 'no'."""
    while True:
        answer = input(prompt).strip()
        if answer == "Y":
            return "yes"
        if answer == "S":
            return "somewhat"
        if answer == "N":
            return "no"
        print("  Please enter Y, S, or N (capital only).")


def annotate_file(path: Path, sample_nodes: list, session_id: str) -> dict | None:
    stem = path.stem  # e.g. "argkp-0_DeepSeek-R1"

    if not sample_nodes:
        return None

    print()
    print("=" * 70)
    print(f"FILE: {stem}")
    topic = sample_nodes[0].get("topic", "(unknown topic)")
    print(f"TOPIC: {topic}")
    print(f"Evaluating {len(sample_nodes)} node(s)")
    print("=" * 70)

    node_results = []

    for node_idx, node in enumerate(sample_nodes, 1):
        node_id = node["node_id"]
        node_label = node.get("node_label", "?")
        node_text = node.get("node_text", "").strip()
        arg_list = node.get("argument_list", [])

        print()
        print(f"  ── Node {node_idx}/{len(sample_nodes)}  [{node_label}]  id={node_id}")
        preview = node_text[:300].replace("\n", " ")
        if len(node_text) > 300:
            preview += "..."
        print(f"  Node text: {preview}")
        print(f"  ({len(arg_list)} matched argument(s))")

        arg_results = []
        for arg_idx, arg in enumerate(arg_list, 1):
            matched = arg.get("matched_argument", "").strip()
            mace_p = arg.get("mace_p")
            stance_raw = arg.get("stance")
            stance_str = "for" if stance_raw == 1 else ("against" if stance_raw == -1 else "?")

            print()
            print(f"    [{arg_idx}/{len(arg_list)}] Matched argument (stance={stance_str}, mace_p={mace_p:.3f}):")
            print(f"    \"{matched}\"")
            print()

            verdict = prompt_ysn("    Is this a valid match for the node text? [Y/S/N]: ")
            score = {"yes": 1.0, "somewhat": 0.5, "no": 0.0}[verdict]
            arg_results.append({
                "matched_argument": matched,
                "mace_p": mace_p,
                "wa": arg.get("wa"),
                "stance": stance_raw,
                "human_verdict": verdict,
                "score": score,
            })
            print(f"    -> Recorded: {verdict}")

        node_score = sum(r["score"] for r in arg_results)
        node_results.append({
            "node_id": node_id,
            "node_label": node_label,
            "node_text": node_text,
            "argument_results": arg_results,
            "total_matched": len(arg_results),
            "n_yes": sum(1 for r in arg_results if r["human_verdict"] == "yes"),
            "n_somewhat": sum(1 for r in arg_results if r["human_verdict"] == "somewhat"),
            "n_no": sum(1 for r in arg_results if r["human_verdict"] == "no"),
            "score": node_score,
        })

    total_matched = sum(r["total_matched"] for r in node_results)
    total_score = sum(r["score"] for r in node_results)
    n_yes = sum(r["n_yes"] for r in node_results)
    n_somewhat = sum(r["n_somewhat"] for r in node_results)
    n_no = sum(r["n_no"] for r in node_results)
    precision = total_score / total_matched if total_matched else None

    print()
    print(f"  File: Y={n_yes}  S={n_somewhat}  N={n_no}  "
          + (f"precision={precision:.2f}" if precision is not None else "precision=n/a"))

    return {
        "session_id": session_id,
        "filename": stem,
        "topic": topic,
        "node_results": node_results,
        "total_matched": total_matched,
        "n_yes": n_yes,
        "n_somewhat": n_somewhat,
        "n_no": n_no,
        "score": total_score,
        "precision": precision,
    }


def save_result(result: dict, session_id: str):
    ANNOT_DIR.mkdir(exist_ok=True)
    out_path = ANNOT_DIR / f"{session_id}_{result['filename']}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_path}")


def main():
    rng = random.Random(RANDOM_SEED)

    print("Argumentation Match Verification Tool")
    print(f"Selecting up to {TARGET_FILES} files, {TARGET_NODES} nodes total...\n")

    nodes_by_file = select_files_and_nodes(rng)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_args_est = sum(
        sum(len(n.get("argument_list", [])) for n in nodes)
        for nodes in nodes_by_file.values()
    )
    print(f"Session ID: {session_id}")
    print(f"Files with sampled nodes : {len(nodes_by_file)}")
    print(f"Total nodes to evaluate  : {sum(len(v) for v in nodes_by_file.values())}")
    print(f"Total argument judgements: ~{total_args_est}")
    print()
    print("Instructions:")
    print("  - For each matched argument:")
    print("      Y = valid/equivalent match")
    print("      S = somewhat related but not equivalent")
    print("      N = not a match")
    print("  - Capital Y, S, or N only.")
    print("  - Results are saved after each file.")
    print("  - Press Ctrl+C at any time to stop; progress up to that point is saved.\n")

    total_matched = 0
    total_score = 0.0
    total_yes = total_somewhat = total_no = 0
    files_done = 0
    file_list = sorted(nodes_by_file.keys())

    for idx, path in enumerate(file_list, 1):
        print(f"\n--- File {idx}/{len(file_list)} ---")
        try:
            result = annotate_file(path, nodes_by_file[path], session_id)
            if result is None:
                print("  (no matched arguments — skipping)")
                continue
            save_result(result, session_id)
            total_matched += result["total_matched"]
            total_score += result["score"]
            total_yes += result["n_yes"]
            total_somewhat += result["n_somewhat"]
            total_no += result["n_no"]
            files_done += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved up to this point.")
            break

    print("\n" + "=" * 70)
    print("Session complete.")
    print(f"  Files evaluated   : {files_done}")
    print(f"  Arguments shown   : {total_matched}")
    print(f"  Yes / Somewhat / No: {total_yes} / {total_somewhat} / {total_no}")
    if total_matched > 0:
        print(f"  Overall precision : {total_score / total_matched:.2f}  "
              f"(Y=1, S=0.5, N=0)")
    print(f"  Results saved in  : {ANNOT_DIR}/")


if __name__ == "__main__":
    main()
