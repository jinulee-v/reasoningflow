"""CLI tool for manually annotating PARC evaluation results.

Selects 30 files with 0 < errors <= 10 (undecidable excluded),
prompts Y/N for each flagged node, saves results to parc_annotations/.
"""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

PARC_DIR = Path("parc_results")
DATA_DIR = Path("data/v1_llm_gemini-3.1-pro-preview")
ANNOT_DIR = Path("parc_annotations")
TARGET_FILES = 30
MAX_ERRORS = 10
RANDOM_SEED = 42


def count_errors(parc_data: list) -> int:
    return sum(1 for item in parc_data if item["correctness"] == "error")


def load_node_text(filename: str, node_id: str) -> str | None:
    data_path = DATA_DIR / filename
    if not data_path.exists():
        return None
    try:
        with open(data_path) as f:
            data = json.load(f)
        for node in data.get("nodes", []):
            if node["id"] == node_id:
                return node.get("text", "")
    except Exception:
        pass
    return None


def select_files() -> list[Path]:
    candidates = []
    for path in sorted(PARC_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        n_errors = count_errors(data)
        if 0 < n_errors <= MAX_ERRORS:
            candidates.append((path, n_errors))

    print(f"Found {len(candidates)} files with 0 < errors <= {MAX_ERRORS}.")
    if len(candidates) < TARGET_FILES:
        print(f"Warning: only {len(candidates)} qualifying files found; using all.")
        selected = candidates
    else:
        rng = random.Random(RANDOM_SEED)
        selected = rng.sample(candidates, TARGET_FILES)
        selected.sort(key=lambda x: x[0].name)

    return selected


def prompt_yns(prompt: str) -> str:
    while True:
        answer = input(prompt).strip()
        if answer in ("Y", "S", "N"):
            return answer
        print("  Please enter Y, S, or N (capital only).")


def annotate_file(path: Path, parc_data: list, session_id: str) -> dict:
    errors = [item for item in parc_data if item["correctness"] == "error"]
    filename = path.name
    results = []

    print()
    print("=" * 70)
    print(f"FILE: {filename}  ({len(errors)} errors)")
    print("=" * 70)

    for i, item in enumerate(errors, 1):
        node_id = item["node_id"]
        reason = item["chain_of_thought"]

        print()
        print(f"  [{i}/{len(errors)}] Node: {node_id}")

        node_text = load_node_text(filename, node_id)
        if node_text:
            preview = node_text[:200].replace("\n", " ")
            if len(node_text) > 200:
                preview += "..."
            print(f"  Text: {preview}")

        print(f"  Reason: {reason}")
        print()

        answer = prompt_yns("  Is this really an error? [Y/S/N] (Y=yes, S=somewhat, N=no): ")
        verdict_map = {"Y": "error", "S": "somewhat", "N": "correct"}
        label_map = {"Y": "confirmed error", "S": "somewhat an error", "N": "not an error"}
        results.append({
            "node_id": node_id,
            "llm_verdict": "error",
            "human_verdict": verdict_map[answer],
            "confirmed": answer == "Y",
            "chain_of_thought": reason,
        })
        print(f"  -> Recorded: {label_map[answer]}")

    return {
        "session_id": session_id,
        "filename": filename,
        "annotations": results,
        "total_errors_flagged": len(errors),
        "confirmed_errors": sum(1 for r in results if r["confirmed"]),
        "partial_errors": sum(1 for r in results if r["human_verdict"] == "somewhat"),
    }


def save_result(result: dict, session_id: str):
    ANNOT_DIR.mkdir(exist_ok=True)
    out_path = ANNOT_DIR / f"{session_id}_{result['filename']}"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_path}")


def print_stats():
    if not ANNOT_DIR.exists():
        print(f"No annotations directory found at {ANNOT_DIR}/")
        sys.exit(1)

    files = sorted(ANNOT_DIR.glob("*.json"))
    if not files:
        print(f"No annotation files found in {ANNOT_DIR}/")
        sys.exit(1)

    total_flagged = 0
    total_confirmed = 0
    total_partial = 0

    for path in files:
        with open(path) as f:
            result = json.load(f)
        total_flagged += result["total_errors_flagged"]
        total_confirmed += result["confirmed_errors"]
        total_partial += result["partial_errors"]

    print("=" * 70)
    print(f"Stats from {ANNOT_DIR}/  ({len(files)} files)")
    print("=" * 70)
    print(f"  Files annotated  : {len(files)}")
    print(f"  Errors flagged   : {total_flagged}")
    print(f"  Errors confirmed : {total_confirmed}")
    print(f"  Errors somewhat  : {total_partial}")
    if total_flagged:
        precision = (total_confirmed + total_partial) / total_flagged
        print(f"  Precision (Y+S)  : {precision:.1%}")


def main():
    if "--stats" in sys.argv:
        print_stats()
        return

    print("PARC Annotation Tool")
    print(f"Selecting {TARGET_FILES} files with 0 < errors <= {MAX_ERRORS}...\n")

    selected = select_files()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nSession ID: {session_id}")
    print(f"Annotating {len(selected)} files.\n")
    print("Instructions:")
    print("  - For each flagged node, enter Y to confirm it is an error,")
    print("    S if it is somewhat an error, or N if the LLM was wrong.")
    print("  - Capital Y, S, or N only.")
    print("  - Results are saved after each file.")

    total_flagged = 0
    total_confirmed = 0
    total_partial = 0

    for idx, (path, n_errors) in enumerate(selected, 1):
        print(f"\n--- File {idx}/{len(selected)} ---")
        with open(path) as f:
            parc_data = json.load(f)

        try:
            result = annotate_file(path, parc_data, session_id)
            save_result(result, session_id)
            total_flagged += result["total_errors_flagged"]
            total_confirmed += result["confirmed_errors"]
            total_partial += result["partial_errors"]
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved up to this point.")
            break

    print("\n" + "=" * 70)
    print("Session complete.")
    print(f"  Files annotated  : {idx}")
    print(f"  Errors flagged   : {total_flagged}")
    print(f"  Errors confirmed : {total_confirmed}")
    print(f"  Errors somewhat  : {total_partial}")
    print(f"  Results saved in: {ANNOT_DIR}/")


if __name__ == "__main__":
    main()
