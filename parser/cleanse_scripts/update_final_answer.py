"""
Measure model accuracy on GPQA-Diamond and AIME 2024 using annotated data
in data/v1_llm_gemini-3-pro-preview/.

Strategy: extract the model's answer from the last contiguous run of
"conclusion" nodes (the final conclusion bulb).
  - GPQA-Diamond: find the first A/B/C/D letter mentioned in the bulb.
  - AIME 2024: find the last \\boxed{...} value, or fall back to the
    last standalone integer in the bulb.
"""

import argparse
import json
import re
import glob
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data/v1_raw_data")

# Build GPQA index → topic mapping
_gpqa_ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
GPQA_TOPIC = {
    i: row["High-level domain"].lower()
    for i, row in enumerate(_gpqa_ds)
}

DOMAIN_TOPIC = {
    "aime2024": "math",
    "bright-theoremqa_theorems": "math",
    "argkp": "argumentation",
}


def get_topic(domain: str, idx: int | None) -> str | None:
    if domain in DOMAIN_TOPIC:
        return DOMAIN_TOPIC[domain]
    if domain == "gpqa-diamond" and idx is not None:
        return GPQA_TOPIC.get(idx)
    return None


def get_last_conclusion_bulb(nodes: list) -> list:
    """Return the trailing run of nodes whose label is 'conclusion'."""
    i = len(nodes) - 1
    bulb = []
    while i >= 0 and nodes[i]["label"] != "conclusion":
        i -= 1
    while i >= 0 and nodes[i]["label"] == "conclusion":
        bulb.insert(0, nodes[i])
        i -= 1
    return bulb


def extract_gpqa_answer(bulb_text: str) -> str | None:
    """Return the last A/B/C/D letter found in the conclusion bulb."""
    m = re.findall(r"\b([A-D])\b", bulb_text)
    return m[-1] if m else None


def extract_aime_answer(bulb_text: str) -> str | None:
    """Return the last \\boxed{} value; fall back to the last integer."""
    boxes = re.findall(r"\\boxed\{([^}]+)\}", bulb_text)
    if boxes:
        return boxes[-1].strip()
    integers = re.findall(r"\b(\d+)\b", bulb_text)
    return integers[-1] if integers else None


EXTRACTORS = {
    "gpqa-diamond": extract_gpqa_answer,
    "aime2024": extract_aime_answer,
}

TARGET_DOMAINS = set(EXTRACTORS)


def main():
    parser = argparse.ArgumentParser(
        description="Update correctness metadata and report model accuracy."
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR), metavar="DIR",
        help=f"Directory of JSON files to process (default: {DATA_DIR}).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # stats[domain][generator] = {"correct": int, "wrong": int, "no_bulb": int}
    stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "wrong": 0, "no_bulb": 0})
    )

    for path in sorted(data_dir.glob("*.json")):
        with open(path) as f:
            datum = json.load(f)

        meta = datum["metadata"]
        domain = meta.get("domain", "")

        # Extract numeric index from filename (e.g. "gpqa-diamond_42_Model.json" → 42)
        name_parts = path.stem.split("_")
        try:
            file_idx = int(name_parts[1])
        except (IndexError, ValueError):
            file_idx = None

        meta["topic"] = get_topic(domain, file_idx)

        if "aime2024" not in path.name and "gpqa-diamond" not in path.name:
            meta["correctness"] = None
            with open(path, "w") as f:
                json.dump(datum, f, ensure_ascii=False, indent=4)
            continue
        # if domain not in TARGET_DOMAINS:
        #     continue

        generator = meta.get("generator", "unknown")
        correct_answer = meta.get("correct_answer")
        if not correct_answer:
            continue

        nodes = datum.get("nodes", [])
        bulb = get_last_conclusion_bulb(nodes)
        if bulb:
            bulb_text = " ".join(n["text"] for n in bulb)
        else:
            # Last 100 characters of the full text
            bulb_text = datum["raw_text"]["response"][-100:]

        predicted = EXTRACTORS[domain](bulb_text)

        is_correct = predicted == correct_answer
        meta["correctness"] = is_correct

        with open(path, "w") as f:
            json.dump(datum, f, ensure_ascii=False, indent=4)

        if is_correct:
            stats[domain][generator]["correct"] += 1
        else:
            stats[domain][generator]["wrong"] += 1

    # ── Report ───────────────────────────────────────────────────────────────────

    for domain in sorted(TARGET_DOMAINS):
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")
        print(f"{'Generator':<30} {'Correct':>7} {'Total':>7} {'Accuracy':>9} {'No-bulb':>8}")
        print(f"{'-'*30} {'-'*7} {'-'*7} {'-'*9} {'-'*8}")

        domain_correct = domain_total = domain_no_bulb = 0
        for generator in sorted(stats[domain]):
            s = stats[domain][generator]
            correct = s["correct"]
            total = s["correct"] + s["wrong"]
            no_bulb = s["no_bulb"]
            acc = f"{correct / total:.1%}" if total else "N/A"
            print(f"{generator:<30} {correct:>7} {total:>7} {acc:>9} {no_bulb:>8}")
            domain_correct += correct
            domain_total += total
            domain_no_bulb += no_bulb

        total_acc = f"{domain_correct / domain_total:.1%}" if domain_total else "N/A"
        print(f"{'TOTAL':<30} {domain_correct:>7} {domain_total:>7} {total_acc:>9} {domain_no_bulb:>8}")


if __name__ == "__main__":
    main()
