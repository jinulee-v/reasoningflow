"""
Count local verification frequency per trace across 5×3 (model, dataset) combinations.

Local verification: a fact/restatement/reasoning node that has at least one outgoing
validate: or plan:verify edge.
Also prints the number of conclusion blocks per trace, where consecutive conclusion nodes
(ordered by start offset) are merged into a single block.
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"

MODEL_ORDER = ["QwQ-32B", "DeepSeek-R1", "gpt-oss-120b", "DeepSeek-V3", "Qwen2.5-32B-Instruct"]
DATASET_ORDER = ["aime2024", "gpqa-diamond", "argkp"]


def count_conclusion_blocks(nodes):
    """Count runs of consecutive conclusion nodes ordered by start offset."""
    ordered = sorted(nodes.values(), key=lambda n: n.get("start", 0))
    blocks = 0
    in_block = False
    for n in ordered:
        if n.get("label") == "conclusion":
            if not in_block:
                blocks += 1
                in_block = True
        else:
            in_block = False
    return blocks


LV_NODE_LABELS = {"fact", "restatement", "reasoning"}


def analyze_trace(datum):
    """Return (local_verify_count, conclusion_block_count) for a single trace."""
    nodes = {n["id"]: n for n in datum.get("nodes", [])}

    # Collect node IDs that have at least one outgoing validate: or plan:verify edge
    has_lv_out = set()
    for e in datum.get("edges", []):
        label = e.get("label", "")
        if label.startswith("validate:") or label == "plan:verify":
            has_lv_out.add(e["source_node_id"])

    local_verify_count = sum(
        1 for nid in has_lv_out
        if nodes.get(nid, {}).get("label") in LV_NODE_LABELS
    )
    return local_verify_count, count_conclusion_blocks(nodes)


def load_and_analyze():
    # stats[model][dataset] = list of (local_verify_count, conclusion_count) per trace
    stats = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".json"):
            continue
        parts = fname.replace(".json", "").split("_")
        # filename format: {dataset}_{idx}_{model}.json
        # dataset may contain "-" (e.g. gpqa-diamond), model may contain "-" too
        # parts[0] is always a single token; parts[-1] and parts[-2] form the model
        # more robustly: split on first "_<digits>_" to get dataset and model
        import re
        m = re.match(r"^(.+?)_\d+_(.+)\.json$", fname)
        if not m:
            continue
        dataset, model = m.group(1), m.group(2)

        with open(os.path.join(DATA_DIR, fname)) as f:
            datum = json.load(f)

        lv, conc = analyze_trace(datum)
        stats[model][dataset].append((lv, conc))

    return stats


def print_results(stats):
    # Per-trace detail header
    print("=" * 110)
    print(f"{'Model':<30} {'Dataset':<15} {'Traces':>7} "
          f"{'LV/trace (avg)':>16} {'LV/trace (sum)':>15} "
          f"{'ConcBlk/trace (avg)':>20} {'ConcBlk/trace (sum)':>20}")
    print("-" * 110)

    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            traces = stats.get(model, {}).get(dataset, [])
            if not traces:
                continue
            lv_counts = [t[0] for t in traces]
            conc_counts = [t[1] for t in traces]
            n = len(traces)
            print(
                f"{model:<30} {dataset:<15} {n:>7} "
                f"{sum(lv_counts)/n:>16.3f} {sum(lv_counts):>15} "
                f"{sum(conc_counts)/n:>20.3f} {sum(conc_counts):>20}"
            )
        print()

    # Per-dataset aggregate
    print("=" * 110)
    print("DATASET TOTALS")
    print("-" * 110)
    print(f"{'Dataset':<15} {'Traces':>7} {'LV/trace (avg)':>16} {'ConcBlk/trace (avg)':>20}")
    print("-" * 110)
    for dataset in DATASET_ORDER:
        all_traces = [t for model in MODEL_ORDER for t in stats.get(model, {}).get(dataset, [])]
        if not all_traces:
            continue
        n = len(all_traces)
        avg_lv = sum(t[0] for t in all_traces) / n
        avg_conc = sum(t[1] for t in all_traces) / n
        print(f"{dataset:<15} {n:>7} {avg_lv:>16.3f} {avg_conc:>20.3f}")

    # Per-model aggregate
    print()
    print("MODEL TOTALS")
    print("-" * 110)
    print(f"{'Model':<30} {'Traces':>7} {'LV/trace (avg)':>16} {'ConcBlk/trace (avg)':>20}")
    print("-" * 110)
    for model in MODEL_ORDER:
        all_traces = [t for ds in DATASET_ORDER for t in stats.get(model, {}).get(ds, [])]
        if not all_traces:
            continue
        n = len(all_traces)
        avg_lv = sum(t[0] for t in all_traces) / n
        avg_conc = sum(t[1] for t in all_traces) / n
        print(f"{model:<30} {n:>7} {avg_lv:>16.3f} {avg_conc:>20.3f}")


def plot_grouped_bar(stats, out_path="plots/localverify_grouped_bar.svg"):
    PLOT_GROUPS = [
        ("aime2024", "QwQ-32B",    "AIME-QwQ"),
        ("aime2024", "DeepSeek-R1","AIME-R1"),
        ("gpqa-diamond", "QwQ-32B",    "GPQA-QwQ"),
        ("gpqa-diamond", "DeepSeek-R1","GPQA-R1"),
    ]

    lv_avgs, gv_avgs, labels = [], [], []
    for dataset, model, label in PLOT_GROUPS:
        traces = stats.get(model, {}).get(dataset, [])
        n = len(traces)
        lv_avgs.append(sum(t[0] for t in traces) / n if n else 0)
        # global verification = avg(conclusion_blocks - 1), floored at 0
        gv_avgs.append(sum(max(t[1] - 1, 0) for t in traces) / n if n else 0)
        labels.append(label)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_lv = ax.bar(x - width / 2, lv_avgs, width, label="Local verification",  color="#ffc6c6")
    bars_gv = ax.bar(x + width / 2, gv_avgs, width, label="Global verification", color="#c3b1e1")

    ax.bar_label(bars_lv, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars_gv, fmt="%.2f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg count per trace")
    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    stats = load_and_analyze()
    print_results(stats)
    plot_grouped_bar(stats)
