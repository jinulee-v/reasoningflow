"""
Analyze argumentation structure per model for the argkp dataset.

1. Find all planning nodes that collect for/against arguments using Gemini flash.
2. Analyze DAG depth/width initiated from those planning nodes (all edge types, forward BFS).
3. Determine model stance via Gemini flash from the first Conclusion node;
   compute average argumentation scores of for/against argument chains;
   plot score-diff histograms (agree vs. disagree stance) per model.
"""

import json
import glob
import os
import sys
from collections import defaultdict, deque
import statistics
from typing import List, Optional

from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

MODEL_COLORS = {
    "Qwen2.5-32B-Instruct": "#8ED7D7",
    "QwQ-32B":              "#007A84",
    "DeepSeek-V3":          "#FF8CA1",
    "DeepSeek-R1":          "#E42741",
    "gpt-oss-120b":         "#98C126",
}
PLOTS_DIR = "plots"

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"
ARGKP_PATTERN = os.path.join(DATA_DIR, "argkp_*.json")


# ── Pydantic schemas for Gemini structured output ─────────────────────────────

class HighLevelPlans(BaseModel):
    for_planning_node_ids: List[str] = []
    against_planning_node_ids: List[str] = []


class ModelStance(BaseModel):
    stance: str  # "for" or "against"


class ArgumentMatch(BaseModel):
    selected_indices: List[int] = []


# ── LLM helpers ───────────────────────────────────────────────────────────────

def find_high_level_planning_nodes(planning_nodes: list) -> dict:
    """Use Gemini flash to identify all planning nodes that collect FOR or
    AGAINST arguments.  Returns {for_planning_node_ids, against_planning_node_ids}
    where each value is a list of node IDs.
    """
    from parser.utils.vertexai import call_llm

    if not planning_nodes:
        return {"for_planning_node_ids": [], "against_planning_node_ids": []}

    node_list_str = "\n".join(
        f"ID={n['id']}: {n.get('text', '').replace(chr(10), ' ').strip()[:200]}"
        for n in planning_nodes
    )
    prompt = (
        "You are given a list of planning nodes from a model's reasoning trace on a debate topic.\n"
        "Identify **all** planning nodes that collect or list arguments:\n"
        "  1. **for_planning_node_ids**: every planning node that initiates or organises the "
        "listing of arguments **in favor of** (supporting) the topic — e.g. 'Let me list "
        "the arguments in favor of the topic', 'Arguments for:', 'First argument supporting…'.\n"
        "  2. **against_planning_node_ids**: every planning node that initiates or organises "
        "the listing of arguments **against** (counterarguments to) the topic — e.g. 'Now I "
        "will go through the arguments against', 'Counter-arguments:', 'Argument opposing…'.\n"
        "Include both high-level organising nodes AND any sub-item nodes that each introduce "
        "a single argument in the list. Exclude nodes that merely restate the topic or are "
        "unrelated to argument collection.\n"
        "Return empty lists if no such nodes exist.\n\n"
        f"Planning nodes:\n{node_list_str}\n\n"
        'Return JSON: {"for_planning_node_ids": [<id>, ...], '
        '"against_planning_node_ids": [<id>, ...]}'
    )
    try:
        result = call_llm(prompt, schema=HighLevelPlans)
    except Exception as e:
        print(f"    [LLM error finding planning nodes: {e}]")
        return {"for_planning_node_ids": [], "against_planning_node_ids": []}

    return {
        "for_planning_node_ids": result.get("for_planning_node_ids", []),
        "against_planning_node_ids": result.get("against_planning_node_ids", []),
    }


def determine_model_stance(conclusion_text: str) -> str:
    """Use Gemini flash to determine whether the model's conclusion is 'for'
    or 'against' the debate topic.  Returns 'for' or 'against'.
    """
    from parser.utils.vertexai import call_llm

    prompt = (
        "The following is the conclusion from a model's reasoning trace on a debate topic.\n"
        "Determine whether the model's final stance is IN FAVOR OF (for) or AGAINST the "
        "topic proposition.\n\n"
        f"Conclusion text:\n{conclusion_text}\n\n"
        'Return JSON: {"stance": "for" or "against"}'
    )
    try:
        result = call_llm(prompt, schema=ModelStance)
        stance = result.get("stance", "").lower()
        if stance not in ("for", "against"):
            stance = "for"
        return stance
    except Exception as e:
        print(f"    [LLM error determining stance: {e}]")
        return "for"


# ── DAG utilities ─────────────────────────────────────────────────────────────

def prethink_node_ids(nodes):
    """Return the set of node IDs that appear at or before the </think> node
    (by character offset).  If no </think> node exists, returns all node IDs.
    """
    think_node = next(
        (n for n in nodes if "</think>" in n.get("text", "")), None
    )
    if think_node is None:
        return {n["id"] for n in nodes}
    cutoff = think_node["start"]
    return {n["id"] for n in nodes if n["start"] <= cutoff}


def build_forward_adj(edges):
    """Build forward adjacency from ALL edges (source -> dest)."""
    adj = defaultdict(list)
    for e in edges:
        adj[e["source_node_id"]].append(e["dest_node_id"])
    return adj


def dag_depth_width(start_id, adj):
    """BFS from start_id following all forward edges.
    Returns (depth, largest_antichain) where:
      - depth = longest shortest path from start_id to any descendant
      - largest_antichain = size of the largest set of pairwise non-reachable nodes
        (computed via Dilworth's theorem: n - max_bipartite_matching on reachability DAG)
    """
    # BFS to collect all reachable nodes and their levels
    visited = {start_id: 0}
    queue = deque([start_id])
    while queue:
        node = queue.popleft()
        level = visited[node]
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = level + 1
                queue.append(neighbor)

    depth = max(visited.values()) if visited else 0
    nodes = list(visited.keys())
    node_set = set(nodes)

    # Transitive closure restricted to the reachable subgraph
    reachable = {}
    for u in nodes:
        reach = set()
        q = deque(adj.get(u, []))
        while q:
            v = q.popleft()
            if v in node_set and v not in reach:
                reach.add(v)
                q.extend(adj.get(v, []))
        reachable[u] = reach

    # Maximum bipartite matching (Kuhn's algorithm)
    # Edge (u -> v) exists iff u can transitively reach v
    match_r = {}  # right node -> matched left node

    def try_augment(u, seen):
        for v in reachable[u]:
            if v not in seen:
                seen.add(v)
                if v not in match_r or try_augment(match_r[v], seen):
                    match_r[v] = u
                    return True
        return False

    matching_size = sum(1 for u in nodes if try_augment(u, set()))
    largest_antichain = len(nodes) - matching_size
    return depth, largest_antichain


def reachable_from(start_id, adj):
    """Return set of node IDs reachable from start_id (including itself)."""
    visited = set()
    queue = deque([start_id])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited


# ── Score computation ─────────────────────────────────────────────────────────

def compute_avg_mace_p(planning_node_ids, adj, node_annotations):
    """Return mean MACE-P score over all nodes reachable from any of the given
    planning node IDs that have matched arguments.
    Returns None if no matched arguments are found.
    """
    if not planning_node_ids:
        return None
    reach: set = set()
    for pid in planning_node_ids:
        reach |= reachable_from(pid, adj)
    scores = []
    for nid in reach:
        if nid in node_annotations:
            for arg in node_annotations[nid].get("argument_list", []):
                v = arg.get("mace_p")
                if v is not None:
                    scores.append(v)
    return statistics.mean(scores) if scores else None


# ── SVG export ────────────────────────────────────────────────────────────────

def export_planning_dag_svgs(plan_entries, out_dir="plots/planning_dag_examples"):
    """Draw the DAG reachable from each identified planning node.
    Only validate: and reason: edges are drawn (reflect: and plan: are excluded).
    """
    from generate_svg import draw_graph_2d

    os.makedirs(out_dir, exist_ok=True)

    for entry in plan_entries:
        path = entry["path"]
        with open(path) as f:
            raw = json.load(f)

        node_by_id = {n["id"]: n for n in raw["nodes"]}
        allowed_ids = prethink_node_ids(raw["nodes"])
        # restrict to validate: and reason: edges only, within the pre-</think> subgraph
        filtered_edges = [
            e for e in raw["edges"]
            if (e.get("label", "").startswith("validate:") or e.get("label", "").startswith("reason:"))
            and e["source_node_id"] in allowed_ids and e["dest_node_id"] in allowed_ids
        ]
        adj = build_forward_adj(filtered_edges)

        for side, pids in [
            ("for",     entry.get("for_planning_node_ids", [])),
            ("against", entry.get("against_planning_node_ids", [])),
        ]:
            if not pids:
                continue

            reach: set = set()
            for pid in pids:
                if pid not in allowed_ids:
                    continue
                if pid in node_by_id:
                    reach |= reachable_from(pid, adj)
            if not reach:
                continue

            nodes_out = [node_by_id[nid] for nid in reach if nid in node_by_id]
            edges_out = [
                e for e in filtered_edges
                if e["source_node_id"] in reach and e["dest_node_id"] in reach
            ]

            safe_model = entry["model"].replace("/", "_")
            fname = f"{entry['question_id']}_{safe_model}_{side}.svg"
            draw_graph_2d({"nodes": nodes_out, "edges": edges_out},
                          os.path.join(out_dir, fname))


# ── Violin plots for DAG depth/width ─────────────────────────────────────────

def plot_dag_depth_width_violin(metrics_by_model):
    """Violin plots for planning-DAG depth and width per model."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_order = ["Qwen2.5-32B-Instruct", "QwQ-32B", "DeepSeek-V3", "DeepSeek-R1"]
    models = [m for m in model_order if m in metrics_by_model]

    depth_data = [[e["dag_depth"] for e in metrics_by_model[m] if e["dag_depth"] is not None]
                  for m in models]
    width_data = [[e["dag_width"] for e in metrics_by_model[m] if e["dag_width"] is not None]
                  for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    positions = list(range(len(models)))

    for ax, data, ylabel, title in [
        (ax1, depth_data, "DAG Depth",     "Planning-DAG Depth per Model"),
        (ax2, width_data, "DAG Max Width",  "Planning-DAG Max Width per Model"),
    ]:
        non_empty = [i for i, d in enumerate(data) if len(d) > 1]
        if non_empty:
            parts = ax.violinplot(
                [data[i] for i in non_empty],
                positions=[positions[i] for i in non_empty],
                showmedians=True, showextrema=True,
            )
            for pc, i in zip(parts["bodies"], non_empty):
                color = MODEL_COLORS.get(models[i], "#888888")
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.7)
            for key in ("cbars", "cmaxes", "cmins", "cmedians"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(1.2)
        ax.set_xticks(positions)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "argumentation_planning_dag_depth_width_violin.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Score-diff histograms ─────────────────────────────────────────────────────

def plot_score_diff_histograms(metrics_by_model):
    """For each model, plot two overlapping histograms of
    (for_avg_mace_p - against_avg_mace_p):
      - green: traces where model decided 'for'
      - red:   traces where model decided 'against'
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    models = sorted(metrics_by_model.keys())
    n_models = len(models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = [axes] if n_models == 1 else list(axes.flatten())

    for ax, model in zip(axes, models):
        for_diffs = []
        against_diffs = []
        for e in metrics_by_model[model]:
            fs = e.get("for_avg_mace_p")
            as_ = e.get("against_avg_mace_p")
            stance = e.get("model_stance")
            if fs is None or as_ is None or stance is None:
                continue
            diff = fs - as_
            if stance == "for":
                for_diffs.append(diff)
            else:
                against_diffs.append(diff)

        bins = 15
        all_diffs = for_diffs + against_diffs
        x_min = min(all_diffs) - 0.05 if all_diffs else -1.0
        x_max = max(all_diffs) + 0.05 if all_diffs else 1.0
        xs = np.linspace(x_min, x_max, 300)

        # Gaussian KDE with Silverman's bandwidth rule (Silverman 1986,
        # "Density Estimation for Statistics and Data Analysis", Chapman & Hall).
        for diffs, color, label_prefix in [
            (for_diffs,     "#2ecc71", "agree"),
            (against_diffs, "#e74c3c", "disagree"),
        ]:
            if not diffs:
                continue
            ax.hist(diffs, bins=bins, alpha=0.25, color=color, density=True)
            if len(diffs) > 1:
                kde = gaussian_kde(diffs, bw_method="silverman")
                ax.plot(xs, kde(xs), color=color, linewidth=2,
                        label=f"{label_prefix} (n={len(diffs)})")
            else:
                ax.axvline(diffs[0], color=color, linewidth=2,
                           label=f"{label_prefix} (n=1)")

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("for_avg_mace_p − against_avg_mace_p")
        ax.set_ylabel("Density")
        ax.set_title(f"{model}")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle("Score diff: for-args vs. against-args (by model stance)", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "argumentation_score_diff_histograms.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── IBM ArgQ-30k helpers (unchanged) ─────────────────────────────────────────

def load_hf_dataset():
    """Load ibm-research/argument_quality_ranking_30k, merge all splits.
    Returns dict: topic -> list of {"argument": str, "mace_p": float}
    """
    from datasets import load_dataset
    ds = load_dataset("ibm-research/argument_quality_ranking_30k", "argument_quality_ranking")
    topic_args = defaultdict(list)
    for split in ["train", "validation", "test"]:
        for row in ds[split]:
            topic_args[row["topic"]].append({
                "argument": row["argument"],
                "mace_p": row["MACE-P"],
                "wa": row["WA"],
                "stance_WA": row["stance_WA"],
            })
    return dict(topic_args)


def build_qid_topic_map():
    """Map question_id -> topic by reading raw argkp data files."""
    raw_pattern = os.path.join("data/v1_raw_data", "argkp_*.json")
    qid_topic = {}
    for raw_path in sorted(glob.glob(raw_pattern)):
        with open(raw_path) as f:
            d = json.load(f)
        qid = d["metadata"].get("question_id", d["doc_id"])
        if qid in qid_topic:
            continue
        question = d["raw_text"]["question"]
        if "? " in question:
            topic = question.split("? ", 1)[1].split("\n")[0].strip()
        else:
            topic = question.split("\n")[0].strip()
        qid_topic[qid] = topic
    return qid_topic


def match_argument_with_gemini(node_text: str, topic: str, topic_args: dict) -> dict:
    """Use Gemini flash to find matching arguments for node_text."""
    from parser.utils.vertexai import call_llm

    args = topic_args.get(topic, [])
    if not args:
        return {"topic": topic, "argument_list": []}

    arg_list_str = "\n".join(f"{i + 1}. {a['argument']}" for i, a in enumerate(args))
    prompt = (
        f"You are given a reasoning node text and a list of short arguments on the same debate topic.\n"
        f"Select arguments that are logically **equivalent** with the given node text.\n"
        f"By equivalent, we mean arguments that share the same keywords, stance, and possible premises that can support the argument.\n"
        f"The ones that you choose should be almost identical to each other.\n"
        f"If the sentence is too general that it is in the level of topic, or there is no good match at all, return an empty list.\n\n"
        f"Node text: {node_text}\n\n"
        f"Topic: {topic}\n\n"
        f"Arguments:\n{arg_list_str}\n\n"
        f'Return JSON: {{"selected_indices": <list of indices of the best matching arguments, or empty list>}}'
    )

    try:
        result = call_llm(prompt, schema=ArgumentMatch)
        print(result)
    except Exception as e:
        print(f"    [LLM error: {e}]")
        return {"topic": topic, "argument_list": []}

    indices = result.get("selected_indices", [])
    argument_list = []
    for idx in indices:
        if isinstance(idx, int) and 1 <= idx <= len(args):
            a = args[idx - 1]
            argument_list.append({
                "matched_argument": a["argument"],
                "mace_p": a["mace_p"],
                "wa": a["wa"],
                "stance": a["stance_WA"],
            })

    return {"topic": topic, "argument_list": argument_list}


# ── Existing violin plots (unchanged) ────────────────────────────────────────

def plot_reflect_violin():
    """Violin plots: WA and MACE-P distributions per edge type for QwQ-32B, DeepSeek-R1, and gpt-oss-120b."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    EDGE_TYPES = ["reflect:positive", "reflect:uncertain", "reflect:negative"]
    EDGE_COLORS = {
        "reflect:positive":  "#2ecc71",
        "reflect:uncertain": "#f39c12",
        "reflect:negative":  "#e74c3c",
    }
    TARGET_MODELS = ["QwQ-32B", "DeepSeek-R1", "gpt-oss-120b"]
    REFLECT_PRIORITY = {"reflect:uncertain": 0, "reflect:positive": 1, "reflect:negative": 2}

    wa_data    = {m: {et: [] for et in EDGE_TYPES} for m in TARGET_MODELS}
    macep_data = {m: {et: [] for et in EDGE_TYPES} for m in TARGET_MODELS}

    for graph_path in sorted(glob.glob(ARGKP_PATTERN)):
        with open(graph_path) as f:
            graph = json.load(f)
        model = graph["metadata"]["generator"]
        if model not in TARGET_MODELS:
            continue
        question_id = graph["metadata"].get("question_id", graph["doc_id"])

        merged_path = os.path.join("argumentation_results", f"{question_id}_{model}.json")
        if not os.path.exists(merged_path):
            continue
        with open(merged_path) as f:
            merged_data = json.load(f)
        node_annot = {e["node_id"]: e for e in merged_data.get("nodes", [])}

        node_best_edge = {}
        for e in graph["edges"]:
            label = e["label"]
            if label not in REFLECT_PRIORITY:
                continue
            priority = REFLECT_PRIORITY[label]
            src = e["source_node_id"]
            if src not in node_best_edge or priority < node_best_edge[src][0]:
                node_best_edge[src] = (priority, label)

        for nid, (_, edge_label) in node_best_edge.items():
            if nid not in node_annot:
                continue
            entry = node_annot[nid]
            arg_list = entry.get("argument_list", [])
            wa_vals = [a["wa"] for a in arg_list if a.get("wa") is not None]
            mp_vals = [a["mace_p"] for a in arg_list if a.get("mace_p") is not None]
            if wa_vals:
                wa_data[model][edge_label].append(statistics.mean(wa_vals))
            if mp_vals:
                macep_data[model][edge_label].append(statistics.mean(mp_vals))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_configs = [
        (axes[0], wa_data["QwQ-32B"],        "WA Score",    "QwQ-32B — WA Score per Edge Type"),
        (axes[1], wa_data["DeepSeek-R1"],    "WA Score",    "DeepSeek-R1 — WA Score per Edge Type"),
        (axes[2], wa_data["gpt-oss-120b"],   "WA Score",    "gpt-oss-120b — WA Score per Edge Type"),
        (axes[3], macep_data["QwQ-32B"],     "MACE-P Score","QwQ-32B — MACE-P Score per Edge Type"),
        (axes[4], macep_data["DeepSeek-R1"], "MACE-P Score","DeepSeek-R1 — MACE-P Score per Edge Type"),
        (axes[5], macep_data["gpt-oss-120b"],"MACE-P Score","gpt-oss-120b — MACE-P Score per Edge Type"),
    ]

    for ax, score_by_type, ylabel, title in plot_configs:
        data_lists = [score_by_type[et] for et in EDGE_TYPES]
        positions  = list(range(len(EDGE_TYPES)))
        non_empty  = [i for i, d in enumerate(data_lists) if d]

        if non_empty:
            parts = ax.violinplot(
                [data_lists[i] for i in non_empty],
                positions=[positions[i] for i in non_empty],
                showmedians=True, showextrema=True,
            )
            for pc, i in zip(parts["bodies"], non_empty):
                color = EDGE_COLORS[EDGE_TYPES[i]]
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.7)
            for key in ("cbars", "cmaxes", "cmins", "cmedians"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(1.2)

        ax.set_xticks(positions)
        ax.set_xticklabels([et.replace("reflect:", "") for et in EDGE_TYPES], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for i, et in enumerate(EDGE_TYPES):
            n = len(score_by_type[et])
            ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.05,
                    f"n={n}", ha="center", va="top", fontsize=8, color="#555555")

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "argumentation_reflect_scores_violin.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_validate_violin():
    """Violin plots: WA and MACE-P distributions per validate edge type."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    EDGE_TYPES = ["validate:support", "validate:attack"]
    EDGE_COLORS = {
        "validate:support": "#2ecc71",
        "validate:attack":  "#e74c3c",
    }
    TARGET_MODELS = ["QwQ-32B", "DeepSeek-R1", "gpt-oss-120b"]

    wa_data    = {m: {et: [] for et in EDGE_TYPES} for m in TARGET_MODELS}
    macep_data = {m: {et: [] for et in EDGE_TYPES} for m in TARGET_MODELS}

    for graph_path in sorted(glob.glob(ARGKP_PATTERN)):
        with open(graph_path) as f:
            graph = json.load(f)
        model = graph["metadata"]["generator"]
        if model not in TARGET_MODELS:
            continue
        question_id = graph["metadata"].get("question_id", graph["doc_id"])

        merged_path = os.path.join("argumentation_results", f"{question_id}_{model}.json")
        if not os.path.exists(merged_path):
            continue
        with open(merged_path) as f:
            merged_data = json.load(f)
        node_annot = {e["node_id"]: e for e in merged_data.get("nodes", [])}

        VALIDATE_PRIORITY = {"validate:support": 0, "validate:attack": 1}
        node_best_edge = {}
        for e in graph["edges"]:
            label = e["label"]
            if label not in VALIDATE_PRIORITY:
                continue
            priority = VALIDATE_PRIORITY[label]
            src = e["source_node_id"]
            if src not in node_best_edge or priority < node_best_edge[src][0]:
                node_best_edge[src] = (priority, label)

        for nid, (_, edge_label) in node_best_edge.items():
            if nid not in node_annot:
                continue
            entry = node_annot[nid]
            arg_list = entry.get("argument_list", [])
            wa_vals = [a["wa"] for a in arg_list if a.get("wa") is not None]
            mp_vals = [a["mace_p"] for a in arg_list if a.get("mace_p") is not None]
            if wa_vals:
                wa_data[model][edge_label].append(statistics.mean(wa_vals))
            if mp_vals:
                macep_data[model][edge_label].append(statistics.mean(mp_vals))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_configs = [
        (axes[0], wa_data["QwQ-32B"],        "WA Score",    "QwQ-32B — WA Score per Validate Edge Type"),
        (axes[1], wa_data["DeepSeek-R1"],    "WA Score",    "DeepSeek-R1 — WA Score per Validate Edge Type"),
        (axes[2], wa_data["gpt-oss-120b"],   "WA Score",    "gpt-oss-120b — WA Score per Validate Edge Type"),
        (axes[3], macep_data["QwQ-32B"],     "MACE-P Score","QwQ-32B — MACE-P Score per Validate Edge Type"),
        (axes[4], macep_data["DeepSeek-R1"], "MACE-P Score","DeepSeek-R1 — MACE-P Score per Validate Edge Type"),
        (axes[5], macep_data["gpt-oss-120b"],"MACE-P Score","gpt-oss-120b — MACE-P Score per Validate Edge Type"),
    ]

    for ax, score_by_type, ylabel, title in plot_configs:
        data_lists = [score_by_type[et] for et in EDGE_TYPES]
        positions  = list(range(len(EDGE_TYPES)))
        non_empty  = [i for i, d in enumerate(data_lists) if d]

        if non_empty:
            parts = ax.violinplot(
                [data_lists[i] for i in non_empty],
                positions=[positions[i] for i in non_empty],
                showmedians=True, showextrema=True,
            )
            for pc, i in zip(parts["bodies"], non_empty):
                color = EDGE_COLORS[EDGE_TYPES[i]]
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.7)
            for key in ("cbars", "cmaxes", "cmins", "cmedians"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(1.2)

        ax.set_xticks(positions)
        ax.set_xticklabels([et.replace("validate:", "") for et in EDGE_TYPES], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for i, et in enumerate(EDGE_TYPES):
            n = len(score_by_type[et])
            ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.05,
                    f"n={n}", ha="center", va="top", fontsize=8, color="#555555")

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "argumentation_validate_scores_violin.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Migration helper ──────────────────────────────────────────────────────────

def migrate_split_files(results_dir="argumentation_results"):
    """Merge legacy _nodes.json + _high-level-plans.json pairs into a single
    {qid}_{model}.json file, then remove the originals."""
    nodes_files = glob.glob(os.path.join(results_dir, "*_nodes.json"))
    for nodes_path in nodes_files:
        base = os.path.basename(nodes_path)[:-len("_nodes.json")]
        plans_path = os.path.join(results_dir, f"{base}_high-level-plans.json")
        merged_path = os.path.join(results_dir, f"{base}.json")

        if os.path.exists(merged_path):
            continue  # already migrated

        with open(nodes_path) as f:
            nodes_data = json.load(f)

        merged = {}
        if os.path.exists(plans_path):
            with open(plans_path) as f:
                merged = json.load(f)

        merged["nodes"] = nodes_data

        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=4)

        os.remove(nodes_path)
        if os.path.exists(plans_path):
            os.remove(plans_path)
        print(f"  Migrated {base}")

    # Also remove orphaned _high-level-plans.json files that have no _nodes.json pair
    for plans_path in glob.glob(os.path.join(results_dir, "*_high-level-plans.json")):
        base = os.path.basename(plans_path)[:-len("_high-level-plans.json")]
        merged_path = os.path.join(results_dir, f"{base}.json")
        if os.path.exists(merged_path):
            continue
        with open(plans_path) as f:
            merged = json.load(f)
        merged.setdefault("nodes", [])
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=4)
        os.remove(plans_path)
        print(f"  Migrated (plans-only) {base}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    files = sorted(glob.glob(ARGKP_PATTERN))
    print(f"Found {len(files)} argkp files\n")

    os.makedirs("argumentation_results", exist_ok=True)
    migrate_split_files()

    # ── Existing: node-level argument annotation ──────────────────────────────
    TARGET_NODE_LABELS = {"fact", "reasoning"}

    print("--- Argument alignment for all fact/reasoning nodes ---")
    print("Loading IBM ArgQ-30k dataset...")
    hf_topic_args = load_hf_dataset()
    qid_topic_map = build_qid_topic_map()
    print(f"Dataset loaded: {sum(len(v) for v in hf_topic_args.values())} arguments "
          f"across {len(hf_topic_args)} topics\n")

    for path in sorted(glob.glob(ARGKP_PATTERN)):
        with open(path) as f:
            data = json.load(f)
        model = data["metadata"]["generator"]
        question_id = data["metadata"].get("question_id", data["doc_id"])

        merged_path = os.path.join("argumentation_results",
                                   f"{question_id}_{model}.json")
        merged = {}
        if os.path.exists(merged_path):
            with open(merged_path) as f:
                merged = json.load(f)

        if "nodes" in merged:
            print(f"  {question_id} | {model}  [skipped, file exists]")
            continue

        question_topic = qid_topic_map.get(question_id)
        target_nodes = [
            n for n in data["nodes"]
            if n.get("source") == "response" and n.get("label") in TARGET_NODE_LABELS
        ]
        print(f"\n  {question_id} | {model}  ({len(target_nodes)} nodes to annotate)")

        out_data = []
        for n in target_nodes:
            nid = n["id"]
            node_lbl = n.get("label", "?")
            text = n.get("text", "").replace("\n", " ").strip()
            if question_topic:
                match = match_argument_with_gemini(text, question_topic, hf_topic_args)
            else:
                match = {"topic": question_topic, "argument_list": []}
            out_data.append({
                "node_id": nid,
                "node_label": node_lbl,
                "node_text": text,
                "topic": match["topic"],
                "argument_list": match["argument_list"],
            })

        merged["nodes"] = out_data
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=4)
        print(f"    Saved {merged_path}")

    # ── Task 1 & 3: high-level planning nodes + model stance ─────────────────
    print("\n--- High-level planning nodes + model stance ---")

    # model -> list of per-file metric dicts (for later violin/histogram)
    metrics_by_model = defaultdict(list)
    all_plan_entries = []  # for SVG export

    for path in sorted(glob.glob(ARGKP_PATTERN)):
        with open(path) as f:
            data = json.load(f)
        model = data["metadata"]["generator"]
        question_id = data["metadata"].get("question_id", data["doc_id"])

        merged_path = os.path.join("argumentation_results",
                                   f"{question_id}_{model}.json")

        # Load existing merged file if present
        if os.path.exists(merged_path):
            with open(merged_path) as f:
                merged = json.load(f)
        else:
            merged = {}
        plans = merged  # alias for clarity

        changed = False

        # Task 1: find for/against planning nodes if not yet done
        if "for_planning_node_ids" not in plans or "against_planning_node_ids" not in plans:
            planning_nodes = [
                n for n in data["nodes"]
                if n.get("source") == "response" and n.get("label") == "planning"
            ]
            print(f"  {question_id} | {model}  [task1: {len(planning_nodes)} planning nodes]")
            result = find_high_level_planning_nodes(planning_nodes)
            plans["question_id"]               = question_id
            plans["model"]                     = model
            plans["for_planning_node_ids"]     = result["for_planning_node_ids"]
            plans["against_planning_node_ids"] = result["against_planning_node_ids"]
            # Store node texts for reference
            node_by_id = {n["id"]: n for n in data["nodes"]}
            for key in ("for_planning_node_ids", "against_planning_node_ids"):
                text_key = key.replace("_ids", "_texts")
                pids = plans[key]
                plans[text_key] = [
                    node_by_id[pid].get("text", "") for pid in pids if pid in node_by_id
                ]
            changed = True
        else:
            print(f"  {question_id} | {model}  [task1: loaded from file]")

        # Task 3a: model stance from first conclusion node
        if "model_stance" not in plans:
            conclusion_nodes = [n for n in data["nodes"] if n.get("label") == "conclusion"]
            if conclusion_nodes:
                first_conclusion_text = conclusion_nodes[0].get("text", "")
                print(f"  {question_id} | {model}  [task3: determining stance]")
                stance = determine_model_stance(first_conclusion_text)
                plans["model_stance"]                 = stance
                plans["model_stance_conclusion_text"] = first_conclusion_text
            else:
                plans["model_stance"]                 = None
                plans["model_stance_conclusion_text"] = None
            changed = True
        else:
            print(f"  {question_id} | {model}  [task3: stance loaded from file]")

        # Task 3b: compute average MACE-P scores from planning node subtrees
        if "for_avg_mace_p" not in plans or "against_avg_mace_p" not in plans:
            node_annotations = {e["node_id"]: e for e in merged.get("nodes", [])}

            adj = build_forward_adj(data["edges"])
            plans["for_avg_mace_p"]     = compute_avg_mace_p(
                plans.get("for_planning_node_ids", []), adj, node_annotations)
            plans["against_avg_mace_p"] = compute_avg_mace_p(
                plans.get("against_planning_node_ids", []), adj, node_annotations)
            changed = True

        if changed:
            with open(merged_path, "w") as f:
                json.dump(merged, f, indent=4)
            print(f"    Saved {merged_path}")

        # Task 2: compute DAG depth/width for violin plot (reason:/validate: edges only)
        allowed_ids = prethink_node_ids(data["nodes"])
        filtered_edges = [
            e for e in data["edges"]
            if (e.get("label", "").startswith("reason:") or e.get("label", "").startswith("validate:"))
            and e["source_node_id"] in allowed_ids and e["dest_node_id"] in allowed_ids
        ]
        adj = build_forward_adj(filtered_edges)

        group_max_depths = []
        group_max_widths = []
        for pids in (plans.get("for_planning_node_ids", []), plans.get("against_planning_node_ids", [])):
            if not pids:
                continue
            group_depths, group_widths = zip(*(dag_depth_width(pid, adj) for pid in pids))
            group_max_depths.append(max(group_depths))
            group_max_widths.append(sum(group_widths))

        dag_depth = statistics.mean(group_max_depths) if group_max_depths else None
        dag_width = statistics.mean(group_max_widths) if group_max_widths else None

        entry = {
            "question_id":               question_id,
            "path":                      path,
            "model":                     model,
            "dag_depth":                 dag_depth,
            "dag_width":                 dag_width,
            "for_planning_node_ids":     plans.get("for_planning_node_ids", []),
            "against_planning_node_ids": plans.get("against_planning_node_ids", []),
            "model_stance":              plans.get("model_stance"),
            "for_avg_mace_p":            plans.get("for_avg_mace_p"),
            "against_avg_mace_p":        plans.get("against_avg_mace_p"),
        }
        metrics_by_model[model].append(entry)
        all_plan_entries.append(entry)

    # ── Task 2: violin plots for DAG depth/width ──────────────────────────────
    print("\n--- Plotting DAG depth/width violins ---")
    plot_dag_depth_width_violin(metrics_by_model)

    # ── Task 2: export planning DAG SVGs ─────────────────────────────────────
    print("\n--- Exporting planning DAG SVGs ---")
    export_planning_dag_svgs(all_plan_entries)

    # ── Task 3: score-diff histograms ─────────────────────────────────────────
    print("\n--- Plotting score-diff histograms ---")
    plot_score_diff_histograms(metrics_by_model)

    # ── Existing: reflect/validate violin plots ───────────────────────────────
    plot_reflect_violin()
    plot_validate_violin()


if __name__ == "__main__":
    main()
