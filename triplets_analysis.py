import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/v1_llm_gemini-3.1-pro-preview")
SCHEMA_DIR = Path("schema")
PLOTS_DIR  = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams["svg.fonttype"] = "none"

# ── Model config ──────────────────────────────────────────────────────────────
MODELS = ["Qwen2.5-32B-Instruct", "QwQ-32B", "DeepSeek-V3", "DeepSeek-R1", "gpt-oss-120b"]

FAMILY_MAP = {
    "Qwen2.5-32B-Instruct": "Qwen",
    "QwQ-32B":              "Qwen",
    "DeepSeek-V3":          "DeepSeek",
    "DeepSeek-R1":          "DeepSeek",
    "gpt-oss-120b":         "oss",
}

SHORT = {
    "Qwen2.5-32B-Instruct": "Qwen2.5",
    "QwQ-32B":              "QwQ",
    "DeepSeek-V3":          "DS-V3",
    "DeepSeek-R1":          "DS-R1",
    "gpt-oss-120b":         "oss",
}

MODEL_COLORS = {
    "Qwen2.5-32B-Instruct": "#8ED7D7",
    "QwQ-32B":              "#007A84",
    "DeepSeek-V3":          "#FF8CA1",
    "DeepSeek-R1":          "#E42741",
    "gpt-oss-120b":         "#98C126",
}

# ── Schema ────────────────────────────────────────────────────────────────────
with open(SCHEMA_DIR / "node_labels.yaml") as f:
    _ns = yaml.safe_load(f)
VALID_NODE_LABELS = {n["name"] for n in _ns["nodes"]}

with open(SCHEMA_DIR / "edge_labels.yaml") as f:
    _es = yaml.safe_load(f)
VALID_EDGE_LABELS = {e["name"] for e in _es["edges"]}

TRANSPARENT = {"context", "planning", "restatement", "reflection"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_graphs():
    records = []
    for path in sorted(DATA_DIR.glob("*.json")):
        if "bright" in str(path):
            continue
        with open(path) as f:
            datum = json.load(f)
        gen = datum["metadata"]["generator"]
        if gen not in FAMILY_MAP:
            continue

        nodes = [n for n in datum.get("nodes", []) if n.get("label") in VALID_NODE_LABELS]
        edges = [e for e in datum.get("edges", []) if e.get("label") in VALID_EDGE_LABELS]

        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["id"], label=n["label"], source=n.get("source", ""))
        for e in edges:
            s, d = e["source_node_id"], e["dest_node_id"]
            if G.has_node(s) and G.has_node(d):
                G.add_edge(s, d, label=e["label"])

        records.append({
            "doc_id": datum["doc_id"],
            "model":  gen,
            "domain": datum["metadata"]["domain"],
            "G":      G,
            "nodes":  nodes,
            "edges":  edges,
        })
    return records


# ── Graph statistics ──────────────────────────────────────────────────────────

def graph_depth_bfs(G, node_labels: dict):
    conclusion_ids = [n for n, lbl in node_labels.items() if lbl == "conclusion"]
    if not conclusion_ids:
        return None

    adj = defaultdict(list)
    for u, v, data in G.edges(data=True):
        if data.get("label", "").startswith("reason:infer"):
            adj[v].append(u)

    dist: dict[str, int] = {}
    dq = deque()
    for cid in conclusion_ids:
        if cid in node_labels:
            dist[cid] = 0
            dq.append(cid)

    while dq:
        node = dq.popleft()
        for nb in adj[node]:
            w  = 0 if node_labels.get(nb) in TRANSPARENT else 1
            nd = dist[node] + w
            if nb not in dist or nd < dist[nb]:
                dist[nb] = nd
                (dq.appendleft if w == 0 else dq.append)(nb)

    resp_dists = {
        nid: d for nid, d in dist.items()
        if G.nodes[nid].get("source") == "response"
        and node_labels.get(nid) not in TRANSPARENT
    }
    return max(resp_dists.values()) if resp_dists else None


def compute_graph_stats(rec: dict):
    G = rec["G"]
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0 or m == 0:
        return None

    node_labels = {nd: G.nodes[nd]["label"] for nd in G.nodes}
    in_degs  = [d for _, d in G.in_degree()]
    out_degs = [d for _, d in G.out_degree()]

    largest_wcc = len(max(nx.weakly_connected_components(G), key=len)) / n
    clustering  = nx.average_clustering(G.to_undirected())
    depth       = graph_depth_bfs(G, node_labels)

    try:
        longest_path = len(nx.dag_longest_path(G)) - 1
    except nx.NetworkXUnfeasible:
        longest_path = None

    return {
        "n_nodes":          n,
        "n_edges":          m,
        "edge_density":     m / n,
        "max_in_degree":    max(in_degs),
        "max_out_degree":   max(out_degs),
        "reciprocity":      nx.reciprocity(G) if m > 0 else 0.0,
        "n_wcc":            nx.number_weakly_connected_components(G),
        "largest_wcc_frac": largest_wcc,
        "clustering":       clustering,
        "depth":            depth,
        "longest_path":     longest_path,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    records = load_graphs()

    stats_by_model: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        s = compute_graph_stats(rec)
        if s:
            stats_by_model[rec["model"]].append(s)

    PLOT_KEYS = [
        ("n_nodes",      "Node count"),
        ("edge_density", "Edge density (m/n)"),
        ("clustering",   "Avg clustering coeff."),
        ("reciprocity",  "Reciprocity"),
        ("depth",        "Reasoning depth"),
        ("longest_path", "Longest DAG path"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, (key, title) in zip(axes.flat, PLOT_KEYS):
        data_per_model = [
            [s[key] for s in stats_by_model[m] if s.get(key) is not None]
            for m in MODELS
        ]
        parts = ax.violinplot(data_per_model, positions=range(len(MODELS)),
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(MODEL_COLORS[MODELS[i]])
            pc.set_alpha(0.75)
        for key in ("cbars", "cmaxes", "cmins", "cmedians"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1.2)

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=MODEL_COLORS[m], label=SHORT[m]) for m in MODELS]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Graph-level statistics per model", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = PLOTS_DIR / "model_graph_stats.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"Saved plot → {out}")


if __name__ == "__main__":
    main()
