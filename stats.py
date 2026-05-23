"""
Generate plots for v1_llm_gemini-3.1-pro-preview data.
Plots are saved in plots/ as SVG with text as text entities.
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
from scipy.ndimage import gaussian_filter1d
import yaml

# ── SVG text as text entities (not converted to paths) ──────────────────────
plt.rcParams["svg.fonttype"] = "none"

DATA_DIR = Path("data/v1_llm_gemini-3.1-pro-preview")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

SCHEMA_DIR = Path("schema")

# ── Load color maps from schema ──────────────────────────────────────────────
with open(SCHEMA_DIR / "node_labels.yaml") as f:
    node_schema = yaml.safe_load(f)
NODE_COLORS = {n["name"]: n["color"] for n in node_schema["nodes"]}
SCHEMA_NODE_LABELS = set(NODE_COLORS)
SCHEMA_NODE_ORDER = [n["name"] for n in node_schema["nodes"]]

with open(SCHEMA_DIR / "edge_labels.yaml") as f:
    edge_schema = yaml.safe_load(f)
EDGE_COLORS = {e["name"]: e["color"] for e in edge_schema["edges"]}
SCHEMA_EDGE_LABELS = set(EDGE_COLORS)
SCHEMA_EDGE_ORDER = [e["name"] for e in edge_schema["edges"]]

# ── Model colors for plot 7 ──────────────────────────────────────────────────
# Similar colors for DeepSeek-R1/V3, similar for Qwen/QwQ
MODEL_COLORS = {
    "Qwen2.5-32B-Instruct": "#8ED7D7",
    "QwQ-32B":              "#007A84",
    "DeepSeek-V3":          "#FF8CA1",
    "DeepSeek-R1":          "#E42741",
    "gpt-oss-120b":         "#98C126",
}

# ── Crossing-edge analysis helpers ───────────────────────────────────────────

def _min_edges_to_erase_for_noncrossing_tree(nodes: list, edges: list) -> tuple[int, int]:
    """Return (min_edges_to_erase, total_valid_edges).

    Nodes are ordered by their text `start` position.  Two edges *cross* when
    their node-index intervals interleave: for intervals [a,b] and [c,d]
    (a<b, c<d) they cross iff a<c<b<d or c<a<d<b.

    We find the *maximum* subset of edges that (a) is non-crossing (intervals
    form a laminar family) and (b) forms an undirected forest (no cycles,
    checked with Union-Find).  The minimum number of edges to erase is then
    total_valid_edges − max_kept.

    Greedy strategy: sort edges by span (ascending) so shorter, more local
    edges are preferred; ties broken by left endpoint.  For each edge, keep it
    if it neither crosses any already-kept edge nor creates an undirected cycle.
    """
    sorted_nodes = sorted(nodes, key=lambda n: (n.get("start", 0), n["id"]))
    node_to_idx = {n["id"]: i for i, n in enumerate(sorted_nodes)}
    n = len(sorted_nodes)

    intervals: list[tuple[int, int]] = []
    for e in edges:
        s, d = e.get("source_node_id"), e.get("dest_node_id")
        if s not in node_to_idx or d not in node_to_idx:
            continue
        l, r = sorted([node_to_idx[s], node_to_idx[d]])
        if l == r:
            continue
        intervals.append((l, r))

    # Sort: shorter span first, then by left endpoint
    intervals.sort(key=lambda iv: (iv[1] - iv[0], iv[0]))

    # Union-Find (path-compressed)
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    kept: list[tuple[int, int]] = []
    for l, r in intervals:
        # Check if [l,r] crosses any already-kept interval
        if any((l2 < l < r2 < r) or (l < l2 < r < r2) for l2, r2 in kept):
            continue
        # Check if adding [l,r] creates an undirected cycle
        if not union(l, r):
            continue
        kept.append((l, r))

    return len(intervals) - len(kept), len(intervals)


# ── Read all data ─────────────────────────────────────────────────────────────
all_node_labels: list[str] = []
all_edge_labels: list[str] = []
node_labels_per_gen: dict[str, list[str]] = defaultdict(list)
edge_labels_per_gen: dict[str, list[str]] = defaultdict(list)
node_counts_per_gen: dict[str, list[int]] = defaultdict(list)
node_counts_per_gen_per_domain: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
node_labels_per_domain: dict[str, list[str]] = defaultdict(list)
edge_labels_per_domain: dict[str, list[str]] = defaultdict(list)
n_datapoints_total: int = 0
n_datapoints_per_gen: dict[str, int] = defaultdict(int)
n_datapoints_per_domain: dict[str, int] = defaultdict(int)

# Crossing-edge data per file
crossing_erased_per_gen: dict[str, list[int]] = defaultdict(list)
crossing_total_per_gen: dict[str, list[int]] = defaultdict(list)
crossing_erased_per_domain: dict[str, list[int]] = defaultdict(list)
crossing_total_per_domain: dict[str, list[int]] = defaultdict(list)
all_erased: list[int] = []
all_totals: list[int] = []

# Node/edge count and incoming-degree data per file
# Each entry: (n_nodes, n_nonctx_nodes, n_edges, avg_indegree_nonctx)
graph_stats_per_gen: dict[str, list[tuple]] = defaultdict(list)
graph_stats_per_domain: dict[str, list[tuple]] = defaultdict(list)
graph_stats_per_gen_domain: dict[str, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))
all_graph_stats: list[tuple] = []
node_labels_per_gen_domain: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
edge_labels_per_gen_domain: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

for path in sorted(DATA_DIR.glob("*.json")):
    with open(path) as f:
        datum = json.load(f)
    generator = datum["metadata"]["generator"]
    domain = datum["metadata"]["domain"]
    nodes = datum.get("nodes", [])
    edges = datum.get("edges", [])

    node_labels = [n["label"] for n in nodes if n.get("label") in SCHEMA_NODE_LABELS]
    edge_labels = [e["label"] for e in edges if e.get("label") in SCHEMA_EDGE_LABELS]

    all_node_labels.extend(node_labels)
    all_edge_labels.extend(edge_labels)
    node_labels_per_gen[generator].extend(node_labels)
    edge_labels_per_gen[generator].extend(edge_labels)
    node_labels_per_gen_domain[domain][generator].extend(node_labels)
    edge_labels_per_gen_domain[domain][generator].extend(edge_labels)
    node_counts_per_gen[generator].append(len(node_labels))
    node_counts_per_gen_per_domain[domain][generator].append(len(node_labels))
    node_labels_per_domain[domain].extend(node_labels)
    edge_labels_per_domain[domain].extend(edge_labels)
    n_datapoints_total += 1
    n_datapoints_per_gen[generator] += 1
    n_datapoints_per_domain[domain] += 1

    erased, total = _min_edges_to_erase_for_noncrossing_tree(nodes, edges)
    crossing_erased_per_gen[generator].append(erased)
    crossing_total_per_gen[generator].append(total)
    crossing_erased_per_domain[domain].append(erased)
    crossing_total_per_domain[domain].append(total)
    all_erased.append(erased)
    all_totals.append(total)

    # Graph structure stats
    schema_nodes = [n for n in nodes if n.get("label") in SCHEMA_NODE_LABELS]
    nonctx_ids = {n["id"] for n in schema_nodes if n.get("label") != "context"}
    schema_edges = [e for e in edges if e.get("label") in SCHEMA_EDGE_LABELS]
    incoming_nonctx = sum(1 for e in schema_edges if e.get("dest_node_id") in nonctx_ids)
    avg_indeg = incoming_nonctx / len(nonctx_ids) if nonctx_ids else 0.0
    stat = (len(schema_nodes), len(nonctx_ids), len(schema_edges), avg_indeg)
    graph_stats_per_gen[generator].append(stat)
    graph_stats_per_domain[domain].append(stat)
    graph_stats_per_gen_domain[generator][domain].append(stat)
    all_graph_stats.append(stat)

GENERATORS = sorted(node_labels_per_gen.keys())
DOMAINS = sorted(node_labels_per_domain.keys())
GEN_ORDER = ["Qwen2.5-32B-Instruct", "QwQ-32B", "DeepSeek-V3", "DeepSeek-R1", "gpt-oss-120b"]


# ── Helper: draw a single pie chart ──────────────────────────────────────────
def draw_pie(ax, counter: Counter, color_map: dict, title: str,
             label_order: list | None = None, legend_on_side: bool = False,
             n_datapoints: int | None = None):
    if label_order is not None:
        labels_sorted = [l for l in label_order if l in counter]
        labels_sorted += sorted(k for k in counter if k not in label_order)
    else:
        labels_sorted = sorted(counter.keys())
    sizes = [counter[l] for l in labels_sorted]
    colors = [color_map.get(l, "#CCCCCC") for l in labels_sorted]
    total = sum(sizes)

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(linewidth=0.5, edgecolor="white"),
    )
    ax.set_title(title, fontsize=9, pad=6)

    # Font size proportional to pie radius in figure inches
    fig = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    ax_pos = ax.get_position()
    pie_radius_in = min(ax_pos.width * fig_w, ax_pos.height * fig_h) * 0.45
    fontsize = max(7.0, min(12.0, pie_radius_in * 4.0))

    INNER_THRESHOLD = 5.0  # pct; slices above this get text placed inside

    if legend_on_side:
        # Annotate each slice with ratio + per-response avg (label names go in the shared figure legend)
        for wedge, size in zip(wedges, sizes):
            angle = (wedge.theta2 + wedge.theta1) / 2
            cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            pct = size / total * 100
            count_str = f"{size / n_datapoints:.2f}/resp" if n_datapoints else f"{size:,}"
            if pct >= INNER_THRESHOLD:
                ax.annotate(
                    f"{pct:.1f}%\n{count_str}",
                    xy=(0, 0),
                    xytext=(0.6 * cos_a, 0.6 * sin_a),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
            else:
                ha = "left" if cos_a >= 0 else "right"
                ax.annotate(
                    f"{pct:.1f}%\n{count_str}",
                    xy=(cos_a, sin_a),
                    xytext=(1.3 * cos_a, 1.3 * sin_a),
                    ha=ha,
                    va="center",
                    fontsize=fontsize,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
                )
    else:
        # Original: label + ratio + count around the pie with leader lines
        for wedge, label, size in zip(wedges, labels_sorted, sizes):
            angle = (wedge.theta2 + wedge.theta1) / 2
            cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            pct = size / total * 100
            count_str = f"{size / n_datapoints:.2f}/resp" if n_datapoints else f"{size:,}"
            if pct >= INNER_THRESHOLD:
                ax.annotate(
                    f"{label}\n{pct:.1f}% ({count_str})",
                    xy=(0, 0),
                    xytext=(0.6 * cos_a, 0.6 * sin_a),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
            else:
                ha = "left" if cos_a >= 0 else "right"
                ax.annotate(
                    f"{label}\n{pct:.1f}% ({count_str})",
                    xy=(cos_a, sin_a),
                    xytext=(1.35 * cos_a, 1.35 * sin_a),
                    ha=ha,
                    va="center",
                    fontsize=fontsize,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
                )


# ── Helper: add a shared wide node legend below a figure ─────────────────────
def add_node_legend(fig, fontsize: float = 9):
    """2-row × 4-col legend of the 8 non-context node labels, placed below the figure.

    Desired display (row-major):
      planning   fact        reasoning   restatement
      assumption example     reflection  conclusion

    Matplotlib fills ncol=4 column-by-column, so supply items in column-major
    order: col0=[planning,assumption], col1=[fact,example], ...
    """
    row0 = ["planning", "fact", "reasoning", "restatement"]
    row1 = ["assumption", "example", "reflection", "conclusion"]
    # column-major order: (row0[0],row1[0]), (row0[1],row1[1]), ...
    ordered = [item for pair in zip(row0, row1) for item in pair]
    handles = [mpatches.Patch(facecolor=NODE_COLORS[l], edgecolor="white", label=l) for l in ordered]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=fontsize,
        frameon=True,
        framealpha=0.9,
        handlelength=1.5,
        handleheight=1.0,
        columnspacing=1.2,
    )


# ── Helper: add a shared wide edge legend below a figure ─────────────────────
def add_edge_legend(fig, fontsize: float = 9):
    """3-col × 5-row legend: col 1 = 5 reason, col 2 = 4 plan (+blank), col 3 = 5 rest.

    Matplotlib fills ncol=3 column-by-column, so supply all of col0 first,
    then col1, then col2 (simple concatenation of the three groups).
    """
    reason = [l for l in SCHEMA_EDGE_ORDER if l.startswith("reason:")]   # 5
    plan   = [l for l in SCHEMA_EDGE_ORDER if l.startswith("plan:")]     # 4
    rest   = [l for l in SCHEMA_EDGE_ORDER
              if not l.startswith("reason:") and not l.startswith("plan:")]  # 5
    plan_padded = plan + [None]  # pad to 5 so all columns have equal height

    handles, labels = [], []
    for item in reason + plan_padded + rest:
        if item is None:
            handles.append(mpatches.Patch(visible=False))
            labels.append("")
        else:
            handles.append(mpatches.Patch(facecolor=EDGE_COLORS[item], edgecolor="white", label=item))
            labels.append(item)

    fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=fontsize,
        frameon=True,
        framealpha=0.9,
        handlelength=1.5,
        handleheight=1.0,
        columnspacing=1.5,
    )


# ── Plot 1: % node labels – all data (excluding "context") ───────────────────
node_counter_no_ctx = Counter({k: v for k, v in Counter(all_node_labels).items() if k != "context"})
fig, ax = plt.subplots(figsize=(8, 6))
draw_pie(ax, node_counter_no_ctx, NODE_COLORS, "Node Label Distribution (All Data)", SCHEMA_NODE_ORDER, legend_on_side=True, n_datapoints=n_datapoints_total)
fig.tight_layout()
add_node_legend(fig)
fig.savefig(PLOTS_DIR / "01_node_labels_all.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 01_node_labels_all.svg")


# ── Plot 2: % node labels – per generator (5 sub-plots, excluding "context") ─
fig, axes = plt.subplots(1, 5, figsize=(32, 6))
fig.suptitle("Node Label Distribution per Generator", fontsize=11, y=1.02)
for ax, gen in zip(axes, GENERATORS):
    per_gen_no_ctx = Counter({k: v for k, v in Counter(node_labels_per_gen[gen]).items() if k != "context"})
    draw_pie(ax, per_gen_no_ctx, NODE_COLORS, gen, SCHEMA_NODE_ORDER, legend_on_side=True, n_datapoints=n_datapoints_per_gen[gen])
fig.tight_layout()
add_node_legend(fig)
fig.savefig(PLOTS_DIR / "03_node_labels_per_gen.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 03_node_labels_per_gen.svg")


# ── Plot 3: % edge labels – all data ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
draw_pie(ax, Counter(all_edge_labels), EDGE_COLORS, "Edge Label Distribution (All Data)", SCHEMA_EDGE_ORDER, legend_on_side=True, n_datapoints=n_datapoints_total)
fig.tight_layout()
add_edge_legend(fig)
fig.savefig(PLOTS_DIR / "02_edge_labels_all.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 02_edge_labels_all.svg")


# ── Plot 4: % edge labels – per generator (5 sub-plots) ─────────────────────
fig, axes = plt.subplots(1, 5, figsize=(36, 7))
fig.suptitle("Edge Label Distribution per Generator", fontsize=11, y=1.02)
for ax, gen in zip(axes, GENERATORS):
    draw_pie(ax, Counter(edge_labels_per_gen[gen]), EDGE_COLORS, gen, SCHEMA_EDGE_ORDER, legend_on_side=True, n_datapoints=n_datapoints_per_gen[gen])
fig.tight_layout()
add_edge_legend(fig)
fig.savefig(PLOTS_DIR / "04_edge_labels_per_gen.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 04_edge_labels_per_gen.svg")


# ── Plot 5: % node labels – per domain (excluding "context") ─────────────────
n_domains = len(DOMAINS)
print(f"Domains found: {DOMAINS} (total {n_domains})")
fig, axes = plt.subplots(1, n_domains, figsize=(8 * n_domains, 6))
if n_domains == 1:
    axes = [axes]
fig.suptitle("Node Label Distribution per Domain", fontsize=11, y=1.02)
for ax, domain in zip(axes, DOMAINS):
    per_domain_no_ctx = Counter({k: v for k, v in Counter(node_labels_per_domain[domain]).items() if k != "context"})
    draw_pie(ax, per_domain_no_ctx, NODE_COLORS, domain, SCHEMA_NODE_ORDER, legend_on_side=True, n_datapoints=n_datapoints_per_domain[domain])
fig.tight_layout()
add_node_legend(fig)
fig.savefig(PLOTS_DIR / "05_node_labels_per_domain.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 05_node_labels_per_domain.svg")


# ── Plot 6: % edge labels – per domain ───────────────────────────────────────
fig, axes = plt.subplots(1, n_domains, figsize=(9 * n_domains, 7))
if n_domains == 1:
    axes = [axes]
fig.suptitle("Edge Label Distribution per Domain", fontsize=11, y=1.02)
for ax, domain in zip(axes, DOMAINS):
    draw_pie(ax, Counter(edge_labels_per_domain[domain]), EDGE_COLORS, domain, SCHEMA_EDGE_ORDER, legend_on_side=True, n_datapoints=n_datapoints_per_domain[domain])
fig.tight_layout()
add_edge_legend(fig)
fig.savefig(PLOTS_DIR / "06_edge_labels_per_domain.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 06_edge_labels_per_domain.svg")


# ── Plot 7: Node count distribution per generator (overlapped line chart) ────
# Log-scale x-axis with log-spaced bins.
# print([(x, len(y)) for x, y in node_counts_per_gen.items()])
all_counts = [c for counts in node_counts_per_gen.values() for c in counts]
X_MAX = max(all_counts)
X_MIN = max(1, min(all_counts))

# Log-spaced bins from X_MIN to X_MAX
bins = np.logspace(np.log10(X_MIN), np.log10(X_MAX + 1), num=15)
bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric midpoints

fig, ax = plt.subplots(figsize=(8, 5))

for gen in GENERATORS:
    counts = node_counts_per_gen[gen]
    hist, _ = np.histogram(counts, bins=bins)
    hist = hist / hist.sum()
    color = MODEL_COLORS.get(gen, "#888888")
    ax.plot(bin_centers, hist, color=color, label=gen, linewidth=1.8, markersize=3)

ax.set_xscale("log")

# Ticks: 1..9 × 10^k covering the data range
tick_vals = []
k = 0
while True:
    for d in range(1, 10):
        v = d * (10 ** k)
        if v > X_MAX * 1.1:
            break
        tick_vals.append(v)
    else:
        k += 1
        continue
    break
tick_vals = [v for v in tick_vals if v >= X_MIN * 0.9]

ax.set_xticks(tick_vals)
ax.set_xticklabels([str(v) for v in tick_vals], fontsize=7, rotation=45, ha="right")
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

ax.set_xlabel("Number of nodes per reasoning trace (log scale)")
ax.set_ylabel("Proportion")
ax.set_title("Number of Nodes per Reasoning Trace by Generator")
ax.legend(loc="upper right", fontsize=8, frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "07_node_count_per_trace.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 07_node_count_per_trace.svg")


# ── Plot 8: Node count distribution per generator, separated by domain ────────
fig, axes = plt.subplots(1, n_domains, figsize=(8 * n_domains, 5), sharey=False)
if n_domains == 1:
    axes = [axes]
fig.suptitle("Number of Nodes per Reasoning Trace by Generator (per Domain)", fontsize=11, y=1.02)

for ax, domain in zip(axes, DOMAINS):
    domain_counts = node_counts_per_gen_per_domain[domain]
    all_domain_counts = [c for counts in domain_counts.values() for c in counts]
    if not all_domain_counts:
        ax.set_title(domain, fontsize=9)
        continue
    # Use global X_MIN/X_MAX so all subplots share the same x range
    d_bins = np.logspace(np.log10(X_MIN), np.log10(X_MAX + 1), num=15)
    d_centers = np.sqrt(d_bins[:-1] * d_bins[1:])

    for gen in GENERATORS:
        if gen not in domain_counts:
            continue
        counts = domain_counts[gen]
        hist, _ = np.histogram(counts, bins=d_bins)
        hist = hist / hist.sum()
        color = MODEL_COLORS.get(gen, "#888888")
        ax.plot(d_centers, hist, color=color, label=gen, linewidth=1.8, markersize=3)

    ax.set_xscale("log")
    ax.set_xlim(X_MIN * 0.9, (X_MAX + 1) * 1.1)

    ax.set_xticks(tick_vals)
    ax.set_xticklabels([str(v) for v in tick_vals], fontsize=7, rotation=45, ha="right")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.set_xlabel("Number of nodes per reasoning trace (log scale)", fontsize=8)
    ax.set_ylabel("Proportion", fontsize=8)
    ax.set_title(domain, fontsize=9)
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(PLOTS_DIR / "08_node_count_per_trace_per_domain.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 08_node_count_per_trace_per_domain.svg")


# ── Analysis: crossing-edge statistics ────────────────────────────────────────
print("\n── Crossing-edge analysis ──────────────────────────────────────────────────")
global_avg_erased = np.mean(all_erased)
global_avg_total  = np.mean(all_totals)
global_avg_ratio  = np.mean([e / t if t > 0 else 0.0 for e, t in zip(all_erased, all_totals)])
print(f"Global average edges to erase : {global_avg_erased:.3f} / {global_avg_total:.3f} "
      f"({global_avg_ratio * 100:.1f}%) across {n_datapoints_total} files")

print("\nPer generator:")
for gen in GENERATORS:
    erased_list = crossing_erased_per_gen[gen]
    total_list  = crossing_total_per_gen[gen]
    avg_e = np.mean(erased_list)
    avg_t = np.mean(total_list)
    avg_r = np.mean([e / t if t > 0 else 0.0 for e, t in zip(erased_list, total_list)])
    print(f"  {gen:30s}  {avg_e:6.2f} / {avg_t:6.2f}  ({avg_r * 100:5.1f}%)")

print("\nPer domain:")
for domain in DOMAINS:
    erased_list = crossing_erased_per_domain[domain]
    total_list  = crossing_total_per_domain[domain]
    avg_e = np.mean(erased_list)
    avg_t = np.mean(total_list)
    avg_r = np.mean([e / t if t > 0 else 0.0 for e, t in zip(erased_list, total_list)])
    print(f"  {domain:30s}  {avg_e:6.2f} / {avg_t:6.2f}  ({avg_r * 100:5.1f}%)")


def _graph_stat_summary(stats: list[tuple]) -> tuple[float, float, float, float]:
    """Return (avg_nodes, avg_nonctx_nodes, avg_edges, avg_indegree) over a list of per-file stat tuples."""
    n = len(stats)
    return (
        sum(s[0] for s in stats) / n,
        sum(s[1] for s in stats) / n,
        sum(s[2] for s in stats) / n,
        sum(s[3] for s in stats) / n,
    )

print("\n── Graph structure statistics ──────────────────────────────────────────────")
print(f"{'':30s}  {'nodes':>7}  {'non-ctx':>7}  {'edges':>7}  {'avg in-deg':>10}  {'N':>5}")

avg_nodes, avg_nc, avg_edges, avg_indeg = _graph_stat_summary(all_graph_stats)
print(f"{'TOTAL':30s}  {avg_nodes:7.2f}  {avg_nc:7.2f}  {avg_edges:7.2f}  {avg_indeg:10.3f}  {n_datapoints_total:5d}")

print("\nPer model:")
for gen in GENERATORS:
    stats = graph_stats_per_gen[gen]
    avg_nodes, avg_nc, avg_edges, avg_indeg = _graph_stat_summary(stats)
    print(f"  {gen:28s}  {avg_nodes:7.2f}  {avg_nc:7.2f}  {avg_edges:7.2f}  {avg_indeg:10.3f}  {n_datapoints_per_gen[gen]:5d}")

print("\nPer dataset:")
for domain in DOMAINS:
    stats = graph_stats_per_domain[domain]
    avg_nodes, avg_nc, avg_edges, avg_indeg = _graph_stat_summary(stats)
    print(f"  {domain:28s}  {avg_nodes:7.2f}  {avg_nc:7.2f}  {avg_edges:7.2f}  {avg_indeg:10.3f}  {n_datapoints_per_domain[domain]:5d}")

print("\nPer (model, dataset):")
for gen in GENERATORS:
    for domain in DOMAINS:
        stats = graph_stats_per_gen_domain[gen].get(domain)
        if not stats:
            continue
        avg_nodes, avg_nc, avg_edges, avg_indeg = _graph_stat_summary(stats)
        label = f"{gen} / {domain}"
        print(f"  {label:45s}  {avg_nodes:7.2f}  {avg_nc:7.2f}  {avg_edges:7.2f}  {avg_indeg:10.3f}  {len(stats):5d}")

# ── Plot 9 & 10: Node/Edge label distribution – per (domain × generator) ──────
NODE_ORDER_NO_CTX = [l for l in SCHEMA_NODE_ORDER if l != "context"]


def _stacked_hbar(ax, label_order, color_map, per_domain_gen_data, domains, generators, title):
    """100% stacked horizontal bar chart, rows ordered domain → generator."""
    row_data = []
    for gen in generators:
        for domain in domains:
            labels_list = per_domain_gen_data[domain].get(gen, [])
            if not labels_list:
                continue
            counter = Counter(labels_list)
            total = sum(counter.get(l, 0) for l in label_order)
            props = [counter.get(l, 0) / total if total > 0 else 0.0 for l in label_order]
            row_data.append((gen, domain, props))

    n_rows = len(row_data)
    y = np.arange(n_rows)
    lefts = np.zeros(n_rows)

    for i, label in enumerate(label_order):
        values = np.array([row[2][i] for row in row_data])
        ax.barh(y, values, left=lefts, color=color_map.get(label, "#CCCCCC"), height=0.65)
        lefts += values

    ax.set_yticks(y)
    ax.set_yticklabels([row[1] for row in row_data], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Proportion")
    ax.set_title(title, fontsize=11, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Domain group separators and bold left-side labels
    domain_bounds: dict[str, list[int]] = {}
    for i, (domain, _, _) in enumerate(row_data):
        if domain not in domain_bounds:
            domain_bounds[domain] = [i, i]
        else:
            domain_bounds[domain][1] = i

    trans = ax.get_yaxis_transform()
    for domain, (start, end) in domain_bounds.items():
        mid = (start + end) / 2.0
        ax.text(-0.02, mid, domain, transform=trans,
                ha="right", va="center", fontsize=9, fontweight="bold", clip_on=False)
        if start > 0:
            ax.axhline(start - 0.5, color="gray", lw=0.8, linestyle="--", alpha=0.5)


fig, ax = plt.subplots(figsize=(12, 8))
_stacked_hbar(ax, NODE_ORDER_NO_CTX, NODE_COLORS,
              node_labels_per_gen_domain, DOMAINS, GEN_ORDER,
              "Node Label Distribution by Domain × Generator (excl. context)")
fig.tight_layout()
add_node_legend(fig)
fig.savefig(PLOTS_DIR / "09_node_labels_domain_gen.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 09_node_labels_domain_gen.svg")

fig, ax = plt.subplots(figsize=(12, 8))
_stacked_hbar(ax, SCHEMA_EDGE_ORDER, EDGE_COLORS,
              edge_labels_per_gen_domain, DOMAINS, GEN_ORDER,
              "Edge Label Distribution by Domain × Generator")
fig.tight_layout()
add_edge_legend(fig)
fig.savefig(PLOTS_DIR / "10_edge_labels_domain_gen.svg", format="svg", bbox_inches="tight")
plt.close(fig)
print("Saved 10_edge_labels_domain_gen.svg")

print("\nAll plots saved to", PLOTS_DIR.resolve())
