"""
Compute (node_label, edge_label, node_label) triplet distributions per (model, dataset),
project to 2D via MDS using Jensen-Shannon divergence, and plot with color=model, shape=dataset.
Also compares model pairs via Mann-Whitney U p-value and Jensen-Shannon divergence heatmap.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.special import rel_entr
from scipy.stats import mannwhitneyu

plt.rcParams["svg.fonttype"] = "none"

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"


def load_triplets():
    """Return dict: (model, dataset) -> Counter of (src_label, edge_label, dst_label)."""
    counts = defaultdict(lambda: defaultdict(int))

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json"):
            continue
        if "bright" in fname:
            continue
        m = re.match(r"(.+)_(\d+)_(.+)\.json$", fname)
        if not m:
            continue
        dataset, _, model = m.group(1), m.group(2), m.group(3)

        with open(os.path.join(DATA_DIR, fname)) as f:
            doc = json.load(f)

        node_label = {n["id"]: n["label"] for n in doc.get("nodes", [])}

        for edge in doc.get("edges", []):
            src = node_label.get(edge["source_node_id"])
            dst = node_label.get(edge["dest_node_id"])
            elabel = edge["label"]
            if src and dst:
                triplet = (src, elabel, dst)
                counts[(model, dataset)][triplet] += 1

    return counts


def build_distributions(counts):
    """Convert raw counts to probability distributions over a shared vocabulary."""
    # Collect all triplet types
    vocab = sorted({t for c in counts.values() for t in c})
    vocab_idx = {t: i for i, t in enumerate(vocab)}

    keys = sorted(counts.keys())
    dists = np.zeros((len(keys), len(vocab)))
    for i, key in enumerate(keys):
        for t, v in counts[key].items():
            dists[i, vocab_idx[t]] += v
        total = dists[i].sum()
        if total > 0:
            dists[i] /= total

    return keys, dists


def jsd(p, q, eps=1e-10):
    """Jensen-Shannon divergence: (KL(p||m) + KL(q||m)) / 2, where m = (p+q)/2."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (rel_entr(p, m).sum() + rel_entr(q, m).sum())


def build_distance_matrix(dists):
    n = len(dists)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jsd(dists[i], dists[j])
            D[i, j] = D[j, i] = d
    return D


def plot(keys, coords):
    models = sorted({k[0] for k in keys})
    datasets = sorted({k[1] for k in keys})

    model_color = {
        "Qwen2.5-32B-Instruct": "#8ED7D7",
        "QwQ-32B":              "#007A84",
        "DeepSeek-V3":          "#FF8CA1",
        "DeepSeek-R1":          "#E42741",
        "gpt-oss-120b":         "#98C126",
    }

    # matplotlib marker styles
    marker_styles = ["o", "s", "^", "D", "P", "*", "X", "v"]
    dataset_marker = {d: marker_styles[i % len(marker_styles)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8,5))

    for i, (model, dataset) in enumerate(keys):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            color=model_color[model],
            marker=dataset_marker[dataset],
            s=120,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )

    # Legend: colors for models
    model_handles = [
        mpatches.Patch(color=model_color[m], label=m) for m in models
    ]
    # Legend: shapes for datasets
    dataset_handles = [
        plt.Line2D(
            [0], [0],
            marker=dataset_marker[d],
            color="w",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="k",
            label=d,
        )
        for d in datasets
    ]

    legend1 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        framealpha=0.9,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0,
        framealpha=0.9,
    )

    ax.set_title("MDS projection of (model, dataset) pairs\nby Jensen-Shannon divergence of triplet distributions")
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig("plots/triplet_mds.svg")
    print("Saved plots/triplet_mds.svg")
    plt.show()


def build_prob_vectors(counts):
    """Build Hellinger embedding vectors (sqrt(P)) over the shared triplet vocabulary."""
    vocab = sorted({t for c in counts.values() for t in c})
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    keys = sorted(counts.keys())
    mat = np.zeros((len(keys), len(vocab)))
    for i, key in enumerate(keys):
        for t, v in counts[key].items():
            mat[i, vocab_idx[t]] = v
        total = mat[i].sum()
        if total > 0:
            mat[i] /= total
    return keys, np.sqrt(mat), vocab


def print_pca_directions(vocab, components, n=5):
    """Print top-n triplets by absolute loading for each principal component."""
    for pc_idx, loadings in enumerate(components):
        top = np.argsort(np.abs(loadings))[-n:][::-1]
        print(f"\n  PC{pc_idx + 1} top {n} dimensions (by |loading|):")
        for rank, i in enumerate(top, 1):
            src, edge, dst = vocab[i]
            print(f"    {rank}. {loadings[i]:+.4f}  {src} –[{edge}]– {dst}")


def plot_pca(keys, coords):
    models = sorted({k[0] for k in keys})
    datasets = sorted({k[1] for k in keys})

    model_color = {
        "Qwen2.5-32B-Instruct": "#8ED7D7",
        "QwQ-32B":              "#007A84",
        "DeepSeek-V3":          "#FF8CA1",
        "DeepSeek-R1":          "#E42741",
        "gpt-oss-120b":         "#98C126",
    }
    marker_styles = ["o", "s", "^", "D", "P", "*", "X", "v"]
    dataset_marker = {d: marker_styles[i % len(marker_styles)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model, dataset) in enumerate(keys):
        ax.scatter(
            coords[i, 0], coords[i, 1],
            color=model_color.get(model, "gray"),
            marker=dataset_marker[dataset],
            s=120, edgecolors="k", linewidths=0.5, zorder=3,
        )

    model_handles = [mpatches.Patch(color=model_color.get(m, "gray"), label=m) for m in models]
    dataset_handles = [
        plt.Line2D([0], [0], marker=dataset_marker[d], color="w",
                   markerfacecolor="gray", markersize=10, markeredgecolor="k", label=d)
        for d in datasets
    ]

    legend1 = ax.legend(handles=model_handles, title="Model",
                        loc="upper left", bbox_to_anchor=(1.02, 1),
                        borderaxespad=0, framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=dataset_handles, title="Dataset",
              loc="lower left", bbox_to_anchor=(1.02, 0),
              borderaxespad=0, framealpha=0.9)

    ax.set_title("PCA of (model, dataset) pairs\nby Hellinger embedding of triplet distributions")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/triplet_pca.svg")
    print("Saved plots/triplet_pca.svg")
    plt.show()


def aggregate_counts_per_model(counts):
    """Aggregate raw triplet counts across all datasets per model."""
    model_counts = defaultdict(lambda: defaultdict(int))
    for (model, dataset), triplet_counts in counts.items():
        for triplet, cnt in triplet_counts.items():
            model_counts[model][triplet] += cnt
    return model_counts


def build_model_count_matrix(model_counts):
    """Build a matrix of raw counts (models x vocab) with shared vocabulary."""
    vocab = sorted({t for mc in model_counts.values() for t in mc})
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    models = sorted(model_counts.keys())

    mat = np.zeros((len(models), len(vocab)))
    for i, model in enumerate(models):
        for t, v in model_counts[model].items():
            mat[i, vocab_idx[t]] += v
    return models, mat


MODEL_ORDER = [
    "Qwen2.5-32B-Instruct",
    "DeepSeek-V3",
    "QwQ-32B",
    "DeepSeek-R1",
    "gpt-oss-120b",
]

SHORT_NAMES = {
    "Qwen2.5-32B-Instruct": "Qwen32B",
    "QwQ-32B":              "QwQ",
    "DeepSeek-V3":          "DS-V3",
    "DeepSeek-R1":          "DS-R1",
    "gpt-oss-120b":         "gpt-oss",
}


def sig_stars(pv):
    if pv < 0.001:
        return "***"
    if pv < 0.01:
        return "**"
    if pv < 0.05:
        return "*"
    return ""


def plot_model_heatmap(models, p_values, jsd_values, show_significance=True):
    """
    Heatmap colored by Jensen-Shannon divergence.
    When show_significance=True, appends Mann-Whitney U stars to each cell and saves
    to triplet_model_heatmap.svg; otherwise saves to triplet_model_heatmap_jsd.svg.
    """
    ordered = [m for m in MODEL_ORDER if m in models]
    idx = [models.index(m) for m in ordered]
    n = len(ordered)

    p_ord = p_values[np.ix_(idx, idx)]
    jsd_ord = jsd_values[np.ix_(idx, idx)]
    labels = [SHORT_NAMES.get(m, m) for m in ordered]

    display = jsd_ord.copy().astype(float)
    np.fill_diagonal(display, np.nan)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "white_orange", ["#ffffff", "#fd901c"]
    )
    cmap.set_bad("#e8e8e8")

    fig, ax = plt.subplots(figsize=(n * 1.4, n * 1))
    im = ax.imshow(display, cmap=cmap, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Jensen-Shannon divergence", fontsize=10)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=13, color="#999999")
                continue
            js = jsd_ord[i, j]
            if show_significance:
                label = f"{js:.3f}{sig_stars(p_ord[i, j])}"
            else:
                label = f"{js:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="black")

    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=1.5)
        ax.axvline(k - 0.5, color="white", linewidth=1.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)

    sig_text = "* p<0.05   ** p<0.01   *** p<0.001"
    ax.text(
        1.02, 0.02, sig_text,
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
    ) # Add this text for JSD-only plot too, to match the layout
        
    if show_significance:
        title_suffix = "\n(Jensen-Shannon divergence, Mann-Whitney U significance)"
        out_path = "plots/triplet_model_heatmap.svg"
    else:
        title_suffix = "\n(Jensen-Shannon divergence)"
        out_path = "plots/triplet_model_heatmap_jsd.svg"

    ax.set_title(
        "Pairwise model comparison of triplet distributions" + title_suffix,
        fontsize=11, pad=30,
    )
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


def main():
    print("Loading triplet counts...")
    counts = load_triplets()
    print(f"  Found {len(counts)} (model, dataset) pairs")

    print("Building distributions...")
    keys, dists = build_distributions(counts)

    print("Computing distance matrix...")
    D = build_distance_matrix(dists)

    print("Running MDS...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
    coords = mds.fit_transform(D)

    print("Plotting MDS...")
    plot(keys, coords)

    print("Building Hellinger embedding vectors for PCA...")
    pca_keys, prob_mat, vocab = build_prob_vectors(counts)
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(prob_mat)
    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    print_pca_directions(vocab, pca.components_)
    print("Plotting PCA...")
    plot_pca(pca_keys, pca_coords)

    # --- Pairwise model heatmap ---
    print("Aggregating counts per model...")
    model_counts = aggregate_counts_per_model(counts)
    models, count_mat = build_model_count_matrix(model_counts)
    n = len(models)
    print(f"  {n} models: {models}")

    # Normalize to distributions for KL
    dist_mat = count_mat / count_mat.sum(axis=1, keepdims=True).clip(min=1e-10)

    p_values = np.zeros((n, n))
    jsd_values = np.zeros((n, n))

    print("Computing pairwise Mann-Whitney U and Jensen-Shannon divergence...")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Mann-Whitney U on raw count vectors (one value per triplet type)
            stat, pv = mannwhitneyu(count_mat[i], count_mat[j], alternative="two-sided")
            p_values[i, j] = pv
            jsd_values[i, j] = jsd(dist_mat[i], dist_mat[j])

    print("Plotting model heatmaps...")
    plot_model_heatmap(models, p_values, jsd_values, show_significance=True)
    plot_model_heatmap(models, p_values, jsd_values, show_significance=False)


if __name__ == "__main__":
    main()
