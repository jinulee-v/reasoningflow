import json
import os
import re
import yaml
import numpy as np
import krippendorff

# ============================================================
# Load schema
# ============================================================

with open("schema/node_labels.yaml", "r") as f:
    NODE_LABELS = [node["name"] for node in yaml.safe_load(f)["nodes"]]

with open("schema/edge_labels.yaml", "r") as f:
    EDGE_DATA = yaml.safe_load(f)
    EDGE_LABELS = [edge["name"] for edge in EDGE_DATA["edges"]]

NO_EDGE_LABEL = "__NO_EDGE__"
JACCARD_THRESHOLD = 0.9

def node_word_tokens(text):
    return set(re.findall(r'[a-zA-Z0-9]+', text.lower()))

def jaccard_sim(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0

def align_nodes_monotone(ref_nodes, other_nodes):
    """Order-preserving Jaccard DP; returns [(ref_idx, other_idx), ...] in order."""
    ref_tok   = [node_word_tokens(n["text"]) for n in ref_nodes]
    other_tok = [node_word_tokens(n["text"]) for n in other_nodes]
    n_r, n_o = len(ref_tok), len(other_tok)
    if n_r == 0 or n_o == 0:
        return []
    sim_mat = np.array([[jaccard_sim(rt, ot) for ot in other_tok] for rt in ref_tok])
    dp = np.zeros((n_r + 1, n_o + 1))
    for i in range(1, n_r + 1):
        for j in range(1, n_o + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            if sim_mat[i - 1][j - 1] > JACCARD_THRESHOLD:
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + sim_mat[i - 1][j - 1])
    pairs = []
    i, j = n_r, n_o
    while i > 0 and j > 0:
        s = sim_mat[i - 1][j - 1]
        if s > JACCARD_THRESHOLD and dp[i - 1][j - 1] + s >= dp[i][j] - 1e-10:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs

# ============================================================
# Load all annotations grouped by doc_id and annotator
# ============================================================

# DIRECTORIES = ["data/v0_human_1", "data/v0_human_2"]
DIRECTORIES = ["data/v0_human_1", "data/v0_llm_gpt-5.1-2025-11-13_groundseg"]

all_annotations = {}  # {doc_id: {annotator: datum}}

for directory in DIRECTORIES:
    if not os.path.isdir(directory):
        continue
    for file in sorted(os.listdir(directory)):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(directory, file), "r") as f:
            datum = json.load(f)
        doc_id = datum["doc_id"]
        annotator = datum["metadata"]["annotator"]
        if doc_id not in all_annotations:
            all_annotations[doc_id] = {}
        all_annotations[doc_id][annotator] = datum

# ============================================================
# Collect ratings per unit across all annotators
# ============================================================

# {unit_key: {annotator: label}}
node_ratings = {}
edge_unlabeled_ratings = {}
edge_labeled_ratings = {}

n_docs = 0
node_alignment_f1_scores = []  # only populated for 2-annotator documents

for doc_id, annotator_data in all_annotations.items():
    if len(annotator_data) < 2:
        continue
    n_docs += 1

    annotators = list(annotator_data.keys())
    datums = [annotator_data[a] for a in annotators]

    # Align every annotator's nodes to the first (reference) annotator using
    # order-preserving Jaccard DP — same algorithm as evaluate_llm_annot.py.
    ref_nodes = datums[0]["nodes"]
    alignments = []  # alignments[k]: ref_idx -> other_idx (for annotators[k+1])
    for datum in datums[1:]:
        pairs = align_nodes_monotone(ref_nodes, datum["nodes"])
        alignments.append({r: o for r, o in pairs})

    # Node alignment F1 (pairwise, only when exactly 2 annotators)
    if len(annotators) == 2:
        n_matched = len(alignments[0])
        n_h = len(ref_nodes)
        n_o = len(datums[1]["nodes"])
        prec = n_matched / n_o if n_o > 0 else 0
        rec  = n_matched / n_h if n_h > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        node_alignment_f1_scores.append(f1)

    # Common reference indices: those matched in ALL other annotators
    common_ref_indices = set(range(len(ref_nodes)))
    for alignment in alignments:
        common_ref_indices &= set(alignment.keys())

    # Node ratings (unit key: reference node id)
    for ref_idx in common_ref_indices:
        unit_key = (doc_id, ref_nodes[ref_idx]["id"])
        if unit_key not in node_ratings:
            node_ratings[unit_key] = {}
        node_ratings[unit_key][annotators[0]] = ref_nodes[ref_idx]["label"]
        for ann, alignment, datum in zip(annotators[1:], alignments, datums[1:]):
            node_ratings[unit_key][ann] = datum["nodes"][alignment[ref_idx]]["label"]

    # Edge lookups keyed by (ref_src_idx, ref_dst_idx) for each annotator.
    # Only include edges where both endpoints are in common_ref_indices.
    ref_node_id_to_ref_idx = {n["id"]: i for i, n in enumerate(ref_nodes)}

    ref_edge_lookup = {}
    for edge in datums[0]["edges"]:
        si = ref_node_id_to_ref_idx.get(edge["source_node_id"])
        di = ref_node_id_to_ref_idx.get(edge["dest_node_id"])
        if si in common_ref_indices and di in common_ref_indices:
            ref_edge_lookup[(si, di)] = edge["label"]

    other_edge_lookups = []
    for datum, alignment in zip(datums[1:], alignments):
        rev_alignment = {o: r for r, o in alignment.items()}
        other_node_id_to_idx = {n["id"]: i for i, n in enumerate(datum["nodes"])}
        edge_lookup = {}
        for edge in datum["edges"]:
            si_o = other_node_id_to_idx.get(edge["source_node_id"])
            di_o = other_node_id_to_idx.get(edge["dest_node_id"])
            if si_o is None or di_o is None:
                continue
            si_r = rev_alignment.get(si_o)
            di_r = rev_alignment.get(di_o)
            if si_r in common_ref_indices and di_r in common_ref_indices:
                edge_lookup[(si_r, di_r)] = edge["label"]
        other_edge_lookups.append(edge_lookup)

    # Edge ratings for all ordered pairs of common reference nodes
    common_ref_list = sorted(common_ref_indices)
    for src_idx in common_ref_list:
        for dst_idx in common_ref_list:
            if src_idx == dst_idx:
                continue
            unit_key = (doc_id, ref_nodes[src_idx]["id"], ref_nodes[dst_idx]["id"])
            if unit_key not in edge_unlabeled_ratings:
                edge_unlabeled_ratings[unit_key] = {}
                edge_labeled_ratings[unit_key] = {}

            ref_label = ref_edge_lookup.get((src_idx, dst_idx), NO_EDGE_LABEL)
            edge_unlabeled_ratings[unit_key][annotators[0]] = "edge" if ref_label != NO_EDGE_LABEL else NO_EDGE_LABEL
            edge_labeled_ratings[unit_key][annotators[0]] = ref_label

            for ann, edge_lookup in zip(annotators[1:], other_edge_lookups):
                label = edge_lookup.get((src_idx, dst_idx), NO_EDGE_LABEL)
                edge_unlabeled_ratings[unit_key][ann] = "edge" if label != NO_EDGE_LABEL else NO_EDGE_LABEL
                edge_labeled_ratings[unit_key][ann] = label

# ============================================================
# Build reliability matrices and compute Krippendorff's alpha
# ============================================================

def build_reliability_matrix(ratings_dict, categories):
    """
    ratings_dict: {unit_key: {annotator: label}}
    Returns a matrix of shape (n_annotators, n_units) with category indices,
    using np.nan for missing values.
    """
    all_annotators = sorted({ann for ratings in ratings_dict.values() for ann in ratings})
    units = list(ratings_dict.keys())
    cat_to_idx = {cat: float(i) for i, cat in enumerate(categories)}

    matrix = np.full((len(all_annotators), len(units)), np.nan)
    for j, unit_key in enumerate(units):
        for i, ann in enumerate(all_annotators):
            label = ratings_dict[unit_key].get(ann)
            if label is not None and label in cat_to_idx:
                matrix[i, j] = cat_to_idx[label]
    return matrix

print(f"Evaluated {n_docs} documents")

print("=======================================")
print("Node Alignment F1 (2-annotator documents only)")
print("-----------------------------------------------")
if node_alignment_f1_scores:
    avg_f1 = sum(node_alignment_f1_scores) / len(node_alignment_f1_scores)
    print(f"Node Segmentation F1 (macro/doc): {avg_f1:.4f}  (N={len(node_alignment_f1_scores)} docs)")
else:
    print("Node Segmentation F1:             N/A (no 2-annotator documents)")

print("")
print("=======================================")
print("Inter-Annotator Agreement (Krippendorff's α)")
print("-----------------------------------------------")

# Node classification α
if node_ratings:
    matrix = build_reliability_matrix(node_ratings, NODE_LABELS)
    alpha_node = krippendorff.alpha(reliability_data=matrix, level_of_measurement='nominal')
    print(f"Node Classification α:        {alpha_node:.4f}  (N={len(node_ratings)})")
else:
    print("Node Classification α:        N/A")

# Edge unlabeled α (binary)
if edge_unlabeled_ratings:
    matrix = build_reliability_matrix(edge_unlabeled_ratings, ["edge", NO_EDGE_LABEL])
    alpha_edge_unlabeled = krippendorff.alpha(reliability_data=matrix, level_of_measurement='nominal')
    print(f"Unlabeled Edge Detection α:   {alpha_edge_unlabeled:.4f}  (N={len(edge_unlabeled_ratings)})")
else:
    print("Unlabeled Edge Detection α:   N/A")

# Edge labeled α (N + 1 classes)
if edge_labeled_ratings:
    matrix = build_reliability_matrix(edge_labeled_ratings, EDGE_LABELS + [NO_EDGE_LABEL])
    alpha_edge_labeled = krippendorff.alpha(reliability_data=matrix, level_of_measurement='nominal')
    print(f"Labeled Edge Detection α:     {alpha_edge_labeled:.4f}  (N={len(edge_labeled_ratings)})")
else:
    print("Labeled Edge Detection α:     N/A")

print("")
print("Legend:")
print("  - Node α: agreement on node labels (aligned nodes only)")
print("  - Unlabeled α: agreement on existence of edge per node pair")
print("  - Labeled α: agreement on full edge label per node pair (N + no-edge)")
