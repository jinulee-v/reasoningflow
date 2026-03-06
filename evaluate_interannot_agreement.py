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

def normalize(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text)

# ============================================================
# Load all annotations grouped by doc_id and annotator
# ============================================================

# DIRECTORIES = ["data/v0_human_D", "data/v0_human_A", "data/v0_human_B", "data/v0_human_C"]
DIRECTORIES = ["data/v0_human_D", "data/v0_llm_gemini-3-pro-preview"]

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

    # Build node lookup by normalized text for each annotator
    node_by_norm = []
    for datum in datums:
        lookup = {}
        for node in datum["nodes"]:
            lookup[normalize(node["text"])] = node
        node_by_norm.append(lookup)

    # Common nodes across all annotators
    common_norms = set(node_by_norm[0].keys())
    for lookup in node_by_norm[1:]:
        common_norms &= set(lookup.keys())

    # Node alignment F1 (pairwise, only when exactly 2 annotators)
    if len(annotators) == 2:
        norms_a = set(node_by_norm[0].keys())
        norms_b = set(node_by_norm[1].keys())
        tp = len(common_norms)
        fn = len(norms_a - common_norms)
        fp = len(norms_b - common_norms)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        node_alignment_f1_scores.append(f1)

    # Node ratings
    for norm in common_norms:
        unit_key = (doc_id, norm)
        if unit_key not in node_ratings:
            node_ratings[unit_key] = {}
        for ann, lookup in zip(annotators, node_by_norm):
            node_ratings[unit_key][ann] = lookup[norm]["label"]

    # Build edge lookup by normalized (src, dst) text pair for each annotator
    # Only include edges where both endpoints are among the aligned (common) nodes,
    # mirroring the `corresponding_nodes_h2l`/`corresponding_nodes_l2h` guard in evaluate_llm_annot.py
    edge_by_norm = []
    for datum, node_lookup in zip(datums, node_by_norm):
        node_id_to_norm = {node["id"]: normalize(node["text"]) for node in datum["nodes"]}
        lookup = {}
        for edge in datum["edges"]:
            src_norm = node_id_to_norm.get(edge["source_node_id"])
            dst_norm = node_id_to_norm.get(edge["dest_node_id"])
            if src_norm in common_norms and dst_norm in common_norms:
                lookup[(src_norm, dst_norm)] = edge["label"]
        edge_by_norm.append(lookup)

    # Edge ratings for all ordered pairs of common nodes
    common_norms_list = list(common_norms)
    for src_norm in common_norms_list:
        for dst_norm in common_norms_list:
            if src_norm == dst_norm:
                continue
            unit_key = (doc_id, src_norm, dst_norm)
            if unit_key not in edge_unlabeled_ratings:
                edge_unlabeled_ratings[unit_key] = {}
                edge_labeled_ratings[unit_key] = {}
            for ann, edge_lookup in zip(annotators, edge_by_norm):
                label = edge_lookup.get((src_norm, dst_norm), NO_EDGE_LABEL)
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
