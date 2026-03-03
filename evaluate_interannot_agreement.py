import json
import os
import re
import yaml
from sklearn.metrics import cohen_kappa_score

# ============================================================
# Load schema
# ============================================================

with open("schema/node_labels.yaml", "r") as f:
    NODE_LABELS = [node["name"] for node in yaml.safe_load(f)["nodes"]]

with open("schema/edge_labels.yaml", "r") as f:
    EDGE_DATA = yaml.safe_load(f)
    EDGE_LABELS = [edge["name"] for edge in EDGE_DATA["edges"]]

NO_EDGE_LABEL = "__NO_EDGE__"

# ============================================================
# Containers for global IAA computation
# ============================================================

n_docs = 0

# Node classification (only for aligned nodes)
node_true_labels = []
node_pred_labels = []

# Edge unlabeled (binary: edge / no_edge)
edge_unlabeled_true = []
edge_unlabeled_pred = []

# Edge labeled (multi-class: N edge labels + no_edge)
edge_labeled_true = []
edge_labeled_pred = []

# ============================================================
# Iterate over documents
# ============================================================

for file in sorted(os.listdir("data/v0_human_jinu_lee")):
    if not file.endswith(".json"):
        continue

    try:
        with open(os.path.join("data/v0_human_jinu_lee", file), "r") as f:
            datum_a = json.load(f)
        with open(os.path.join("data/v0_human_A", file), "r") as f:
        # with open(os.path.join("data/v0_llm_gemini-3-pro-preview", file), "r") as f:
            datum_b = json.load(f)

        assert datum_a["doc_id"] == datum_b["doc_id"]
        n_docs += 1
    except Exception:
        continue

    # --------------------------------------------------------
    # Align nodes by normalized text
    # --------------------------------------------------------

    corresponding_nodes_a2b = {}
    corresponding_nodes_b2a = {}

    for na in datum_a["nodes"]:
        for nb in datum_b["nodes"]:
            if re.sub(r'[^a-zA-Z0-9]', '', na["text"]) == \
               re.sub(r'[^a-zA-Z0-9]', '', nb["text"]):
                corresponding_nodes_a2b[na["id"]] = nb
                corresponding_nodes_b2a[nb["id"]] = na

    # --------------------------------------------------------
    # Node classification agreement (Cohen's kappa)
    # --------------------------------------------------------

    for na in datum_a["nodes"]:
        if na["id"] in corresponding_nodes_a2b:
            nb = corresponding_nodes_a2b[na["id"]]
            node_true_labels.append(na["label"])
            node_pred_labels.append(nb["label"])

    # --------------------------------------------------------
    # Build edge lookup tables restricted to aligned nodes
    # --------------------------------------------------------

    # A edges
    edges_a = {}
    for ea in datum_a["edges"]:
        if ea["source_node_id"] in corresponding_nodes_a2b and \
           ea["dest_node_id"] in corresponding_nodes_a2b:
            src_b = corresponding_nodes_a2b[ea["source_node_id"]]["id"]
            dst_b = corresponding_nodes_a2b[ea["dest_node_id"]]["id"]
            edges_a[(ea["source_node_id"], ea["dest_node_id"])] = ea["label"]

    # B edges (indexed by A-space node ids)
    edges_b = {}
    for eb in datum_b["edges"]:
        if eb["source_node_id"] in corresponding_nodes_b2a and \
           eb["dest_node_id"] in corresponding_nodes_b2a:
            src_a = corresponding_nodes_b2a[eb["source_node_id"]]["id"]
            dst_a = corresponding_nodes_b2a[eb["dest_node_id"]]["id"]
            edges_b[(src_a, dst_a)] = eb["label"]

    # --------------------------------------------------------
    # Enumerate all ordered node pairs for agreement
    # --------------------------------------------------------

    aligned_node_ids = list(corresponding_nodes_a2b.keys())

    for src in aligned_node_ids:
        for dst in aligned_node_ids:
            if src == dst:
                continue

            label_a = edges_a.get((src, dst), NO_EDGE_LABEL)
            label_b = edges_b.get((src, dst), NO_EDGE_LABEL)

            # ----- Unlabeled (binary) -----
            edge_unlabeled_true.append(
                "edge" if label_a != NO_EDGE_LABEL else NO_EDGE_LABEL
            )
            edge_unlabeled_pred.append(
                "edge" if label_b != NO_EDGE_LABEL else NO_EDGE_LABEL
            )

            # ----- Labeled (N + 1 classes) -----
            edge_labeled_true.append(label_a)
            edge_labeled_pred.append(label_b)

# ============================================================
# Compute Cohen's kappa
# ============================================================

print(f"Evaluated {n_docs} documents")

print("=======================================")
print("Inter-Annotator Agreement (Cohen's κ)")
print("---------------------------------------")

# Node classification κ
if node_true_labels:
    kappa_node = cohen_kappa_score(
        node_true_labels,
        node_pred_labels,
        labels=NODE_LABELS
    )
    print(f"Node Classification κ:        {kappa_node:.4f}")
else:
    print("Node Classification κ:        N/A")

# Edge unlabeled κ (binary)
if edge_unlabeled_true:
    kappa_edge_unlabeled = cohen_kappa_score(
        edge_unlabeled_true,
        edge_unlabeled_pred,
        labels=["edge", NO_EDGE_LABEL]
    )
    print(f"Unlabeled Edge Detection κ:   {kappa_edge_unlabeled:.4f}")
else:
    print("Unlabeled Edge Detection κ:   N/A")

# Edge labeled κ (N + 1 classes)
if edge_labeled_true:
    kappa_edge_labeled = cohen_kappa_score(
        edge_labeled_true,
        edge_labeled_pred,
        labels=EDGE_LABELS + [NO_EDGE_LABEL]
    )
    print(f"Labeled Edge Agreement κ:     {kappa_edge_labeled:.4f}")
else:
    print("Labeled Edge Agreement κ:     N/A")

print("")
print("Legend:")
print("  - Node κ: agreement on node labels (aligned nodes only)")
print("  - Unlabeled κ: agreement on existence of edge per node pair")
print("  - Labeled κ: agreement on full edge label per node pair (N + no-edge)")