import json
import os
from collections import defaultdict
import re
from matplotlib import pyplot as plt
import yaml

with open("schema/node_labels.yaml", "r") as f:
    NODE_LABELS = [node["name"] for node in yaml.safe_load(f)["nodes"]]
    print(NODE_LABELS)
with open("schema/edge_labels.yaml", "r") as f:
    EDGE_DATA = yaml.safe_load(f)
    EDGE_LABELS = [edge["name"] for edge in EDGE_DATA["edges"]]
    print(EDGE_LABELS)

node_segmentation_f1_scores = []
node_classification_acc_scores = []
edge_detection_precision_scores = []
edge_detection_recall_scores = []
edge_detection_f1_scores = []
edge_detcls_precision_scores = []
edge_detcls_recall_scores = []
edge_detcls_f1_scores = []
edge_label_f1_scores = []
global_node_label_confusion_matrix = defaultdict(lambda: defaultdict(int))
global_edge_label_confusion_matrix = defaultdict(lambda: defaultdict(int))
global_edge_label_confusion_matrix_filtered = defaultdict(lambda: defaultdict(int))

for file in sorted(os.listdir("data/v0_human_jinu_lee")):
    if not file.endswith(".json"):
        continue
    try:
        with open(os.path.join("data/v0_human_jinu_lee", file), "r") as f:
            datum_human = json.load(f)
        with open(os.path.join("data/v0_llm_gemini-2.5-flash", file), "r") as f:
            datum_llm = json.load(f)
        assert datum_human["doc_id"] == datum_llm["doc_id"]
    except FileNotFoundError:
        # filenotexist
        continue
    except Exception as e:
        print(e.__class__, e)
        continue
    print(file)
    
    corresponding_nodes_h2l = {} # hn idx to ln
    corresponding_nodes_l2h = {} # ln idx to hn
    for hn in datum_human["nodes"]:
        for ln in datum_llm["nodes"]:
            if re.sub(r'[^a-zA-Z0-9]', '', hn["text"]) == re.sub(r'[^a-zA-Z0-9]', '', ln["text"]):
                corresponding_nodes_h2l[hn["id"]] = ln
                corresponding_nodes_l2h[ln["id"]] = hn
            
    # Node segmentation F1
    tp, fp, fn = 0, 0, 0
    for hn in datum_human["nodes"]:
        if hn["id"] in corresponding_nodes_h2l:
            tp += 1
        else:
            fn += 1
    for ln in datum_llm["nodes"]:
        if ln["id"] not in corresponding_nodes_l2h:
            fp += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    # print("  node segmentation:", round(prec, 4), round(rec, 4), round(f1, 4))
    node_segmentation_f1_scores.append(f1)
    
    # Node labels F1
    node_acc_correct = 0
    node_acc_total = 0
    for hn in datum_human["nodes"]:
        if hn["id"] in corresponding_nodes_h2l:
            ln = corresponding_nodes_h2l[hn["id"]]
            global_node_label_confusion_matrix[hn["label"]][ln["label"]] += 1
            
            node_acc_total += 1
            if hn["label"] == ln["label"]:
                node_acc_correct += 1
    node_acc = node_acc_correct / node_acc_total if node_acc_total > 0 else 0
    node_classification_acc_scores.append(node_acc)
    
    # Find valid edges
    corresponding_edges_h2l = {}
    corresponding_edges_l2h = {}
    for he in datum_human["edges"]:
        if he["source_node_id"] in corresponding_nodes_h2l and he["dest_node_id"] in corresponding_nodes_h2l:
            le_from = corresponding_nodes_h2l[he["source_node_id"]]
            le_to = corresponding_nodes_h2l[he["dest_node_id"]]
            # Check if such an edge exists in LLM data
            for le in datum_llm["edges"]:
                if le["source_node_id"] == le_from["id"] and le["dest_node_id"] == le_to["id"]:
                    corresponding_edges_h2l[he["id"]] = le
                    corresponding_edges_l2h[le["id"]] = he
                    break
    
    # Edge unlabeled F1
    tp, fp, fn = 0, 0, 0
    for he in datum_human["edges"]:
        if not (he["source_node_id"] in corresponding_nodes_h2l and he["dest_node_id"] in corresponding_nodes_h2l):
            continue
        if he["id"] in corresponding_edges_h2l:
            tp += 1
        else:
            fn += 1
    for le in datum_llm["edges"]:
        if not (le["source_node_id"] in corresponding_nodes_l2h and le["dest_node_id"] in corresponding_nodes_l2h):
            continue
        if le["id"] not in corresponding_edges_l2h:
            fp += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    # print("  edge detection:", round(prec, 4), round(rec, 4), round(f1, 4))
    edge_detection_precision_scores.append(prec)
    edge_detection_recall_scores.append(rec)
    edge_detection_f1_scores.append(f1)
    
    # Edge labeled F1
    tp, fp, fn = 0, 0, 0
    for he in datum_human["edges"]:
        if not (he["source_node_id"] in corresponding_nodes_h2l and he["dest_node_id"] in corresponding_nodes_h2l):
            continue
        if he["id"] in corresponding_edges_h2l and corresponding_edges_h2l[he["id"]]["label"] == he["label"]:
            tp += 1
        else:
            fn += 1
    for le in datum_llm["edges"]:
        if not (le["source_node_id"] in corresponding_nodes_l2h and le["dest_node_id"] in corresponding_nodes_l2h):
            continue
        if le["id"] not in corresponding_edges_l2h or corresponding_edges_l2h[le["id"]]["label"] != le["label"]:
            fp += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    edge_detcls_precision_scores.append(prec)
    edge_detcls_recall_scores.append(rec)
    edge_detcls_f1_scores.append(f1)
    
    # Global edge label confusion matrix
    for he in datum_human["edges"]:
        if he["source_node_id"] not in corresponding_nodes_h2l or he["dest_node_id"] not in corresponding_nodes_h2l:
            continue
        if he["id"] in corresponding_edges_h2l:
            le = corresponding_edges_h2l[he["id"]]
            global_edge_label_confusion_matrix[he["label"]][le["label"]] += 1
            
            le_label_entity = [edge for edge in EDGE_DATA["edges"] if edge["name"] == le["label"]]
            if not le_label_entity:
                continue
            le_label_entity = le_label_entity[0]
            le_possible_sources = set(le_label_entity.get("source", []))
            le_possible_dests = set(le_label_entity.get("dest", []))
            le_source_node_label = corresponding_nodes_l2h[le["source_node_id"]]["label"]
            le_dest_node_label = corresponding_nodes_l2h[le["dest_node_id"]]["label"]
            if (not le_possible_sources or le_source_node_label in le_possible_sources) and \
               (not le_possible_dests or le_dest_node_label in le_possible_dests):
                global_edge_label_confusion_matrix_filtered[he["label"]][le["label"]] += 1
        else:
            pass
            # global_edge_label_confusion_matrix[he["label"]]["no_edge"] += 1
            # global_edge_label_confusion_matrix_filtered[he["label"]]["no_edge"] += 1

node_classification_prec_scores = []
node_classification_rec_scores = []
node_classification_f1_scores = []
for label in NODE_LABELS:
    tp = global_node_label_confusion_matrix[label][label]
    fp = sum(global_node_label_confusion_matrix[other][label] for other in NODE_LABELS if other != label)
    fn = sum(global_node_label_confusion_matrix[label][other] for other in NODE_LABELS if other != label)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    node_classification_prec_scores.append(prec)
    node_classification_rec_scores.append(rec)
    node_classification_f1_scores.append(f1)
    # print(f"    Edge label {label}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")


edge_classification_prec_scores = []
edge_classification_rec_scores = []
edge_classification_f1_scores = []
for label in EDGE_LABELS:
    tp = global_edge_label_confusion_matrix[label][label]
    fp = sum(global_edge_label_confusion_matrix[other][label] for other in EDGE_LABELS if other != label)
    fn = sum(global_edge_label_confusion_matrix[label][other] for other in EDGE_LABELS if other != label)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    edge_classification_prec_scores.append(prec)
    edge_classification_rec_scores.append(rec)
    edge_classification_f1_scores.append(f1)
    # print(f"    Edge label {label}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

print("---------------------------------------")

print(f"Average Node Segmentation F1:    {sum(node_segmentation_f1_scores) / len(node_segmentation_f1_scores):.4f}")
print( "cf. Macro-averaged per document")
print("")

print(f"Average Node Classification Acc: {sum(node_classification_acc_scores) / len(node_classification_acc_scores):.4f}")
print( "cf. Among correctly segmented nodes, Macro-averaged per document")

print(f"Average Node Classification F1:  {sum(node_classification_f1_scores) / len(node_classification_f1_scores):.4f}")
for label, f1, prec, rec in zip(NODE_LABELS, node_classification_f1_scores, node_classification_prec_scores, node_classification_rec_scores):
    print(f"  {label}: F1 {f1:.4f} (P {prec:.4f}, R {rec:.4f})")
print( "cf. Among correctly segmented nodes, Macro-averaged per label, Micro-averaged per document")
print("")
print(f"Average Unlabeled Edge F1:          {sum(edge_detection_f1_scores) / len(edge_detection_f1_scores):.4f}")
print(f"- Average Unlabeled Edge Precision: {sum(edge_detection_precision_scores) / len(edge_detection_precision_scores):.4f}")
print(f"- Average Unlabeled Edge Recall:    {sum(edge_detection_recall_scores) / len(edge_detection_recall_scores):.4f}")
print(f"Average Labeled Edge F1:            {sum(edge_detcls_f1_scores) / len(edge_detcls_f1_scores):.4f}")
print(f"- Average Labeled Edge Precision:   {sum(edge_detcls_precision_scores) / len(edge_detcls_precision_scores):.4f}")
print(f"- Average Labeled Edge Recall:      {sum(edge_detcls_recall_scores) / len(edge_detcls_recall_scores):.4f}")
print( "cf. Among correctly segmented nodes, Macro-averaged per label, Macro-averaged per document")
print("")

print(f"Average Edge Classification F1: {sum(edge_classification_f1_scores) / len(edge_classification_f1_scores):.4f}")
for label, f1, prec, rec in zip(EDGE_LABELS, edge_classification_f1_scores, edge_classification_prec_scores, edge_classification_rec_scores):
    print(f"  {label}: F1 {f1:.4f} (P {prec:.4f}, R {rec:.4f})")
print( "cf. Among correctly identified edges, Macro-averaged per label, Micro-averaged per document")

# Density heatmaps for confusion matrices
def plot_confusion_matrix(confusion_matrix, labels, title, filename):
    matrix = []
    for row_label in labels:
        row = []
        for col_label in labels:
            row.append(confusion_matrix[row_label][col_label])
        row = [x / sum(row) for x in row] if sum(row) > 0 else row
        matrix.append(row)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    
plot_confusion_matrix(global_node_label_confusion_matrix, NODE_LABELS, "Node Label Confusion Matrix", "node_label_confusion_matrix.svg")
plot_confusion_matrix(global_edge_label_confusion_matrix, EDGE_LABELS, "Edge Label Confusion Matrix", "edge_label_confusion_matrix.svg")
plot_confusion_matrix(global_edge_label_confusion_matrix_filtered, EDGE_LABELS, "Edge Label Confusion Matrix (Source/Dest Condition Met)", "edge_label_confusion_matrix_filtered.svg")
