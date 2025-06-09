import json
import os
from collections import Counter

label_types_per_node = {}
with open("web/static/labels.json", "r") as f:
    labels = json.load(f)
    label_types_per_node = {x: Counter() for x in labels["node_labels"]}

label_types_per_node = {
    "planning": ["plan:frontier-plan", "plan:plan-subplan", "plan:plan-nextplan", "plan:frontier-verify", "plan:plan-alternative"],
    "fact": ["reason:plan-step", "reason:fact-detail"],
    "reasoning": ["reason:premise-conclusion", "reason:plan-step", "reason:stmt-correction", "evaluate:support", "evaluate:refute"],
    "restatement": ["reason:stmt-restatement", "reason:plan-step"],
    "assumption": ["reason:plan:plan-subplan", "reason:premise-conclusion", "reason:plan-step", "plan:frontier-plan", "plan:plan-nextplan"],
    "example": ["reason:concept-example", "reason:plan-step"],
    "reflection": ["evaluate:uncertainty", "evaluate:support", "evaluate:refute", "reason:premise-conclusion"],
    "conclusion": ["reason:premise-conclusion", "reason:stmt-restatement", "reason:plan-step", "reason:stmt-correction"],
}

# Select examples that cover all edge types per node type.
few_shot_examples = {x: list() for x in label_types_per_node.keys()}
for file in os.listdir("web/data"):
    if not file.endswith(".json"):
        continue
    with open(os.path.join("web/data", file), "r") as f:
        data = json.load(f)
        
        # Check if the nodes and edges are empty
        if len(data["edges"]) < 20:
            # print(f"Skipping {file} due to empty 'nodes' and 'edges'.")
            continue
        
        # For each node and type, check if edges with "to_node_id" have edge labels included in label_types_per_node
        for i,node in enumerate(data["nodes"]):
            valid_example = True
            node_type = node["label"]
            if node_type not in label_types_per_node:
                continue
            
            # Check if edge label included in label_types_per_node[node_type]
            for edge in data["edges"]:
                if edge["to_node_id"] == node["id"]:
                    edge_label = edge["label"]
                    if edge_label not in label_types_per_node[node_type]:
                        valid_example = False
                        break

            if valid_example:
                # If the example is valid, add it to the few-shot examples
                few_shot_examples[node["label"]].append({
                    "prev_steps": [
                        {"id": x["id"], "label": x["label"], "text": x["text"]}
                        for x in data["nodes"][:i]
                    ],
                    "current_step": {
                        "label": node["label"],
                        "text": node["text"],
                    },
                    "edges": [
                        {
                            "from_node_id": x["from_node_id"],
                            "label": x["label"]
                        }
                        for x in data["edges"] if x["to_node_id"] == node["id"]
                    ],
                })
print("Few-shot examples selected:")
for node_type, examples in few_shot_examples.items():
    print(f"{node_type}: {len(examples)} examples")

# Select 5 examples per node type.
# It should contain less than 20 prev steps.
# The selected examples should contain all edge labels as defined in label_types_per_node for a node type.

def select_best_examples(examples, required_labels, max_examples=5):
    """
    Select up to max_examples that:
    1. Have fewer than 20 prev_steps
    2. Collectively cover all required edge labels
    """
    # Filter examples with fewer than 20 prev_steps
    # valid_examples = [ex for ex in examples if len(ex["prev_steps"]) < 20]
    valid_examples = sorted([ex for ex in examples if len(ex["prev_steps"]) >= 10], key=lambda x: len(x["prev_steps"]))
    # valid_examples = examples
    
    if not valid_examples:
        return []
    
    # If we have fewer examples than required, return all
    if len(valid_examples) <= max_examples:
        return valid_examples
    
    # Greedy selection to cover all edge labels
    selected = []
    covered_labels = set()
    remaining_examples = valid_examples.copy()
    
    # First pass: greedily select examples that cover the most uncovered labels
    while len(selected) < max_examples and remaining_examples:
        best_example = None
        best_new_labels = 0
        best_idx = -1
        
        for i, example in enumerate(remaining_examples):
            # Get edge labels for this example
            example_labels = set(edge["label"] for edge in example["edges"])
            # Count how many new labels this example would add
            new_labels = len(example_labels - covered_labels)
            
            if new_labels > best_new_labels:
                best_new_labels = new_labels
                best_example = example
                best_idx = i
        
        if best_example is not None:
            selected.append(best_example)
            # Update covered labels
            example_labels = set(edge["label"] for edge in best_example["edges"])
            covered_labels.update(example_labels)
            remaining_examples.pop(best_idx)
        else:
            # No more examples add new labels, just add remaining examples
            break
    
    # If we still have slots and examples, fill them
    while len(selected) < max_examples and remaining_examples:
        selected.append(remaining_examples.pop(0))
    
    return selected

# Apply selection for each node type
selected_examples = {}
for node_type, examples in few_shot_examples.items():
    required_labels = label_types_per_node[node_type]
    selected = select_best_examples(examples, required_labels, max_examples=3)
    selected_examples[node_type] = selected

print("\nSelected examples with constraints:")
for node_type, examples in selected_examples.items():
    print(f"{node_type}: {len(examples)} examples")
    
    # Check coverage of edge labels
    all_edge_labels = set()
    for example in examples:
        edge_labels = set(edge["label"] for edge in example["edges"])
        all_edge_labels.update(edge_labels)
    
    required_labels = set(label_types_per_node[node_type])
    covered_labels = all_edge_labels.intersection(required_labels)
    missing_labels = required_labels - covered_labels
    
    print(f"  Required labels: {len(required_labels)}")
    print(f"  Covered labels: {len(covered_labels)}")
    if missing_labels:
        print(f"  Missing labels: {missing_labels}")
    
    # Check prev_steps constraint
    for i, example in enumerate(examples):
        prev_steps_count = len(example["prev_steps"])
        print(f"  Example {i+1}: {prev_steps_count} prev_steps")
    print()

with open("parser/edge_fewshot_examples.json", "w") as f:
    json.dump(selected_examples, f, indent=4)
