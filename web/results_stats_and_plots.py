import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from transformers import AutoTokenizer\

with open("static/labels.json", 'r', encoding='utf-8') as f:
    label_config = json.load(f)
    node_color_map = label_config["node_colors"]
    edge_color_map = label_config["edge_colors"]

node_valid_data = []
edge_valid_data = []
for filename in os.listdir('data'):
    if not filename.endswith('.json'):
        continue
    with open(os.path.join('data', filename), 'r', encoding='utf-8') as f:
        data = json.load(f)
        node_valid = True
        edge_valid = True
        for node in data['nodes']:
            if node['label'] == "":
                node_valid = False
                break
        if len(data["edges"]) < len(data["nodes"]):
            edge_valid = False
        if edge_valid:
            edge_valid_data.append(data)
        if node_valid:
            node_valid_data.append(data)
        else:
            print(f"Invalid data in {filename}")

print("-" * 20)
print(f"Node valid data: {len(node_valid_data)}")
print(f"Edge valid data: {len(edge_valid_data)}")

# Node label stats
node_labels = {n:0 for n in node_color_map.keys()}
for data in node_valid_data + edge_valid_data:
    for node in data['nodes']:
        # if node['label'] == 'assumption':
        #     print("ASSUMPTION", data["doc_id"], node["id"])
        if node['label'] == "":
            print(data["doc_id"], node["id"])
        else:
            node_labels[node['label']] += 1
print(node_labels)
del node_labels["context"]
# Plot node label stats
plt.figure(figsize=(4, 4))
# print(plt.pie(node_labels.values()))
plt.pie(
    node_labels.values(),
    # labels=[f"{round(count/sum(node_labels.values())*100, 1)}%" for label, count in node_labels.items()],
    autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
    pctdistance=1.2,
    startangle=90,
    colors=node_color_map.values(),
)
# output as svg
plt.savefig("node_label_stat.svg", format='svg', bbox_inches='tight')


# Edge label stats
edge_labels = {n:0 for n in edge_color_map.keys()}
for data in edge_valid_data:
    for edge in data['edges']:            
        if edge['label'] == "":
            print(data["doc_id"], edge["from_node_id"], edge["to_node_id"])
        else:
            edge_labels[edge['label']] += 1
print(edge_labels)
# Plot edge label stats
plt.figure(figsize=(5, 5))
# print(plt.pie(node_labels.values()))
plt.pie(
    [edge_labels[label] for label in edge_color_map.keys()],
    # labels=[f"{round(count/sum(edge_labels.values())*100, 1)}%" for label, count in edge_labels.items()],
    autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
    pctdistance=1.2,
    startangle=90,
    colors=[edge_color_map[x] for x in edge_color_map.keys()],
)
# output as svg
plt.savefig("edge_label_stat.svg", format='svg', bbox_inches='tight')

# Estimate the number of nodes and edges over 60 data
print("avg. nodes:", round(sum(node_labels.values()) / len(node_valid_data) * 60) / 60)
print("avg. edges:", round(sum(edge_labels.values()) / sum([len(data['nodes']) for data in edge_valid_data]) * sum(node_labels.values()) / len(node_valid_data) * 60) / 60)

# Transition probability between node labels
transition_matrix = {n: {m: 0 for m in node_color_map.keys()} for n in node_color_map.keys()}
for data in node_valid_data:
    nodes = data['nodes']
    for i in range(len(nodes)-1):
        transition_matrix[nodes[i]['label']][nodes[i+1]['label']] += 1
# Print a table
print("Transition matrix:")
print(" " * 10 + " ".join([f"{label:>10}" for label in node_color_map.keys()]))
for n in node_color_map.keys():
    print(f"{n:>10}", end=" ")
    for m in node_color_map.keys():
        print(f"{round(transition_matrix[n][m] / sum(transition_matrix[n].values()) * 100, 2):>10}", end=" ")
    print()

# Analyze token length of all responses
tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B-Preview")
def count_tokens(text):
    return len(tokenizer(text)["input_ids"])
token_lengths = []
for data in node_valid_data:
    token_lengths.append(count_tokens(data['raw_text']['response']))
print("Average token length:", round(sum(token_lengths) / len(token_lengths), 2))
# Plot token length distribution
plt.figure(figsize=(5, 3))
plt.hist(token_lengths, bins=8)
plt.xlabel("Token length")
plt.ylabel("Frequency")
plt.savefig("token_length_distribution.svg", format='svg', bbox_inches='tight')

first_two_tokens = {n: [] for n in node_color_map.keys()}
for data in node_valid_data:
    for node in data['nodes']:
        if node['label'] == "":
            continue
        tokens = tokenizer.tokenize(node['text'], add_special_tokens=False)
        if len(tokens) < 2:
            continue
        first_two_tokens[node['label']].append((tokens[0].replace("Ġ", "_"), tokens[1].replace("Ġ", "_")))

# Sunburst plot
for label in first_two_tokens.keys():
    # Count the first two tokens
    counts = Counter(first_two_tokens[label])
    # Create a dataframe
    df = []
    for tokens, count in counts.items():
        df.append({"first": tokens[0], "second": tokens[1], "count": count})
    df = pd.DataFrame(df)
    # Create a sunburst plot
    fig = px.sunburst(df, path=['first', 'second'], values='count', color='count', color_continuous_scale='RdBu')
    fig.update_traces(textinfo="label+percent entry")
    fig.update_layout(title=f"Sunburst plot of {label} tokens")
    # fig.show()
    # Save as svg
    fig.write_image(f"sunburst_{label}.svg")

# "Wait," label distribution
print("**Wait,** label distribution:")
labels = {n: first_two_tokens[n].count(("Wait", ",")) for n in node_color_map.keys()}
print(labels)
print("**Alternatively,** label distribution:")
labels = {n: first_two_tokens[n].count(("Alternatively", ",")) for n in node_color_map.keys()}
print(labels)