import os
import json
from collections import Counter
import matplotlib.pyplot as plt

def graph_merge_same_unit(graph):
    # node_dict
    node_dict = {}
    for node in graph['nodes']:
        node_dict[node['id']] = node
    for edge in graph['edges']:
        if edge['label'] == 'same_unit':
            from_node = edge['from_node_id']
            to_node = edge['to_node_id']
            assert node_dict[from_node]['label'] == node_dict[to_node]['label'], f"Node {from_node} and {to_node} have different labels: {node_dict[from_node]['label']} and {node_dict[to_node]['label']}"
            # create new node
            new_node = {
                'id': from_node,
                'annotation': True,
                'start': node_dict[from_node]['start'],
                'end': node_dict[to_node]['end'],
                'label': node_dict[from_node]['label'],
                'text': graph['raw_text'][node_dict[from_node]['start']:node_dict[to_node]['end']],
            }
            # remove to_node from graph['nodes']
            graph['nodes'].remove(node_dict[to_node])
            # reconnect from graph['edges']
            for edge in graph['edges']:
                if edge['from_node_id'] == to_node:
                    edge['from_node_id'] = from_node
                if edge['to_node_id'] == to_node:
                    edge['to_node_id'] = from_node
    # remove same_unit edge
    graph['edges'] = [edge for edge in graph['edges'] if edge['label'] != 'same_unit']

with open("web/static/labels.json", 'r', encoding='utf-8') as f:
    label_config = json.load(f)
    node_color_map = label_config["node_colors"]
    edge_color_map = label_config["edge_colors"]

node_valid_data = []
edge_valid_data = []
for filename in os.listdir('web/data'):
    if not filename.endswith('.json'):
        continue
    with open(os.path.join('web/data', filename), 'r', encoding='utf-8') as f:
        data = json.load(f)
        graph_merge_same_unit(data)
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
        elif node_valid:
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
        if node['label'] == 'assumption':
            print("ASSUMPTION", data["doc_id"], node["id"])
        if node['label'] == "":
            print(data["doc_id"], node["id"])
        else:
            node_labels[node['label']] += 1
print(node_labels)
del node_labels["context"]
# Plot node label stats
plt.figure(figsize=(4, 4))
# print(plt.pie(node_labels.values()))
wedges, autotexts = plt.pie(
    node_labels.values(),
    # labels: percentage
    labels=[f"{round(count/sum(node_labels.values())*100, 1)}%" for label, count in node_labels.items()],
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
wedges, autotexts = plt.pie(
    [edge_labels[label] for label in edge_color_map.keys()],
    # labels: percentage
    labels=[f"{round(edge_labels[x]/sum(edge_labels.values())*100, 1)}%" for x in edge_color_map.keys()],
    startangle=90,
    colors=[edge_color_map[x] for x in edge_color_map.keys()],
)
# output as svg
plt.savefig("edge_label_stat.svg", format='svg', bbox_inches='tight')

# Estimate the number of nodes and edges over 60 data
print("avg. nodes:", round(sum(node_labels.values()) / len(node_valid_data) * 60) / 60)
print("avg. edges:", round(sum(edge_labels.values()) / sum([len(data['nodes']) for data in edge_valid_data]) * sum(node_labels.values()) / len(node_valid_data) * 60) / 60)