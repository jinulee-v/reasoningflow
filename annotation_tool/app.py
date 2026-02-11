from flask import Flask, request, jsonify, render_template
import os
import json
import re
import yaml

###############################################################################
# Setup and utility functions
###############################################################################

app = Flask(__name__)
DATA_DIR = "data"
labels = {"node_labels": [], "edge_labels": [], "node_colors": {}}
with open("schema/node_labels.yaml", 'r', encoding='utf-8') as f:
    nodes = yaml.safe_load(f)
    labels["node_labels"] = [node['name'] for node in nodes['nodes']]
    for node in nodes['nodes']:
        labels["node_colors"][node['name']] = node.get('color', '#FFFFFF')
with open("schema/edge_labels.yaml", 'r', encoding='utf-8') as f:
    edges = yaml.safe_load(f)
    labels["edge_labels"] = [edge['name'] for edge in edges['edges']]

print("Loaded node labels:", labels["node_labels"])
print("Loaded edge labels:", labels["edge_labels"])

def load_file(directory, filename):
    file_path = os.path.join(DATA_DIR, directory, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_file(directory, filename, data):
    file_path = os.path.join(DATA_DIR, directory, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def reorganize_doc(doc):# reindex
    id_map = {}
    ctx_count = 0; trace_count = 0
    id_order_dict = {}
    for i, node in enumerate(doc["nodes"]):
        if node["source"] == "question":
            id_map[node["id"]] = f"ctx{ctx_count}"
            ctx_count += 1
        else:
            id_map[node["id"]] = f"resp{trace_count}"
            trace_count += 1
        assert doc["raw_text"][node["source"]][node["start"]:node["end"]] == node["text"], \
            f"Mismatch in text for node {node['id']}: expected '{doc['raw_text'][node['source']][node['start']:node['end']]}' but got '{node['text']}'"
        id_order_dict[id_map[node["id"]]] = i
    
    for source in doc["raw_text"].keys():
        assert "".join([node["text"] for node in doc["nodes"] if node["source"] == source]) == doc["raw_text"][source], \
            f"Mismatch in raw_text for source '{source}'"

    # Reorder data
    new_data = {
        "doc_id": doc["doc_id"],
        "raw_text": doc["raw_text"],
        "metadata": doc["metadata"],
        "nodes": [
            {
                "id": id_map[d["id"]],
                "annotation": d["annotation"],
                "start": d["start"],
                "end": d["end"],
                "label": d["label"],
                "text": d["text"],
                "source": d["source"]
            } for d in doc["nodes"]
        ],
        "edges": sorted([
            {
                "id": d["id"],
                "source_node_id": id_map.get(d["source_node_id"], ""),
                "dest_node_id": id_map.get(d["dest_node_id"], ""),
                "label": d["label"]
            } for d in doc["edges"] if d["source_node_id"] in id_map and d["dest_node_id"] in id_map
        ], key=lambda x: (id_order_dict.get(x["dest_node_id"], -1), id_order_dict.get(x["source_node_id"], -1)))
    }
    return new_data


###############################################################################
# Initialize annotation tool
###############################################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_labels', methods=['GET'])
def get_node_labels():
    return jsonify(labels)

@app.route('/load_file_list', methods=['GET'])
def load_file_list():
    result = []
    directories = os.listdir(DATA_DIR)
    if not directories:
        return jsonify({"error": "No directories found in data directory"}), 500

    for directory in directories:
        dir_path = os.path.join(DATA_DIR, directory)
        if not os.path.isdir(dir_path):
            continue
        if "raw_data" in dir_path:
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        if not files:
            return jsonify({"error": f"No JSON files found in directory '{directory}'"}), 500
        # files: chemistry_0.json, chemistry_1.json, ...
        # sort by domain -> numbers in the filename
        def sort_key(file_name):
            match = re.match(r'(\D+)(\d+)', file_name)
            if match:
                return (match.group(1), int(match.group(2)))
            return (file_name, 0)
        files.sort(key=sort_key)
        files = [{"filename": f} for f in files]
        result.append({
            "directory": directory,
            "files": files
        })
        
    return jsonify(result)

###############################################################################
# Load and save annotations
###############################################################################

@app.route('/load_annotation', methods=['GET'])
def load_annotation():
    directory = request.args.get('directory')
    filename = request.args.get('filename')
    data = load_file(directory, filename)
    return jsonify(data)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    raw_data = request.json
    directory = raw_data.get('directory')
    filename = raw_data.get('filename')
    data = raw_data.get('doc')
    
    if not directory or not filename:
        return jsonify({"error": "Invalid document ID"}), 400
    
    # Reorder data
    new_data = reorganize_doc(data)

    save_file(directory, filename, new_data)
    return jsonify({"message": "Annotation saved successfully"})

###############################################################################
# Manipulating graph structures
###############################################################################

@app.route('/split_node', methods=['POST'])
def split_node():
    data = request.json
    doc = data.get('doc')
    directory = data.get('directory')
    filename = data.get('filename')
    doc_id = doc.get('doc_id')
    node_id = data.get('node_id')
    new_text = data.get('new_text')
    if new_text.replace("|", "") not in "".join(doc['raw_text'].values()):
        return jsonify({"error": "Invalid split text format: must add single '|' character"}), 400
    new_text = new_text.split("|")
    if len(new_text) != 2:
        return jsonify({"error": "Invalid split text format: must add single '|' character"}), 400

    file_path = os.path.join(DATA_DIR, doc_id + ".json")
    print(file_path)
    
    if not doc_id:
        return jsonify({"error": "Invalid document ID"}), 400

    # Add new splitted nodes
    for i, node in enumerate(doc["nodes"]):
        if node["id"] == node_id:
            end = node["end"]
            node["end"] = node["start"] + len(new_text[0])
            node["text"] = doc['raw_text'][node['source']][node["start"]:node["end"]]
            doc["nodes"].insert(
                i + 1,
                {
                    "id": node["id"] + "-1",
                    "annotation": True,
                    "start": node["start"] + len(new_text[0]),
                    "end": end,
                    "label": node["label"],
                    "source": node['source'],
                    "text": doc['raw_text'][node['source']][node["start"] + len(new_text[0]):end]
                }
            )
            break
    
    new_data = reorganize_doc(doc)
    save_file(directory, filename, new_data)
    return jsonify(new_data)


@app.route('/merge_node', methods=['POST'])
def merge_node():
    data = request.json
    doc = data.get('doc')
    directory = data.get('directory')
    filename = data.get('filename')
    doc_id = doc.get('doc_id')
    node_id = data.get('node_id')

    file_path = os.path.join(DATA_DIR, directory, filename)
    print(file_path)
    
    if not doc_id:
        return jsonify({"error": "Invalid document ID"}), 400

    # select node
    node_to_merge_i = None
    node_to_merge = None
    for i, node in enumerate(doc["nodes"]):
        if node["id"] == node_id:
            node_to_merge_i = i
            node_to_merge = node
    if node_to_merge_i is None:
        return jsonify({"error": "Node ID not found"}), 404
    node_merged = doc["nodes"][node_to_merge_i - 1]
    # merge nodes
    doc["nodes"] = [
        node for i, node in enumerate(doc["nodes"]) 
        if i != node_to_merge_i
    ]
    node_merged["end"] = node_to_merge["end"]
    node_merged["text"] = doc['raw_text'][node_merged['source']][node_merged["start"]:node_merged["end"]]

    # replace edge labels
    for edge in doc["edges"]:
        if edge["source_node_id"] == node_id:
            edge["source_node_id"] = node_merged["id"]
        if edge["dest_node_id"] == node_id:
            edge["dest_node_id"] = node_merged["id"]
    
    new_data = reorganize_doc(doc)
    save_file(directory, filename, new_data)
    return jsonify(new_data)

###############################################################################
# Utilities
###############################################################################

@app.route('/graph', methods=['GET'])
def visualize_graph():
    doc_id = request.args.get('doc_id')
    if not doc_id:
        return jsonify({"error": "Document ID is required"}), 400
    
    file_path = os.path.join(DATA_DIR, doc_id)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare the data for visualization
    graph_data = {
        'nodes': [],
        'edges': []
    }
    nodes = sorted(data.get('nodes', []), key=lambda x: x['id'])
    node_ids = [node['id'] for node in data.get('nodes', [])]
    for node in nodes:
        graph_data['nodes'].append({
            'id': node['id'],
            'label': f"{node['id']}: {node['text']}",
            'color': {'background': labels["node_colors"].get(node['label'], '#FFFFFF')}
        })
    
    for edge in data.get('edges', []):
        graph_data['edges'].append({
            'from': edge['source_node_id'],
            'to': edge['dest_node_id'],
            'label': edge.get('label', ''),
            'arrows': 'to',
            "smooth": {"forceDirection": "vertical"}
        })
    graph_data['edges'] = sorted(graph_data['edges'], key=lambda x: (node_ids.index(x['from']), node_ids.index(x['to'])))
    
    return render_template('graph.html', graph_data=json.dumps(graph_data), doc_id=doc_id)


@app.route('/node_examples')
def index_node_examples():
    return render_template('node_examples.html')

@app.route('/load_node_examples', methods=['GET'])
def load_node_examples():
    node_label = request.args.get('node_id')
    node_examples = []

    # Get file list directly without creating a Response object
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]

    # Process files one at a time to reduce memory usage
    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract only matching nodes, then discard the full data
            for node in data.get('nodes', []):
                if node['label'] == node_label:
                    node_examples.append({
                        'doc_id': file,
                        'id': node["id"],
                        'text': node["text"],
                        'source': node["source"],
                        'start': node["start"],
                        'end': node["end"],
                        'annotation': node["annotation"],
                        'label': node["label"],
                    })

            # Explicitly delete data to free memory before next iteration
            del data
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    return jsonify(node_examples)

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(debug=True)
