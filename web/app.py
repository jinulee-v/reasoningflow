from flask import Flask, request, jsonify, render_template
import os
import json
import re

app = Flask(__name__)
DATA_DIR = "data"
with open("static/labels.json", 'r', encoding='utf-8') as f:
    labels = json.load(f)

def reorganize_doc(doc):# reindex
    id_map = {}
    ctx_count = 0; trace_count = 0
    id_order_dict = {}
    for i, node in enumerate(doc["nodes"]):
        if node["source"] == "question":
            id_map[node["id"]] = f"ctx{ctx_count}"
            ctx_count += 1
        else:
            id_map[node["id"]] = f"trace{trace_count}"
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
                "from_node_id": id_map[d["from_node_id"]],
                "to_node_id": id_map[d["to_node_id"]],
                "label": d["label"]
            } for d in doc["edges"]
        ], key=lambda x: (id_order_dict[x["to_node_id"]], id_order_dict[x["from_node_id"]]))
    }
    return new_data

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load_file_list', methods=['GET'])
def load_file_list():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    files = sorted(files)
    return jsonify(files)

@app.route('/load_document', methods=['GET'])
def load_document():
    doc_id = request.args.get('doc_id')
    file_path = os.path.join(DATA_DIR, doc_id)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    doc_id = data.get('doc_id')
    file_path = os.path.join(DATA_DIR, doc_id + ".json")
    
    if not doc_id:
        return jsonify({"error": "Invalid document ID"}), 400
    
    # Reorder data
    new_data = reorganize_doc(data)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)
    return jsonify({"message": "Annotation saved successfully"})

@app.route('/upload_raw', methods=['POST'])
def upload_raw():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    data = json.load(file)
    doc_id = data.get('doc_id')
    if not doc_id:
        return jsonify({"error": "Invalid data format"}), 400
    
    file_path = os.path.join(DATA_DIR, f"{doc_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    return jsonify({"message": "Raw data uploaded successfully"})

@app.route('/get_labels', methods=['GET'])
def get_node_labels():
    return jsonify(labels)

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
            'from': edge['from_node_id'],
            'to': edge['to_node_id'],
            'label': edge.get('label', ''),
            'arrows': 'to',
            "smooth": {"forceDirection": "vertical"}
        })
    graph_data['edges'] = sorted(graph_data['edges'], key=lambda x: (node_ids.index(x['from']), node_ids.index(x['to'])))
    
    return render_template('graph.html', graph_data=json.dumps(graph_data), doc_id=doc_id)


@app.route('/split_node', methods=['POST'])
def split_node():
    data = request.json
    doc = data.get('doc')
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
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)
    return jsonify(new_data)


@app.route('/merge_node', methods=['POST'])
def merge_node():
    data = request.json
    doc = data.get('doc')
    doc_id = doc.get('doc_id')
    node_id = data.get('node_id')

    file_path = os.path.join(DATA_DIR, doc_id + ".json")
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
        if edge["from_node_id"] == node_id:
            edge["from_node_id"] = node_merged["id"]
        if edge["to_node_id"] == node_id:
            edge["to_node_id"] = node_merged["id"]
    
    new_data = reorganize_doc(doc)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)
    return jsonify(new_data)


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(debug=True)
