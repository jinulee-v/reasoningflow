from flask import Flask, request, jsonify, render_template
import os
import json

app = Flask(__name__)
DATA_DIR = "data"
with open("static/labels.json", 'r', encoding='utf-8') as f:
    labels = json.load(f)

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
    new_data = {
        "doc_id": data["doc_id"],
        "raw_text": data["raw_text"],
        "metadata": data["metadata"],
        "nodes": [
            {
                "id": d["id"],
                "annotation": d["annotation"],
                "start": d["start"],
                "end": d["end"],
                "label": d["label"],
                "text": d["text"]
            } for d in data["nodes"]
        ],
        "edges": [
            {
                "id": d["id"],
                "from_node_id": d["from_node_id"],
                "to_node_id": d["to_node_id"],
                "label": d["label"]
            } for d in data["edges"]
        ],
    }

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

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    app.run(debug=True)
