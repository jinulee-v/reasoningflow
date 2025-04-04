import json
import os
import re

for file in os.listdir("web/data"):
    print(file)
    with open("web/data/" + file) as f:
        data = json.load(f)
    
    # ID map
    id_map = {}
    ctx_count = 0; trace_count = 0
    id_order_dict = {}
    for i, node in enumerate(data["nodes"]):
        if node["label"] == "context":
            id_map[node["id"]] = f"ctx{ctx_count}"
            ctx_count += 1
        else:
            id_map[node["id"]] = f"trace{trace_count}"
            trace_count += 1
        assert data["raw_text"][node["start"]:node["end"]] == node["text"], \
            f"Mismatch in text for node {node['id']}: expected '{data['raw_text'][node['start']:node['end']]}' but got '{node['text']}'"
        id_order_dict[id_map[node["id"]]] = i
    new_data = {
        "doc_id": data["doc_id"],
        "raw_text": data["raw_text"],
        "metadata": {
            "source": "NuminaMath" if "math" in file else "STILL-2",
            "generator": "Qwen/QwQ-32B-Preview",
            "domain": file.split("_")[0],
            "batch": 0
        },
        "nodes": [
            {
                "id": id_map[d["id"]],
                "annotation": d["annotation"],
                "start": d["start"],
                "end": d["end"],
                "label": d["label"],
                "text": d["text"]
            } for d in data["nodes"]
        ],
        "edges": sorted([
            {
                "id": d["id"],
                "from_node_id": id_map[d["from_node_id"]],
                "to_node_id": id_map[d["to_node_id"]],
                "label": d["label"]
            } for d in data["edges"]
        ], key=lambda x: (id_order_dict[x["to_node_id"]], id_order_dict[x["from_node_id"]]))
    }
    with open("web/data/" + file, "w") as f:
        json.dump(new_data, f, indent=4)