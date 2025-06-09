import json
import os
from collections import Counter

selected_examples = []
for file in ["web/data/math_14_QwQ-32B-Preview_long_correct.json", "web/data/math_16_QwQ-32B-Preview_long_correct.json", "web/data/chemistry_16_QwQ-32B-Preview_long_correct.json"]:
    with open(file, "r") as f:
        data = json.load(f)
    
    selected_examples.append({
        "raw_text": data["raw_text"]["response"],
        "nodes": [
            {
                "text": x["text"],
                "label": x["label"]
            }
            for x in data["nodes"] if x["source"] == "response"
        ]
    })

with open("parser/node_fewshot_examples.json", "w") as f:
    json.dump(selected_examples, f, indent=4, ensure_ascii=False)
