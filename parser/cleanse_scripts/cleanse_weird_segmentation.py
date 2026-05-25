#!/usr/bin/env python3
"""Delete documents in data/v1_llm_gemini-3-pro-preview that have nodes
longer than 300 characters and/or more than 2 double-newlines."""

import argparse
import json
import re
from pathlib import Path

DATA_DIR = Path("data/v1_llm_gemini-3.1-pro-preview")
RESULT_DIRS = [Path("prm_results"), Path("parc_results"), Path("argumentation_results")]

DELETE_ANOMALY_NODES = False


def main():
    parser = argparse.ArgumentParser(
        description="Scan (and optionally delete) documents with anomalous segmentation."
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR), metavar="DIR",
        help=f"Directory of JSON files to process (default: {DATA_DIR}).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if DELETE_ANOMALY_NODES:
        print("Deleting documents with anomaly nodes. This action is irreversible.")
        input_char = input("Press Y to continue...")
        if input_char != "Y":
            print("Aborted.")
            return

    docs_with_long_nodes = 0
    docs_with_many_newlines = 0
    docs_with_either = 0
    docs_with_anomaly_nodes = 0
    total_docs = 0
    docs_per_dataset = {}

    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file) as f:
            doc = json.load(f)

        dataset = doc["metadata"]["domain"]

        nodes = doc.get("nodes", [])

        def has_long_fn(node_text):
            # Strip non-alphanumeric characters
            # return node_text.strip().count(".\n\n") > 0
            return len(re.findall("[a-z](\.\?)\s+[A-Z]", node_text.strip(), re.MULTILINE)) > 0
        has_long = [has_long_fn(node["text"]) for node in nodes].count(True) > 0
        # if has_long:
        #     print(f"{json_file} has a long node:")
        #     for node in nodes:
        #         if has_long_fn(node["text"]):
        #             print(f"  Node {node['id']} (length {len(node['text'])}):\n{node['text']}")
        #             print("----------------")

        has_short = [len(node["text"]) == 1 for node in nodes].count(True) > 0
        if has_short:
            print(f"{json_file} has a short node:")
            for node in nodes:
                if len(node["text"]) == 1:
                    print(f"  Node {node['id']} (length {len(node['text'])}):\n{node['text']}")
                    print("----------------")

        # If two adjacent nodes: first node ends with small letter, second node starts with small letter
        has_interword_cut = False
        for i in range(len(nodes) - 1):
            if nodes[i]["text"] and nodes[i+1]["text"]:
                if nodes[i]["text"][-1].islower() and nodes[i+1]["text"][0].islower():
                    has_interword_cut = True
                    print(f"{json_file} has an inter-word cut between node {nodes[i]['id']} and node {nodes[i+1]['id']}.")
                    print(f"  Node {nodes[i]['id']} (length {len(nodes[i]['text'])}):\n{nodes[i]['text']}")
                    print(f"  Node {nodes[i+1]['id']} (length {len(nodes[i+1]['text'])}):\n{nodes[i+1]['text']}")
                    print("----------------")

        if has_long or has_short or has_interword_cut:
            print(f"{json_file} (long node: {has_long}, short node: {has_short}, inter-word cut: {has_interword_cut})")
            if DELETE_ANOMALY_NODES:
                print(f"Deleting {json_file} due to anomaly.")
                json_file.unlink()
                for result_dir in RESULT_DIRS:
                    result_file = result_dir / json_file.name
                    if result_file.exists():
                        print(f"  Deleting {result_file}")
                        result_file.unlink()
            docs_with_anomaly_nodes += 1

            docs_per_dataset[dataset] = docs_per_dataset.get(dataset, 0) + 1
        total_docs += 1

    print(f"Total documents:              {total_docs}")
    print(f"Docs with anomaly nodes:      {docs_with_anomaly_nodes}")
    for dataset, count in docs_per_dataset.items():
        print(f"  {dataset}: {count}")


if __name__ == "__main__":
    main()
