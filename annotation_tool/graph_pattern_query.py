import json
import subprocess
import os
CLINGO_PATH = os.environ.get("CLINGO_PATH", "clingo")

def query_reasoningflow(data, query, arity, verbose=False):    
    # Convert nodes and edges to Answer Set Programming
    ASP = """
connected(X, Y) :- edge(X, Y, _).
connected(X, Z) :- edge(X, Y, _), connected(Y, Z).
connected(X, Y, Label) :- edge(X, Y, Label).
connected(X, Z, Label) :- edge(X, Y, Label), connected(Y, Z, Label).
distance(X, X, 0) :- node(X, _).
distance(X, Y, D) :- edge(X, Y, _), D = 1.
distance(X, Y, D) :- edge(X, Z, _), distance(Z, Y, D1), D = D1 + 1.
"""
    for node in data["nodes"]:
        ASP += f"node({node['id']}, \"{node['label']}\").\n"
    for edge in data["edges"]:
        ASP += f"edge({edge['source_node_id']}, {edge['dest_node_id']}, \"{edge['label']}\").\n"
    
    for query in query:
        ASP += "\n" + query + "\n"
    ASP += f"#show query/{arity}."

    if verbose:
        print("%" * 50)
        print("% Generated ASP:")
        print(ASP)
        print("%" * 50)

    process = subprocess.Popen(
        [CLINGO_PATH, '--quiet=1', '--opt-mode=opt', "--restart-on-model", "--time-limit=600"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    clingo_output, _ = process.communicate(input=ASP)
    if verbose:
        print("%" * 50)
        print("Clingo raw output:")
        print(clingo_output)
        print("%" * 50)
    clingo_output = [sent for sent in clingo_output.split("\n") if sent.startswith("query")] # All predicates are listed in a single line
    if len(clingo_output) == 0:
        return []
    clingo_output = clingo_output[0]
    clingo_output = clingo_output.split(" ")
    clingo_output = [item.replace("query", "") for item in clingo_output if item.strip()]
    if verbose:
        print("Clingo output:")
        for line in clingo_output:
            print(line)
    return clingo_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, required=False, help='Path to the JSON file containing the data. If provided, runs on single file mode; if not, runs on stat mode')
    parser.add_argument('--query', '-q', type=str, required=True, action='append', help='Query to run in ASP format. Can be provided multiple times. Refer to README.md for further instructions')
    parser.add_argument('--arity', type=int, required=True, help='Arity of the `query` predicate')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()


    if args.file:
        FILE = args.file
        print("Loading data from", FILE)
        with open(FILE) as f:
            data = json.load(f)
        result = query_reasoningflow(data, args.query, args.arity, verbose=args.verbose)
        print(result)
    else:
        import os
        import glob
        total = 0; match_files = []
        data_files = glob.glob(os.path.join('data', '*.json'))
        for FILE in data_files:
            with open(FILE) as f:
                data = json.load(f)
            # Leave sound data
            if len(data["edges"]) < len(data["nodes"]):
                continue
            total += 1
            result = query_reasoningflow(data, args.query, args.arity, verbose=args.verbose)
            if len(result) > 0:
                match_files.append(FILE)
        print(f"{len(match_files)}/{total} files including matching patterns:")
        for FILE in match_files:
            print("-", FILE)