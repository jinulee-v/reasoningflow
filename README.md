# 🌊 ReasoningFlow

Large reasoning models (LRMs) generate complex reasoning traces with planning, reflection, verification, and backtracking. ReasoningFlow is a **unified schema for analyzing the semantic structures of these complex traces.**

ReasoningFlow parses traces into directed acyclic graphs, enabling the characterization of distinct reasoning patterns as subgraph structures.

This repository includes the **preview** version of ReasoningFlow, as presented in ArgMining 2025 Workshop, with:
- 31 manually labeled data
- Web annotation interface `annotator_webapp.py`
- Graph visualizer `visualize.py`
- Graph querying code/examples `graph_pattern_query_*`

We will soon add:
- More data
- Automatic parsers

> Link to project page: https://tinyurl.com/reasoningflow

## Web annotator / Visualizer

### Requirements

```bash
cd web
pip install -r requirements.txt
# Annotation tool can be accessed at http://127.0.0.1:5000
```

### Start web annotator

Note: Any edit will modify the content inside `data/` directory.

```bash
python annotator_webapp.py
# http://127.0.0.1:5000
```

### Visualize into graphs:

```bash
python visualize.py --file data/chemistry_0_QwQ-32B-Preview.json # Can change --file to something else
# Generates graph_test.html; can download SVG file from the HTML file (via pyvis)
```

## Setup `Clingo`

Clingo is an efficient Answer Set Programming solver. Clingo is required for running `graph_pattern_query.py`, a script that queries subgraph patterns.

### Installation

For installation, refer to this page: [INSTALL.md](https://github.com/potassco/clingo/blob/master/INSTALL.md#build-install-and-test)

### Guide for writing graph queries (graph_pattern_query.py)

There are four predicates you can use: `node/2`, `edge/3`, `connected/2`, `connected/3`. `node/2` and `edge/3` represents ReasoningFlow graph in a straightforward manner:
```
node(trace0, "restatement").
node(trace1, "planning").
node(trace2, "fact").
edge(trace0, trace1, "plan:frontier-plan").
edge(trace1, trace2, "reason:plan-step).
```

`connected/2` defines a transitive closure with arbitrary edge labels (*Are two nodes connected?*), and `connencted/3` defines a transitive closure of a specific label (*Are two nodes connected using only 'Label' edges?*). For instance,
```
connected(trace0, trace2). % True
connected(trace0, trace1, "plan:frontier-plan"). % True
connected(trace0, trace2, "plan:frontier-plan"). % False
```

Using these predicates, one can write a query using the walrus symbol `:-`. Basically, the predicate in LHS turns true when all predicates on the RHS are satisfied. For instance, this query detects two nodes connected with "reason:plan-step" edge and the preceding node is "planning":
```
query(X, Y) :- edge(X, Y, "reason:plan-step"), node(X, "planning").
```

This will find all unique `(X, Y)` tuples of node IDs that satisfy the stated condition. Note that if you add multiple queries by repeating `-q` command line args, it will OR the queries, finding tuples that satisfy any of them.

Finally, one must specify the arity (number of predicates) of the `query` predicate. In this case, it is 2 because query has two arguments X and Y. One can query for bigger or smaller tuples, as long as the variables appear on the RHS and the `--arity` is correcly provided.

```sh
python -f "path/to/file.json" -q 'query(X, Y) :- edge(X, Y, "reason:plan-step"), node(X, "planning").' --arity 2
```

Refer to [this link](https://potassco.org/doc/start/) for an introduction on Clingo syntax, and `graph_pattern_query_examples.sh` for examples.