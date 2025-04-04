import json
from pvsvg import Network
import networkx as nx
import numpy as np
import textwrap
from copy import deepcopy

with open("web/static/labels.json", 'r', encoding='utf-8') as f:
    label_config = json.load(f)
    node_color_map = label_config["node_colors"]
    edge_color_map = label_config["edge_colors"]

def draw_graph(data):

    # Create a NetworkX graph to assist with topological sorting
    G = nx.DiGraph()

    # Add nodes
    for node in data["nodes"]:
        wrapped_text = "\n".join(textwrap.wrap(node['id'] + ": " + node['text'], width=30))  # Wrap text
        G.add_node(
            node["id"],
            label=wrapped_text,  # Default label
            shape='box',  # Default shape
            borderWidth=0.3,  # Default border width
            color={"background": node_color_map.get(node["label"], "#F0F0F0"), "border": "black"},  # Default color, fallback to gray if label not found
            # physics=False
            fixed={"y": True}
        )

    # Add edges
    for edge in data["edges"]:
        G.add_edge(edge["from_node_id"], edge["to_node_id"], label=edge["label"], smooth={'type': 'straightCross'}, arrows={'to':{'enabled':True}}) # physics=False, 

    # Perform topological sorting to assign vertical positions
    layers = list(nx.topological_generations(G))
    for i, layer in enumerate(layers):
        for node in layer:
            G.nodes[node]['subset'] = i

    # Initialize X-positions using multipartite layout
    pos = nx.multipartite_layout(G)

    # Normalize positions and apply median heuristic for alignment
    layer_x_positions = {i: [] for i in range(len(layers))}
    for node, (x, y) in pos.items():
        layer = next(i for i, layer in enumerate(layers) if node in layer)
        layer_x_positions[layer].append(x)
    max_node_per_layer = max(len(layer) for layer in layers)

    # Adjust X-positions using median heuristic
    for i, layer in enumerate(layers):
        # i = len(layers) - i - 1  # Reverse order for downward arrows
        parent_x_positions = []
        for node in layer:
            parents = list(G.predecessors(node))
            if parents:
                parent_x_positions.append(np.median([G.nodes[p]['x'] for p in parents]))
            else:
                parent_x_positions.append(0)
        
        sorted_indices = np.argsort(parent_x_positions)
        for j, node in enumerate(np.array(layer)[sorted_indices]):
            xsplit = 200
            if len(layer) <= max_node_per_layer//2:
                xsplit = 400
            ysplit = 120
            G.nodes[node]['x'] = j * xsplit - round((len(sorted_indices) - 1) / 2 * xsplit)
            G.nodes[node]['y'] = i * ysplit

    # Initialize PyVis network
    net = Network(G, width='100%', height='1000px', physics_kwargs={
            "enabled": True,
            "stabilization": {"enabled": True, "iterations": 200},
            "hierarchicalRepulsion": {
            "centralGravity": 0.0,
            "springLength": 100,
            "springConstant": 0.01,
            "nodeDistance": 120,
            "damping": 0.09
            }
        }
    )

    # # Add nodes with custom labels and positions
    # for node in data["nodes"]:
    #     x, y = node_positions[node['id']]
    #     wrapped_text = "\n".join(textwrap.wrap(node['id'] + ": " + node['text'], width=30))  # Wrap text
    #     net.add_node(
    #         node['id'],
    #         label=wrapped_text,
    #         shape='box',
    #         x=x,
    #         y=-y,  # Negate Y to ensure downward arrows
    #         physics=True,
    #         borderWidth=0.3,
    #         color={"background": color_map[node["label"]], "border": "black"}
    #     )

    # # Add edges
    # for edge in data["edges"]:
    #     net.add_edge(edge["from_node_id"], edge["to_node_id"], label=edge["label"], smooth={'type': 'straightCross'})

    # net.force_atlas_2based(gravity=-50, central_gravity=0.5, spring_length=200)
    # Show graph
    return net

# def subgraph_premise_conclusion(graph: Network):
#     # Remove all edges where the label is not "reasoning:premise-conclusion"
#     edges_to_remove = [edge for edge in graph.edges if edge['label'] != "reason:premise-conclusion"]
#     for edge in edges_to_remove:
#         # set line and label invisible
#         edge['hidden'] = True
#         edge['label'] = ""
    
#     # Change background color of nodes where no edges with labels "reason:premise-conclusion" are connected
#     for node in graph.nodes:
#         if not any(edge['label'] == "reason:premise-conclusion" and (edge['from'] == node['id'] or edge['to'] == node['id']) for edge in graph.edges):
#             node['color']['background'] = '#F0F0F0'
    
#     return graph

# def subgraph_restatement(graph: Network):
#     # Remove all edges where the label is not "reasoning:premise-conclusion"
#     edges_to_remove = [edge for edge in graph.edges if edge['label'] != "reason:stmt-restatement"]
#     for edge in edges_to_remove:
#         # set line and label invisible
#         edge['hidden'] = True
#         edge['label'] = ""

#     # Change background color of nodes where no edges with labels "reason:premise-conclusion" are connected
#     for node in graph.nodes:
#         if not any(edge['label'] == "reason:stmt-restatement" and (edge['from'] == node['id'] or edge['to'] == node['id']) for edge in graph.edges):
#             node['color']['background'] = '#F0F0F0'
    
#     return graph

# def subgraph_planning(graph: Network):
#     # Remove all edges where the label is not "reasoning:premise-conclusion"
#     edges_to_remove = [edge for edge in graph.edges if not edge['label'].startswith("plan")]
#     for edge in edges_to_remove:
#         # set line and label invisible
#         # edge['hidden'] = True
#         edge['label'] = ""
#         edge['color'] = "#D0D0D0"
#     # Change background color of nodes where no edges with labels "reason:premise-conclusion" are connected
#     for node in graph.nodes:
#         if not node['color']['background'] == color_map["planning"]:
#             node['color']['background'] = '#F0F0F0'
    
#     return graph

FILE = "web/data/math_0_QwQ-32B-Preview_long_correct.json"
with open(FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
    graph = draw_graph(data)  # Call the function to draw the graph
    graph.draw("graph_test.html")  # Save the graph to an HTML file

    # graph = draw_graph(data)  # Call the function to draw the graph
    # graph = subgraph_premise_conclusion(graph)
    # graph.show("graph_entailment.html")
    
    # graph = draw_graph(data)  # Call the function to draw the graph
    # graph = subgraph_restatement(graph)
    # graph.show("graph_restatement.html")

    
    # graph = draw_graph(data)  # Call the function to draw the graph
    # graph = subgraph_planning(graph)
    # graph.show("graph_planning.html")
    
    # graph = draw_graph(data)  # Call the function to draw the graph
    # graph = subgraph_selected_nodes(graph, ["trace8"])
    # graph.show("graph_selected.html")