import json
from networkx import Graph
from pyvis.network import Network
import textwrap
import pvsvg

with open("static/labels.json", 'r', encoding='utf-8') as f:
    label_config = json.load(f)
    node_color_map = label_config["node_colors"]
    edge_color_map = label_config["edge_colors"]

def draw_graph(data):
    net = Network(height="1000px", width="100%")
    net.toggle_physics(False)

    # Approximate function for box height (assuming ~20px per line, 50 chars per line)
    def estimate_node_height(text, line_height=18, padding=1):
        lines = len(text.split('\n'))
        return lines * line_height + padding

    # Layout with consistent top-to-bottom box margin
    margin = 15  # pixels between bottom of one box and top of next
    current_y = 0
    x_pos = 0

    for node in data["nodes"]:
        label = "\n".join(textwrap.wrap(node['id'] + ": " + node['text'].replace("\n", " "), width=50))
        node_width = 390
        node_height = estimate_node_height(label)
        y_pos = current_y + node_height//2  # center point of the box

        net.add_node(
            node["id"],
            label=label,
            shape="box",
            widthConstraint={"minimum": node_width, "maximum": node_width},
            heightConstraint={"minimum": node_height, "maximum":node_height},
            color=node_color_map[node["label"]],
            borderRadius=10,
            x=x_pos,
            y=y_pos,
            fixed=True,
            margin=1
        )
        net.add_node(
            node["id"] + "-ghost",
            label=" ",
            shape="dot", size=1,
            color="gray",
            x=x_pos + node_width//2,
            y=y_pos,
            fixed=True,
        )
        current_y += node_height + margin

    # Add edges with styled labels
    for edge in data["edges"]:
        net.add_edge(
            edge["from_node_id"] + "-ghost",
            edge["to_node_id"] + "-ghost",
            label=edge["label"].split(":")[-1],
            arrows={'to':{'enabled':True}},
            stroke="4",
            font={
                "background": edge_color_map[edge["label"]],
                "strokeWidth": 0,
                "multi": "html"
            },
            smooth={
                "type": "curvedCW",
                "roundness": 0.6
            }
        )
    return net

# FILE = "data/math_0_QwQ-32B-Preview_long_correct.json"
FILE = "data/math_14_QwQ-32B-Preview_long_correct.json"
with open(FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
    net = draw_graph(data)  # Call the function to draw the graph

    graph= Graph()
    for node in net.nodes:
        graph.add_node(
            node["id"],
            **node
        )
    for edge in net.edges:
        graph.add_edge(
            edge["from"],
            edge["to"],
            **edge
        )

    new_network = pvsvg.Network(
        graph,
        height="1000px",
        width="100%",
        physics_kwargs={"enabled": False},
    )
    new_network.draw("graph_test.html")  # Save the graph to an HTML file

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