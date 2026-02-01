import yaml
from pathlib import Path
from textwrap import indent

NODE_YAML = "schema/node_labels.yaml"
NODE_CLS_MD = "parser/prompts/2_node_classification_prompt.txt"


def md_escape(text: str) -> str:
    """Minimal escaping for Markdown."""
    return text.replace("\n\n", "\n").strip()


def render_examples(examples):
    blocks = []
    for ex in examples:
        nodes = ex.get("nodes", [])
        if not nodes:
            continue

        block = []
        for n in nodes:
            node_type = n.get("type", "null")
            text = n.get("text", "").strip()
            if node_type and text:
                block.append(f"- {text} -> {node_type}")
        if block:
            blocks.append("\n".join(block))

    if not blocks:
        return ""

    return "\n\n".join(f"> {b.replace(chr(10), chr(10) + '> ')}" for b in blocks)


def render_subtypes(subtypes):
    out = []
    for st in subtypes:
        out.append(f"#### {st['type']}")
        if st.get("description"):
            out.append(md_escape(st["description"]))

        examples = render_examples(st.get("examples", []))
        if examples:
            out.append("**Examples:**")
            out.append(examples)

        out.append("")  # spacing
    return "\n".join(out)

# def legacy_generate_node_classification_prompt():
#     data = yaml.safe_load(Path(NODE_YAML).read_text())

#     md = []
#     md.append("# Node Classification Guidelines")
#     md.append("")
#     md.append(
#         """Reasoning traces are segmented into nodes, which are syntactically non-overlapping segments ranging from clause to one sentence (exceptionally multi-line equations), and are semantically atomic. Each node is assigned a distinct label based on its semantic role.\n"""
#         """Segment into atomic units that are uniform in their semantic roles, often a sentence or a clause. If not sure, just segment into sentence. Make the node "text" as short as possible. You don't have token limits so feel free to list all fine-grained nodes.\n"""
#         """The "text" part must EXACTLY copy the given text without any difference, so that `text in raw_text == True` always holds."""
#     )
#     md.append("")
#     md.append("# Labels")
#     md.append("")
    
#     exclude_labels = {"context", "conclusion"}
#     nodes = [n for n in data["nodes"] if n["name"] not in exclude_labels]

#     for label in nodes:
#         md.append(f"## {label['name'].capitalize()}")
#         md.append("")
#         if label.get("description"):
#             md.append(md_escape(label["description"]))
#             md.append("")

#         if label.get("subtypes"):
#             md.append("")
#             md.append(render_subtypes(label["subtypes"]))

#         md.append("")

#     md.append("## Classification Rules")
#     md.append("")
#     md.append(
#         """Classify this text, using the examples above and previous steps provided for better contex of the step's role. Respond in JSON Dict {"label": "(label)"}.\n"""
#         f"""You must choose one of these labels: {', '.join([n['name'] for n in nodes])}."""
#         "\n\n"
#         """<<input>>"""
#     )
    
#     full_md = "\n".join(md)
#     full_md = full_md.replace("**", "") # Remove unnecessary bolding for LLMs

#     Path(NODE_CLS_MD).write_text(full_md)
#     print(f"Markdown prompt written to: {NODE_CLS_MD}")

def generate_node_classification_prompt():
    data = yaml.safe_load(Path(NODE_YAML).read_text())

    md = []
    md.append("# Node Classification Guidelines")
    md.append("")
    md.append(
        """Reasoning traces are already segmented into nodes, which are syntactically non-overlapping segments ranging from clause to one sentence (exceptionally multi-line equations), and are semantically atomic.\n"""
        """Your job is to assign labels to all nodes according to the provide description and examples below. Make sure you do not miss any nodes.\n"""
        """Respond by a list of JSON Dicts {"responses": [{"node_id": "(node id)", "label": "(label)"}, ...]}\n"""
    )
    md.append("")
    md.append("# Labels")
    md.append("")
    
    exclude_labels = {"context", "conclusion"}
    nodes = [n for n in data["nodes"] if n["name"] not in exclude_labels]

    for label in nodes:
        md.append(f"## {label['name'].capitalize()}")
        md.append("")
        if label.get("description"):
            md.append(md_escape(label["description"]))
            md.append("")

        if label.get("subtypes"):
            md.append("")
            md.append(render_subtypes(label["subtypes"]))

        md.append("")

    md.append("## Classification Rules")
    md.append("")
    md.append(
        """Reasoning traces are already segmented into nodes, which are syntactically non-overlapping segments ranging from clause to one sentence (exceptionally multi-line equations), and are semantically atomic.\n"""
        """Your job is to assign labels to all nodes according to the provide description and examples below. Make sure you do not miss any nodes.\n"""
        """Respond by a list of JSON Dicts {"responses": [{"node_id": "(node id)", "label": "(label)"}, ...]}\n"""
    )
    md.append("")
    md.append("<<input>>")
    
    full_md = "\n".join(md)
    full_md = full_md.replace("**", "") # Remove unnecessary bolding for LLMs

    Path(NODE_CLS_MD).write_text(full_md)
    print(f"Markdown prompt written to: {NODE_CLS_MD}")

def generate_update_conclusion_prompt():
    data = yaml.safe_load(Path(NODE_YAML).read_text())
    
    md = []
    md.append("# Conclusion Node Update Guidelines")
    md.append("")
    for label in data["nodes"]:
        if label["name"] != "conclusion":
            continue
        md.append(f"## {label['name'].capitalize()}")
        md.append("")
        if label.get("description"):
            md.append(md_escape(label["description"]))
            md.append("")

        if label.get("subtypes"):
            md.append("")
            md.append(render_subtypes(label["subtypes"]))

        md.append("")

    md.append(
        """Annotate which of the nodes should be changed to "conclusion" nodes.\n"""
        """Respond in JSON Dict {"conclusion_node_ids": ["(node id)", "(node id)", ...]}.\n"""
        "\n\n"
        """<<input>>"""
    )
    
    full_md = "\n".join(md)
    conclusion_md_path = "parser/prompts/3_update_conclusion_prompt.txt"
    Path(conclusion_md_path).write_text(full_md)
    print(f"Markdown prompt written to: {conclusion_md_path}")

if __name__ == "__main__":
    generate_node_classification_prompt()
    
    generate_update_conclusion_prompt()
