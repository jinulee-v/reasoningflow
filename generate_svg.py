import json
import yaml
import io
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Schema loading ─────────────────────────────────────────────────────────────
with open("schema/node_labels.yaml", 'r', encoding='utf-8') as f:
    _node_schema = yaml.safe_load(f)
    node_color_map = {n['name']: n['color'] for n in _node_schema['nodes']}

with open("schema/edge_labels.yaml", 'r', encoding='utf-8') as f:
    _edge_schema = yaml.safe_load(f)
    edge_color_map = {e['name']: e['color'] for e in _edge_schema['edges']}

# ── Layout constants ───────────────────────────────────────────────────────────
NODE_W      = 320   # node box width in pixels
DPI         = 96    # rendering resolution
NODE_MARGIN = 15    # vertical gap between nodes
FONT_SIZE   = 8     # points
LINE_H      = 13    # pixels per rendered line
TEXT_PAD    = 5     # pixels of padding inside node boxes
WRAP_CHARS  = 55    # target max characters per line (non-math tokens)

# ── LaTeX helpers ──────────────────────────────────────────────────────────────
def to_mathtext(text: str) -> str:
    """Convert \\(...\\) and \\[...\\] LaTeX delimiters to matplotlib $...$."""
    text = re.sub(r'\\\[(.+?)\\\]', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text


def wrap_latex_aware(text: str, max_chars: int = WRAP_CHARS) -> list:
    """Word-wrap text while keeping $...$ math blocks as atomic tokens."""
    math_re = re.compile(r'\$[^$]+\$')
    segments = math_re.split(text)
    maths = math_re.findall(text)

    tokens = []
    for i, seg in enumerate(segments):
        tokens.extend(w for w in seg.split() if w)
        if i < len(maths):
            tokens.append(maths[i])

    lines, cur, cur_len = [], [], 0
    for tok in tokens:
        tl = len(tok)
        if cur and cur_len + 1 + tl > max_chars:
            lines.append(' '.join(cur))
            cur, cur_len = [tok], tl
        else:
            cur_len = (cur_len + 1 + tl) if cur else tl
            cur.append(tok)
    if cur:
        lines.append(' '.join(cur))
    return lines or ['']


# ── SVG rendering ──────────────────────────────────────────────────────────────
def _strip_math(lines: list) -> list:
    """Remove $...$ math markup, leaving plain text."""
    return [re.sub(r'\$([^$]+)\$', r'\1', line) for line in lines]


def _prefix_ids(svg_str: str, prefix: str) -> str:
    """Rename every id and its references in an SVG string with a unique prefix.

    A single pass per pattern avoids double-prefixing: the bare 'href="#'
    pattern already matches inside 'xlink:href="#', so no separate rule is
    needed for xlink:href.
    """
    svg_str = re.sub(r'\bid="', f'id="{prefix}-',  svg_str)
    svg_str = re.sub(r'href="#', f'href="#{prefix}-', svg_str)  # covers both href and xlink:href
    svg_str = re.sub(r'url\(#',  f'url(#{prefix}-',  svg_str)
    return svg_str


def _mpl_to_svg_bytes(lines: list, bg_color: str) -> bytes:
    """Render text lines (may contain mathtext) to SVG bytes via matplotlib."""
    h_px = len(lines) * LINE_H + 2 * TEXT_PAD
    w_in = NODE_W / DPI
    h_in = max(h_px / DPI, 0.15)

    fig = plt.figure(figsize=(w_in, h_in), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, NODE_W)
    ax.set_ylim(h_px, 0)          # y=0 at top
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for i, line in enumerate(lines):
        ax.text(TEXT_PAD, TEXT_PAD + i * LINE_H, line,
                fontsize=FONT_SIZE, va='top', ha='left', color='#111111')

    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches=None,
                facecolor=bg_color, edgecolor='none')
    plt.close(fig)
    return buf.getvalue()


def render_node_svg(lines: list, bg_color: str, prefix: str) -> tuple:
    """Render node text to SVG, with fallback to plain text on mathtext error.

    Returns (defs_str, figure_str, vb_w, vb_h, h_px):
      defs_str    – raw inner content of matplotlib's <defs> block
      figure_str  – raw <g> block holding the rendered figure content
      vb_w, vb_h  – matplotlib SVG viewBox dimensions
      h_px        – intended node height in pixels

    String extraction is used instead of ElementTree serialization to avoid
    namespace-prefix corruption when re-serializing parsed SVG elements.
    """
    h_px = len(lines) * LINE_H + 2 * TEXT_PAD

    try:
        raw = _mpl_to_svg_bytes(lines, bg_color)
    except Exception:
        plt.close('all')
        raw = _mpl_to_svg_bytes(_strip_math(lines), bg_color)

    # Prefix all IDs so multiple nodes don't share glyph definition names
    svg_str = _prefix_ids(raw.decode('utf-8'), prefix)

    # viewBox (matplotlib reports in points: width_pt x height_pt)
    vb_match = re.search(r'viewBox=["\']([^"\']+)["\']', svg_str)
    if vb_match:
        vb = [float(x) for x in vb_match.group(1).split()]
        vb_w, vb_h = vb[2], vb[3]
    else:
        vb_w, vb_h = float(NODE_W), float(h_px)

    # Inner content of <defs> (glyph paths, clip regions, style)
    defs_match = re.search(r'<defs>(.*?)</defs>', svg_str, re.DOTALL)
    defs_str = defs_match.group(1).strip() if defs_match else ''

    # Figure <g> block: everything between </defs> and </svg>
    after_defs = svg_str.split('</defs>', 1)[-1] if '</defs>' in svg_str else svg_str
    g_match = re.search(r'(<g\b.*)\s*</svg>', after_defs, re.DOTALL)
    figure_str = g_match.group(1).strip() if g_match else ''

    return defs_str, figure_str, vb_w, vb_h, h_px


# ── SVG helpers ────────────────────────────────────────────────────────────────
def xml_escape(s: str) -> str:
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
             .replace('"', '&quot;'))


def color_to_id(color: str) -> str:
    """Make a safe XML id fragment from a hex color like '#FFADAD'."""
    return 'c' + color.lstrip('#')


# ── Main draw function ─────────────────────────────────────────────────────────
def draw_graph(data: dict, output_path: str = "graph.svg"):
    nodes = data["nodes"]
    edges = data["edges"]

    # 1. Render each node's text to SVG and record layout
    node_info: dict = {}
    all_defs_parts: list = []   # raw inner-<defs> strings from all node SVGs
    current_y = 0

    for idx, node in enumerate(nodes):
        nid   = node["id"]
        label = node.get("label", "")
        text  = node.get("text", "")
        bg    = node_color_map.get(label, "#EEEEEE")

        raw = to_mathtext(f"{nid}: {text.replace(chr(10), ' ')}")
        lines = wrap_latex_aware(raw)

        prefix = f"n{idx}"
        defs_str, figure_str, vb_w, vb_h, h_px = render_node_svg(lines, bg, prefix)
        all_defs_parts.append(defs_str)

        node_info[nid] = {
            "y":          current_y,
            "h":          h_px,
            "center_y":   current_y + h_px // 2,
            "figure_str": figure_str,
            "vb_w":       vb_w,
            "vb_h":       vb_h,
            "bg":         bg,
        }
        current_y += h_px + NODE_MARGIN

    total_h = current_y

    # 2. Compute SVG width from the maximum edge bulge
    attach_x = NODE_W + 4
    BULGE_RATIO = 0.45
    LABEL_MARGIN = 60  # extra room to the right of the widest label/path

    max_bulge = 30  # minimum
    for edge in edges:
        src_id = edge.get("source_node_id", "")
        dst_id = edge.get("dest_node_id", "")
        if src_id not in node_info or dst_id not in node_info:
            continue
        dy = abs(node_info[src_id]["center_y"] - node_info[dst_id]["center_y"])
        max_bulge = max(max_bulge, max(30, int(dy * BULGE_RATIO)))

    svg_w = attach_x + max_bulge + LABEL_MARGIN

    # 3. Build SVG
    parts = []
    parts.append(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{svg_w}" height="{total_h}">'
    )

    # <defs>: hoisted glyph/font defs from node SVGs only
    parts.append('<defs>')
    for defs_str in all_defs_parts:
        parts.append(defs_str)
    parts.append('</defs>')

    # Node boxes: background rect + nested <svg> for vector text
    for node in nodes:
        nid  = node["id"]
        info = node_info[nid]
        y_n  = info["y"]
        h_n  = info["h"]
        bg   = info["bg"]
        vb_w = info["vb_w"]
        vb_h = info["vb_h"]
        figure_str = info["figure_str"]

        # Background fill (no stroke – border drawn after content so it's on top)
        parts.append(
            f'<rect x="0" y="{y_n}" width="{NODE_W}" height="{h_n}" '
            f'fill="{bg}"/>'
        )
        # Nested SVG: maps matplotlib's coordinate space onto our pixel box
        parts.append(
            f'<svg x="0" y="{y_n}" width="{NODE_W}" height="{h_n}" '
            f'viewBox="0 0 {vb_w:.4f} {vb_h:.4f}" overflow="hidden">'
        )
        parts.append(figure_str)
        parts.append('</svg>')
        # Border on top of content
        parts.append(
            f'<rect x="0" y="{y_n}" width="{NODE_W}" height="{h_n}" '
            f'fill="none" stroke="#999999" stroke-width="1" rx="6" ry="6"/>'
        )

    # Edges as cubic Bezier curves routed to the right of the nodes.
    # Pass 1: draw all arrow paths (each grouped with its arrowhead marker)
    valid_edges = []
    for edge in edges:
        src_id = edge.get("source_node_id", "")
        dst_id = edge.get("dest_node_id", "")
        if src_id not in node_info or dst_id not in node_info:
            continue
        valid_edges.append(edge)

    for edge in valid_edges:
        src_id  = edge.get("source_node_id", "")
        dst_id  = edge.get("dest_node_id", "")
        e_label = edge.get("label", "")
        color   = edge_color_map.get(e_label, "#888888")
        cid     = color_to_id(color)

        y0 = node_info[src_id]["center_y"]
        y1 = node_info[dst_id]["center_y"]
        bulge = max(30, int(abs(y1 - y0) * BULGE_RATIO))

        cx1, cy1 = attach_x + bulge, y0
        cx2, cy2 = attach_x + bulge, y1
        path = f"M{attach_x},{y0} C{cx1},{cy1} {cx2},{cy2} {attach_x},{y1}"

        parts.append('<g>')
        parts.append(
            f'  <defs><marker id="arr-{cid}" markerWidth="8" markerHeight="6" '
            f'refX="7" refY="3" orient="auto">'
            f'<path d="M0,0 L0,6 L8,3 z" fill="{color}"/></marker></defs>'
        )
        parts.append(
            f'  <path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
            f'marker-end="url(#arr-{cid})"/>'
        )
        parts.append('</g>')

    # Pass 2: draw all edge labels on top of the paths
    for edge in valid_edges:
        src_id  = edge.get("source_node_id", "")
        dst_id  = edge.get("dest_node_id", "")
        e_label = edge.get("label", "")
        color   = edge_color_map.get(e_label, "#888888")
        short   = xml_escape(e_label.split(":")[-1])

        y0 = node_info[src_id]["center_y"]
        y1 = node_info[dst_id]["center_y"]
        bulge = max(30, int(abs(y1 - y0) * BULGE_RATIO))

        mid_x = int(attach_x + 0.75 * bulge)
        mid_y = int((y0 + y1) / 2)
        lw    = len(short) * 5 + 8

        parts.append('<g>')
        parts.append(
            f'  <rect x="{mid_x - lw // 2}" y="{mid_y - 7}" '
            f'width="{lw}" height="13" fill="{color}" rx="3"/>'
        )
        parts.append(
            f'  <text x="{mid_x}" y="{mid_y + 1}" '
            f'font-family="sans-serif" font-size="9" '
            f'fill="#111111" text-anchor="middle" dominant-baseline="middle">'
            f'{short}</text>'
        )
        parts.append('</g>')

    parts.append('</svg>')

    svg_content = '\n'.join(parts)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"Saved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
FILE = "data/v0_human_D/physics_4_QwQ-32B-Preview.json"
with open(FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

draw_graph(data, "graph.svg")
