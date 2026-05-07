#!/usr/bin/env python3
"""Side-by-side 1D node diff for physics_6_QwQ-32B-Preview."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
from generate_svg import node_color_map, edge_color_map, xml_escape, color_to_id, NODE_W

# ── Data ───────────────────────────────────────────────────────────────────────
with open("data/v0_human_D/physics_6_QwQ-32B-Preview.json") as f:
    datum_h = json.load(f)
with open("data/v0_llm_gemini-3.1-pro-preview/physics_6_QwQ-32B-Preview.json") as f:
    datum_l = json.load(f)

h_nodes = datum_h["nodes"]
l_nodes = datum_l["nodes"]

BEFORE  = [(20, 22), (21, 23), (22, 24)]
H_GAP   = [23, 24, 25]
L_GAP   = [25, 26, 27]
AFTER   = [(26, 28), (27, 29), (28, 30)]

H_RANGE = list(range(20, 29))
L_RANGE = list(range(22, 31))

h_id_to_idx = {h_nodes[i]["id"]: i for i in H_RANGE}
l_id_to_idx = {l_nodes[i]["id"]: i for i in L_RANGE}

h_edges_vis = [e for e in datum_h["edges"]
               if e["source_node_id"] in h_id_to_idx and e["dest_node_id"] in h_id_to_idx]
l_edges_vis = [e for e in datum_l["edges"]
               if e["source_node_id"] in l_id_to_idx and e["dest_node_id"] in l_id_to_idx]

# ── Text layout ────────────────────────────────────────────────────────────────
FONT_SIZE  = 11    # px
LINE_H     = 15    # px per line
TEXT_PAD   = 5     # px inside box
WRAP_CHARS = 47    # approx chars per line at NODE_W=320, font-size=11

def wrap(text, max_chars=WRAP_CHARS):
    words = text.split()
    lines, cur, cur_len = [], [], 0
    for w in words:
        wl = len(w)
        if cur and cur_len + 1 + wl > max_chars:
            lines.append(' '.join(cur))
            cur, cur_len = [w], wl
        else:
            cur_len = cur_len + 1 + wl if cur else wl
            cur.append(w)
    if cur:
        lines.append(' '.join(cur))
    return lines or ['']

def node_h(lines):
    return len(lines) * LINE_H + 2 * TEXT_PAD

# ── Layout constants ──────────────────────────────────────────────────────────
LEFT_PAD   = 120
COL_GAP    = 140
RIGHT_PAD  = 170
ROW_PAD    = 5
TOP_MARGIN = 30

BULGE_RATIO = 0.45
MIN_BULGE   = 30
EDGE_OFF    = 3

LEFT_X  = LEFT_PAD
RIGHT_X = LEFT_X + NODE_W + COL_GAP

MATCH_COLOR = "#BBBBBB"
MATCH_DASH  = "5,4"
LEFT_LABEL  = "v0_human_D"
RIGHT_LABEL = "v0_llm_gemini-3.1-pro-preview"

# ── Build node_info (no matplotlib) ──────────────────────────────────────────
node_info = {}

def pre_render(side, idx, nodes):
    key = (side, idx)
    if key in node_info:
        return
    node    = nodes[idx]
    node_id = node.get("id", f"n{idx}")
    text    = node.get("text", "").replace("\n", " ").strip()
    bg      = node_color_map.get(node.get("label", ""), "#EEEEEE")
    lines   = wrap(f"{node_id}: {text}")
    node_info[key] = {
        "h_px": node_h(lines), "bg": bg, "lines": lines,
        "x": LEFT_X if side == "h" else RIGHT_X,
        "y": 0, "center_y": 0,
    }

for hi, li in BEFORE + AFTER:
    pre_render("h", hi, h_nodes)
    pre_render("l", li, l_nodes)
for hi in H_GAP:
    pre_render("h", hi, h_nodes)
for li in L_GAP:
    pre_render("l", li, l_nodes)

# ── Layout ────────────────────────────────────────────────────────────────────
cur_y       = TOP_MARGIN
draw_order  = []
match_lines = []

def set_y(key, y):
    node_info[key]["y"]        = y
    node_info[key]["center_y"] = y + node_info[key]["h_px"] // 2

def place_matched_row(hi, li):
    global cur_y
    hk, lk = ("h", hi), ("l", li)
    row_h = max(node_info[hk]["h_px"], node_info[lk]["h_px"])
    set_y(hk, cur_y + (row_h - node_info[hk]["h_px"]) // 2)
    set_y(lk, cur_y + (row_h - node_info[lk]["h_px"]) // 2)
    draw_order.extend([hk, lk])
    cy = cur_y + row_h // 2
    match_lines.append((LEFT_X + NODE_W, cy, RIGHT_X, cy))
    cur_y += row_h + ROW_PAD

for hi, li in BEFORE:
    place_matched_row(hi, li)

# Unaligned section
gap_y   = cur_y
h_col_h = sum(node_info[("h", hi)]["h_px"] for hi in H_GAP) + ROW_PAD * (len(H_GAP) - 1)
l_col_h = sum(node_info[("l", li)]["h_px"] for li in L_GAP) + ROW_PAD * (len(L_GAP) - 1)
gap_h   = max(h_col_h, l_col_h)

hy = gap_y + (gap_h - h_col_h) // 2
for hi in H_GAP:
    k = ("h", hi)
    set_y(k, hy)
    draw_order.append(k)
    hy += node_info[k]["h_px"] + ROW_PAD

ly = gap_y + (gap_h - l_col_h) // 2
for li in L_GAP:
    k = ("l", li)
    set_y(k, ly)
    draw_order.append(k)
    ly += node_info[k]["h_px"] + ROW_PAD

cur_y = gap_y + gap_h + ROW_PAD

for hi, li in AFTER:
    place_matched_row(hi, li)

total_h = cur_y + 10
total_w = RIGHT_X + NODE_W + RIGHT_PAD

# ── Edge routing ──────────────────────────────────────────────────────────────
def bulge(y0, y1):
    return max(MIN_BULGE, int(abs(y1 - y0) * BULGE_RATIO))

def left_path(y0, y1):
    ax = LEFT_X - EDGE_OFF
    b  = bulge(y0, y1)
    return f"M{ax},{y0} C{ax-b},{y0} {ax-b},{y1} {ax},{y1}"

def right_path(y0, y1):
    ax = RIGHT_X + NODE_W + EDGE_OFF
    b  = bulge(y0, y1)
    return f"M{ax},{y0} C{ax+b},{y0} {ax+b},{y1} {ax},{y1}"

def left_label_pos(y0, y1):
    return LEFT_X - EDGE_OFF - int(0.75 * bulge(y0, y1)), int((y0 + y1) / 2)

def right_label_pos(y0, y1):
    return RIGHT_X + NODE_W + EDGE_OFF + int(0.75 * bulge(y0, y1)), int((y0 + y1) / 2)

def draw_edges(parts, edges, id_to_idx, side, path_fn, label_fn, pfx):
    for e in edges:
        si    = id_to_idx[e["source_node_id"]]
        di    = id_to_idx[e["dest_node_id"]]
        color = edge_color_map.get(e["label"], "#888888")
        cid   = f"{pfx}-{color_to_id(color)}"
        y0    = node_info[(side, si)]["center_y"]
        y1    = node_info[(side, di)]["center_y"]
        parts.append(
            f'<g><defs><marker id="arr-{cid}" markerWidth="8" markerHeight="6" '
            f'refX="7" refY="3" orient="auto">'
            f'<path d="M0,0 L0,6 L8,3 z" fill="{color}"/></marker></defs>'
            f'<path d="{path_fn(y0, y1)}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" marker-end="url(#arr-{cid})"/></g>'
        )
    for e in edges:
        si    = id_to_idx[e["source_node_id"]]
        di    = id_to_idx[e["dest_node_id"]]
        color = edge_color_map.get(e["label"], "#888888")
        short = xml_escape(e["label"].split(":")[-1])
        y0    = node_info[(side, si)]["center_y"]
        y1    = node_info[(side, di)]["center_y"]
        lx, ly = label_fn(y0, y1)
        lw     = len(short) * 5 + 8
        parts.append(
            f'<g><rect x="{lx - lw//2}" y="{ly-7}" width="{lw}" height="13" '
            f'fill="{color}" rx="3"/>'
            f'<text x="{lx}" y="{ly+1}" font-family="" font-size="9" '
            f'fill="#111111" text-anchor="middle" dominant-baseline="middle">'
            f'{short}</text></g>'
        )

# ── Assemble SVG ──────────────────────────────────────────────────────────────
parts = []
parts.append(
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{total_w}" height="{total_h}">'
)

# White background
parts.append(f'<rect width="{total_w}" height="{total_h}" fill="#FFFFFF"/>')

# Column headers
for label, cx in [(LEFT_LABEL, LEFT_X + NODE_W // 2), (RIGHT_LABEL, RIGHT_X + NODE_W // 2)]:
    parts.append(
        f'<text x="{cx}" y="{TOP_MARGIN - 6}" font-family="" font-size="11" '
        f'font-weight="bold" fill="#333333" text-anchor="middle">{label}</text>'
    )

# Matching lines
for x0, y0, x1, y1 in match_lines:
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
        f'stroke="{MATCH_COLOR}" stroke-width="1.5" stroke-dasharray="{MATCH_DASH}"/>'
    )

# Edges (drawn before nodes)
draw_edges(parts, h_edges_vis, h_id_to_idx, "h", left_path,  left_label_pos,  "lft")
draw_edges(parts, l_edges_vis, l_id_to_idx, "l", right_path, right_label_pos, "rgt")

# Node boxes with native SVG text
for key in draw_order:
    info = node_info[key]
    x_n, y_n, h_n = info["x"], info["y"], info["h_px"]
    parts.append(
        f'<rect x="{x_n}" y="{y_n}" width="{NODE_W}" height="{h_n}" fill="{info["bg"]}"/>'
    )
    for i, line in enumerate(info["lines"]):
        ty = y_n + TEXT_PAD + FONT_SIZE + i * LINE_H
        parts.append(
            f'<text x="{x_n + TEXT_PAD}" y="{ty}" '
            f'font-family="" font-size="{FONT_SIZE}" fill="#111111">'
            f'{xml_escape(line)}</text>'
        )
    parts.append(
        f'<rect x="{x_n}" y="{y_n}" width="{NODE_W}" height="{h_n}" '
        f'fill="none" stroke="#999999" stroke-width="1" rx="4" ry="4"/>'
    )

parts.append('</svg>')

out = "segmentation_diff_physics_6_QwQ-32B-Preview.svg"
with open(out, "w", encoding="utf-8") as f:
    f.write('\n'.join(parts))
print(f"Saved → {out}")
