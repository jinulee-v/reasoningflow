#!/usr/bin/env python3
"""Print unaligned regions from Hungarian-matched trace pairs, side by side."""

import json
import os
import re
import argparse
from itertools import zip_longest
import numpy as np

JACCARD_THRESHOLD = 0.9
CONTEXT_SIZE = 3
COL_WIDTH = 72


def node_word_tokens(text):
    return set(re.findall(r'[a-zA-Z0-9]+', text.lower()))


def jaccard_sim(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def align_nodes(h_nodes, l_nodes):
    """Return (h2l, l2h) dicts mapping index->index for order-preserving matched pairs.

    Uses a weighted LCS DP so that if h[i]->l[j] and h[i']->l[j'] are both matched,
    i < i' implies j < j' (no crossing matches).
    """
    h_tok = [node_word_tokens(n["text"]) for n in h_nodes]
    l_tok = [node_word_tokens(n["text"]) for n in l_nodes]
    n_h, n_l = len(h_tok), len(l_tok)
    h2l, l2h = {}, {}
    if n_h == 0 or n_l == 0:
        return h2l, l2h

    sim_mat = np.array([[jaccard_sim(ht, lt) for lt in l_tok] for ht in h_tok])

    # dp[i][j] = max total similarity for an order-preserving matching
    # using only h[0..i-1] and l[0..j-1], including only pairs above threshold.
    dp = np.zeros((n_h + 1, n_l + 1))
    for i in range(1, n_h + 1):
        for j in range(1, n_l + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            if sim_mat[i - 1][j - 1] > JACCARD_THRESHOLD:
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + sim_mat[i - 1][j - 1])

    # Backtrack: prefer matching over skipping when the match explains dp[i][j].
    i, j = n_h, n_l
    while i > 0 and j > 0:
        s = sim_mat[i - 1][j - 1]
        if s > JACCARD_THRESHOLD and dp[i - 1][j - 1] + s >= dp[i][j] - 1e-10:
            h2l[i - 1] = j - 1
            l2h[j - 1] = i - 1
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return h2l, l2h


def fmt_node(node, idx, width):
    """Format a node as '[idx|label] text...' truncated to width."""
    label = node.get("label", "?")
    text = node["text"].replace("\n", " ").strip()
    header = f"[{idx}|{label}] "
    avail = width - len(header)
    if avail <= 0:
        return header
    if len(text) <= avail:
        return header + text
    return header + text[:avail - 3] + "..."


def find_unaligned_regions(h_nodes, l_nodes, h2l):
    """
    Identify contiguous unaligned gaps between consecutive aligned pairs.

    Returns a list of dicts:
        h_gap:      list of h_node indices in the gap
        l_gap:      list of l_node indices in the gap
        before_pairs: up to CONTEXT_SIZE aligned (h_idx, l_idx) pairs before the gap
        after_pairs:  up to CONTEXT_SIZE aligned (h_idx, l_idx) pairs after the gap
    """
    aligned_pairs = sorted(h2l.items())  # [(h_idx, l_idx), ...] sorted by h_idx
    regions = []

    for i in range(-1, len(aligned_pairs)):
        h_prev, l_prev = aligned_pairs[i] if i >= 0 else (-1, -1)
        h_next, l_next = (
            aligned_pairs[i + 1] if i + 1 < len(aligned_pairs)
            else (len(h_nodes), len(l_nodes))
        )

        h_gap = list(range(h_prev + 1, h_next))
        l_gap = list(range(l_prev + 1, l_next))

        if not h_gap and not l_gap:
            continue

        before_pairs = aligned_pairs[max(0, i - CONTEXT_SIZE + 1): i + 1]
        after_pairs  = aligned_pairs[i + 1: i + 1 + CONTEXT_SIZE]

        regions.append({
            "h_gap":        h_gap,
            "l_gap":        l_gap,
            "before_pairs": before_pairs,
            "after_pairs":  after_pairs,
        })

    return regions


def print_region(region, h_nodes, l_nodes, left_label, right_label,
                 doc_id, col_width=COL_WIDTH):
    h_gap = region["h_gap"]
    l_gap = region["l_gap"]
    before = region["before_pairs"]
    after  = region["after_pairs"]

    total_width = col_width * 2 + 3
    bar = "─" * col_width

    print(f"\n{'═' * total_width}")
    print(f"  {doc_id}")
    left_indices  = h_gap  if h_gap  else ["(none)"]
    right_indices = l_gap  if l_gap  else ["(none)"]
    print(f"  LEFT  unaligned: {left_indices}")
    print(f"  RIGHT unaligned: {right_indices}")
    print(f"{'═' * total_width}")
    print(f"{left_label:^{col_width}} | {right_label:^{col_width}}")
    print(f"{bar}─+─{bar}")

    if before:
        label_line = "── context before ──"
        print(f"{label_line:^{col_width}} | {label_line:^{col_width}}")
        for h_idx, l_idx in before:
            left  = fmt_node(h_nodes[h_idx], h_idx, col_width)
            right = fmt_node(l_nodes[l_idx], l_idx, col_width)
            print(f"{left:<{col_width}} | {right}")

    if h_gap or l_gap:
        label_line = "── unaligned ──"
        print(f"{bar}─+─{bar}")
        print(f"{label_line:^{col_width}} | {label_line:^{col_width}}")
        left_lines  = [fmt_node(h_nodes[i], i, col_width) for i in h_gap]
        right_lines = [fmt_node(l_nodes[i], i, col_width) for i in l_gap]
        for left, right in zip_longest(left_lines, right_lines, fillvalue=""):
            print(f"{left:<{col_width}} | {right}")

    if after:
        label_line = "── context after ──"
        print(f"{bar}─+─{bar}")
        print(f"{label_line:^{col_width}} | {label_line:^{col_width}}")
        for h_idx, l_idx in after:
            left  = fmt_node(h_nodes[h_idx], h_idx, col_width)
            right = fmt_node(l_nodes[l_idx], l_idx, col_width)
            print(f"{left:<{col_width}} | {right}")


def main():
    parser = argparse.ArgumentParser(
        description="Show unaligned regions from Hungarian-matched trace pairs.")
    parser.add_argument("dir_left",  help="Left annotation directory")
    parser.add_argument("dir_right", help="Right annotation directory")
    parser.add_argument(
        "--max-regions", type=int, default=None,
        help="Stop after this many unaligned regions total")
    parser.add_argument(
        "--file", default=None,
        help="Only process this specific filename (e.g. math_0_QwQ-32B-Preview.json)")
    parser.add_argument(
        "--col-width", type=int, default=COL_WIDTH,
        help="Column width for each side (default: %(default)s)")
    parser.add_argument(
        "--both-sides", action="store_true",
        help="Only show regions where BOTH sides have unaligned nodes "
             "(skip pure insertion/deletion)")
    args = parser.parse_args()

    left_label  = os.path.basename(args.dir_left.rstrip("/"))
    right_label = os.path.basename(args.dir_right.rstrip("/"))

    files = [args.file] if args.file else sorted(os.listdir(args.dir_left))
    region_count = 0

    for fname in files:
        if not fname.endswith(".json"):
            continue
        left_path  = os.path.join(args.dir_left,  fname)
        right_path = os.path.join(args.dir_right, fname)
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            continue

        with open(left_path)  as f: datum_left  = json.load(f)
        with open(right_path) as f: datum_right = json.load(f)

        h_nodes = datum_left["nodes"]
        l_nodes = datum_right["nodes"]
        h2l, _  = align_nodes(h_nodes, l_nodes)

        regions = find_unaligned_regions(h_nodes, l_nodes, h2l)

        for region in regions:
            if args.both_sides and not (region["h_gap"] and region["l_gap"]):
                continue
            print_region(
                region, h_nodes, l_nodes,
                left_label, right_label,
                datum_left.get("doc_id", fname),
                col_width=args.col_width,
            )
            region_count += 1
            if args.max_regions and region_count >= args.max_regions:
                return


if __name__ == "__main__":
    main()
