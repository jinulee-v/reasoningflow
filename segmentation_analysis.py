import json
import os
import re
import numpy as np

JACCARD_THRESHOLD = 0.9

def node_word_tokens(text):
    return set(re.findall(r'[a-zA-Z0-9]+', text.lower()))

def jaccard_sim(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0

def find_one_to_many_in_gap(h_toks, l_toks, threshold=JACCARD_THRESHOLD):
    """
    Within a gap (contiguous unmatched h and l nodes), find which nodes participate
    in a one-to-many alignment (one node = concatenation of multiple on the other side).

    Returns sets of indices (within h_toks / l_toks) that participate in:
      h_one: h nodes that are the "one" in one h -> many l
      h_many: h nodes that are the "many" in many h -> one l
      l_one: l nodes that are the "one" in one l -> many h
      l_many: l nodes that are the "many" in many l -> one h
    """
    nh, nl = len(h_toks), len(l_toks)
    if nh == 0 or nl == 0:
        return set(), set(), set(), set()

    # Precompute cumulative token unions for fast range queries
    h_cum = [set()] * (nh + 1)
    for i in range(nh):
        h_cum[i + 1] = h_cum[i] | h_toks[i]

    l_cum = [set()] * (nl + 1)
    for j in range(nl):
        l_cum[j + 1] = l_cum[j] | l_toks[j]

    def h_union(i1, i2):
        return h_cum[i2] - (h_cum[i1] - h_cum[i1])  # can't subtract properly; recompute
    def l_union_range(j1, j2):
        u = set()
        for k in range(j1, j2):
            u |= l_toks[k]
        return u
    def h_union_range(i1, i2):
        u = set()
        for k in range(i1, i2):
            u |= h_toks[k]
        return u

    # DP: dp[i][j] = max nodes explained by one-to-many alignment for h[0..i-1], l[0..j-1]
    dp = np.full((nh + 1, nl + 1), -1, dtype=float)
    bp = [[None] * (nl + 1) for _ in range(nh + 1)]
    dp[0][0] = 0.0

    for i in range(nh + 1):
        for j in range(nl + 1):
            if dp[i][j] < 0:
                continue
            cur = dp[i][j]

            # Skip h[i] (unresolved misalignment)
            if i < nh:
                if cur > dp[i + 1][j]:
                    dp[i + 1][j] = cur
                    bp[i + 1][j] = (i, j, 'skip_h')

            # Skip l[j] (unresolved misalignment)
            if j < nl:
                if cur > dp[i][j + 1]:
                    dp[i][j + 1] = cur
                    bp[i][j + 1] = (i, j, 'skip_l')

            # One h[i] -> many l[j..j2-1] (j2 >= j+2)
            if i < nh:
                for j2 in range(j + 2, nl + 1):
                    lu = l_union_range(j, j2)
                    if jaccard_sim(h_toks[i], lu) > threshold:
                        explained = 1 + (j2 - j)
                        val = cur + explained
                        if val > dp[i + 1][j2]:
                            dp[i + 1][j2] = val
                            bp[i + 1][j2] = (i, j, ('h_to_many_l', j2))

            # Many h[i..i2-1] -> one l[j] (i2 >= i+2)
            if j < nl:
                for i2 in range(i + 2, nh + 1):
                    hu = h_union_range(i, i2)
                    if jaccard_sim(hu, l_toks[j]) > threshold:
                        explained = (i2 - i) + 1
                        val = cur + explained
                        if val > dp[i2][j + 1]:
                            dp[i2][j + 1] = val
                            bp[i2][j + 1] = (i, j, ('many_h_to_l', i2))

    # Backtrack
    h_one = set()   # h nodes that are "one" in one-h -> many-l
    h_many = set()  # h nodes that are "many" in many-h -> one-l
    l_one = set()   # l nodes that are "one" in one-l -> many-h
    l_many = set()  # l nodes that are "many" in many-l -> one-h

    i, j = nh, nl
    while i > 0 or j > 0:
        move = bp[i][j]
        if move is None:
            break
        pi, pj, action = move
        if action == 'skip_h' or action == 'skip_l':
            pass
        elif isinstance(action, tuple):
            atype, end = action
            if atype == 'h_to_many_l':
                h_one.add(pi)
                for k in range(pj, end):
                    l_many.add(k)
            elif atype == 'many_h_to_l':
                for k in range(pi, end):
                    h_many.add(k)
                l_one.add(pj)
        i, j = pi, pj

    return h_one, h_many, l_one, l_many


# Aggregate counters
total_misaligned_h = 0
total_misaligned_l = 0
one_to_many_h = 0  # h nodes in any one-to-many pattern (as either "one" or "many")
one_to_many_l = 0  # l nodes in any one-to-many pattern

files_processed = 0

for file in sorted(os.listdir("data/v0_human_D")):
    if not file.endswith(".json"):
        continue
    try:
        with open(os.path.join("data/v0_human_D", file), "r") as f:
            datum_human = json.load(f)
        with open(os.path.join("data/v0_llm_gemini-3.1-pro-preview", file), "r") as f:
            datum_llm = json.load(f)
        assert datum_human["doc_id"] == datum_llm["doc_id"]
    except FileNotFoundError:
        continue
    except Exception as e:
        print(e.__class__, e)
        continue

    files_processed += 1
    h_nodes = datum_human["nodes"]
    l_nodes = datum_llm["nodes"]
    h_tok = [node_word_tokens(hn["text"]) for hn in h_nodes]
    l_tok = [node_word_tokens(ln["text"]) for ln in l_nodes]
    n_h, n_l = len(h_tok), len(l_tok)

    # Order-preserving weighted LCS alignment (from evaluate_llm_annot.py)
    corresponding_nodes_h2l = {}
    corresponding_nodes_l2h = {}
    if n_h > 0 and n_l > 0:
        sim_mat = np.array([[jaccard_sim(ht, lt) for lt in l_tok] for ht in h_tok])
        dp = np.zeros((n_h + 1, n_l + 1))
        for i in range(1, n_h + 1):
            for j in range(1, n_l + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                if sim_mat[i - 1][j - 1] > JACCARD_THRESHOLD:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + sim_mat[i - 1][j - 1])
        i, j = n_h, n_l
        while i > 0 and j > 0:
            s = sim_mat[i - 1][j - 1]
            if s > JACCARD_THRESHOLD and dp[i - 1][j - 1] + s >= dp[i][j] - 1e-10:
                corresponding_nodes_h2l[h_nodes[i - 1]["id"]] = l_nodes[j - 1]
                corresponding_nodes_l2h[l_nodes[j - 1]["id"]] = h_nodes[i - 1]
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

    # Build matched anchor pairs (sorted by h index)
    h_id_to_idx = {hn["id"]: i for i, hn in enumerate(h_nodes)}
    l_id_to_idx = {ln["id"]: j for j, ln in enumerate(l_nodes)}

    matched_pairs = []
    for hn in h_nodes:
        if hn["id"] in corresponding_nodes_h2l:
            ln = corresponding_nodes_h2l[hn["id"]]
            matched_pairs.append((h_id_to_idx[hn["id"]], l_id_to_idx[ln["id"]]))
    matched_pairs.sort()

    # Build gaps between consecutive anchor pairs
    anchors = [(-1, -1)] + matched_pairs + [(n_h, n_l)]
    file_mis_h = 0
    file_mis_l = 0
    file_otm_h = 0
    file_otm_l = 0

    for k in range(len(anchors) - 1):
        h_gap = list(range(anchors[k][0] + 1, anchors[k + 1][0]))
        l_gap = list(range(anchors[k][1] + 1, anchors[k + 1][1]))
        if not h_gap and not l_gap:
            continue

        nh_g = len(h_gap)
        nl_g = len(l_gap)
        file_mis_h += nh_g
        file_mis_l += nl_g

        h_toks_g = [h_tok[i] for i in h_gap]
        l_toks_g = [l_tok[j] for j in l_gap]

        h_one, h_many, l_one, l_many = find_one_to_many_in_gap(h_toks_g, l_toks_g)

        # Nodes participating in any one-to-many pattern
        h_explained = h_one | h_many
        l_explained = l_one | l_many
        file_otm_h += len(h_explained)
        file_otm_l += len(l_explained)

    total_misaligned_h += file_mis_h
    total_misaligned_l += file_mis_l
    one_to_many_h += file_otm_h
    one_to_many_l += file_otm_l
    print(f"{file}: mis_h={file_mis_h}, mis_l={file_mis_l}, otm_h={file_otm_h}, otm_l={file_otm_l}")

print()
print("=" * 50)
print(f"Files processed: {files_processed}")
print(f"Total misaligned nodes (human side): {total_misaligned_h}")
print(f"Total misaligned nodes (LLM side):   {total_misaligned_l}")
print(f"Total misaligned:                    {total_misaligned_h + total_misaligned_l}")
print()
print(f"One-to-many (human side):  {one_to_many_h} / {total_misaligned_h} "
      f"({100 * one_to_many_h / total_misaligned_h:.1f}%)" if total_misaligned_h else "N/A")
print(f"One-to-many (LLM side):    {one_to_many_l} / {total_misaligned_l} "
      f"({100 * one_to_many_l / total_misaligned_l:.1f}%)" if total_misaligned_l else "N/A")
total_mis = total_misaligned_h + total_misaligned_l
total_otm = one_to_many_h + one_to_many_l
print(f"One-to-many (combined):    {total_otm} / {total_mis} "
      f"({100 * total_otm / total_mis:.1f}%)" if total_mis else "N/A")
print(f"Arbitrary misalignment:    {total_mis - total_otm} / {total_mis} "
      f"({100 * (total_mis - total_otm) / total_mis:.1f}%)" if total_mis else "N/A")
