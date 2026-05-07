"""
Classify assumption nodes in v1_llm_gemini-3.1-pro-preview data into two types:

  Type 1 – switch_case:
    The assumption is one of multiple alternatives being explored (switch/case,
    proof-by-contradiction, or exhaustive enumeration).  The model intends to
    find which possibility is correct.

  Type 2 – underspecified:
    The assumption fills in an underspecified gap in the problem so that solving
    can proceed (WLOG coordinate placement, simplifying physical conditions,
    ignoring higher-order terms, etc.).

Edge-structure criteria
-----------------------
Type 1 signals (any one is sufficient):

  S1. plan:backtrack outgoing edge
        The assumption was tried and explicitly abandoned.

  S2. validate:attack outgoing edge
        A reasoning node immediately tests the assumption for contradiction
        (proof-by-contradiction pattern).

  S3. Predecessor is another assumption via plan:proceed
        AND that predecessor was NOT itself seeded by a plan:decompose from a
        non-assumption node.
        → The current node is the *next* alternative in a chain.
        (Exception: WLOG-sequential chains like coordinate placement, where
         the root assumption was introduced by plan:decompose from planning.)

  S4. Successor is an assumption via plan:proceed AND no reason:infer /
      reason:execute outgoing
        → Pure alternative-chain starter: the node only hands off to the next
        case without being used as a base fact itself.

  S5. Successor is an assumption via plan:proceed AND has reason:infer /
      reason:execute outgoing AND no plan:decompose incoming from a
      non-assumption
        → Trying an alternative interpretation while immediately inferring from
        it, but not a WLOG setup (which would have plan:decompose from
        planning).

  S6. The parent node uses plan:decompose to reach MULTIPLE assumption nodes
        → The planning node explicitly enumerates parallel cases.

Type 2: everything that does not satisfy any Type 1 signal.
  Typical patterns observed:
  – plan:decompose from a single planning node (WLOG / "let's set up coords")
  – Outgoing edges are only reason:infer / reason:execute (used as a base fact)
  – Assumption isolated in a local sub-graph with no alternative structure
"""

import json
import os
import glob
from collections import defaultdict, Counter
from random import shuffle


DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "data",
    "v1_llm_gemini-3.1-pro-preview",
)


def load_graph(fpath):
    with open(fpath) as f:
        d = json.load(f)
    nodes_by_id = {n["id"]: n for n in d["nodes"]}
    in_edges = defaultdict(list)   # node_id -> [edge, ...]
    out_edges = defaultdict(list)
    for e in d["edges"]:
        out_edges[e["source_node_id"]].append(e)
        in_edges[e["dest_node_id"]].append(e)
    # For each non-assumption node: how many assumption nodes it plan:decomposes to
    multi_decompose_parents = set()
    for node in d["nodes"]:
        assump_decompose_targets = [
            e for e in out_edges[node["id"]]
            if e["label"] == "plan:decompose"
            and nodes_by_id.get(e["dest_node_id"], {}).get("label") == "assumption"
        ]
        if len(assump_decompose_targets) > 1:
            multi_decompose_parents.add(node["id"])
    return d, nodes_by_id, in_edges, out_edges, multi_decompose_parents


def classify_assumption(node_id, nodes_by_id, in_edges, out_edges, multi_decompose_parents):
    """
    Returns ('switch_case', reason_str) or ('underspecified', reason_str).
    """
    ie = in_edges[node_id]
    oe = out_edges[node_id]

    out_labels = {e["label"] for e in oe}

    # ── S1: assumption was tried and explicitly failed ──────────────────────
    if "plan:backtrack" in out_labels:
        return "switch_case", "S1:backtrack_out"

    # ── S2: assumption is being tested for contradiction ────────────────────
    if "validate:attack" in out_labels:
        return "switch_case", "S2:attack_out"

    # ── S6: parent plan:decomposes to multiple assumptions ──────────────────
    for e in ie:
        if e["label"] == "plan:decompose" and e["source_node_id"] in multi_decompose_parents:
            return "switch_case", "S6:multi_decompose_parent"

    # ── S3: predecessor is another assumption (alternative chain) ───────────
    pred_assumption_nodes = [
        nodes_by_id[e["source_node_id"]]
        for e in ie
        if e["label"] == "plan:proceed"
        and nodes_by_id.get(e["source_node_id"], {}).get("label") == "assumption"
    ]
    if pred_assumption_nodes:
        # Check if the predecessor was itself seeded by plan:decompose from a
        # non-assumption node (WLOG-sequential chain root → exempt from S3).
        pred_is_wlog_root = any(
            any(
                pe["label"] == "plan:decompose"
                and nodes_by_id.get(pe["source_node_id"], {}).get("label") != "assumption"
                for pe in in_edges[pn["id"]]
            )
            for pn in pred_assumption_nodes
        )
        if not pred_is_wlog_root:
            return "switch_case", "S3:pred_is_assumption"

    # ── S4 / S5: successor is another assumption ────────────────────────────
    succ_is_assumption = any(
        e["label"] == "plan:proceed"
        and nodes_by_id.get(e["dest_node_id"], {}).get("label") == "assumption"
        for e in oe
    )
    if succ_is_assumption:
        has_infer_out = any(e["label"] in {"reason:infer", "reason:execute"} for e in oe)
        has_decompose_in = any(
            e["label"] == "plan:decompose"
            and nodes_by_id.get(e["source_node_id"], {}).get("label") != "assumption"
            for e in ie
        )
        if not has_infer_out:
            # S4: pure alternative chain starter
            return "switch_case", "S4:succ_assumption_no_infer"
        if not has_decompose_in:
            # S5: trying alternative while immediately inferring, no WLOG setup
            return "switch_case", "S5:succ_assumption_with_infer_no_decompose"
        # else: has_infer_out AND has_decompose_in → ambiguous; treat as Type 2

    return "underspecified", "no_type1_signal"


def analyse_file(fpath):
    d, nodes_by_id, in_edges, out_edges, multi_decompose_parents = load_graph(fpath)
    results = []
    for node in d["nodes"]:
        if node["label"] != "assumption":
            continue
        cls, reason = classify_assumption(
            node["id"], nodes_by_id, in_edges, out_edges, multi_decompose_parents
        )
        results.append(
            dict(
                file=os.path.basename(fpath),
                node_id=node["id"],
                cls=cls,
                reason=reason,
                text=node["text"],
                in_edges=[
                    (e["label"], nodes_by_id.get(e["source_node_id"], {}).get("label", "?"))
                    for e in in_edges[node["id"]]
                ],
                out_edges=[
                    (e["label"], nodes_by_id.get(e["dest_node_id"], {}).get("label", "?"))
                    for e in out_edges[node["id"]]
                ],
            )
        )
    return results


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    all_results = []
    for fpath in files:
        all_results.extend(analyse_file(fpath))

    # ── Global statistics ────────────────────────────────────────────────────
    total = len(all_results)
    by_cls = Counter(r["cls"] for r in all_results)
    by_reason = Counter(r["reason"] for r in all_results)

    print(f"{'='*70}")
    print(f"Total assumption nodes : {total}")
    print(f"  switch_case          : {by_cls['switch_case']} ({100*by_cls['switch_case']/total:.1f}%)")
    print(f"  underspecified       : {by_cls['underspecified']} ({100*by_cls['underspecified']/total:.1f}%)")
    print()
    print("Triggered signals (Type 1 only):")
    for sig in ["S1:backtrack_out", "S2:attack_out", "S6:multi_decompose_parent",
                "S3:pred_is_assumption", "S4:succ_assumption_no_infer",
                "S5:succ_assumption_with_infer_no_decompose"]:
        c = by_reason[sig]
        print(f"  {sig:<42} {c:5d}  ({100*c/total:.1f}%)")
    print()

    # ── Example switch_case nodes ────────────────────────────────────────────
    print(f"{'='*70}")
    print("SWITCH_CASE examples (one per signal type)")
    printed = set()
    shuffle(all_results)
    for target_reason in [
        "S1:backtrack_out",
        "S2:attack_out",
        "S3:pred_is_assumption",
        "S4:succ_assumption_no_infer",
        "S5:succ_assumption_with_infer_no_decompose",
        "S6:multi_decompose_parent",
    ]:
        for r in all_results:
            if r["reason"] == target_reason and target_reason not in printed:
                printed.add(target_reason)
                _print_node(r, target_reason)
                break

    # ── Example underspecified nodes ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("UNDERSPECIFIED examples")
    shown = 0
    # Pick varied structural patterns
    for r in all_results:
        if r["cls"] != "underspecified":
            continue
        if "gpqa" in r["file"]:
            continue
        in_l = {el for el, _ in r["in_edges"]}
        # Vary by incoming edge type for diversity
        key = frozenset(in_l)
        if key not in {
            frozenset({"plan:decompose"}),
            frozenset({"reason:infer"}),
            frozenset({"plan:proceed"}),
            frozenset({"reason:execute"}),
        }:
            continue
        _print_node(r, "underspecified")
        shown += 1
        if shown >= 15:
            break


def _print_node(r, label):
    print(f"\n  [{label}] {r['file']} / {r['node_id']}")
    print(f"  Text: {r['text'].strip()[:200]}")
    print(f"  In : {r['in_edges']}")
    print(f"  Out: {r['out_edges']}")


if __name__ == "__main__":
    main()
