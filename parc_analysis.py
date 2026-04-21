"""
Analyze PARC data for reasoning error criticality.

Rebuts the belief that all reasoning errors are critical to overall correctness,
especially in reasoning models (QwQ, R1, gpt-oss).

Analyses (all over AIME+GPQA, split by domain and model):

1. Out of erroneous nodes: does it connect to a correct/incorrect conclusion
   via reason: edges, or no conclusion at all?

2. Out of erroneous nodes: is the node reachable from an assumption/example
   node via reason: edges?

3. Out of all PARC-evaluated nodes: correlate outgoing reflect:* edge type
   with node correctness.

4. Out of all PARC-evaluated nodes: correlate outgoing validate:* edge
   types with node correctness.
"""

import json
import glob
import os
import re
import sys
from collections import defaultdict, deque, Counter

DATA_DIR = "data/v1_llm_gemini-3.1-pro-preview"
PARC_DIR = "parc_results"

REASONING_MODELS = {"QwQ-32B", "DeepSeek-R1", "gpt-oss-120b"}
PARC_TARGET_LABELS = {"fact", "reasoning", "restatement", "conclusion"}
MODEL_ORDER = ["QwQ-32B", "DeepSeek-R1", "gpt-oss-120b", "DeepSeek-V3", "Qwen2.5-32B-Instruct"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_files():
    """Load paired graph + PARC result files from DATA_DIR and PARC_DIR."""
    records = []
    for parc_path in sorted(glob.glob(os.path.join(PARC_DIR, "*.json"))):
        basename = os.path.basename(parc_path)
        graph_path = os.path.join(DATA_DIR, basename)
        if not os.path.exists(graph_path):
            continue
        with open(graph_path) as f:
            graph = json.load(f)
        with open(parc_path) as f:
            parc = json.load(f)
        meta = graph["metadata"]
        domain_raw = meta.get("domain", "")
        if "aime" in domain_raw.lower():
            domain = "AIME"
        elif "gpqa" in domain_raw.lower():
            domain = "GPQA"
        else:
            domain = domain_raw.upper()
        records.append({
            "doc_id": graph["doc_id"],
            "domain": domain,
            "model": meta.get("generator", "unknown"),
            "correct_answer": str(meta.get("correct_answer", "") or ""),
            "graph": graph,
            "parc": parc,
        })
    return records


# ── Graph utilities ───────────────────────────────────────────────────────────

def build_reason_fwd_adj(edges):
    """Forward adjacency using only reason: edges."""
    adj = defaultdict(set)
    for e in edges:
        if e.get("label", "").startswith("reason:"):
            adj[e["source_node_id"]].add(e["dest_node_id"])
    return adj


def bfs_reachable(start_ids, adj):
    """Return all node IDs reachable from start_ids via adj (including starts)."""
    visited = set(start_ids)
    queue = deque(start_ids)
    while queue:
        node = queue.popleft()
        for nbr in adj.get(node, set()):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return visited


def conclusion_contains_answer(text, answer):
    """True if text contains answer as a whole word (case-insensitive)."""
    if not answer:
        return False
    return bool(re.search(r'\b' + re.escape(answer) + r'\b', text, re.IGNORECASE))


# ── Core analyses ─────────────────────────────────────────────────────────────

def run_analyses(records, debug_connects_correct=None):
    """
    Returns Counter dicts keyed by (domain, model):

    ana0: for each conclusion node, is it reachable from any erroneous node
      via reason: edges, and is the conclusion itself correct?
      bins: reachable_correct | reachable_incorrect | clean_correct | clean_incorrect

    ana1: erroneous node connectivity to correct/incorrect conclusion
      bins: connects_correct | connects_incorrect | no_connect

    ana1_1: among no_connect erroneous nodes, are they reachable from
      an assumption/example node via reason: edges?
      bins: reachable | not_reachable

    ana2: erroneous node reachability from assumption/example nodes
      bins: reachable | not_reachable

    ana3: all PARC nodes — reflect edge type × node correctness
      bins: (reflect_type, 'correct'|'error')

    ana4: all PARC nodes — validate type × node correctness
      bins: (validate_type, 'corr'|'prop'|'error')
        validate_type: 'support' | 'attack' | 'none'
        prop: labeled 'correct' but reachable from an erroneous node via reason: edges
        (nodes with both support+attack are counted in both support and attack bins)
    """
    ana0 = defaultdict(Counter)
    ana1 = defaultdict(Counter)
    ana1_1 = defaultdict(Counter)
    ana1_2 = defaultdict(Counter)
    ana2 = defaultdict(Counter)
    ana3 = defaultdict(Counter)
    ana4 = defaultdict(Counter)

    for rec in records:
        key = (rec["domain"], rec["model"])
        graph = rec["graph"]
        parc_list = rec["parc"]
        answer = rec["correct_answer"]

        nodes = graph["nodes"]
        edges = graph["edges"]
        node_by_id = {n["id"]: n for n in nodes}

        parc_map = {p["node_id"]: p["correctness"] for p in parc_list}

        erroneous_ids = {
            nid for nid, corr in parc_map.items()
            if corr == "error"
            and node_by_id.get(nid, {}).get("label") in PARC_TARGET_LABELS
        }

        conc_nodes = [n for n in nodes if n.get("label") == "conclusion"]
        conc_ids = {n["id"] for n in conc_nodes}
        conc_correct_ids = {
            n["id"] for n in conc_nodes
            if conclusion_contains_answer(n.get("text", ""), answer)
        }
        reason_adj = build_reason_fwd_adj(edges)

        # Analysis 0 ──────────────────────────────────────────────────────────
        concs_reachable_from_err = set()
        for eid in erroneous_ids:
            concs_reachable_from_err |= bfs_reachable([eid], reason_adj) & conc_ids

        for cn in conc_nodes:
            cid = cn["id"]
            is_correct = cid in conc_correct_ids
            if cid in concs_reachable_from_err:
                ana0[key]["reachable_correct" if is_correct else "reachable_incorrect"] += 1
            else:
                ana0[key]["clean_correct" if is_correct else "clean_incorrect"] += 1

        # Pre-compute assumption/example reachability (used in ana1_1 and ana2)
        a_ids = [n["id"] for n in nodes if n.get("label") == "assumption"]
        e_ids = [n["id"] for n in nodes if n.get("label") == "example"]
        ae_ids = a_ids + e_ids
        ae_id_set = set(ae_ids)
        reachable_from_a = bfs_reachable(a_ids, reason_adj)
        reachable_from_e = bfs_reachable(e_ids, reason_adj)
        reachable_from_ae = reachable_from_a | reachable_from_e

        # Build forward adj for ALL edges to check outgoing labels per node
        all_outgoing = defaultdict(set)
        for e in edges:
            all_outgoing[e["source_node_id"]].add(e.get("label", ""))

        # Analysis 1, 1-1 & 1-2 ───────────────────────────────────────────────
        for eid in erroneous_ids:
            reachable = bfs_reachable([eid], reason_adj)
            reached_concs = reachable & conc_ids
            if reached_concs:
                if reached_concs - conc_correct_ids:  # any incorrect conclusion reachable
                    ana1[key]["connects_incorrect"] += 1
                else:
                    ana1[key]["connects_correct"] += 1
                    if debug_connects_correct and rec["model"] == debug_connects_correct:
                        conc_texts = {
                            n["id"]: n.get("text", "")[:80]
                            for n in conc_nodes if n["id"] in reached_concs
                        }
                        err_text = node_by_id.get(eid, {}).get("text", "")[:80]
                        print(f"  [DBG connects_correct] doc={rec['doc_id']}  answer={answer}")
                        print(f"    err_node={eid}: {err_text!r}")
                        for cid, ctxt in conc_texts.items():
                            print(f"    conc={cid}: {ctxt!r}")
                        print()
            else:
                ana1[key]["no_connect"] += 1
                from_ae = eid in reachable_from_ae and eid not in ae_id_set
                ana1_1[key]["reachable" if from_ae else "not_reachable"] += 1
                # Check if any node reachable from eid via reason: edges carries
                # a self-correction signal (signals may overlap)
                connected = bfs_reachable([eid], reason_adj)
                any_va = any("validate:attack" in all_outgoing[n] for n in connected)
                any_rn = any("reflect:negative" in all_outgoing[n] for n in connected)
                any_pv = any("plan:verify" in all_outgoing[n] for n in connected)
                ana1_2[key]["total"] += 1
                if any_va:
                    ana1_2[key]["val_attack"] += 1
                if any_rn:
                    ana1_2[key]["ref_neg"] += 1
                if any_pv:
                    ana1_2[key]["plan_verify"] += 1
                if any_va or any_rn or any_pv:
                    ana1_2[key]["any_signal"] += 1

        # Analysis 2 ─────────────────────────────────────────────────────────

        for eid in erroneous_ids:
            if eid in ae_id_set:
                ana2[key]["not_reachable"] += 1
                continue
            from_a = eid in reachable_from_a
            from_e = eid in reachable_from_e
            if from_a:
                ana2[key]["reachable_assumption"] += 1
            if from_e:
                ana2[key]["reachable_example"] += 1
            if from_a or from_e:
                ana2[key]["reachable"] += 1
            else:
                ana2[key]["not_reachable"] += 1

        # Analysis 3 & 4 — per-node outgoing edge inventory ──────────────────
        outgoing = all_outgoing  # already built above

        # Nodes reachable from any erroneous node via reason: edges (for ana4 prop)
        reachable_from_err = bfs_reachable(list(erroneous_ids), reason_adj)

        for nid, corr in parc_map.items():
            if node_by_id.get(nid, {}).get("label") not in PARC_TARGET_LABELS:
                continue
            if corr == "undecidable":
                continue
            corr_str = "correct" if corr == "correct" else "error"
            node_out = outgoing[nid]

            # Analysis 3 — split correct into corr (clean) vs prop (reachable from error)
            if corr == "correct":
                corr3_str = "prop" if (nid in reachable_from_err and nid not in erroneous_ids) else "corr"
            else:
                corr3_str = "error"

            for rt in ("reflect:positive", "reflect:uncertain", "reflect:negative"):
                if rt in node_out:
                    ana3[key][(rt.split(":")[1], corr3_str)] += 1

            # Analysis 4 — split correct into corr (clean) vs prop (reachable from error)
            if corr == "correct":
                corr4_str = "prop" if (nid in reachable_from_err and nid not in erroneous_ids) else "corr"
            else:
                corr4_str = "error"

            has_vs = "validate:support" in node_out
            has_va = "validate:attack" in node_out

            if has_vs:
                ana4[key][("support", corr4_str)] += 1
            if has_va:
                ana4[key][("attack", corr4_str)] += 1
            if not has_vs and not has_va:
                ana4[key][("none", corr4_str)] += 1

    return ana0, ana1, ana1_1, ana1_2, ana2, ana3, ana4


# ── Printing utilities ────────────────────────────────────────────────────────

def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "  N/A"


def aggregate(counters_by_key, domain=None, model=None):
    """Sum counters matching the given domain/model filters."""
    total = Counter()
    for (d, m), c in counters_by_key.items():
        if domain and d != domain:
            continue
        if model and m != model:
            continue
        total += c
    return total


def print_analysis_0(ana0, domains, models):
    print("=" * 72)
    print("ANALYSIS 0  —  Out of all conclusion nodes")
    print("  Is the conclusion reachable from any erroneous node via reason: edges?")
    print("  Is the conclusion correct (contains ground-truth answer)?")
    print()

    cols = ["reachable_correct", "reachable_incorrect", "clean_correct", "clean_incorrect"]
    col_labels = ["reach ✓", "reach ✗", "clean ✓", "clean ✗"]

    header = f"  {'Domain':<8} {'Model':<26} {'Total':>6}  " + "  ".join(f"{l:>10}" for l in col_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    def print_row(domain_label, model_label, c):
        total = sum(c[k] for k in cols)
        if total == 0:
            return
        vals = "  ".join(f"{c[k]:>5} {pct(c[k], total):>5}" for k in cols)
        print(f"  {domain_label:<8} {model_label:<26} {total:>6}  {vals}")

    for domain in domains:
        for model in sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
            c = aggregate(ana0, domain=domain, model=model)
            if sum(c.values()) == 0:
                continue
            is_rm = "*" if model in REASONING_MODELS else " "
            print_row(domain, f"{is_rm}{model}", c)
        c = aggregate(ana0, domain=domain)
        if sum(c.values()) > 0:
            print_row(domain, "(all models)", c)
        print()

    c = aggregate(ana0)
    print_row("ALL", "(all models)", c)
    print()
    print("  * = reasoning model (QwQ-32B, DeepSeek-R1, gpt-oss-120b)")
    print()


def print_analysis_1(ana1, domains, models):
    print("=" * 72)
    print("ANALYSIS 1  —  Out of all erroneous nodes")
    print("  Does the erroneous node connect (via reason: edges) to a")
    print("  correct/incorrect conclusion, or no conclusion at all?")
    print()

    cols = ["connects_correct", "connects_incorrect", "no_connect"]
    col_labels = ["→conc ✓", "→conc ✗", "⇏conc", "✓/(✓+✗)"]

    header = f"  {'Domain':<8} {'Model':<26} {'Total':>6}  " + "  ".join(f"{l:>10}" for l in col_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    def print_row(domain_label, model_label, c):
        total = sum(c[k] for k in cols[:3])
        if total == 0:
            return
        vals = "  ".join(f"{c[k]:>5} {pct(c[k], total):>5}" for k in cols[:3])
        reaches_conc = c["connects_correct"] + c["connects_incorrect"]
        ratio = pct(c["connects_correct"], reaches_conc)
        print(f"  {domain_label:<8} {model_label:<26} {total:>6}  {vals}  {'':>5} {ratio:>5}")

    for domain in domains:
        for model in sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
            c = aggregate(ana1, domain=domain, model=model)
            if sum(c.values()) == 0:
                continue
            is_rm = "*" if model in REASONING_MODELS else " "
            print_row(domain, f"{is_rm}{model}", c)
        # Domain subtotal
        c = aggregate(ana1, domain=domain)
        if sum(c.values()) > 0:
            print_row(domain, "(all models)", c)
        print()

    # Grand total
    c = aggregate(ana1)
    print_row("ALL", "(all models)", c)
    print()
    print("  * = reasoning model (QwQ-32B, DeepSeek-R1, gpt-oss-120b)")
    print()


def print_analysis_1_1(ana1_1, domains, models):
    print("=" * 72)
    print("ANALYSIS 1-1  —  Out of ⇏conc erroneous nodes")
    print("  Are they reachable from an assumption/example node via reason: edges?")
    print()

    header = f"  {'Domain':<8} {'Model':<26} {'Total':>6}  {'reachable':>12}  {'not_reachable':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def print_row(domain_label, model_label, c):
        total = c["reachable"] + c["not_reachable"]
        if total == 0:
            return
        print(f"  {domain_label:<8} {model_label:<26} {total:>6}"
              f"  {c['reachable']:>5} {pct(c['reachable'], total):>5}"
              f"  {c['not_reachable']:>5} {pct(c['not_reachable'], total):>7}")

    for domain in domains:
        for model in sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
            c = aggregate(ana1_1, domain=domain, model=model)
            if sum(c.values()) == 0:
                continue
            is_rm = "*" if model in REASONING_MODELS else " "
            print_row(domain, f"{is_rm}{model}", c)
        c = aggregate(ana1_1, domain=domain)
        if sum(c.values()) > 0:
            print_row(domain, "(all models)", c)
        print()

    c = aggregate(ana1_1)
    print_row("ALL", "(all models)", c)
    print()
    print("  * = reasoning model")
    print()


def print_analysis_1_2(ana1_2, domains, models):
    print("=" * 72)
    print("ANALYSIS 1-2  —  Out of ⇏conc erroneous nodes")
    print("  Among nodes reachable via reason: edges, does any carry a")
    print("  self-correction signal? (counts overlap)")
    print()

    signals = ["val_attack", "ref_neg", "plan_verify"]
    sig_labels = ["val:attack", "ref:neg", "plan:verify"]

    header = (f"  {'Domain':<8} {'Model':<26} {'Total':>6}  "
              + "  ".join(f"{l:>14}" for l in sig_labels)
              + f"  {'any signal':>14}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    def print_row(domain_label, model_label, c):
        total = c["total"]
        if total == 0:
            return
        vals = "  ".join(f"{c[s]:>5} {pct(c[s], total):>7}" for s in signals)
        any_s = c["any_signal"]
        print(f"  {domain_label:<8} {model_label:<26} {total:>6}  {vals}  {any_s:>5} {pct(any_s, total):>7}")

    for domain in domains:
        for model in sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
            c = aggregate(ana1_2, domain=domain, model=model)
            if c["total"] == 0:
                continue
            is_rm = "*" if model in REASONING_MODELS else " "
            print_row(domain, f"{is_rm}{model}", c)
        c = aggregate(ana1_2, domain=domain)
        if c["total"] > 0:
            print_row(domain, "(all models)", c)
        print()

    c = aggregate(ana1_2)
    print_row("ALL", "(all models)", c)
    print()
    print("  * = reasoning model (QwQ-32B, DeepSeek-R1, gpt-oss-120b)")
    print()


def print_analysis_2(ana2, domains, models):
    print("=" * 72)
    print("ANALYSIS 2  —  Out of all erroneous nodes")
    print("  Is the node reachable from an assumption/example node")
    print("  via reason: edges?")
    print("  (from_assump + from_example may sum > reachable due to overlap)")
    print()

    header = (f"  {'Domain':<8} {'Model':<26} {'Total':>6}"
              f"  {'reachable':>12}  {'from_assump':>12}  {'from_example':>13}  {'not_reachable':>14}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    def print_row(domain_label, model_label, c):
        total = c["reachable"] + c["not_reachable"]
        if total == 0:
            return
        print(f"  {domain_label:<8} {model_label:<26} {total:>6}"
              f"  {c['reachable']:>5} {pct(c['reachable'], total):>5}"
              f"  {c['reachable_assumption']:>5} {pct(c['reachable_assumption'], total):>5}"
              f"  {c['reachable_example']:>6} {pct(c['reachable_example'], total):>5}"
              f"  {c['not_reachable']:>5} {pct(c['not_reachable'], total):>7}")

    for domain in domains:
        for model in sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
            c = aggregate(ana2, domain=domain, model=model)
            if sum(c.values()) == 0:
                continue
            is_rm = "*" if model in REASONING_MODELS else " "
            print_row(domain, f"{is_rm}{model}", c)
        c = aggregate(ana2, domain=domain)
        if sum(c.values()) > 0:
            print_row(domain, "(all models)", c)
        print()

    c = aggregate(ana2)
    print_row("ALL", "(all models)", c)
    print()
    print("  * = reasoning model")
    print()


def print_analysis_3(ana3, domains, models):
    print("=" * 72)
    print("ANALYSIS 3  —  Out of all PARC-evaluated nodes")
    print("  Outgoing reflect:* edge type vs. node correctness")
    print("  (nodes with no reflect edges are excluded)")
    print()

    reflect_types = ["positive", "uncertain", "negative"]
    cs_vals = ["corr", "prop", "error"]
    cell_w = 20

    def cell(c, rt):
        n_c = c[(rt, "corr")]
        n_p = c[(rt, "prop")]
        n_e = c[(rt, "error")]
        n_t = n_c + n_p + n_e
        return f"{n_c}/{n_p}/{n_e}({pct(n_c, n_t)})"

    def print_row(domain_label, model_label, c):
        total = sum(c[(rt, cs)] for rt in reflect_types for cs in cs_vals)
        if total == 0:
            return
        cells = [cell(c, rt) for rt in reflect_types]
        print(f"  {domain_label:<8} {model_label:<26} {total:>6}  " + "  ".join(f"{v:>{cell_w}}" for v in cells))

    prefix = f"  {'':8} {'':26} {'':6}  "
    header = prefix + "  ".join(f"{'ref:'+rt:>{cell_w}}" for rt in reflect_types)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for domain in domains:
        for model in sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
            c = aggregate(ana3, domain=domain, model=model)
            if sum(c.values()) == 0:
                continue
            is_rm = "*" if model in REASONING_MODELS else " "
            print_row(domain, f"{is_rm}{model}", c)
        c = aggregate(ana3, domain=domain)
        if sum(c.values()) > 0:
            print_row(domain, "(all models)", c)
        print()

    c = aggregate(ana3)
    print_row("ALL", "(all models)", c)
    print()
    print("  Format: corr/prop/err(accuracy%)  prop=correct-labeled but reachable from error node")
    print("  * = reasoning model")
    print()


def print_analysis_4(ana4, domains, models):
    print("=" * 72)
    print("ANALYSIS 4  —  Out of all PARC-evaluated nodes")
    print("  validate:* edge type vs. node correctness")
    print()

    vt_vals = ["support", "attack", "none"]
    cs_vals = ["corr", "prop", "error"]
    cell_w = 20

    def cell(c, vt):
        n_c = c[(vt, "corr")]
        n_p = c[(vt, "prop")]
        n_e = c[(vt, "error")]
        n_t = n_c + n_p + n_e
        return f"{n_c}/{n_p}/{n_e}({pct(n_c, n_t)})"

    def print_row(domain_label, model_label, c):
        if sum(c.values()) == 0:
            return
        cells = [cell(c, vt) for vt in vt_vals]
        print(f"  {domain_label:<8} {model_label:<26}  " + "  ".join(f"{v:>{cell_w}}" for v in cells))

    prefix = f"  {'':8} {'':26}  "
    header = prefix + "  ".join(f"{'val:'+vt:>{cell_w}}" for vt in vt_vals)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for domain in domains:
        for model in sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99):
            c = aggregate(ana4, domain=domain, model=model)
            if sum(c.values()) == 0:
                continue
            is_rm = "*" if model in REASONING_MODELS else " "
            print_row(domain, f"{is_rm}{model}", c)
        c = aggregate(ana4, domain=domain)
        if sum(c.values()) > 0:
            print_row(domain, "(all models)", c)
        print()

    c = aggregate(ana4)
    print_row("ALL", "(all models)", c)
    print()
    print("  Format: corr/prop/err(accuracy%)  prop=correct-labeled but reachable from error node")
    print("  * = reasoning model")
    print()


# ── Analysis 1 example sampler ───────────────────────────────────────────────

def _node_idx(node_id):
    """Extract trailing integer from a node ID (e.g. 'resp35' → 35)."""
    m = re.search(r'(\d+)$', node_id)
    return int(m.group(1)) if m else None


def sample_analysis_1_examples(records, n=5, seed=42):
    """Print n randomly sampled examples for Analysis 1 →conc ✓ and ⇏conc bins.

    For →conc ✓: prints all nodes by index between the erroneous node and the
    nearest correct conclusion (inclusive), parc correctness annotated.
    For ⇏conc: prints the erroneous node alone (span = 1).
    Sampling priority: examples with total nodes to print < 10 are preferred.
    """
    import random
    rng = random.Random(seed)

    connects_correct_examples = []
    no_connect_examples = []

    for rec in records:
        graph = rec["graph"]
        parc_list = rec["parc"]
        answer = rec["correct_answer"]
        nodes = graph["nodes"]
        edges = graph["edges"]
        node_by_id = {n["id"]: n for n in nodes}
        # index → node for fast range lookup
        nodes_by_idx = {}
        for nd in nodes:
            idx = _node_idx(nd["id"])
            if idx is not None:
                nodes_by_idx[idx] = nd
        parc_map = {p["node_id"]: p["correctness"] for p in parc_list}

        erroneous_ids = {
            nid for nid, corr in parc_map.items()
            if corr == "error"
            and node_by_id.get(nid, {}).get("label") in PARC_TARGET_LABELS
        }

        conc_nodes = [n for n in nodes if n.get("label") == "conclusion"]
        conc_ids = {n["id"] for n in conc_nodes}
        conc_correct_ids = {
            n["id"] for n in conc_nodes
            if conclusion_contains_answer(n.get("text", ""), answer)
        }
        reason_adj = build_reason_fwd_adj(edges)

        for eid in erroneous_ids:
            reachable = bfs_reachable([eid], reason_adj)
            reached_concs = reachable & conc_ids
            err_node = node_by_id.get(eid, {})
            err_idx = _node_idx(eid)

            if reached_concs and not (reached_concs - conc_correct_ids):
                # Pick the nearest correct conclusion by index distance
                best_conc_id = None
                best_span = None
                for cid in reached_concs:
                    cidx = _node_idx(cid)
                    if err_idx is not None and cidx is not None:
                        span = abs(cidx - err_idx) + 1
                        if best_span is None or span < best_span:
                            best_span = span
                            best_conc_id = cid
                if best_conc_id is None:
                    best_span = 1

                # Collect index-range nodes (inclusive)
                if err_idx is not None and best_conc_id is not None:
                    conc_idx = _node_idx(best_conc_id)
                    lo, hi = min(err_idx, conc_idx), max(err_idx, conc_idx)
                    path_nodes = [
                        nodes_by_idx[i] for i in range(lo, hi + 1)
                        if i in nodes_by_idx
                    ]
                else:
                    path_nodes = [err_node]

                connects_correct_examples.append({
                    "doc_id": rec["doc_id"],
                    "model": rec["model"],
                    "domain": rec["domain"],
                    "answer": answer,
                    "err_node_id": eid,
                    "conc_node_id": best_conc_id,
                    "path_nodes": path_nodes,
                    "parc_map": parc_map,
                    "span": best_span if best_span is not None else 1,
                })

            elif not reached_concs:
                if err_idx is not None:
                    path_nodes = [
                        nodes_by_idx[i] for i in range(err_idx, err_idx + 10)
                        if i in nodes_by_idx
                    ]
                else:
                    path_nodes = [err_node]
                no_connect_examples.append({
                    "doc_id": rec["doc_id"],
                    "model": rec["model"],
                    "domain": rec["domain"],
                    "answer": answer,
                    "err_node_id": eid,
                    "conc_node_id": None,
                    "path_nodes": path_nodes,
                    "parc_map": parc_map,
                    "span": len(path_nodes),
                })

    def priority_sample(examples, sample_n):
        """Sample with priority: 5<=span<10, then span<5, then span>=10."""
        mid   = [e for e in examples if 5 <= e["span"] < 10]
        short = [e for e in examples if e["span"] < 5]
        long_ = [e for e in examples if e["span"] >= 10]
        for tier in (mid, short, long_):
            rng.shuffle(tier)
        pool = mid + short + long_
        return pool[:sample_n]

    def print_node(nd, parc_map, err_id, conc_id):
        nid = nd["id"]
        label = nd.get("label", "?")
        corr = parc_map.get(nid, "-")
        marker = ""
        if nid == err_id:
            marker = "  ← ERROR"
        elif nid == conc_id:
            marker = "  ← CONCLUSION"
        print(f"      [{nid}] ({label}, parc={corr}){marker}")
        for line in nd.get("text", "").split("\n"):
            print(f"        {line}")

    def print_examples(label, examples, sample_n):
        print(f"{'=' * 72}")
        print(f"ANALYSIS 1 EXAMPLES  —  {label}  (n={min(sample_n, len(examples))} of {len(examples)})")
        print()
        sampled = priority_sample(examples, sample_n)
        for i, ex in enumerate(sampled, 1):
            conc_id = ex.get("conc_node_id")
            print(f"  [{i}] doc={ex['doc_id']}  model={ex['model']}  domain={ex['domain']}"
                  f"  answer={ex['answer']!r}  span={ex['span']}")
            for nd in ex["path_nodes"]:
                print_node(nd, ex["parc_map"], ex["err_node_id"], conc_id)
            print()

    aime_cc = [e for e in connects_correct_examples if e["domain"] == "AIME"]
    aime_nc = [e for e in no_connect_examples if e["domain"] == "AIME"]
    print_examples("→conc ✓  (erroneous node reaches only correct conclusions) [AIME]", aime_cc, n)
    print_examples("⇏conc    (erroneous node reaches no conclusion) [AIME]", aime_nc, n)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading files from '{PARC_DIR}/' matched against '{DATA_DIR}/'...")
    records = load_all_files()
    print(f"Loaded {len(records)} file(s)\n")
    if not records:
        print("No paired PARC + graph files found. Exiting.")
        sys.exit(1)

    domains = sorted({r["domain"] for r in records})
    models = sorted({r["model"] for r in records})
    print(f"Domains: {domains}")
    print(f"Models:  {models}")
    print(f"Files per (domain, model):")
    cnt = Counter((r["domain"], r["model"]) for r in records)
    for (d, m), n in sorted(cnt.items()):
        print(f"  {d:<8} {m:<30} {n:>3} files")
    print()

    ana0, ana1, ana1_1, ana1_2, ana2, ana3, ana4 = run_analyses(records)

    print_analysis_0(ana0, domains, models)
    print_analysis_1(ana1, domains, models)
    print_analysis_1_1(ana1_1, domains, models)
    print_analysis_1_2(ana1_2, domains, models)
    print_analysis_2(ana2, domains, models)
    print_analysis_3(ana3, domains, models)
    print_analysis_4(ana4, domains, models)
    sample_analysis_1_examples(records, n=5)


if __name__ == "__main__":
    main()
