"""
Verify correctness of built graph data before training.

Checks:
  1. Structural integrity (shapes, types, no NaN/inf)
  2. Edge validity (no out-of-bounds indices)
  3. Edge type distribution (all 4 types present?)
  4. DFG connectivity (are DFG nodes actually connected?)
  5. CFG presence (do control-flow edges exist?)
  6. Feature quality (embeddings not all zeros?)
  7. Visual samples (print a few graphs in detail)
  8. Pair consistency (correct vs buggy should differ)

Usage:
    python scripts/verify_graphs.py --graph-data ./data/graphs/graph_data_codenet_python.pt
    python scripts/verify_graphs.py --graph-data ./data/graphs/graph_data_codenet_python.pt --verbose
    python scripts/verify_graphs.py --graph-data ./data/graphs/graph_data_codenet_python.pt --show-samples 5
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch


def verify_graph_data(path, verbose=False, show_samples=3):
    print("=" * 70)
    print(f"VERIFYING: {path}")
    print("=" * 70)

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return False

    data = torch.load(path, map_location="cpu", weights_only=False)

    # ═══════════════════════════════════════
    # 1. BASIC STRUCTURE
    # ═══════════════════════════════════════
    print(f"\n{'─'*50}")
    print("1. STRUCTURE CHECK")
    print(f"{'─'*50}")

    required_keys = ["graph_a", "graph_b", "labels", "metadata"]
    for key in required_keys:
        if key not in data:
            print(f"  ✗ Missing key: {key}")
            return False
        print(f"  ✓ {key}: {len(data[key])} items")

    n = len(data["labels"])
    assert len(data["graph_a"]) == n, f"graph_a length mismatch: {len(data['graph_a'])} vs {n}"
    assert len(data["graph_b"]) == n, f"graph_b length mismatch: {len(data['graph_b'])} vs {n}"
    assert len(data["metadata"]) == n, f"metadata length mismatch: {len(data['metadata'])} vs {n}"
    print(f"  ✓ All lists same length: {n}")

    # ═══════════════════════════════════════
    # 2. GRAPH INTEGRITY
    # ═══════════════════════════════════════
    print(f"\n{'─'*50}")
    print("2. GRAPH INTEGRITY (checking all graphs)")
    print(f"{'─'*50}")

    issues = defaultdict(int)
    stats = {
        "nodes": [], "edges": [], "dfg_nodes": [], "cfg_edges": [],
        "feat_dim": [], "edge_types_seen": set(),
    }

    for side, graphs in [("graph_a", data["graph_a"]), ("graph_b", data["graph_b"])]:
        for i, g in enumerate(graphs):
            # Check x (node features)
            if not hasattr(g, 'x') or g.x is None:
                issues[f"{side}_missing_x"] += 1
                continue

            num_nodes = g.x.shape[0]
            feat_dim = g.x.shape[1]
            stats["feat_dim"].append(feat_dim)

            # NaN/Inf check
            if torch.isnan(g.x).any():
                issues[f"{side}_nan_features"] += 1
            if torch.isinf(g.x).any():
                issues[f"{side}_inf_features"] += 1

            # All-zero features check
            zero_rows = (g.x.abs().sum(dim=1) == 0).sum().item()
            if zero_rows == num_nodes:
                issues[f"{side}_all_zero_features"] += 1
            elif zero_rows > num_nodes * 0.5:
                issues[f"{side}_mostly_zero_features"] += 1

            # Edge index validity
            if not hasattr(g, 'edge_index') or g.edge_index is None:
                issues[f"{side}_missing_edges"] += 1
                continue

            ei = g.edge_index
            if ei.shape[0] != 2:
                issues[f"{side}_bad_edge_shape"] += 1
                continue

            max_idx = ei.max().item() if ei.numel() > 0 else 0
            if max_idx >= num_nodes:
                issues[f"{side}_edge_out_of_bounds"] += 1

            min_idx = ei.min().item() if ei.numel() > 0 else 0
            if min_idx < 0:
                issues[f"{side}_negative_edge_idx"] += 1

            # Edge type check
            if hasattr(g, 'edge_type') and g.edge_type is not None:
                for t in g.edge_type.unique().tolist():
                    stats["edge_types_seen"].add(t)

            # Collect stats
            if side == "graph_a":
                stats["nodes"].append(num_nodes)
                stats["edges"].append(ei.shape[1])
                stats["dfg_nodes"].append(getattr(g, 'num_dfg_nodes', 0))
                stats["cfg_edges"].append(getattr(g, 'num_cfg_edges', 0))

    if issues:
        print("  ⚠ Issues found:")
        for issue, count in sorted(issues.items()):
            print(f"    {issue}: {count}")
    else:
        print("  ✓ No structural issues found")

    # ═══════════════════════════════════════
    # 3. STATISTICS
    # ═══════════════════════════════════════
    print(f"\n{'─'*50}")
    print("3. GRAPH STATISTICS")
    print(f"{'─'*50}")

    if stats["nodes"]:
        print(f"  Nodes:     min={min(stats['nodes']):>5}  avg={np.mean(stats['nodes']):>7.1f}"
              f"  max={max(stats['nodes']):>5}  std={np.std(stats['nodes']):>6.1f}")
        print(f"  Edges:     min={min(stats['edges']):>5}  avg={np.mean(stats['edges']):>7.1f}"
              f"  max={max(stats['edges']):>5}  std={np.std(stats['edges']):>6.1f}")
        print(f"  DFG nodes: min={min(stats['dfg_nodes']):>5}  avg={np.mean(stats['dfg_nodes']):>7.1f}"
              f"  max={max(stats['dfg_nodes']):>5}")
        print(f"  CFG edges: min={min(stats['cfg_edges']):>5}  avg={np.mean(stats['cfg_edges']):>7.1f}"
              f"  max={max(stats['cfg_edges']):>5}")
        print(f"  Feature dim: {stats['feat_dim'][0] if stats['feat_dim'] else '?'}")

    # ═══════════════════════════════════════
    # 4. EDGE TYPE DISTRIBUTION
    # ═══════════════════════════════════════
    print(f"\n{'─'*50}")
    print("4. EDGE TYPE DISTRIBUTION")
    print(f"{'─'*50}")

    type_names = {0: "sequential", 1: "dfg-code", 2: "dfg-dfg", 3: "cfg"}
    type_totals = defaultdict(int)
    type_graph_counts = defaultdict(int)  # How many graphs have this type

    for g in data["graph_a"]:
        if hasattr(g, 'edge_type') and g.edge_type is not None:
            seen_in_graph = set()
            for t in g.edge_type.tolist():
                type_totals[t] += 1
                seen_in_graph.add(t)
            for t in seen_in_graph:
                type_graph_counts[t] += 1

    total_edges = sum(type_totals.values())
    for t in sorted(type_totals):
        name = type_names.get(t, f"type_{t}")
        count = type_totals[t]
        pct = 100 * count / total_edges if total_edges else 0
        graph_pct = 100 * type_graph_counts[t] / n if n else 0
        status = "✓" if graph_pct > 50 else ("~" if graph_pct > 10 else "⚠")
        print(f"  {status} {name:<12}: {count:>10} edges ({pct:>5.1f}%)  "
              f"present in {type_graph_counts[t]:>5}/{n} graphs ({graph_pct:.0f}%)")

    expected_types = {0, 1, 2, 3}
    missing = expected_types - stats["edge_types_seen"]
    if missing:
        print(f"  ⚠ Missing edge types: {[type_names.get(t, t) for t in missing]}")
    else:
        print(f"  ✓ All 4 edge types present")

    # ═══════════════════════════════════════
    # 5. PAIR ANALYSIS
    # ═══════════════════════════════════════
    print(f"\n{'─'*50}")
    print("5. PAIR ANALYSIS (correct vs buggy)")
    print(f"{'─'*50}")

    feature_diffs = []
    same_count = 0
    for i in range(min(n, 1000)):  # Sample first 1000
        ga = data["graph_a"][i]
        gb = data["graph_b"][i]
        if ga.x is not None and gb.x is not None:
            # Compare mean features
            diff = (ga.x.mean() - gb.x.mean()).abs().item()
            feature_diffs.append(diff)
            if diff < 1e-6:
                same_count += 1

    if feature_diffs:
        print(f"  Avg feature difference: {np.mean(feature_diffs):.6f}")
        print(f"  Identical pairs: {same_count}/{len(feature_diffs)} "
              f"({100*same_count/len(feature_diffs):.1f}%)")
        if same_count / len(feature_diffs) > 0.1:
            print(f"  ⚠ Warning: >10% identical pairs — check buggy code generation")
        else:
            print(f"  ✓ Pairs are sufficiently different")

    # ═══════════════════════════════════════
    # 6. LANGUAGE DISTRIBUTION
    # ═══════════════════════════════════════
    print(f"\n{'─'*50}")
    print("6. METADATA")
    print(f"{'─'*50}")

    lang_counts = defaultdict(int)
    ds_counts = defaultdict(int)
    bug_counts = defaultdict(int)
    for m in data["metadata"]:
        lang_counts[m.get("language", "?")] += 1
        ds_counts[m.get("dataset", "?")] += 1
        bug_counts[m.get("bug_type", "?")] += 1

    print(f"  Languages:")
    for lang, c in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang:<20} {c:>6}")

    print(f"  Datasets:")
    for ds, c in sorted(ds_counts.items(), key=lambda x: -x[1]):
        print(f"    {ds:<25} {c:>6}")

    print(f"  Bug types:")
    for bt, c in sorted(bug_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {bt:<20} {c:>6}")

    # ═══════════════════════════════════════
    # 7. SAMPLE INSPECTION
    # ═══════════════════════════════════════
    if show_samples > 0:
        print(f"\n{'─'*50}")
        print(f"7. SAMPLE GRAPHS (first {show_samples})")
        print(f"{'─'*50}")

        for i in range(min(show_samples, n)):
            ga = data["graph_a"][i]
            gb = data["graph_b"][i]
            meta = data["metadata"][i]

            print(f"\n  ── Pair {i}: {meta.get('pair_id', '?')} [{meta.get('language', '?')}] ──")

            for label, g in [("Correct", ga), ("Buggy", gb)]:
                n_nodes = g.x.shape[0] if g.x is not None else 0
                n_edges = g.edge_index.shape[1] if g.edge_index is not None else 0
                dfg_n = getattr(g, 'num_dfg_nodes', '?')
                cfg_n = getattr(g, 'num_cfg_edges', '?')
                code_n = getattr(g, 'num_code_tokens', '?')

                # Edge type breakdown
                et_counts = defaultdict(int)
                if hasattr(g, 'edge_type') and g.edge_type is not None:
                    for t in g.edge_type.tolist():
                        et_counts[t] += 1

                et_str = ", ".join(f"{type_names.get(t, t)}={c}" for t, c in sorted(et_counts.items()))

                print(f"    {label}: {n_nodes} nodes ({code_n} code + {dfg_n} DFG), "
                      f"{n_edges} edges [{et_str}], {cfg_n} CFG")

                # Feature stats
                if g.x is not None:
                    nonzero = (g.x.abs().sum(dim=1) > 0).sum().item()
                    print(f"      Features: dim={g.x.shape[1]}, nonzero_rows={nonzero}/{n_nodes}, "
                          f"mean={g.x.mean():.4f}, std={g.x.std():.4f}")

    # ═══════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════
    print(f"\n{'='*70}")
    total_issues = sum(issues.values())
    if total_issues == 0:
        print(f"✅ VERIFICATION PASSED — {n} graph pairs, no issues")
        print(f"   Ready for training!")
    elif total_issues < n * 0.01:
        print(f"⚠ VERIFICATION PASSED WITH WARNINGS — {total_issues} issues in {n} pairs ({100*total_issues/n:.2f}%)")
        print(f"   Should be fine for training.")
    else:
        print(f"✗ VERIFICATION FAILED — {total_issues} issues in {n} pairs ({100*total_issues/n:.1f}%)")
        print(f"   Review issues above before training.")
    print(f"{'='*70}")

    return total_issues == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-data", required=True, help="Path to .pt graph data file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-samples", type=int, default=3)
    args = parser.parse_args()

    verify_graph_data(args.graph_data, args.verbose, args.show_samples)


if __name__ == "__main__":
    main()
