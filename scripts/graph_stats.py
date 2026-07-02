"""
Extract graph statistics from all .pt files for paper tables.
Reports: avg nodes, avg edges by type, edge type distribution.

Usage:
    python scripts/graph_stats.py
"""
import gc
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

GRAPH_DIR = "/data/workzone/siamese_gat_journal/data/graphs"


def stats_for_file(fpath):
    """Extract stats from one .pt file without keeping it in memory."""
    data = torch.load(fpath, weights_only=False, map_location="cpu")
    n = len(data["labels"])

    stats = {
        "n_pairs": n,
        "avg_nodes_a": 0, "avg_edges_a": 0,
        "avg_dfg_nodes": 0, "avg_cfg_edges": 0,
        "edge_type_counts": defaultdict(int),
        "total_edges": 0,
    }

    nodes_a, edges_a, dfg_n, cfg_e = [], [], [], []
    for g in data["graph_a"]:
        nodes_a.append(g.num_nodes)
        edges_a.append(g.edge_index.shape[1])
        dfg_n.append(g.num_dfg_nodes)
        cfg_e.append(g.num_cfg_edges)
        for t in g.edge_type.tolist():
            stats["edge_type_counts"][t] += 1
            stats["total_edges"] += 1

    stats["avg_nodes_a"] = float(np.mean(nodes_a))
    stats["avg_edges_a"] = float(np.mean(edges_a))
    stats["avg_dfg_nodes"] = float(np.mean(dfg_n))
    stats["avg_cfg_edges"] = float(np.mean(cfg_e))

    del data
    gc.collect()
    return stats


def main():
    # Find all graph files
    all_files = sorted(glob.glob(os.path.join(GRAPH_DIR, "graph_data_*.pt")))
    all_files = [f for f in all_files if "all.pt" not in f and "backup" not in f]

    # Group by variant
    variants = {"full": [], "dfg_only": [], "cfg_only": [], "seq_only": []}
    for f in all_files:
        bn = os.path.basename(f)
        if "_dfg_only" in bn:
            variants["dfg_only"].append(f)
        elif "_cfg_only" in bn:
            variants["cfg_only"].append(f)
        elif "_seq_only" in bn:
            variants["seq_only"].append(f)
        elif "spec_" not in bn:
            variants["full"].append(f)

    type_names = {0: "sequential", 1: "dfg-code", 2: "dfg-dfg", 3: "cfg"}
    results = {}

    for variant, files in variants.items():
        if not files:
            continue
        print(f"\n{'='*60}")
        print(f"VARIANT: {variant} ({len(files)} files)")
        print(f"{'='*60}")
        print(f"{'File':<45} {'Pairs':>6} {'Nodes':>7} {'Edges':>7} {'DFG-n':>6} {'CFG-e':>6}")
        print(f"{'─'*80}")

        variant_results = {}
        for fpath in files:
            bn = os.path.basename(fpath)
            # Extract language from filename
            name = bn.replace("graph_data_", "").replace(f"_{variant}", "").replace(".pt", "")
            print(f"  {bn:<43}", end=" ", flush=True)

            s = stats_for_file(fpath)
            print(f"{s['n_pairs']:>6} {s['avg_nodes_a']:>7.1f} {s['avg_edges_a']:>7.1f} "
                  f"{s['avg_dfg_nodes']:>6.1f} {s['avg_cfg_edges']:>6.1f}")

            # Edge type breakdown
            total = s["total_edges"]
            if total > 0:
                pcts = []
                for t in sorted(s["edge_type_counts"]):
                    c = s["edge_type_counts"][t]
                    pcts.append(f"{type_names.get(t, f'type_{t}')}: {100*c/total:.1f}%")
                print(f"    {'  '.join(pcts)}")

            variant_results[name] = {
                "n_pairs": s["n_pairs"],
                "avg_nodes": round(s["avg_nodes_a"], 1),
                "avg_edges": round(s["avg_edges_a"], 1),
                "avg_dfg_nodes": round(s["avg_dfg_nodes"], 1),
                "avg_cfg_edges": round(s["avg_cfg_edges"], 1),
                "edge_pcts": {type_names.get(t, f"type_{t}"): round(100*c/total, 1)
                              for t, c in s["edge_type_counts"].items()} if total > 0 else {},
            }

        results[variant] = variant_results

    # Save
    out_path = os.path.join("outputs", "graph_statistics.json")
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")

    # Summary table for paper
    print(f"\n{'='*60}")
    print("PAPER TABLE: Edge Distribution (Full Graphs)")
    print(f"{'='*60}")
    print(f"{'Language':<15} {'Seq%':>6} {'DFG-C%':>7} {'DFG-D%':>7} {'CFG%':>6} {'Total':>7}")
    print(f"{'─'*50}")
    if "full" in results:
        for name, s in sorted(results["full"].items()):
            p = s["edge_pcts"]
            print(f"{name:<15} {p.get('sequential',0):>6.1f} {p.get('dfg-code',0):>7.1f} "
                  f"{p.get('dfg-dfg',0):>7.1f} {p.get('cfg',0):>6.1f} {s['avg_edges']:>7.1f}")


if __name__ == "__main__":
    main()