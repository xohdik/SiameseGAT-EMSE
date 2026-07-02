"""
Create ablation variants by filtering edge types from existing graph files.
No GPU needed — just removes edges from existing PyG Data objects.

Edge types:
    0 = sequential (token order)
    1 = DFG-code (DFG node ↔ code token)
    2 = DFG-DFG (data flow between variables)
    3 = CFG (control flow)

Variants:
    full       = {0,1,2,3}  (already have this)
    dfg_only   = {0,1,2}    removes CFG
    cfg_only   = {0,3}      removes DFG
    seq_only   = {0}         sequential baseline

Usage:
    python scripts/ablation_edges.py --lang python
    python scripts/ablation_edges.py --all
"""
import argparse
import gc
import os
import sys
from copy import deepcopy

import torch
from torch_geometric.data import Data
from tqdm import tqdm

GRAPH_DIR = "/data/workzone/siamese_gat_journal/data/graphs"

VARIANTS = {
    "dfg_only": {0, 1, 2},
    "cfg_only": {0, 3},
    "seq_only": {0},
}

LANGS = ["python", "cpp", "java", "c", "ruby", "javascript"]


def filter_edges(graph: Data, keep_types: set) -> Data:
    """Filter edges by type, return new Data object."""
    et = graph.edge_type
    mask = torch.zeros(len(et), dtype=torch.bool)
    for t in keep_types:
        mask |= (et == t)

    new_ei = graph.edge_index[:, mask]
    new_et = graph.edge_type[mask]

    # Recount CFG edges
    cfg_count = (new_et == 3).sum().item()

    return Data(
        x=graph.x,
        edge_index=new_ei,
        edge_type=new_et,
        num_nodes=graph.num_nodes,
        node_types=graph.node_types,
        num_code_tokens=graph.num_code_tokens,
        num_dfg_nodes=graph.num_dfg_nodes,
        num_cfg_edges=cfg_count,
    )


def process_file(input_path, variant_name, keep_types):
    """Filter one .pt file and save the variant."""
    basename = os.path.basename(input_path)
    # graph_data_codenet_python.pt → graph_data_codenet_python_dfg_only.pt
    out_name = basename.replace(".pt", f"_{variant_name}.pt")
    out_path = os.path.join(os.path.dirname(input_path), out_name)

    if os.path.exists(out_path):
        print(f"  Already exists: {out_name} — skipping")
        return out_path

    print(f"  Loading {basename}...")
    data = torch.load(input_path, weights_only=False, map_location="cpu")
    n = len(data["labels"])

    # Filter edges in all graphs
    for key in ["graph_a", "graph_b"]:
        for i in tqdm(range(n), desc=f"    {key}", leave=False):
            data[key][i] = filter_edges(data[key][i], keep_types)

    # Stats
    avg_edges_a = sum(g.edge_index.shape[1] for g in data["graph_a"]) / n
    print(f"  Avg edges after filter: {avg_edges_a:.1f}")

    torch.save(data, out_path)
    print(f"  Saved: {out_name}")
    del data
    gc.collect()
    return out_path


def process_lang(lang, graph_dir):
    """Create all ablation variants for one language."""
    # Find files for this language
    files = []
    for prefix in ["graph_data_codenet_", "graph_data_humanevalfix_"]:
        fpath = os.path.join(graph_dir, f"{prefix}{lang}.pt")
        if os.path.exists(fpath):
            files.append(fpath)

    if not files:
        print(f"No files found for {lang}")
        return

    print(f"\n{'='*60}")
    print(f"Language: {lang} ({len(files)} files)")
    print(f"{'='*60}")

    for variant_name, keep_types in VARIANTS.items():
        print(f"\n  Variant: {variant_name} (keep types: {keep_types})")
        for fpath in files:
            process_file(fpath, variant_name, keep_types)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, default=None, help="Single language")
    ap.add_argument("--all", action="store_true", help="All 6 languages")
    ap.add_argument("--graph-dir", default=GRAPH_DIR)
    args = ap.parse_args()

    if args.all:
        for lang in LANGS:
            process_lang(lang, args.graph_dir)
    elif args.lang:
        process_lang(args.lang, args.graph_dir)
    else:
        print("Specify --lang <language> or --all")
        return

    print(f"\n{'='*60}")
    print("DONE. Now train ablation variants:")
    print("  python scripts/train_spec.py --lang python --model-type siamese_gat --device cuda:1")
    print("  (with --graph-dir pointing to ablation files)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()