"""
Fix label imbalance in existing graph files.

Problem: All pairs have label=0 (A=correct, B=buggy always).
Fix: Randomly swap A↔B for 50% of pairs, setting label=1 for swapped pairs.

The graph embeddings are unchanged — only the assignment to A/B position changes.
This is mathematically equivalent to what build_graphs.py now does for new builds.

Usage:
    python scripts/fix_labels.py
    python scripts/fix_labels.py --graph-dir /data/workzone/siamese_gat_journal/data/graphs
"""
import glob
import os
import random
import sys

import torch
import numpy as np

GRAPH_DIR = "/data/workzone/siamese_gat_journal/data/graphs"


def fix_file(fpath, seed=42):
    """Fix one graph file: randomly swap 50% of pairs."""
    print(f"\n{'─'*60}")
    print(f"Fixing: {os.path.basename(fpath)}")
    
    data = torch.load(fpath, weights_only=False, map_location="cpu")
    n = len(data["labels"])
    
    # Check current state
    n_label0 = data["labels"].count(0)
    n_label1 = data["labels"].count(1) if 1 in data["labels"] else 0
    print(f"  Before: {n} pairs, label-0={n_label0}, label-1={n_label1}")
    
    if n_label1 > 0:
        print(f"  Already has balanced labels — skipping")
        return False
    
    # Deterministic random swap based on file + seed
    rng = random.Random(seed + hash(os.path.basename(fpath)))
    
    swapped = 0
    for i in range(n):
        if rng.random() < 0.5:
            # Swap A and B
            data["graph_a"][i], data["graph_b"][i] = data["graph_b"][i], data["graph_a"][i]
            data["labels"][i] = 1  # Now B is correct
            swapped += 1
    
    n_label0_after = data["labels"].count(0)
    n_label1_after = data["labels"].count(1)
    print(f"  After:  {n} pairs, label-0={n_label0_after}, label-1={n_label1_after}")
    print(f"  Swapped: {swapped}/{n} pairs ({100*swapped/n:.1f}%)")
    
    # Save (overwrite)
    torch.save(data, fpath)
    print(f"  ✓ Saved")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default=GRAPH_DIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    files = sorted(glob.glob(os.path.join(args.graph_dir, "graph_data_*.pt")))
    files = [f for f in files if "backup" not in f and "all.pt" not in f and "spec_" not in f]
    
    print(f"Found {len(files)} graph files in {args.graph_dir}")
    
    fixed = 0
    total_pairs = 0
    
    for fpath in files:
        if fix_file(fpath, args.seed):
            fixed += 1
        data = torch.load(fpath, weights_only=False, map_location="cpu")
        total_pairs += len(data["labels"])
    
    print(f"\n{'='*60}")
    print(f"DONE: Fixed {fixed}/{len(files)} files, {total_pairs} total pairs")
    print(f"{'='*60}")
    
    if fixed > 0:
        print(f"\nNow rebuild the combined file:")
        print(f"  python scripts/combine_graphs.py")


if __name__ == "__main__":
    main()