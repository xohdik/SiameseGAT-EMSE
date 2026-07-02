"""Combine all per-language graph files into a single graph_data_all.pt"""
import os, sys, glob
import torch
from collections import defaultdict

GRAPH_DIR = "/data/workzone/siamese_gat_journal/data/graphs"
OUTPUT = os.path.join(GRAPH_DIR, "graph_data_all.pt")

# Find all graph files (exclude backups and the output file itself)
files = sorted(glob.glob(os.path.join(GRAPH_DIR, "graph_data_*.pt")))
files = [f for f in files if "backup" not in f and "all.pt" not in f and "spec_" not in f]

print(f"Found {len(files)} graph files:")
for f in files:
    print(f"  {os.path.basename(f)}")

combined = {"graph_a": [], "graph_b": [], "labels": [], "metadata": []}
lang_counts = defaultdict(int)
ds_counts = defaultdict(int)

for fpath in files:
    print(f"\nLoading: {os.path.basename(fpath)}...")
    data = torch.load(fpath, weights_only=False, map_location="cpu")
    n = len(data["labels"])
    print(f"  {n} pairs")
    
    combined["graph_a"].extend(data["graph_a"])
    combined["graph_b"].extend(data["graph_b"])
    combined["labels"].extend(data["labels"])
    combined["metadata"].extend(data["metadata"])
    
    for m in data["metadata"]:
        lang_counts[m.get("language", "?")] += 1
        ds_counts[m.get("dataset", "?")] += 1

total = len(combined["labels"])
print(f"\n{'='*60}")
print(f"TOTAL: {total} pairs")
print(f"{'='*60}")

print(f"\nBy language:")
for lang, c in sorted(lang_counts.items(), key=lambda x: -x[1]):
    print(f"  {lang:<15} {c:>7} pairs ({100*c/total:.1f}%)")

print(f"\nBy dataset:")
for ds, c in sorted(ds_counts.items(), key=lambda x: -x[1]):
    print(f"  {ds:<30} {c:>7} pairs")

print(f"\nLabel distribution:")
labels = combined["labels"]
n0 = labels.count(0)
n1 = labels.count(1) if 1 in labels else 0
print(f"  0 (correct pair): {n0}")
print(f"  1 (other):        {n1}")

print(f"\nSaving to: {OUTPUT}")
torch.save(combined, OUTPUT)
size_mb = os.path.getsize(OUTPUT) / 1e6
print(f"  Size: {size_mb:.1f} MB")
print(f"\n✓ Done! Now run:")
print(f"  python scripts/train_spec.py --data data/graphs/graph_data_all.pt --model-type siamese_gat --device cuda:1")