"""
Sharded dataset for training on per-language .pt files without loading all into RAM.

Instead of one 99GB file, loads individual .pt files (8-27GB each) one at a time.
Uses shard-aware batch sampling: all batches from one shard are processed before
moving to the next, so only 1 shard is in memory at any time.

Usage:
    from sharded_dataset import ShardedPairDataset, ShardBatchSampler, build_index

    index = build_index(shard_files)  # Scans files, extracts labels+metadata
    dataset = ShardedPairDataset(index, train_indices, swap_aug=True)
    sampler = ShardBatchSampler(dataset, batch_size=16, shuffle=True)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_pairs)
"""
import gc
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


# ═══════════════════════════════════════
# INDEX BUILDER
# ═══════════════════════════════════════

def build_index(shard_files: List[str], spec_shard_files: Optional[List[str]] = None,
                verbose: bool = True) -> Dict:
    """
    Scan all .pt files and extract lightweight metadata.
    Loads each file one at a time, keeps only labels + metadata, frees graphs.
    
    Returns dict with:
        shards: [{path, n, labels, metadata}, ...]
        index: [(shard_idx, local_idx), ...]  -- global ordering
        labels: [int, ...]  -- all labels in global order
        metadata: [dict, ...]  -- all metadata in global order
        has_spec: bool
        spec_shards: [{path, n}, ...] if spec files provided
    """
    shards = []
    index = []
    all_labels = []
    all_metadata = []
    
    if verbose:
        print(f"Scanning {len(shard_files)} shard files...")
    
    for si, fpath in enumerate(shard_files):
        if verbose:
            print(f"  [{si+1}/{len(shard_files)}] {os.path.basename(fpath)}...", end=" ", flush=True)
        
        # Load just to extract metadata — this loads the full file but we
        # immediately discard graph tensors to free memory
        data = torch.load(fpath, weights_only=False, map_location="cpu")
        n = len(data["labels"])
        labels = list(data["labels"])
        metadata = list(data["metadata"])
        
        shards.append({
            "path": fpath,
            "n": n,
        })
        
        for li in range(n):
            index.append((si, li))
            all_labels.append(labels[li])
            all_metadata.append(metadata[li])
        
        if verbose:
            n0 = labels.count(0)
            n1 = labels.count(1)
            print(f"{n} pairs (label-0={n0}, label-1={n1})")
        
        # Free the heavy graph data
        del data
        gc.collect()
    
    # Handle spec shards
    has_spec = False
    spec_shards = []
    if spec_shard_files:
        if verbose:
            print(f"\nScanning {len(spec_shard_files)} spec shard files...")
        for si, fpath in enumerate(spec_shard_files):
            if verbose:
                print(f"  [{si+1}/{len(spec_shard_files)}] {os.path.basename(fpath)}...", end=" ", flush=True)
            data = torch.load(fpath, weights_only=False, map_location="cpu")
            n = len(data.get("graph_spec", data.get("graph_a", [])))
            spec_shards.append({"path": fpath, "n": n})
            if verbose:
                print(f"{n} specs")
            del data
            gc.collect()
        has_spec = True
    
    total = len(index)
    n0 = all_labels.count(0)
    n1 = all_labels.count(1)
    if verbose:
        print(f"\nTotal: {total} pairs (label-0={n0}, label-1={n1})")
        print(f"Shards: {len(shards)}, Spec: {'YES' if has_spec else 'NO'}")
    
    return {
        "shards": shards,
        "index": index,
        "labels": all_labels,
        "metadata": all_metadata,
        "has_spec": has_spec,
        "spec_shards": spec_shards,
    }


# ═══════════════════════════════════════
# SHARDED DATASET
# ═══════════════════════════════════════

class ShardedPairDataset:
    """
    Lazily loads graph pairs from per-language .pt files.
    Only one shard is in memory at a time (via LRU cache of size 1).
    Use with ShardBatchSampler for efficient shard-sequential access.
    """
    
    def __init__(self, full_index: Dict, subset_indices: List[int],
                 swap_aug: bool = False):
        """
        full_index: output of build_index()
        subset_indices: which global indices to include (train or test split)
        swap_aug: if True, doubles dataset by swapping A/B
        """
        self.shards = full_index["shards"]
        self.swap_aug = swap_aug
        
        # Map from local dataset index → (shard_idx, local_idx_in_shard)
        self.items = [full_index["index"][i] for i in subset_indices]
        self.labels_list = [full_index["labels"][i] for i in subset_indices]
        self.meta_list = [full_index["metadata"][i] for i in subset_indices]
        
        # Shard cache (keeps 1 shard loaded)
        self._cached_shard_data = None
        self._cached_shard_idx = -1
    
    @property
    def labels(self):
        return self.labels_list
    
    @property
    def metadata(self):
        return self.meta_list
    
    def _load_shard(self, shard_idx: int):
        """Load a shard if not already cached."""
        if self._cached_shard_idx != shard_idx:
            # Free previous shard
            self._cached_shard_data = None
            gc.collect()
            
            path = self.shards[shard_idx]["path"]
            self._cached_shard_data = torch.load(path, weights_only=False,
                                                  map_location="cpu")
            self._cached_shard_idx = shard_idx
        return self._cached_shard_data
    
    def release_cache(self):
        """Explicitly free cached shard (call between folds)."""
        self._cached_shard_data = None
        self._cached_shard_idx = -1
        gc.collect()
    
    def __len__(self):
        return len(self.items) * (2 if self.swap_aug else 1)
    
    def __getitem__(self, idx):
        swapped = False
        n = len(self.items)
        if idx >= n:
            idx = idx - n
            swapped = True
        
        shard_idx, local_idx = self.items[idx]
        shard = self._load_shard(shard_idx)
        
        ga = shard["graph_a"][local_idx]
        gb = shard["graph_b"][local_idx]
        label = shard["labels"][local_idx]
        
        if swapped:
            return gb, ga, 1 - label
        return ga, gb, label


class ShardedSpecPairDataset(ShardedPairDataset):
    """Sharded dataset with spec graphs (from separate spec shard files)."""
    
    def __init__(self, full_index: Dict, subset_indices: List[int],
                 swap_aug: bool = False):
        super().__init__(full_index, subset_indices, swap_aug)
        self.spec_shards = full_index.get("spec_shards", [])
        
        # Spec shard cache (separate from code shard cache)
        self._cached_spec_data = None
        self._cached_spec_idx = -1
    
    def _load_spec_shard(self, shard_idx: int):
        """Load spec shard. Shares same shard_idx as code shards."""
        if self._cached_spec_idx != shard_idx:
            self._cached_spec_data = None
            gc.collect()
            if shard_idx < len(self.spec_shards):
                path = self.spec_shards[shard_idx]["path"]
                self._cached_spec_data = torch.load(path, weights_only=False,
                                                     map_location="cpu")
            self._cached_spec_idx = shard_idx
        return self._cached_spec_data
    
    def release_cache(self):
        super().release_cache()
        self._cached_spec_data = None
        self._cached_spec_idx = -1
        gc.collect()
    
    def __getitem__(self, idx):
        swapped = False
        n = len(self.items)
        if idx >= n:
            idx = idx - n
            swapped = True
        
        shard_idx, local_idx = self.items[idx]
        shard = self._load_shard(shard_idx)
        
        ga = shard["graph_a"][local_idx]
        gb = shard["graph_b"][local_idx]
        label = shard["labels"][local_idx]
        
        # Get spec graph
        spec_shard = self._load_spec_shard(shard_idx)
        if spec_shard is not None and "graph_spec" in spec_shard:
            spec = spec_shard["graph_spec"][local_idx]
        else:
            # Fallback: empty spec graph
            spec = _empty_spec_graph()
        
        if swapped:
            return gb, ga, spec, 1 - label
        return ga, gb, spec, label


def _empty_spec_graph():
    """Minimal placeholder spec graph (single CLS node)."""
    from torch_geometric.data import Data
    return Data(
        x=torch.zeros(1, 768),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        num_nodes=1,
    )


# ═══════════════════════════════════════
# SHARD-AWARE BATCH SAMPLER
# ═══════════════════════════════════════

class ShardBatchSampler:
    """
    Yields batches grouped by shard so only one shard needs to be loaded at a time.
    
    Within each epoch:
    1. Shuffle shard order
    2. For each shard, shuffle its indices
    3. Yield batches from that shard
    4. Move to next shard
    
    This means each shard is loaded exactly ONCE per epoch.
    """
    
    def __init__(self, dataset: ShardedPairDataset, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group dataset indices by which shard they come from
        self.shard_groups = defaultdict(list)
        n_real = len(dataset.items)  # Without swap aug
        
        for i, (si, li) in enumerate(dataset.items):
            self.shard_groups[si].append(i)
        
        # If swap_aug, add the swapped indices (offset by n_real) to same shards
        if dataset.swap_aug:
            for i, (si, li) in enumerate(dataset.items):
                self.shard_groups[si].append(i + n_real)
        
        self._total = sum(len(v) for v in self.shard_groups.values())
    
    def __iter__(self):
        shard_order = list(self.shard_groups.keys())
        if self.shuffle:
            random.shuffle(shard_order)
        
        for si in shard_order:
            indices = self.shard_groups[si].copy()
            if self.shuffle:
                random.shuffle(indices)
            
            # Yield batches from this shard
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch
    
    def __len__(self):
        if self.drop_last:
            return sum(len(v) // self.batch_size for v in self.shard_groups.values())
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size
            for v in self.shard_groups.values()
        )


# ═══════════════════════════════════════
# COLLATE FUNCTIONS (same as before)
# ═══════════════════════════════════════

def collate_pairs(batch):
    ga, gb, labels = zip(*batch)
    return (Batch.from_data_list(list(ga)),
            Batch.from_data_list(list(gb)),
            torch.tensor(labels, dtype=torch.long))


def collate_spec_pairs(batch):
    ga, gb, specs, labels = zip(*batch)
    return (Batch.from_data_list(list(ga)),
            Batch.from_data_list(list(gb)),
            Batch.from_data_list(list(specs)),
            torch.tensor(labels, dtype=torch.long))


# ═══════════════════════════════════════
# HELPER: Find shard files
# ═══════════════════════════════════════

def find_shard_files(data_dir: str, prefix: str = "graph_data_",
                     exclude_patterns: List[str] = None) -> List[str]:
    """Find per-language .pt files in a directory."""
    if exclude_patterns is None:
        exclude_patterns = ["all.pt", "backup", "spec_", "checkpoint"]
    
    files = []
    for f in sorted(os.listdir(data_dir)):
        if not f.startswith(prefix) or not f.endswith(".pt"):
            continue
        if any(p in f for p in exclude_patterns):
            continue
        files.append(os.path.join(data_dir, f))
    
    return files


def find_spec_shard_files(data_dir: str, code_shard_files: List[str]) -> List[str]:
    """
    Find spec shard files matching code shard files.
    graph_data_codenet_python.pt → spec_data_codenet_python.pt
    """
    spec_files = []
    for code_file in code_shard_files:
        basename = os.path.basename(code_file)
        spec_name = basename.replace("graph_data_", "spec_data_")
        spec_path = os.path.join(data_dir, spec_name)
        if os.path.exists(spec_path):
            spec_files.append(spec_path)
        else:
            spec_files.append(None)  # No spec for this shard
    return spec_files