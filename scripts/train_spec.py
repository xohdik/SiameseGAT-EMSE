"""
Train Siamese GAT models with optional specification awareness.

Supports:
    - Original SiameseGAT (Paper 1 baseline)
    - SpecAwareSiameseGAT with concat/gate/cross-attention fusion (Paper 2)
    - Ablation: spec model with spec=None (architecture effect vs information effect)
    - 5-fold problem-stratified cross-validation
    - Sharded lazy loading (per-language .pt files, ~8-27GB each, one at a time)

Usage:
    # Paper 1: Per-language baseline (recommended — fits in RAM)
    python scripts/train_spec.py \
        --lang python --device cuda:1

    python scripts/train_spec.py \
        --lang cpp --device cuda:1

    python scripts/train_spec.py \
        --lang java --device cuda:1

    # Paper 2: Spec-aware per-language
    python scripts/train_spec.py \
        --lang python --model-type spec_siamese_gat --device cuda:1

    # Cross-attention fusion
    python scripts/train_spec.py \
        --lang python --model-type spec_siamese_gat \
        --cross-attention --device cuda:1

    # All languages sharded (lazy loading, slow)
    python scripts/train_spec.py \
        --data-dir data/graphs --model-type siamese_gat --device cuda:1

    # Single file mode (small datasets only)
    python scripts/train_spec.py \
        --data data/graphs/graph_data_codenet_python.pt \
        --model-type siamese_gat --device cuda:1
"""
import argparse, json, os, random, sys, time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SiameseGAT, count_parameters
from model_spec import SpecAwareSiameseGAT, SpecAwareSiameseGAT_CodeOnly, build_model
from sharded_dataset import (ShardedPairDataset, ShardedSpecPairDataset,
                              ShardBatchSampler, build_index, find_shard_files,
                              find_spec_shard_files,
                              collate_pairs as sharded_collate_pairs,
                              collate_spec_pairs as sharded_collate_spec_pairs)


# ═══════════════════════════════════════
# LEARNABLE EMBEDDING WRAPPER (no pretrained)
# ═══════════════════════════════════════

class LearnableEmbeddingSiameseGAT(nn.Module):
    """
    Wraps SiameseGAT with trainable token embeddings instead of frozen BERT vectors.
    Small embedding (128-dim) projected to model input dim (768) to keep params manageable.
    """
    
    def __init__(self, base_model, vocab_size=50266, small_dim=128, model_dim=768):
        super().__init__()
        self.base_model = base_model
        self.token_embedding = nn.Embedding(vocab_size, small_dim)
        self.project = nn.Linear(small_dim, model_dim)
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.project.weight)
    
    def _embed(self, data):
        """Replace data.x with learned embeddings from data.token_ids."""
        data.x = self.project(self.token_embedding(data.token_ids))
        return data
    
    def forward(self, data_a, data_b, return_attention=False):
        data_a = self._embed(data_a)
        data_b = self._embed(data_b)
        return self.base_model(data_a, data_b, return_attention)


def count_parameters_wrapper(model):
    """Count parameters for wrapped or unwrapped models."""
    if hasattr(model, 'base_model'):
        from model import count_parameters
        base = count_parameters(model.base_model)
        embed = sum(p.numel() for p in model.token_embedding.parameters())
        total = base["trainable"] + embed
        return {"trainable": total, "trainable_human": f"{total:,}"}
    from model import count_parameters
    return count_parameters(model)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════

class PairDataset:
    """Code pair dataset (no spec)."""
    def __init__(self, ga, gb, labels, meta, swap_aug=False):
        self.ga, self.gb, self.labels, self.meta = ga, gb, labels, meta
        self.swap_aug = swap_aug
    def __len__(self):
        return len(self.labels) * (2 if self.swap_aug else 1)
    def __getitem__(self, idx):
        if idx >= len(self.labels):
            i = idx - len(self.labels)
            return self.gb[i], self.ga[i], 1 - self.labels[i]
        return self.ga[idx], self.gb[idx], self.labels[idx]


class SpecPairDataset:
    """Code pair + spec graph dataset."""
    def __init__(self, ga, gb, specs, labels, meta, swap_aug=False):
        self.ga, self.gb, self.specs = ga, gb, specs
        self.labels, self.meta = labels, meta
        self.swap_aug = swap_aug
    def __len__(self):
        return len(self.labels) * (2 if self.swap_aug else 1)
    def __getitem__(self, idx):
        if idx >= len(self.labels):
            i = idx - len(self.labels)
            # Swap: B becomes A, A becomes B, label flips. Spec stays same.
            return self.gb[i], self.ga[i], self.specs[i], 1 - self.labels[i]
        return self.ga[idx], self.gb[idx], self.specs[idx], self.labels[idx]


def collate_pairs(batch):
    """Collate for code-only mode."""
    ga, gb, labels = zip(*batch)
    return (Batch.from_data_list(list(ga)),
            Batch.from_data_list(list(gb)),
            torch.tensor(labels, dtype=torch.long))


def collate_spec_pairs(batch):
    """Collate for spec-aware mode."""
    ga, gb, specs, labels = zip(*batch)
    return (Batch.from_data_list(list(ga)),
            Batch.from_data_list(list(gb)),
            Batch.from_data_list(list(specs)),
            torch.tensor(labels, dtype=torch.long))


# ═══════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════

def get_problem_groups(metadata):
    groups = [m.get("pair_id", "unk") for m in metadata]
    unique = sorted(set(groups))
    gmap = {g: i for i, g in enumerate(unique)}
    return np.array([gmap[g] for g in groups])


def train_one_epoch(model, loader, optimizer, criterion, device,
                    has_spec=False, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch in loader:
        if has_spec:
            ba, bb, bs, labels = batch
            ba, bb, bs, labels = ba.to(device), bb.to(device), bs.to(device), labels.to(device)
            result = model(ba, bb, bs)
        else:
            ba, bb, labels = batch
            ba, bb, labels = ba.to(device), bb.to(device), labels.to(device)
            result = model(ba, bb)
        
        loss = criterion(result["logits"], labels)
        
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = result["logits"].argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device, has_spec=False):
    model.eval()
    total_loss = 0
    preds_all, labels_all, probs_all = [], [], []
    
    for batch in loader:
        if has_spec:
            ba, bb, bs, labels = batch
            ba, bb, bs, labels = ba.to(device), bb.to(device), bs.to(device), labels.to(device)
            result = model(ba, bb, bs)
        else:
            ba, bb, labels = batch
            ba, bb, labels = ba.to(device), bb.to(device), labels.to(device)
            result = model(ba, bb)
        
        total_loss += criterion(result["logits"], labels).item() * labels.size(0)
        preds_all.extend(result["logits"].argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
        probs_all.extend(F.softmax(result["logits"], dim=-1)[:, 1].cpu().tolist())
    
    n = len(labels_all)
    metrics = {
        "loss": total_loss / n,
        "f1_macro": f1_score(labels_all, preds_all, average="macro"),
        "accuracy": accuracy_score(labels_all, preds_all),
        "precision_macro": precision_score(labels_all, preds_all, average="macro", zero_division=0),
        "recall_macro": recall_score(labels_all, preds_all, average="macro", zero_division=0),
        "preds": preds_all, "labels": labels_all, "probs": probs_all,
    }
    try:
        metrics["auc"] = roc_auc_score(labels_all, probs_all)
    except:
        metrics["auc"] = 0.0
    return metrics


# ═══════════════════════════════════════
# K-FOLD CROSS VALIDATION
# ═══════════════════════════════════════

def run_kfold(graph_data, config, output_dir, device):
    # Set up file logging
    import io
    log_path = os.path.join(output_dir, "train.log")
    log_file = open(log_path, "w")
    _orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print
    def log_print(*args, **kwargs):
        _orig_print(*args, **kwargs)
        kwargs.pop("flush", None)
        _orig_print(*args, **kwargs, file=log_file, flush=True)
    import builtins
    builtins.print = log_print
    
    has_spec = "graph_spec" in graph_data and config.get("model_type") != "siamese_gat"
    n_folds = config.get("n_folds", 5)
    labels = np.array(graph_data["labels"])
    groups = get_problem_groups(graph_data["metadata"])
    
    n_groups = len(set(groups))
    n_unique_labels = len(set(labels.tolist()))
    if n_groups >= n_folds:
        splits = list(GroupKFold(n_splits=n_folds).split(labels, labels, groups))
    elif n_unique_labels >= 2:
        splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=config.get("seed", 42)).split(labels, labels))
    else:
        # All labels identical (e.g., all 0) — use regular KFold
        from sklearn.model_selection import KFold
        splits = list(KFold(n_splits=n_folds, shuffle=True,
                            random_state=config.get("seed", 42)).split(labels))
    
    print(f"\n{'='*60}")
    print(f"MODEL: {config.get('model_type', 'siamese_gat')}")
    print(f"FUSION: {config.get('fusion_mode', 'N/A')}")
    print(f"SPEC: {'YES' if has_spec else 'NO'}")
    print(f"CROSS-ATTN: {'YES' if config.get('cross_attention') else 'NO'}")
    print(f"FOLDS: {n_folds}, PAIRS: {len(labels)}")
    print(f"{'='*60}\n")
    
    fold_results = []
    all_preds, all_labels = [], []
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'─'*50}")
        print(f"FOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"{'─'*50}")
        
        # Build datasets
        # Labels are now balanced (50% label-0, 50% label-1) from fix_labels.py.
        # Train: swap_aug doubles the dataset (valid augmentation).
        # Test: no swap — evaluate on natural balanced distribution.
        if has_spec:
            train_ds = SpecPairDataset(
                [graph_data["graph_a"][i] for i in train_idx],
                [graph_data["graph_b"][i] for i in train_idx],
                [graph_data["graph_spec"][i] for i in train_idx],
                [graph_data["labels"][i] for i in train_idx],
                [graph_data["metadata"][i] for i in train_idx],
                swap_aug=config.get("swap_augmentation", True),
            )
            test_ds = SpecPairDataset(
                [graph_data["graph_a"][i] for i in test_idx],
                [graph_data["graph_b"][i] for i in test_idx],
                [graph_data["graph_spec"][i] for i in test_idx],
                [graph_data["labels"][i] for i in test_idx],
                [graph_data["metadata"][i] for i in test_idx],
                swap_aug=False,
            )
            collate_fn = collate_spec_pairs
        else:
            train_ds = PairDataset(
                [graph_data["graph_a"][i] for i in train_idx],
                [graph_data["graph_b"][i] for i in train_idx],
                [graph_data["labels"][i] for i in train_idx],
                [graph_data["metadata"][i] for i in train_idx],
                swap_aug=config.get("swap_augmentation", True),
            )
            test_ds = PairDataset(
                [graph_data["graph_a"][i] for i in test_idx],
                [graph_data["graph_b"][i] for i in test_idx],
                [graph_data["labels"][i] for i in test_idx],
                [graph_data["metadata"][i] for i in test_idx],
                swap_aug=False,
            )
            collate_fn = collate_pairs
        
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                                 shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        # Build model
        model = build_model(config).to(device)
        if config.get("no_pretrained"):
            model = LearnableEmbeddingSiameseGAT(model, vocab_size=50266,
                                                  model_dim=config.get("embedding_dim", 768)).to(device)
        if fold == 0:
            params = count_parameters_wrapper(model)
            print(f"  Model: {type(model).__name__}")
            print(f"  Parameters: {params['trainable_human']}")
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4)
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"]) \
            if config.get("scheduler") == "cosine" and not config.get("no_scheduler") else None
        
        best_f1, best_metrics, patience_ctr = 0, {}, 0
        
        for epoch in range(config["max_epochs"]):
            t0 = time.time()
            train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                       device, has_spec, config.get("gradient_clip", 1.0))
            test_m = evaluate(model, test_loader, criterion, device, has_spec)
            if scheduler:
                scheduler.step()
            dt = time.time() - t0
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Ep {epoch+1:3d}: tr_loss={train_m['loss']:.4f} tr_acc={train_m['accuracy']:.3f} | "
                      f"te_f1={test_m['f1_macro']:.3f} te_acc={test_m['accuracy']:.3f} "
                      f"te_auc={test_m['auc']:.3f} [{dt:.1f}s]")
            
            if test_m["f1_macro"] > best_f1:
                best_f1 = test_m["f1_macro"]
                best_metrics = test_m
                patience_ctr = 0
                torch.save(model.state_dict(),
                          os.path.join(output_dir, f"best_model_fold{fold}.pt"))
            else:
                patience_ctr += 1
                if patience_ctr >= config["patience"]:
                    print(f"  Early stop at epoch {epoch+1}")
                    break
        
        print(f"\n  FOLD {fold+1} BEST: F1={best_f1:.4f} Acc={best_metrics['accuracy']:.4f} AUC={best_metrics.get('auc',0):.4f}")
        fold_results.append({
            "fold": fold, "f1_macro": best_f1,
            "accuracy": best_metrics["accuracy"],
            "precision_macro": best_metrics["precision_macro"],
            "recall_macro": best_metrics["recall_macro"],
            "auc": best_metrics.get("auc", 0),
        })
        all_preds.extend(best_metrics["preds"])
        all_labels.extend(best_metrics["labels"])
        
        # Per-dataset breakdown
        test_meta = [graph_data["metadata"][i] for i in test_idx]
        ds_m = defaultdict(lambda: {"p": [], "l": []})
        for p, l, m in zip(best_metrics["preds"], best_metrics["labels"], test_meta):
            ds_m[m["dataset"]]["p"].append(p)
            ds_m[m["dataset"]]["l"].append(l)
        for ds, dm in ds_m.items():
            ds_f1 = f1_score(dm['l'], dm['p'], average='macro')
            print(f"    {ds}: F1={ds_f1:.3f} (n={len(dm['l'])})")
    
    # ── Aggregate ──
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS — {config.get('model_type', 'siamese_gat')}")
    print(f"{'='*60}")
    summary = {}
    for metric in ["f1_macro", "accuracy", "precision_macro", "recall_macro", "auc"]:
        vals = [r[metric] for r in fold_results]
        summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Correct', 'Buggy'], digits=4)}")
    
    # Save results
    results = {
        "config": {k: v for k, v in config.items() if not callable(v)},
        "fold_results": fold_results,
        "summary": summary,
        "total_pairs": len(graph_data["labels"]),
        "n_folds": n_folds,
        "model_type": config.get("model_type"),
        "has_spec": has_spec,
        "fusion_mode": config.get("fusion_mode"),
        "cross_attention": config.get("cross_attention", False),
    }
    
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {results_file}")
    print(f"Log saved to: {log_path}")
    
    # Restore print and close log
    import builtins
    builtins.print = _orig_print
    log_file.close()
    
    return results


# ═══════════════════════════════════════
# K-FOLD (SHARDED — lazy loading)
# ═══════════════════════════════════════

def run_kfold_sharded(index, config, output_dir, device):
    """K-fold CV with lazy shard loading. Only 1 shard in RAM at a time."""
    # Set up file logging
    log_path = os.path.join(output_dir, "train.log")
    log_file = open(log_path, "w")
    _orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print
    def log_print(*args, **kwargs):
        _orig_print(*args, **kwargs)
        kwargs.pop("flush", None)
        _orig_print(*args, **kwargs, file=log_file, flush=True)
    import builtins
    builtins.print = log_print
    
    has_spec = index.get("has_spec", False) and config.get("model_type") != "siamese_gat"
    n_folds = config.get("n_folds", 5)
    labels = np.array(index["labels"])
    metadata = index["metadata"]
    
    groups = get_problem_groups(metadata)
    n_groups = len(set(groups))
    n_unique_labels = len(set(labels.tolist()))
    
    if n_groups >= n_folds:
        splits = list(GroupKFold(n_splits=n_folds).split(labels, labels, groups))
    elif n_unique_labels >= 2:
        splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=config.get("seed", 42)).split(labels, labels))
    else:
        from sklearn.model_selection import KFold
        splits = list(KFold(n_splits=n_folds, shuffle=True,
                            random_state=config.get("seed", 42)).split(labels))
    
    print(f"\n{'='*60}")
    print(f"MODEL: {config.get('model_type', 'siamese_gat')}")
    print(f"FUSION: {config.get('fusion_mode', 'N/A')}")
    print(f"SPEC: {'YES' if has_spec else 'NO'}")
    print(f"CROSS-ATTN: {'YES' if config.get('cross_attention') else 'NO'}")
    print(f"FOLDS: {n_folds}, PAIRS: {len(labels)}")
    print(f"MODE: SHARDED (lazy loading)")
    print(f"{'='*60}\n")
    
    fold_results = []
    all_preds, all_labels = [], []
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'─'*50}")
        print(f"FOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"{'─'*50}")
        
        # Build sharded datasets
        DatasetClass = ShardedSpecPairDataset if has_spec else ShardedPairDataset
        collate_fn = sharded_collate_spec_pairs if has_spec else sharded_collate_pairs
        
        train_ds = DatasetClass(index, list(train_idx),
                                swap_aug=config.get("swap_augmentation", True))
        test_ds = DatasetClass(index, list(test_idx), swap_aug=False)
        
        # Shard-aware samplers (batches grouped by shard = minimal I/O)
        train_sampler = ShardBatchSampler(train_ds, config["batch_size"], shuffle=True)
        test_sampler = ShardBatchSampler(test_ds, config["batch_size"], shuffle=False)
        
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                   collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_ds, batch_sampler=test_sampler,
                                  collate_fn=collate_fn, num_workers=0)
        
        # Build model
        model = build_model(config).to(device)
        if fold == 0:
            params = count_parameters(model)
            print(f"  Model: {type(model).__name__}")
            print(f"  Parameters: {params['trainable_human']}")
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4)
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"]) \
            if config.get("scheduler") == "cosine" and not config.get("no_scheduler") else None
        
        best_f1, best_metrics, patience_ctr = 0, {}, 0
        
        for epoch in range(config["max_epochs"]):
            t0 = time.time()
            train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                       device, has_spec, config.get("gradient_clip", 1.0))
            test_m = evaluate(model, test_loader, criterion, device, has_spec)
            if scheduler:
                scheduler.step()
            dt = time.time() - t0
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Ep {epoch+1:3d}: tr_loss={train_m['loss']:.4f} tr_acc={train_m['accuracy']:.3f} | "
                      f"te_f1={test_m['f1_macro']:.3f} te_acc={test_m['accuracy']:.3f} "
                      f"te_auc={test_m['auc']:.3f} [{dt:.1f}s]")
            
            if test_m["f1_macro"] > best_f1:
                best_f1 = test_m["f1_macro"]
                best_metrics = test_m
                patience_ctr = 0
                torch.save(model.state_dict(),
                          os.path.join(output_dir, f"best_model_fold{fold}.pt"))
            else:
                patience_ctr += 1
                if patience_ctr >= config["patience"]:
                    print(f"  Early stop at epoch {epoch+1}")
                    break
        
        print(f"\n  FOLD {fold+1} BEST: F1={best_f1:.4f} Acc={best_metrics['accuracy']:.4f} AUC={best_metrics.get('auc',0):.4f}")
        fold_results.append({
            "fold": fold, "f1_macro": best_f1,
            "accuracy": best_metrics["accuracy"],
            "precision_macro": best_metrics["precision_macro"],
            "recall_macro": best_metrics["recall_macro"],
            "auc": best_metrics.get("auc", 0),
        })
        all_preds.extend(best_metrics["preds"])
        all_labels.extend(best_metrics["labels"])
        
        # Per-dataset breakdown
        test_meta = [metadata[i] for i in test_idx]
        ds_m = defaultdict(lambda: {"p": [], "l": []})
        for p, l, m in zip(best_metrics["preds"], best_metrics["labels"], test_meta):
            ds_m[m["dataset"]]["p"].append(p)
            ds_m[m["dataset"]]["l"].append(l)
        for ds, dm in ds_m.items():
            ds_f1 = f1_score(dm['l'], dm['p'], average='macro')
            print(f"    {ds}: F1={ds_f1:.3f} (n={len(dm['l'])})")
        
        # Free shard caches between folds
        train_ds.release_cache()
        test_ds.release_cache()
        import gc; gc.collect()
    
    # ── Aggregate ──
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS — {config.get('model_type', 'siamese_gat')}")
    print(f"{'='*60}")
    summary = {}
    for metric in ["f1_macro", "accuracy", "precision_macro", "recall_macro", "auc"]:
        vals = [r[metric] for r in fold_results]
        summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Correct', 'Buggy'], digits=4)}")
    
    results = {
        "config": {k: v for k, v in config.items() if not callable(v)},
        "fold_results": fold_results,
        "summary": summary,
        "total_pairs": len(labels),
        "n_folds": n_folds,
        "model_type": config.get("model_type"),
        "has_spec": has_spec,
        "fusion_mode": config.get("fusion_mode"),
        "cross_attention": config.get("cross_attention", False),
        "mode": "sharded",
    }
    
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {results_file}")
    print(f"Log saved to: {log_path}")
    
    # Restore print and close log
    import builtins
    builtins.print = _orig_print
    log_file.close()
    
    return results


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Siamese GAT (with optional spec)")
    
    # Data — single file, directory of shards, or per-language
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data", default=None,
                           help="Single graph data file (.pt) — loads all into RAM")
    data_group.add_argument("--data-dir", default=None,
                           help="Directory of per-language .pt files — lazy shard loading")
    data_group.add_argument("--lang", default=None,
                           help="Language name (e.g. python, cpp, java) — loads that lang's .pt files from data/graphs/")
    parser.add_argument("--graph-dir", default="data/graphs",
                       help="Directory containing .pt files (used with --lang)")
    parser.add_argument("--suffix", default=None,
                       help="Edge ablation suffix (dfg_only, cfg_only, seq_only)")
    parser.add_argument("--edge-filter", default=None,
                       choices=["dfg_only", "cfg_only", "seq_only"],
                       help="Filter edges in memory after loading (no separate files needed)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (auto-generated if not set)")
    
    # Model
    parser.add_argument("--model-type", default="siamese_gat",
                       choices=["siamese_gat", "spec_siamese_gat", "spec_code_only"],
                       help="Model type")
    parser.add_argument("--fusion-mode", default="concat",
                       choices=["concat", "gate"],
                       help="Spec fusion mode (for spec models)")
    parser.add_argument("--cross-attention", action="store_true",
                       help="Use cross-attention fusion")
    parser.add_argument("--shared-spec-encoder", action="store_true",
                       help="Share weights between code and spec encoders")
    
    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pooling", default="attention",
                       choices=["attention", "mean", "max"])
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-swap-augment", action="store_true")
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true",
                       help="Replace GraphCodeBERT embeddings with random vectors (ablation: is signal in BERT or graph?)")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device = f"cuda:{n-1}" if n > 0 else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    set_seed(args.seed)
    
    # Output directory — defer auto-generation until mode is known
    # (--lang mode generates its own dir name below)
    if args.output_dir is None and not args.lang:
        args.output_dir = os.path.join(
            "./outputs",
            f"{args.model_type}_{args.fusion_mode}"
            + ("_ca" if args.cross_attention else "")
            + ("_shared" if args.shared_spec_encoder else "")
        )
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Config
    config = {
        "model_type": args.model_type,
        "fusion_mode": args.fusion_mode,
        "cross_attention": args.cross_attention,
        "shared_spec_encoder": args.shared_spec_encoder,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "attention_dropout": 0.1,
        "mlp_hidden": args.hidden_dim,
        "mlp_dropout": args.dropout,
        "num_classes": 2,
        "pooling": args.pooling,
        "embedding_dim": 768,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "gradient_clip": 1.0,
        "scheduler": "cosine",
        "swap_augmentation": not args.no_swap_augment,
        "no_pretrained": args.no_pretrained,
        "no_scheduler": getattr(args, "no_scheduler", False),
    }
    
    # ── PER-LANGUAGE MODE ──
    if args.lang:
        import glob, gc
        lang = args.lang.lower()
        gdir = args.graph_dir
        
        # Find matching files (codenet + humanevalfix for this language)
        sfx = f"_{args.suffix}" if args.suffix else ""
        patterns = [
            os.path.join(gdir, f"graph_data_codenet_{lang}{sfx}.pt"),
            os.path.join(gdir, f"graph_data_humanevalfix_{lang}{sfx}.pt"),
        ]
        files = [f for f in patterns if os.path.exists(f)]
        if not files:
            print(f"ERROR: No graph files found for language '{lang}' in {gdir}")
            print(f"  Looked for: {patterns}")
            return
        
        # Merge files into one dict
        graph_data = {"graph_a": [], "graph_b": [], "labels": [], "metadata": []}
        for fpath in files:
            print(f"Loading: {os.path.basename(fpath)}...", end=" ", flush=True)
            d = torch.load(fpath, weights_only=False, map_location="cpu")
            n = len(d["labels"])
            graph_data["graph_a"].extend(d["graph_a"])
            graph_data["graph_b"].extend(d["graph_b"])
            graph_data["labels"].extend(d["labels"])
            graph_data["metadata"].extend(d["metadata"])
            # Carry spec graphs if present
            if "graph_spec" in d:
                if "graph_spec" not in graph_data:
                    graph_data["graph_spec"] = []
                graph_data["graph_spec"].extend(d["graph_spec"])
            print(f"{n} pairs")
            del d; gc.collect()
        
        # Apply edge filtering in memory (avoids loading separate ablation files)
        EDGE_FILTERS = {
            "dfg_only": {0, 1, 2},   # remove CFG (type 3)
            "cfg_only": {0, 3},       # remove DFG (types 1,2)
            "seq_only": {0},           # sequential only
        }
        ef = args.edge_filter or args.suffix  # --edge-filter preferred, --suffix as fallback
        if ef and ef in EDGE_FILTERS:
            keep = EDGE_FILTERS[ef]
            print(f"Filtering edges: {ef} (keep types {keep})...", end=" ", flush=True)
            from torch_geometric.data import Data as _Data
            def _filter(g):
                mask = torch.zeros(len(g.edge_type), dtype=torch.bool)
                for t in keep:
                    mask |= (g.edge_type == t)
                return _Data(x=g.x, edge_index=g.edge_index[:, mask],
                             edge_type=g.edge_type[mask], num_nodes=g.num_nodes,
                             node_types=g.node_types, num_code_tokens=g.num_code_tokens,
                             num_dfg_nodes=g.num_dfg_nodes,
                             num_cfg_edges=(g.edge_type[mask] == 3).sum().item())
            for key in ["graph_a", "graph_b"]:
                graph_data[key] = [_filter(g) for g in graph_data[key]]
            print("done")
        
        # Validate token_ids exist for no-pretrained mode
        if args.no_pretrained:
            g0 = graph_data["graph_a"][0]
            if not hasattr(g0, "token_ids"):
                print("ERROR: --no-pretrained requires token_ids in graph data.")
                print("Run: python scripts/enrich_token_ids.py --lang", lang)
                return
            print(f"Learnable embeddings mode: token_ids found (vocab_size={g0.token_ids.max().item() + 1})")
        
        n = len(graph_data["labels"])
        n0 = graph_data["labels"].count(0)
        n1 = graph_data["labels"].count(1)
        print(f"Total: {n} pairs for {lang} (label-0={n0}, label-1={n1})")
        
        has_spec = "graph_spec" in graph_data
        print(f"Spec graphs: {'YES' if has_spec else 'NO'}")
        
        if args.model_type in ("spec_siamese_gat", "spec_code_only") and not has_spec:
            # Try loading spec shard files
            spec_files = [
                os.path.join(gdir, f"spec_data_codenet_{lang}.pt"),
                os.path.join(gdir, f"spec_data_humanevalfix_{lang}.pt"),
            ]
            spec_files = [f for f in spec_files if os.path.exists(f)]
            if spec_files:
                graph_data["graph_spec"] = []
                for fpath in spec_files:
                    print(f"Loading spec: {os.path.basename(fpath)}...", end=" ", flush=True)
                    d = torch.load(fpath, weights_only=False, map_location="cpu")
                    graph_data["graph_spec"].extend(d["graph_spec"])
                    print(f"{len(d['graph_spec'])} specs")
                    del d; gc.collect()
                has_spec = True
                print(f"Spec graphs loaded: {len(graph_data['graph_spec'])}")
            else:
                print("WARNING: No spec shards found, falling back to siamese_gat")
                config["model_type"] = "siamese_gat"
        
        # Auto output dir with language and optional suffix/edge-filter
        if args.output_dir is None:
            dir_name = f"{config['model_type']}_{lang}"
            ef = args.edge_filter or args.suffix
            if ef:
                dir_name += f"_{ef}"
            if args.no_pretrained:
                dir_name += "_no_pretrained"
            if config['model_type'] != 'siamese_gat':
                dir_name += f"_{config['fusion_mode']}"
            if config.get('cross_attention'):
                dir_name += "_ca"
            args.output_dir = os.path.join("./outputs", dir_name)
            os.makedirs(args.output_dir, exist_ok=True)
        
        run_kfold(graph_data, config, args.output_dir, device)
    
    # ── SHARDED MODE ──
    elif args.data_dir:
        print(f"Data dir: {args.data_dir}")
        shard_files = find_shard_files(args.data_dir)
        print(f"Found {len(shard_files)} shard files")
        
        # Check for spec shards
        spec_files = None
        if args.model_type in ("spec_siamese_gat", "spec_code_only"):
            raw_spec = find_spec_shard_files(args.data_dir, shard_files)
            spec_files = [f for f in raw_spec if f is not None]
            if spec_files:
                print(f"Found {len(spec_files)} spec shard files")
            else:
                print("WARNING: No spec shards found, falling back to siamese_gat")
                config["model_type"] = "siamese_gat"
                spec_files = None
        
        index = build_index(shard_files, spec_files)
        run_kfold_sharded(index, config, args.output_dir, device)
    
    # ── SINGLE FILE MODE ──
    else:
        print(f"Loading: {args.data}")
        graph_data = torch.load(args.data, weights_only=False)
        print(f"Loaded {len(graph_data['labels'])} pairs")
        
        has_spec = "graph_spec" in graph_data
        print(f"Spec graphs: {'YES' if has_spec else 'NO'}")
        
        if args.model_type in ("spec_siamese_gat", "spec_code_only") and not has_spec:
            print("\nWARNING: Spec model requested but data has no graph_spec.")
            print("Falling back to siamese_gat.")
            config["model_type"] = "siamese_gat"
        
        run_kfold(graph_data, config, args.output_dir, device)


if __name__ == "__main__":
    main()