"""
Fine-tuned GraphCodeBERT Pair Classifier — Baseline for SiameseGAT comparison.

This is the PROPER baseline. Not frozen CLS cosine — full fine-tuning
on your buggy/correct pairs. Uses the same 5-fold CV as your main experiments.

Key design:
  - Takes (graph_a, graph_b) pair
  - Extracts CLS token from each via GraphCodeBERT
  - Concatenates [CLS_a, CLS_b, |CLS_a - CLS_b|, CLS_a * CLS_b]
  - MLP classifier head
  - Fine-tuned end-to-end on your pairs

Usage:
    python scripts/train_gcbert_baseline.py --lang python --device cuda:1
    python scripts/train_gcbert_baseline.py --lang java --device cuda:1
    python scripts/train_gcbert_baseline.py --lang cpp --device cuda:1

Requirements:
    pip install transformers
"""

import argparse, json, os, random, sys, time, gc
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# ═══════════════════════════════════════
# MODEL
# ═══════════════════════════════════════

class GCBERTPairClassifier(nn.Module):
    """
    Fine-tuned GraphCodeBERT for pair classification.
    
    Input:  Two programs (as token_ids tensors)
    Output: logits for [correct_a, correct_b] binary classification
    
    Architecture:
        GCBERT(program_a) → CLS_a (768)
        GCBERT(program_b) → CLS_b (768)
        features = [CLS_a, CLS_b, |CLS_a - CLS_b|, CLS_a * CLS_b]  (768*4)
        MLP(features) → 2 logits
    """
    
    def __init__(self, model_name="microsoft/graphcodebert-base",
                 hidden_dim=256, dropout=0.3, freeze_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Optionally freeze bottom N transformer layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
        
        enc_dim = self.encoder.config.hidden_size  # 768
        fusion_dim = enc_dim * 4  # concat + diff + product
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
    
    def encode(self, input_ids, attention_mask):
        """Get CLS token embedding for a program."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS token
    
    def forward(self, input_ids_a, attention_mask_a,
                input_ids_b, attention_mask_b):
        cls_a = self.encode(input_ids_a, attention_mask_a)
        cls_b = self.encode(input_ids_b, attention_mask_b)
        
        # Symmetric feature fusion
        features = torch.cat([
            cls_a,
            cls_b,
            torch.abs(cls_a - cls_b),
            cls_a * cls_b,
        ], dim=-1)
        
        logits = self.classifier(features)
        return {"logits": logits, "cls_a": cls_a, "cls_b": cls_b}


# ═══════════════════════════════════════
# DATASET
# ═══════════════════════════════════════

class TokenPairDataset(Dataset):
    """
    Extracts token_ids from graph nodes and prepares them for BERT.
    
    Each graph has node features (data.x = GCBERT embeddings) and
    optionally token_ids (data.token_ids = raw token IDs).
    
    If token_ids not available, falls back to using the first token
    of each node's embedding as a proxy (less ideal).
    """
    
    def __init__(self, graphs_a, graphs_b, labels, tokenizer,
                 max_length=512, swap_aug=False):
        self.graphs_a = graphs_a
        self.graphs_b = graphs_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.swap_aug = swap_aug
        
        # Pre-extract token sequences
        print("  Extracting token sequences...", flush=True)
        self.tokens_a = [self._extract_tokens(g) for g in graphs_a]
        self.tokens_b = [self._extract_tokens(g) for g in graphs_b]
        print(f"  Done. Avg len_a={np.mean([len(t) for t in self.tokens_a]):.0f} "
              f"avg len_b={np.mean([len(t) for t in self.tokens_b]):.0f}")
    
    def _extract_tokens(self, graph):
        """Extract token_ids from graph node data."""
        if hasattr(graph, 'token_ids') and graph.token_ids is not None:
            # Use stored token IDs (best case)
            ids = graph.token_ids
            if ids.dim() > 1:
                ids = ids.squeeze(-1)
            # Only take code token nodes (not DFG/CFG virtual nodes)
            if hasattr(graph, 'num_code_tokens'):
                ids = ids[:graph.num_code_tokens]
            ids = ids.clamp(0, 50264)
            return ids.tolist()
        else:
            # Fallback: we can't re-tokenize without source code
            # Return a placeholder — model will learn nothing useful
            return [1]  # [UNK] token
    
    def _prepare_input(self, token_list):
        """Prepare token_ids for BERT with CLS/SEP and truncation."""
        # Add CLS (0) and SEP (2) tokens, truncate to max_length-2
        max_content = self.max_length - 2
        tokens = token_list[:max_content]
        
        input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long))
    
    def __len__(self):
        return len(self.labels) * (2 if self.swap_aug else 1)
    
    def __getitem__(self, idx):
        if self.swap_aug and idx >= len(self.labels):
            # Swapped pair: B is "correct", A is "buggy"
            i = idx - len(self.labels)
            ids_a, mask_a = self._prepare_input(self.tokens_b[i])
            ids_b, mask_b = self._prepare_input(self.tokens_a[i])
            label = 1 - self.labels[i]
        else:
            ids_a, mask_a = self._prepare_input(self.tokens_a[idx])
            ids_b, mask_b = self._prepare_input(self.tokens_b[idx])
            label = self.labels[idx]
        
        return ids_a, mask_a, ids_b, mask_b, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    ids_a, mask_a, ids_b, mask_b, labels = zip(*batch)
    return (torch.stack(ids_a), torch.stack(mask_a),
            torch.stack(ids_b), torch.stack(mask_b),
            torch.stack(labels))


# ═══════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for ids_a, mask_a, ids_b, mask_b, labels in loader:
        ids_a, mask_a = ids_a.to(device), mask_a.to(device)
        ids_b, mask_b = ids_b.to(device), mask_b.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        out = model(ids_a, mask_a, ids_b, mask_b)
        loss = criterion(out["logits"], labels)
        loss.backward()
        
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * labels.size(0)
        correct += (out["logits"].argmax(-1) == labels).sum().item()
        total += labels.size(0)
    
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all, probs_all = 0, [], [], []
    
    for ids_a, mask_a, ids_b, mask_b, labels in loader:
        ids_a, mask_a = ids_a.to(device), mask_a.to(device)
        ids_b, mask_b = ids_b.to(device), mask_b.to(device)
        labels = labels.to(device)
        
        out = model(ids_a, mask_a, ids_b, mask_b)
        total_loss += criterion(out["logits"], labels).item() * labels.size(0)
        preds_all.extend(out["logits"].argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
        probs_all.extend(F.softmax(out["logits"], dim=-1)[:, 1].cpu().tolist())
    
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
# K-FOLD
# ═══════════════════════════════════════

def get_problem_groups(metadata):
    """Group by PROBLEM id (leakage-safe): 'codenet_p00000_s123' -> 'p00000'."""
    def pid(m):
        p = m.get("problem_id")
        if p:
            return p
        parts = m.get("pair_id", "unk").split("_")
        if len(parts) >= 3 and parts[1].startswith("p") and parts[1][1:].isdigit():
            return parts[1]
        return m.get("pair_id", "unk")
    groups = [pid(m) for m in metadata]
    unique = sorted(set(groups))
    gmap = {g: i for i, g in enumerate(unique)}
    return np.array([gmap[g] for g in groups])


def run_kfold(graph_data, args, output_dir, device):
    model_name = "microsoft/graphcodebert-base"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    labels = np.array(graph_data["labels"])
    metadata = graph_data["metadata"]
    groups = get_problem_groups(metadata)
    n_folds = args.n_folds
    
    n_groups = len(set(groups.tolist()))
    if n_groups >= n_folds:
        splits = list(GroupKFold(n_splits=n_folds).split(labels, labels, groups))
    else:
        splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=42).split(labels, labels))
    
    print(f"\n{'='*60}")
    print(f"MODEL: gcbert_pair_classifier (fine-tuned)")
    print(f"FOLDS: {n_folds}, PAIRS: {len(labels)}")
    print(f"MAX_LENGTH: {args.max_length}, BATCH: {args.batch_size}")
    print(f"{'='*60}\n")
    
    fold_results = []
    all_preds, all_labels = [], []
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'─'*50}")
        print(f"FOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"{'─'*50}")
        
        train_ds = TokenPairDataset(
            [graph_data["graph_a"][i] for i in train_idx],
            [graph_data["graph_b"][i] for i in train_idx],
            [graph_data["labels"][i] for i in train_idx],
            tokenizer, max_length=args.max_length,
            swap_aug=not args.no_swap_aug,
        )
        test_ds = TokenPairDataset(
            [graph_data["graph_a"][i] for i in test_idx],
            [graph_data["graph_b"][i] for i in test_idx],
            [graph_data["labels"][i] for i in test_idx],
            tokenizer, max_length=args.max_length,
            swap_aug=False,
        )
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_fn,
                                 num_workers=2, pin_memory=True)
        
        # Build fresh model each fold
        model = GCBERTPairClassifier(
            model_name=model_name,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            freeze_layers=args.freeze_layers,
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold == 0:
            print(f"  Parameters: {n_params:,}")
        
        # Optimizer: lower LR for BERT, higher for classifier head
        bert_params = list(model.encoder.parameters())
        head_params = list(model.classifier.parameters())
        optimizer = torch.optim.AdamW([
            {"params": bert_params, "lr": args.bert_lr},
            {"params": head_params, "lr": args.head_lr},
        ], weight_decay=1e-4)
        
        # Warmup + cosine schedule
        total_steps = len(train_loader) * args.max_epochs
        warmup_steps = int(0.1 * total_steps)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.CrossEntropyLoss()
        
        best_f1, best_metrics, patience_ctr = 0, {}, 0
        
        for epoch in range(args.max_epochs):
            t0 = time.time()
            train_m = train_epoch(model, train_loader, optimizer, scheduler,
                                   criterion, device, args.grad_clip)
            test_m = evaluate(model, test_loader, criterion, device)
            dt = time.time() - t0
            
            print(f"  Ep {epoch+1:3d}: tr_loss={train_m['loss']:.4f} "
                  f"tr_acc={train_m['accuracy']:.3f} | "
                  f"te_f1={test_m['f1_macro']:.3f} "
                  f"te_acc={test_m['accuracy']:.3f} "
                  f"te_auc={test_m['auc']:.3f} [{dt:.1f}s]")
            
            if test_m["f1_macro"] > best_f1:
                best_f1 = test_m["f1_macro"]
                best_metrics = test_m
                patience_ctr = 0
                torch.save(model.state_dict(),
                          os.path.join(output_dir, f"gcbert_fold{fold}.pt"))
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break
        
        print(f"\n  FOLD {fold+1} BEST: "
              f"F1={best_f1:.4f} "
              f"Acc={best_metrics['accuracy']:.4f} "
              f"AUC={best_metrics.get('auc',0):.4f}")
        
        fold_results.append({
            "fold": fold,
            "f1_macro": best_f1,
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
            ds_f1 = f1_score(dm["l"], dm["p"], average="macro")
            print(f"    {ds}: F1={ds_f1:.3f} (n={len(dm['l'])})")
        
        # Free GPU memory between folds
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Aggregate
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS — gcbert_pair_classifier")
    print(f"{'='*60}")
    summary = {}
    for metric in ["f1_macro", "accuracy", "precision_macro", "recall_macro", "auc"]:
        vals = [r[metric] for r in fold_results]
        summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Correct', 'Buggy'], digits=4)}")
    
    results = {
        "model_type": "gcbert_pair_classifier",
        "language": args.lang,
        "total_pairs": len(graph_data["labels"]),
        "fold_results": fold_results,
        "summary": summary,
        "config": vars(args),
    }
    
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {results_file}")
    return results


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True,
                        help="Language: python, cpp, java, c, ruby, javascript")
    parser.add_argument("--graph-dir", default="data/graphs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:1")
    
    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Freeze bottom N BERT layers (0=train all)")
    
    # Training
    parser.add_argument("--bert-lr", type=float, default=2e-5,
                        help="LR for BERT encoder (keep small)")
    parser.add_argument("--head-lr", type=float, default=1e-4,
                        help="LR for classifier head")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=10,
                        help="Fine-tuning needs fewer epochs than training from scratch")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-swap-aug", action="store_true")
    
    args = parser.parse_args()
    
    # Output dir
    if args.output_dir is None:
        args.output_dir = f"./outputs/gcbert_finetuned_{args.lang}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = args.device
    print(f"Device: {device}")
    
    # Load data (same as your train_spec.py)
    lang = args.lang.lower()
    gdir = args.graph_dir
    patterns = [
        os.path.join(gdir, f"graph_data_codenet_{lang}.pt"),
        os.path.join(gdir, f"graph_data_humanevalfix_{lang}.pt"),
    ]
    files = [f for f in patterns if os.path.exists(f)]
    if not files:
        print(f"ERROR: No files found for {lang} in {gdir}")
        return
    
    graph_data = {"graph_a": [], "graph_b": [], "labels": [], "metadata": []}
    for fpath in files:
        print(f"Loading: {os.path.basename(fpath)}...", end=" ", flush=True)
        d = torch.load(fpath, weights_only=False, map_location="cpu")
        graph_data["graph_a"].extend(d["graph_a"])
        graph_data["graph_b"].extend(d["graph_b"])
        graph_data["labels"].extend(d["labels"])
        graph_data["metadata"].extend(d["metadata"])
        print(f"{len(d['labels'])} pairs")
        del d; gc.collect()
    
    n = len(graph_data["labels"])
    print(f"Total: {n} pairs for {lang}")
    
    # Check token_ids availability
    g0 = graph_data["graph_a"][0]
    if not hasattr(g0, "token_ids"):
        print("\nWARNING: token_ids not found in graph data.")
        print("The model will use a [UNK] fallback — results will be meaningless.")
        print("Run: python scripts/enrich_token_ids.py --lang", lang)
        print("Then re-run this script.\n")
    else:
        print(f"token_ids found. Vocab size: {g0.token_ids.max().item() + 1}")
    
    run_kfold(graph_data, args, args.output_dir, device)


if __name__ == "__main__":
    main()