"""
Train Spec-Grounded GAT for code verification.

Supports:
  - 5-fold cross-validation (split by problem_id)
  - Full model (code graph + IO spec + cross-attention)
  - Ablations: --no-spec, --no-cross-attn, --edge-filter seq_only/dfg_only/cfg_only
  - Edge ablation to prove graph structure matters

Usage:
    # Full model
    python scripts/train_exec.py --data data/exec/exec_dataset_python.pt --device cuda:1

    # Code-only (no spec) ablation
    python scripts/train_exec.py --data data/exec/exec_dataset_python.pt --no-spec --device cuda:1

    # Seq-only edges (prove DFG/CFG matter)
    python scripts/train_exec.py --data data/exec/exec_dataset_python.pt --edge-filter seq_only --device cuda:1
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model_exec import SpecGroundedGAT, CodeOnlyGAT, count_parameters


# ═══════════════════════════════════════
# DATASET
# ═══════════════════════════════════════

class ExecDataset(Dataset):
    def __init__(self, graphs, io_specs, labels, edge_filter=None):
        self.graphs = graphs
        self.io_specs = io_specs
        self.labels = labels
        self.edge_filter = edge_filter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        g = self.graphs[idx]

        # Edge filtering for ablation
        if self.edge_filter and hasattr(g, 'edge_type'):
            mask = self._edge_mask(g.edge_type)
            g = g.clone()
            g.edge_index = g.edge_index[:, mask]
            g.edge_type = g.edge_type[mask]
            if g.edge_index.shape[1] == 0:
                g.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                g.edge_type = torch.tensor([0], dtype=torch.long)

        return g, self.io_specs[idx], self.labels[idx]

    def _edge_mask(self, edge_types):
        if self.edge_filter == "seq_only":
            return edge_types == 0
        elif self.edge_filter == "dfg_only":
            return (edge_types == 0) | (edge_types == 1) | (edge_types == 2)
        elif self.edge_filter == "cfg_only":
            return (edge_types == 0) | (edge_types == 3)
        elif self.edge_filter == "no_seq":
            return edge_types != 0
        return torch.ones(len(edge_types), dtype=torch.bool)


def collate_exec(batch):
    graphs, specs, labels = zip(*batch)
    batched_graph = Batch.from_data_list(list(graphs))
    specs = torch.stack(specs)
    labels = torch.tensor(labels, dtype=torch.long)
    return batched_graph, specs, labels


# ═══════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_n = 0, 0, 0

    for batch_graph, batch_spec, batch_labels in loader:
        batch_graph = batch_graph.to(device)
        batch_spec = batch_spec.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        out = model(batch_graph, batch_spec)
        loss = criterion(out["logits"], batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_labels)
        total_correct += (out["logits"].argmax(1) == batch_labels).sum().item()
        total_n += len(batch_labels)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch_graph, batch_spec, batch_labels in loader:
        batch_graph = batch_graph.to(device)
        batch_spec = batch_spec.to(device)

        out = model(batch_graph, batch_spec)
        probs = torch.softmax(out["logits"], dim=1)
        preds = out["logits"].argmax(1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_labels.tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return {"f1": f1, "accuracy": acc, "auc": auc}


# ═══════════════════════════════════════
# K-FOLD CV
# ═══════════════════════════════════════

def run_kfold(dataset, config, output_dir, device):
    n_folds = config["n_folds"]
    graphs = dataset["graphs"]
    io_specs = dataset["io_specs"]
    labels = dataset["labels"]
    metadata = dataset["metadata"]

    # Group by problem_id for proper CV
    problem_ids = [m.get("problem_id", f"p_{i}") for i, m in enumerate(metadata)]
    unique_problems = sorted(set(problem_ids))
    problem_to_group = {p: i for i, p in enumerate(unique_problems)}
    groups = [problem_to_group[pid] for pid in problem_ids]

    labels_np = np.array(labels)
    groups_np = np.array(groups)

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_results = []

    print(f"\n{'='*60}")
    print(f"MODEL: {config['model_type']}")
    print(f"EDGE FILTER: {config.get('edge_filter', 'full')}")
    print(f"SPEC: {'YES' if not config.get('no_spec') else 'NO'}")
    print(f"CROSS-ATTN: {'YES' if config.get('use_cross_attention') else 'NO'}")
    print(f"FOLDS: {n_folds}, SAMPLES: {len(labels)}")
    print(f"{'='*60}")

    for fold, (train_idx, test_idx) in enumerate(splitter.split(labels_np, labels_np, groups_np)):
        print(f"\n{'─'*50}")
        print(f"FOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"{'─'*50}")

        edge_filter = config.get("edge_filter", None)
        train_ds = ExecDataset(
            [graphs[i] for i in train_idx],
            [io_specs[i] for i in train_idx],
            [labels[i] for i in train_idx],
            edge_filter,
        )
        test_ds = ExecDataset(
            [graphs[i] for i in test_idx],
            [io_specs[i] for i in test_idx],
            [labels[i] for i in test_idx],
            edge_filter,
        )

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                                  shuffle=True, collate_fn=collate_exec, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                                 shuffle=False, collate_fn=collate_exec, num_workers=0)

        # Build model
        if config.get("no_spec"):
            model = CodeOnlyGAT(
                vocab_size=config["vocab_size"],
                embed_dim=config["embed_dim"],
                gat_hidden=config["gat_hidden"],
                gat_heads=config["gat_heads"],
                gat_layers=config["gat_layers"],
                dropout=config["dropout"],
            ).to(device)
        else:
            model = SpecGroundedGAT(
                vocab_size=config["vocab_size"],
                embed_dim=config["embed_dim"],
                gat_hidden=config["gat_hidden"],
                gat_heads=config["gat_heads"],
                gat_layers=config["gat_layers"],
                spec_hidden=config["gat_hidden"],
                spec_layers=config.get("spec_layers", 2),
                spec_heads=config.get("spec_heads", 4),
                dropout=config["dropout"],
                use_cross_attention=config.get("use_cross_attention", True),
            ).to(device)

        if fold == 0:
            params = count_parameters(model)
            print(f"  Model: {type(model).__name__}")
            print(f"  Parameters: {params['trainable_human']}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                     weight_decay=config.get("weight_decay", 1e-4))
        criterion = nn.CrossEntropyLoss()

        best_f1, patience_ctr, best_metrics = 0, 0, {}
        max_epochs = config["max_epochs"]
        patience = config["patience"]

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            te_metrics = evaluate(model, test_loader, device)
            elapsed = time.time() - t0

            if epoch <= 3 or epoch % 5 == 0 or epoch == max_epochs:
                print(f"  Ep {epoch:3d}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                      f"te_f1={te_metrics['f1']:.3f} te_acc={te_metrics['accuracy']:.3f} "
                      f"te_auc={te_metrics['auc']:.3f} [{elapsed:.1f}s]")

            if te_metrics["f1"] > best_f1:
                best_f1 = te_metrics["f1"]
                best_metrics = te_metrics.copy()
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        print(f"  FOLD {fold+1} BEST: F1={best_metrics['f1']:.4f} "
              f"Acc={best_metrics['accuracy']:.4f} AUC={best_metrics['auc']:.4f}")
        all_results.append(best_metrics)

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS (mean ± std across folds)")
    print(f"{'='*60}")
    for metric in ["f1", "accuracy", "auc"]:
        values = [r[metric] for r in all_results]
        print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "config": {k: v for k, v in config.items() if not callable(v)},
        "folds": all_results,
        "mean_f1": float(np.mean([r["f1"] for r in all_results])),
        "std_f1": float(np.std([r["f1"] for r in all_results])),
        "mean_acc": float(np.mean([r["accuracy"] for r in all_results])),
        "mean_auc": float(np.mean([r["auc"] for r in all_results])),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")

    return summary


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/exec/exec_dataset_python.pt")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:1")
    # Model
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--gat-hidden", type=int, default=256)
    ap.add_argument("--gat-heads", type=int, default=4)
    ap.add_argument("--gat-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    # Training
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--n-folds", type=int, default=5)
    # Ablations
    ap.add_argument("--no-spec", action="store_true", help="Code-only, no IO spec")
    ap.add_argument("--no-cross-attn", action="store_true", help="Concat fusion instead of cross-attn")
    ap.add_argument("--edge-filter", type=str, default=None,
                    choices=["seq_only", "dfg_only", "cfg_only", "no_seq"],
                    help="Filter edge types for ablation")
    args = ap.parse_args()

    # Load dataset
    print(f"Loading {args.data}...")
    dataset = torch.load(args.data, weights_only=False, map_location="cpu")
    n = len(dataset["labels"])
    n1 = sum(dataset["labels"])
    print(f"Samples: {n} (accepted={n1}, wrong={n-n1})")
    print(f"Vocab: {dataset['vocab_size']}")

    # Config
    config = {
        "model_type": "code_only" if args.no_spec else "spec_grounded",
        "vocab_size": dataset["vocab_size"],
        "embed_dim": args.embed_dim,
        "gat_hidden": args.gat_hidden,
        "gat_heads": args.gat_heads,
        "gat_layers": args.gat_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "n_folds": args.n_folds,
        "no_spec": args.no_spec,
        "use_cross_attention": not args.no_cross_attn and not args.no_spec,
        "edge_filter": args.edge_filter,
    }

    # Output dir
    if args.output_dir is None:
        parts = ["exec"]
        if args.no_spec:
            parts.append("no_spec")
        if args.edge_filter:
            parts.append(args.edge_filter)
        if args.no_cross_attn:
            parts.append("no_xattn")
        args.output_dir = os.path.join("outputs", "_".join(parts))

    device = args.device
    print(f"Device: {device}")

    run_kfold(dataset, config, args.output_dir, device)


if __name__ == "__main__":
    main()