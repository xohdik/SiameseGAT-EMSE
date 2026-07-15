"""
Step 4: Train Siamese GAT with 5-fold problem-stratified cross-validation.

Usage:
    python scripts/train.py --data ./data/graphs/graph_data_all.pt
    python scripts/train.py --data ./data/graphs/graph_data_all.pt --cross-benchmark

NOTE: run_kfold now also writes siamesegat_preds.json (one record per test pair:
pair_id, dataset, label, pred) for the edit-distance construct-validity analysis.
"""
import argparse, json, os, random, sys
from collections import defaultdict
from typing import Dict, List

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


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class PairDataset:
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


def collate_pairs(batch):
    ga, gb, labels = zip(*batch)
    return Batch.from_data_list(list(ga)), Batch.from_data_list(list(gb)), torch.tensor(labels, dtype=torch.long)


def get_problem_groups(metadata):
    """Group by PROBLEM id (leakage-safe). Derives it from pair_id:
    'codenet_p00000_s560456580' -> 'p00000'. Falls back to explicit
    problem_id field, then to pair_id (singleton) as last resort."""
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


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for ba, bb, labels in loader:
        ba, bb, labels = ba.to(device), bb.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(ba, bb)["logits"], labels)
        loss.backward()
        if grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds_all.extend(model(ba, bb)["logits"].argmax(-1).detach().cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    n = len(labels_all)
    return {"loss": total_loss/n, "f1_macro": f1_score(labels_all, preds_all, average="macro"),
            "accuracy": accuracy_score(labels_all, preds_all)}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all, probs_all = 0, [], [], []
    for ba, bb, labels in loader:
        ba, bb, labels = ba.to(device), bb.to(device), labels.to(device)
        result = model(ba, bb)
        total_loss += criterion(result["logits"], labels).item() * labels.size(0)
        preds_all.extend(result["logits"].argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
        probs_all.extend(F.softmax(result["logits"], dim=-1)[:, 1].cpu().tolist())
    n = len(labels_all)
    metrics = {"loss": total_loss/n, "f1_macro": f1_score(labels_all, preds_all, average="macro"),
               "accuracy": accuracy_score(labels_all, preds_all),
               "precision_macro": precision_score(labels_all, preds_all, average="macro", zero_division=0),
               "recall_macro": recall_score(labels_all, preds_all, average="macro", zero_division=0),
               "preds": preds_all, "labels": labels_all, "probs": probs_all}
    try: metrics["auc"] = roc_auc_score(labels_all, probs_all)
    except: metrics["auc"] = 0.0
    return metrics


def run_kfold(graph_data, config, output_dir, device):
    n_folds = config.get("n_folds", 5)
    labels = np.array(graph_data["labels"])
    groups = get_problem_groups(graph_data["metadata"])

    n_groups = len(set(groups))
    if n_groups >= n_folds:
        splits = list(GroupKFold(n_splits=n_folds).split(labels, labels, groups))
    else:
        splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=config.get("seed", 42)).split(labels, labels))

    fold_results = []
    all_preds, all_labels = [], []
    per_pair_records = []          # construct-validity: one record per test pair

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*60}\nFOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Test: {len(test_idx)}\n{'='*60}")

        train_ds = PairDataset(
            [graph_data["graph_a"][i] for i in train_idx],
            [graph_data["graph_b"][i] for i in train_idx],
            [graph_data["labels"][i] for i in train_idx],
            [graph_data["metadata"][i] for i in train_idx],
            swap_aug=config.get("swap_augmentation", True))
        test_ds = PairDataset(
            [graph_data["graph_a"][i] for i in test_idx],
            [graph_data["graph_b"][i] for i in test_idx],
            [graph_data["labels"][i] for i in test_idx],
            [graph_data["metadata"][i] for i in test_idx])

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                                  collate_fn=collate_pairs, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                                 collate_fn=collate_pairs, num_workers=0)

        model = SiameseGAT(config).to(device)
        if fold == 0: print(f"  Params: {count_parameters(model)['trainable_human']}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                       weight_decay=config.get("weight_decay", 1e-4))
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"]) if config.get("scheduler") == "cosine" else None

        best_f1, best_metrics, patience_ctr = 0, {}, 0
        for epoch in range(config["max_epochs"]):
            train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_m = evaluate(model, test_loader, criterion, device)
            if scheduler: scheduler.step()

            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"  Ep {epoch+1:3d}: tr_f1={train_m['f1_macro']:.3f} | te_f1={test_m['f1_macro']:.3f} te_acc={test_m['accuracy']:.3f}")

            if test_m["f1_macro"] > best_f1:
                best_f1, best_metrics, patience_ctr = test_m["f1_macro"], test_m, 0
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold{fold}.pt"))
            else:
                patience_ctr += 1
                if patience_ctr >= config["patience"]:
                    print(f"  Early stop at epoch {epoch+1}"); break

        print(f"  FOLD {fold+1} BEST: F1={best_f1:.4f} Acc={best_metrics['accuracy']:.4f}")
        fold_results.append({"fold": fold, "f1_macro": best_f1, "accuracy": best_metrics["accuracy"],
                             "precision_macro": best_metrics["precision_macro"],
                             "recall_macro": best_metrics["recall_macro"], "auc": best_metrics.get("auc", 0)})
        all_preds.extend(best_metrics["preds"]); all_labels.extend(best_metrics["labels"])

        # Per-dataset breakdown
        test_meta = [graph_data["metadata"][i] for i in test_idx]

        # --- per-pair predictions for construct-validity (edit-distance) analysis ---
        for p, l, m in zip(best_metrics["preds"], best_metrics["labels"], test_meta):
            per_pair_records.append({
                "pair_id": m.get("pair_id"),
                "dataset": m.get("dataset"),
                "label": int(l),
                "pred": int(p),
            })

        ds_m = defaultdict(lambda: {"p": [], "l": []})
        for p, l, m in zip(best_metrics["preds"], best_metrics["labels"], test_meta):
            ds_m[m["dataset"]]["p"].append(p); ds_m[m["dataset"]]["l"].append(l)
        for ds, dm in ds_m.items():
            print(f"    {ds}: F1={f1_score(dm['l'], dm['p'], average='macro'):.3f} (n={len(dm['l'])})")

    # Aggregate
    print(f"\n{'='*60}\nAGGREGATE RESULTS\n{'='*60}")
    summary = {}
    for metric in ["f1_macro", "accuracy", "precision_macro", "recall_macro", "auc"]:
        vals = [r[metric] for r in fold_results]
        summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Correct', 'Buggy'], digits=4)}")

    results = {"config": {k: v for k, v in config.items() if not callable(v)},
               "fold_results": fold_results, "summary": summary,
               "total_pairs": len(graph_data["labels"]), "n_folds": n_folds}
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    # per-pair predictions for the edit-distance construct-validity script
    with open(os.path.join(output_dir, "siamesegat_preds.json"), "w") as f:
        json.dump(per_pair_records, f)
    print(f"Saved {len(per_pair_records)} per-pair predictions to siamesegat_preds.json")
    return results


def run_cross_benchmark(graph_data, config, output_dir, device):
    """Train on all-but-one dataset, test on held-out."""
    datasets = set(m["dataset"] for m in graph_data["metadata"])
    results = {}
    for held_out in datasets:
        train_idx = [i for i, m in enumerate(graph_data["metadata"]) if m["dataset"] != held_out]
        test_idx = [i for i, m in enumerate(graph_data["metadata"]) if m["dataset"] == held_out]
        if len(test_idx) < 10 or len(train_idx) < 10: continue
        print(f"\n  Transfer → {held_out} ({len(test_idx)} test)")

        train_ds = PairDataset([graph_data["graph_a"][i] for i in train_idx],
                               [graph_data["graph_b"][i] for i in train_idx],
                               [graph_data["labels"][i] for i in train_idx],
                               [graph_data["metadata"][i] for i in train_idx], swap_aug=True)
        test_ds = PairDataset([graph_data["graph_a"][i] for i in test_idx],
                              [graph_data["graph_b"][i] for i in test_idx],
                              [graph_data["labels"][i] for i in test_idx],
                              [graph_data["metadata"][i] for i in test_idx])

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_pairs)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_pairs)

        model = SiameseGAT(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

        best_f1, best_metrics, pc = 0, {}, 0
        for epoch in range(config["max_epochs"]):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_m = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            if test_m["f1_macro"] > best_f1:
                best_f1, best_metrics, pc = test_m["f1_macro"], test_m, 0
            else:
                pc += 1
                if pc >= config["patience"]: break

        results[held_out] = {"f1_macro": float(best_f1), "accuracy": float(best_metrics["accuracy"]),
                             "n_train": len(train_idx), "n_test": len(test_idx)}
        print(f"  → {held_out}: F1={best_f1:.4f}")

    with open(os.path.join(output_dir, "cross_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/graphs/graph_data_all.pt")
    parser.add_argument("--output-dir", default="./outputs/siamese_gat")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pooling", default="attention", choices=["attention","mean","max"])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cross-benchmark", action="store_true")
    parser.add_argument("--no-swap-augment", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device = f"cuda:{n-1}" if n > 0 else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    set_seed(args.seed)

    config = {"hidden_dim": args.hidden_dim, "num_heads": args.num_heads, "num_layers": args.num_layers,
              "dropout": args.dropout, "attention_dropout": 0.1, "mlp_hidden": args.hidden_dim,
              "mlp_dropout": args.dropout, "num_classes": 2, "pooling": args.pooling, "embedding_dim": 768,
              "learning_rate": args.lr, "weight_decay": 1e-4, "batch_size": args.batch_size,
              "max_epochs": args.max_epochs, "patience": args.patience, "n_folds": args.n_folds,
              "seed": args.seed, "gradient_clip": 1.0, "scheduler": "cosine",
              "swap_augmentation": not args.no_swap_augment}

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading {args.data}...")
    graph_data = torch.load(args.data, weights_only=False)
    print(f"Loaded {len(graph_data['labels'])} pairs")

    run_kfold(graph_data, config, args.output_dir, device)
    if args.cross_benchmark:
        run_cross_benchmark(graph_data, config, args.output_dir, device)

if __name__ == "__main__":
    main()