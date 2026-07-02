"""
Cross-lingual transfer experiments.

Train model on language A, test on language B.
Produces a 6×6 transfer matrix showing how well
code verification transfers across languages.

Usage:
    python scripts/cross_lingual.py --device cuda:1
    python scripts/cross_lingual.py --train-lang python --test-lang java --device cuda:1
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
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SiameseGAT, count_parameters

GRAPH_DIR = "/data/workzone/siamese_gat_journal/data/graphs"
LANGS = ["python", "cpp", "java", "c", "ruby", "javascript"]


# ═══════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════

def load_lang(lang, graph_dir):
    """Load all graph data for one language."""
    graph_data = {"graph_a": [], "graph_b": [], "labels": [], "metadata": []}

    for prefix in ["graph_data_codenet_", "graph_data_humanevalfix_"]:
        fpath = os.path.join(graph_dir, f"{prefix}{lang}.pt")
        if not os.path.exists(fpath):
            continue
        d = torch.load(fpath, weights_only=False, map_location="cpu")
        graph_data["graph_a"].extend(d["graph_a"])
        graph_data["graph_b"].extend(d["graph_b"])
        graph_data["labels"].extend(d["labels"])
        graph_data["metadata"].extend(d["metadata"])
        del d
        gc.collect()

    return graph_data


class PairDataset:
    def __init__(self, ga, gb, labels, swap_aug=False):
        self.ga, self.gb, self.labels = ga, gb, labels
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
    return (Batch.from_data_list(list(ga)),
            Batch.from_data_list(list(gb)),
            torch.tensor(labels, dtype=torch.long))


# ═══════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════

def train_model(train_data, config, device):
    """Train model on one language, return trained model."""
    train_ds = PairDataset(
        train_data["graph_a"], train_data["graph_b"],
        train_data["labels"], swap_aug=True
    )
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                               shuffle=True, collate_fn=collate_pairs, num_workers=0)

    model = SiameseGAT(
        embedding_dim=768,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        attention_dropout=0.1,
        mlp_hidden=config["hidden_dim"],
        mlp_dropout=config["dropout"],
        num_classes=2,
        pooling=config["pooling"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                   weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        total_loss, total = 0, 0

        for ba, bb, labels in train_loader:
            ba, bb, labels = ba.to(device), bb.to(device), labels.to(device)
            result = model(ba, bb)
            loss = criterion(result["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        scheduler.step()
        avg_loss = total_loss / total

        if (epoch + 1) % 10 == 0:
            print(f"    Ep {epoch+1}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                print(f"    Early stop at epoch {epoch+1}")
                break

    return model


@torch.no_grad()
def evaluate_model(model, test_data, device, batch_size=32):
    """Evaluate trained model on test language."""
    model.eval()
    test_ds = PairDataset(test_data["graph_a"], test_data["graph_b"],
                           test_data["labels"], swap_aug=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_pairs, num_workers=0)

    preds_all, labels_all, probs_all = [], [], []

    for ba, bb, labels in test_loader:
        ba, bb, labels = ba.to(device), bb.to(device), labels.to(device)
        result = model(ba, bb)
        preds_all.extend(result["logits"].argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
        probs_all.extend(F.softmax(result["logits"], dim=-1)[:, 1].cpu().tolist())

    f1 = f1_score(labels_all, preds_all, average="macro")
    acc = accuracy_score(labels_all, preds_all)
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except:
        auc = 0.0

    return {"f1_macro": f1, "accuracy": acc, "auc": auc,
            "n_test": len(labels_all)}


# ═══════════════════════════════════════
# TRANSFER EXPERIMENTS
# ═══════════════════════════════════════

def run_transfer_pair(train_lang, test_lang, config, device, graph_dir):
    """Train on train_lang, evaluate on test_lang."""
    print(f"\n  {train_lang} → {test_lang}")

    print(f"    Loading {train_lang} (train)...")
    train_data = load_lang(train_lang, graph_dir)
    n_train = len(train_data["labels"])
    print(f"    {n_train} training pairs")

    print(f"    Training...")
    t0 = time.time()
    model = train_model(train_data, config, device)
    train_time = time.time() - t0
    print(f"    Trained in {train_time:.0f}s")

    # Free training data
    del train_data
    gc.collect()

    print(f"    Loading {test_lang} (test)...")
    test_data = load_lang(test_lang, graph_dir)
    n_test = len(test_data["labels"])
    print(f"    {n_test} test pairs")

    metrics = evaluate_model(model, test_data, device, config["batch_size"])
    print(f"    F1={metrics['f1_macro']:.4f}  Acc={metrics['accuracy']:.4f}  AUC={metrics['auc']:.4f}")

    # Free
    del test_data, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "train_lang": train_lang,
        "test_lang": test_lang,
        "n_train": n_train,
        "n_test": n_test,
        "train_time": train_time,
        **metrics,
    }


def run_full_matrix(config, device, graph_dir, output_dir):
    """Run all N×N transfer experiments."""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    matrix = {}

    for train_lang in LANGS:
        matrix[train_lang] = {}
        for test_lang in LANGS:
            r = run_transfer_pair(train_lang, test_lang, config, device, graph_dir)
            results.append(r)
            matrix[train_lang][test_lang] = r["f1_macro"]

    # Print matrix
    print(f"\n{'='*70}")
    print("CROSS-LINGUAL TRANSFER MATRIX (F1)")
    print(f"{'='*70}")
    header = f"{'Train↓ Test→':<14}" + "".join(f"{l:>10}" for l in LANGS)
    print(header)
    print("─" * len(header))
    for train_lang in LANGS:
        row = f"{train_lang:<14}"
        for test_lang in LANGS:
            f1 = matrix[train_lang][test_lang]
            marker = " *" if train_lang == test_lang else "  "
            row += f"{f1:>8.4f}{marker}"
        print(row)

    # Diagonal = in-language (should be best)
    print(f"\nDiagonal (in-language):")
    for lang in LANGS:
        print(f"  {lang}: F1={matrix[lang][lang]:.4f}")

    # Best transfer per test language
    print(f"\nBest transfer source per test language:")
    for test_lang in LANGS:
        best_train = max(LANGS, key=lambda tl: matrix[tl][test_lang] if tl != test_lang else 0)
        in_lang = matrix[test_lang][test_lang]
        transfer = matrix[best_train][test_lang]
        drop = in_lang - transfer
        print(f"  {test_lang}: best={best_train} (F1={transfer:.4f}, drop={drop:+.4f} from in-lang)")

    # Save
    output = {
        "config": config,
        "results": results,
        "matrix": matrix,
        "languages": LANGS,
    }
    rpath = os.path.join(output_dir, "transfer_matrix.json")
    with open(rpath, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved to: {rpath}")

    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-lang", type=str, default=None,
                   help="Single source language")
    ap.add_argument("--test-lang", type=str, default=None,
                   help="Single target language")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--graph-dir", default=GRAPH_DIR)
    ap.add_argument("--output-dir", default="./outputs/cross_lingual")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--pooling", default="attention")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = {
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "pooling": args.pooling,
        "seed": args.seed,
    }

    if args.train_lang and args.test_lang:
        # Single pair
        os.makedirs(args.output_dir, exist_ok=True)
        r = run_transfer_pair(args.train_lang, args.test_lang,
                              config, args.device, args.graph_dir)
        rpath = os.path.join(args.output_dir,
                             f"transfer_{args.train_lang}_to_{args.test_lang}.json")
        with open(rpath, "w") as f:
            json.dump(r, f, indent=2, default=float)
        print(f"Saved to: {rpath}")
    else:
        # Full matrix
        run_full_matrix(config, args.device, args.graph_dir, args.output_dir)


if __name__ == "__main__":
    main()