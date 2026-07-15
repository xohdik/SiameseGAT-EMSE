"""
GraphCodeBERT [CLS] Cosine Similarity Baseline.

Extracts the CLS token embedding (node 0) from existing graph files,
computes cosine similarity between code_a and code_b, and classifies
based on threshold. No training required.

This answers: "Does the graph structure actually help, or is the
pre-trained embedding sufficient?"

Usage:
    python scripts/baseline_cls.py --lang python
    python scripts/baseline_cls.py --all
"""
import argparse
import gc
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, precision_score,
                             recall_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold

GRAPH_DIR = "/data/workzone/siamese_gat_journal/data/graphs"
LANGS = ["python", "cpp", "java", "c", "ruby", "javascript"]


def load_lang_data(lang, graph_dir):
    """Load graph files for a language, extract CLS embeddings."""
    cls_a, cls_b, labels, metadata = [], [], [], []

    for prefix in ["graph_data_codenet_", "graph_data_humanevalfix_"]:
        fpath = os.path.join(graph_dir, f"{prefix}{lang}.pt")
        if not os.path.exists(fpath):
            continue
        print(f"  Loading {os.path.basename(fpath)}...", end=" ", flush=True)
        data = torch.load(fpath, weights_only=False, map_location="cpu")
        n = len(data["labels"])

        for i in range(n):
            # Node 0 is CLS token in every graph
            cls_a.append(data["graph_a"][i].x[0])
            cls_b.append(data["graph_b"][i].x[0])
            labels.append(data["labels"][i])
            metadata.append(data["metadata"][i])

        print(f"{n} pairs")
        del data
        gc.collect()

    cls_a = torch.stack(cls_a)  # (N, 768)
    cls_b = torch.stack(cls_b)  # (N, 768)
    labels = np.array(labels)

    return cls_a, cls_b, labels, metadata


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


def evaluate_cls(lang, graph_dir, output_dir, n_folds=5, seed=42):
    """Run CLS cosine similarity baseline with k-fold CV."""
    print(f"\n{'='*60}")
    print(f"CLS BASELINE: {lang}")
    print(f"{'='*60}")

    cls_a, cls_b, labels, metadata = load_lang_data(lang, graph_dir)
    n = len(labels)
    print(f"Total: {n} pairs")

    # Cosine similarity between CLS embeddings
    sims = F.cosine_similarity(cls_a, cls_b, dim=1).numpy()

    # K-fold to find optimal threshold (same split strategy as training)
    groups = get_problem_groups(metadata)
    n_groups = len(set(groups))

    if n_groups >= n_folds:
        splits = list(GroupKFold(n_splits=n_folds).split(labels, labels, groups))
    else:
        splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=seed).split(labels, labels))

    fold_results = []
    all_preds, all_labels = [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_sims = sims[train_idx]
        train_labels = labels[train_idx]
        test_sims = sims[test_idx]
        test_labels = labels[test_idx]

        # Find optimal threshold on train set
        best_thresh, best_f1 = 0.5, 0
        for thresh in np.arange(0.0, 1.0, 0.01):
            # Higher sim → more similar → predict label 0 (A=correct, both similar)
            # Lower sim → more different → predict label 1
            # Actually: label 0 means A is correct. If sims are high, A≈B, hard to tell.
            # We need: if sim(A, B) is high, they're similar. The correct one should
            # be distinguishable. Let's try both directions and pick the better one.
            preds_0 = (train_sims < thresh).astype(int)
            preds_1 = (train_sims >= thresh).astype(int)
            f1_0 = f1_score(train_labels, preds_0, average="macro")
            f1_1 = f1_score(train_labels, preds_1, average="macro")
            if f1_0 > best_f1:
                best_f1 = f1_0
                best_thresh = thresh
                best_dir = "lt"
            if f1_1 > best_f1:
                best_f1 = f1_1
                best_thresh = thresh
                best_dir = "gte"

        # Apply to test
        if best_dir == "lt":
            test_preds = (test_sims < best_thresh).astype(int)
        else:
            test_preds = (test_sims >= best_thresh).astype(int)

        f1 = f1_score(test_labels, test_preds, average="macro")
        acc = accuracy_score(test_labels, test_preds)
        try:
            auc = roc_auc_score(test_labels, test_sims)
        except:
            auc = 0.0

        print(f"  Fold {fold+1}: F1={f1:.4f} Acc={acc:.4f} AUC={auc:.4f} "
              f"(thresh={best_thresh:.2f}, dir={best_dir})")

        fold_results.append({
            "fold": fold, "f1_macro": f1, "accuracy": acc, "auc": auc,
            "threshold": float(best_thresh), "direction": best_dir,
        })
        all_preds.extend(test_preds.tolist())
        all_labels.extend(test_labels.tolist())

        # Per-dataset breakdown
        test_meta = [metadata[i] for i in test_idx]
        ds_m = defaultdict(lambda: {"p": [], "l": []})
        for p, l, m in zip(test_preds, test_labels, test_meta):
            ds_m[m["dataset"]]["p"].append(p)
            ds_m[m["dataset"]]["l"].append(l)
        for ds, dm in ds_m.items():
            ds_f1 = f1_score(dm['l'], dm['p'], average='macro')
            print(f"    {ds}: F1={ds_f1:.3f} (n={len(dm['l'])})")

    # Aggregate
    summary = {}
    for metric in ["f1_macro", "accuracy", "auc"]:
        vals = [r[metric] for r in fold_results]
        summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    print(f"\n  AGGREGATE: F1={summary['f1_macro']['mean']:.4f} ± {summary['f1_macro']['std']:.4f}  "
          f"AUC={summary['auc']['mean']:.4f} ± {summary['auc']['std']:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Correct', 'Buggy'], digits=4)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "model_type": "cls_cosine_baseline",
        "language": lang,
        "total_pairs": n,
        "fold_results": fold_results,
        "summary": summary,
    }
    rpath = os.path.join(output_dir, "results.json")
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {rpath}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--graph-dir", default=GRAPH_DIR)
    ap.add_argument("--output-dir", default="./outputs")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    langs = LANGS if args.all else ([args.lang] if args.lang else [])
    if not langs:
        print("Specify --lang <language> or --all")
        return

    all_results = {}
    for lang in langs:
        odir = os.path.join(args.output_dir, f"cls_baseline_{lang}")
        r = evaluate_cls(lang, args.graph_dir, odir, args.n_folds)
        all_results[lang] = r["summary"]

    if len(langs) > 1:
        print(f"\n{'='*60}")
        print("CLS BASELINE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Language':<12} {'Pairs':>7} {'F1':>12} {'AUC':>12}")
        print(f"{'─'*45}")
        for lang in langs:
            s = all_results[lang]
            print(f"{lang:<12} {'':>7} "
                  f"{s['f1_macro']['mean']:.4f}±{s['f1_macro']['std']:.4f}  "
                  f"{s['auc']['mean']:.4f}±{s['auc']['std']:.4f}")


if __name__ == "__main__":
    main()