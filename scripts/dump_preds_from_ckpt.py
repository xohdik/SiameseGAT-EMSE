"""
Dump per-pair SiameseGAT predictions from EXISTING fold checkpoints -- no retraining.

Your GroupKFold splits are deterministic (sorted pair_id groups, no random seed),
so reloading a language's graph data and re-splitting reproduces the exact test set
each fold was evaluated on. best_model_foldK.pt is the model for fold K's test pairs.
This script loads each fold's checkpoint, runs inference on that fold's test split,
and writes siamesegat_preds.json (pair_id, dataset, label, pred) for the
edit-distance construct-validity analysis.

Run from the scripts/ folder (needs model.py / model_spec.py on sys.path):

    python dump_preds_from_ckpt.py --lang python \
        --ckpt-dir ../outputs/siamese_gat_python \
        --out ../outputs/cv_python/siamesegat_preds.json

    # or all six at once:
    for L in python cpp java c ruby javascript; do
      python dump_preds_from_ckpt.py --lang $L \
        --ckpt-dir ../outputs/siamese_gat_$L \
        --out ../outputs/cv_$L/siamesegat_preds.json
    done
"""
import argparse, glob, json, os, sys, gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_spec import build_model
# reuse the exact eval + grouping logic from your trainer
from train_spec import evaluate, get_problem_groups, PairDataset, collate_pairs
from torch.utils.data import DataLoader


def load_lang_graph(lang, graph_dir, edge_filter=None):
    """Replicates train_spec.py --lang load+merge (+ optional edge filter)."""
    lang = lang.lower()
    patterns = [os.path.join(graph_dir, f"graph_data_codenet_{lang}.pt"),
                os.path.join(graph_dir, f"graph_data_humanevalfix_{lang}.pt")]
    files = [f for f in patterns if os.path.exists(f)]
    if not files:
        sys.exit(f"No graph files for '{lang}' in {graph_dir} (looked for {patterns})")
    gd = {"graph_a": [], "graph_b": [], "labels": [], "metadata": []}
    for fp in files:
        print(f"  loading {os.path.basename(fp)} ...", end=" ", flush=True)
        d = torch.load(fp, weights_only=False, map_location="cpu")
        for k in ("graph_a", "graph_b", "labels", "metadata"):
            gd[k].extend(d[k])
        print(len(d["labels"]))
        del d; gc.collect()

    if edge_filter:
        EF = {"dfg_only": {0, 1, 2}, "cfg_only": {0, 3}, "seq_only": {0}}
        keep = EF[edge_filter]
        from torch_geometric.data import Data as _Data
        def _f(g):
            mask = torch.zeros(len(g.edge_type), dtype=torch.bool)
            for t in keep: mask |= (g.edge_type == t)
            return _Data(x=g.x, edge_index=g.edge_index[:, mask], edge_type=g.edge_type[mask],
                         num_nodes=g.num_nodes, node_types=g.node_types,
                         num_code_tokens=g.num_code_tokens, num_dfg_nodes=g.num_dfg_nodes,
                         num_cfg_edges=(g.edge_type[mask] == 3).sum().item())
        gd["graph_a"] = [_f(g) for g in gd["graph_a"]]
        gd["graph_b"] = [_f(g) for g in gd["graph_b"]]
    return gd


def make_splits(gd, n_folds, seed=42):
    labels = np.array(gd["labels"])
    groups = get_problem_groups(gd["metadata"])
    if len(set(groups)) >= n_folds:
        return list(GroupKFold(n_splits=n_folds).split(labels, labels, groups))
    if len(set(labels.tolist())) >= 2:
        return list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                    random_state=seed).split(labels, labels))
    return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="dir with best_model_fold{0..4}.pt")
    ap.add_argument("--graph-dir", default="../data/graphs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge-filter", default=None, choices=["dfg_only", "cfg_only", "seq_only"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (f"cuda:{torch.cuda.device_count()-1}" if args.device == "auto"
              and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    print(f"Device: {device}")

    # must match the training config (defaults of train_spec.py full model)
    config = {"model_type": "siamese_gat", "hidden_dim": 256, "num_heads": 4, "num_layers": 2,
              "dropout": 0.3, "attention_dropout": 0.1, "mlp_hidden": 256, "mlp_dropout": 0.3,
              "num_classes": 2, "pooling": "attention", "embedding_dim": 768,
              "fusion_mode": "concat", "cross_attention": False, "shared_spec_encoder": False}

    print(f"[{args.lang}] loading graph data")
    gd = load_lang_graph(args.lang, args.graph_dir, args.edge_filter)
    splits = make_splits(gd, args.n_folds)
    criterion = nn.CrossEntropyLoss()

    records, all_f1 = [], []
    for fold, (_, test_idx) in enumerate(splits):
        ckpt = os.path.join(args.ckpt_dir, f"best_model_fold{fold}.pt")
        if not os.path.exists(ckpt):
            print(f"  fold {fold}: checkpoint missing ({ckpt}); skipping"); continue
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        test_ds = PairDataset([gd["graph_a"][i] for i in test_idx],
                              [gd["graph_b"][i] for i in test_idx],
                              [gd["labels"][i] for i in test_idx],
                              [gd["metadata"][i] for i in test_idx], swap_aug=False)
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pairs, num_workers=0)
        m = evaluate(model, loader, criterion, device, has_spec=False)
        all_f1.append(m["f1_macro"])
        test_meta = [gd["metadata"][i] for i in test_idx]
        for p, l, meta in zip(m["preds"], m["labels"], test_meta):
            records.append({"pair_id": meta.get("pair_id"), "dataset": meta.get("dataset"),
                            "label": int(l), "pred": int(p)})
        print(f"  fold {fold}: F1={m['f1_macro']:.4f}  (n={len(test_idx)})")
        del model; gc.collect()
        if device.startswith("cuda"): torch.cuda.empty_cache()

    if all_f1:
        print(f"[{args.lang}] mean fold F1 = {np.mean(all_f1):.4f}  "
              f"(sanity-check against your results.json)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(records, open(args.out, "w"))
    print(f"[{args.lang}] wrote {len(records)} per-pair predictions -> {args.out}")


if __name__ == "__main__":
    main()