"""
Step 6: Attention-based bug localization.

The key differentiator — uses GAT attention weights to localize bugs:
1. Load trained Siamese GAT model
2. For correctly classified buggy samples, extract attention weights
3. Map highest-attention nodes back to source code tokens
4. Measure overlap with actual bug location (token-level accuracy)
5. Generate visualizations (attention heatmaps on code)

Usage:
    python scripts/bug_localization.py --model ./outputs/siamese_gat/best_model_fold0.pt \
                                        --data ./data/graphs/graph_data_all.pt \
                                        --pairs ./data/processed/pairs_all.json
"""
import argparse, json, os, sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import SiameseGAT


def compute_node_importance(model, data_a, data_b, device="cpu"):
    """
    Extract per-node importance scores from Siamese GAT.
    
    Uses three signals:
    1. GAT attention weights (which nodes attend to which)
    2. Pooling attention (which nodes contribute most to graph representation)
    3. Gradient-based saliency (which nodes most affect the prediction)
    
    Returns:
        importance_a: [num_nodes_a] importance scores for code A
        importance_b: [num_nodes_b] importance scores for code B
    """
    model.eval()
    data_a = data_a.to(device)
    data_b = data_b.to(device)

    # Enable gradient for saliency
    data_a.x.requires_grad_(True)
    data_b.x.requires_grad_(True)

    result = model(data_a, data_b, return_attention=True)
    logits = result["logits"]
    pred = logits.argmax(dim=-1)

    # 1. Pooling attention weights
    pool_attn_a = result.get("pool_attn_a", None)
    pool_attn_b = result.get("pool_attn_b", None)

    # 2. Gradient-based saliency
    target_score = logits[0, pred[0]]
    target_score.backward(retain_graph=True)

    grad_a = data_a.x.grad.abs().mean(dim=-1) if data_a.x.grad is not None else torch.zeros(data_a.num_nodes)
    grad_b = data_b.x.grad.abs().mean(dim=-1) if data_b.x.grad is not None else torch.zeros(data_b.num_nodes)

    # 3. GAT attention (aggregate across layers and heads)
    gat_attn_a = torch.zeros(data_a.num_nodes, device=device)
    gat_attn_b = torch.zeros(data_b.num_nodes, device=device)

    if result.get("attn_a"):
        for edge_idx, attn_weights in result["attn_a"]:
            # attn_weights: [num_edges, num_heads] — average over heads
            attn_mean = attn_weights.mean(dim=-1)
            # Accumulate attention received by each node
            target_nodes = edge_idx[1]
            gat_attn_a.scatter_add_(0, target_nodes, attn_mean)

    if result.get("attn_b"):
        for edge_idx, attn_weights in result["attn_b"]:
            attn_mean = attn_weights.mean(dim=-1)
            target_nodes = edge_idx[1]
            gat_attn_b.scatter_add_(0, target_nodes, attn_mean)

    # Combine signals (normalized)
    def normalize(x):
        if x.max() > 0:
            return x / x.max()
        return x

    importance_a = normalize(grad_a.detach()) * 0.4
    importance_b = normalize(grad_b.detach()) * 0.4

    if pool_attn_a is not None:
        importance_a += normalize(pool_attn_a.detach()) * 0.3
    if pool_attn_b is not None:
        importance_b += normalize(pool_attn_b.detach()) * 0.3

    importance_a += normalize(gat_attn_a.detach()) * 0.3
    importance_b += normalize(gat_attn_b.detach()) * 0.3

    return importance_a.cpu().numpy(), importance_b.cpu().numpy(), pred.item()


def localization_metrics(importance_scores, bug_token_indices, top_k_values=[1, 3, 5, 10]):
    """
    Compute bug localization metrics.
    
    Args:
        importance_scores: [num_nodes] importance per node
        bug_token_indices: set of token indices where the bug is
        top_k_values: list of k values for top-k accuracy
    
    Returns:
        dict with top-k accuracy, mean reciprocal rank, attention entropy
    """
    if not bug_token_indices or len(importance_scores) == 0:
        return {"valid": False}

    # Rank nodes by importance
    ranked = np.argsort(-importance_scores)  # Descending

    metrics = {"valid": True}

    # Top-k accuracy: is any bug token in top-k?
    for k in top_k_values:
        top_k_nodes = set(ranked[:k].tolist())
        hit = bool(top_k_nodes & bug_token_indices)
        metrics[f"top_{k}_hit"] = hit

    # Mean Reciprocal Rank
    for rank_idx, node_idx in enumerate(ranked):
        if node_idx in bug_token_indices:
            metrics["mrr"] = 1.0 / (rank_idx + 1)
            metrics["first_hit_rank"] = rank_idx + 1
            break
    else:
        metrics["mrr"] = 0.0
        metrics["first_hit_rank"] = len(ranked)

    # Attention entropy (lower = more focused)
    probs = importance_scores / (importance_scores.sum() + 1e-10)
    probs = probs[probs > 0]
    metrics["entropy"] = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # Concentration ratio: what fraction of total attention is on bug nodes?
    total_attn = importance_scores.sum()
    if total_attn > 0:
        bug_attn = sum(importance_scores[i] for i in bug_token_indices if i < len(importance_scores))
        metrics["bug_attention_ratio"] = float(bug_attn / total_attn)
    else:
        metrics["bug_attention_ratio"] = 0.0

    return metrics


def find_bug_tokens(correct_code: str, buggy_code: str, tokenizer) -> set:
    """
    Find which token positions differ between correct and buggy code.
    These are the ground-truth bug locations.
    """
    correct_tokens = tokenizer.tokenize(correct_code)[:254]
    buggy_tokens = tokenizer.tokenize(buggy_code)[:254]

    # Find differing positions (with alignment)
    diff_positions = set()
    min_len = min(len(correct_tokens), len(buggy_tokens))

    for i in range(min_len):
        if correct_tokens[i] != buggy_tokens[i]:
            diff_positions.add(i + 1)  # +1 for [CLS] token

    # If lengths differ, mark extra tokens
    if len(buggy_tokens) > min_len:
        for i in range(min_len, len(buggy_tokens)):
            diff_positions.add(i + 1)

    return diff_positions


def run_localization(model_path, graph_data, pairs, config, output_dir, device="cpu", max_samples=200):
    """Run bug localization evaluation."""
    from transformers import AutoTokenizer
    
    print("Loading model and tokenizer...")
    model = SiameseGAT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("/data/workzone/local_models/graphcodebert-base")

    # Build pair lookup
    pair_lookup = {p.get("pair_id", f"pair_{i}"): p for i, p in enumerate(pairs)}

    all_metrics = []
    case_studies = []

    n = min(len(graph_data["labels"]), max_samples)
    print(f"Evaluating localization on {n} samples...")

    for i in range(n):
        ga = graph_data["graph_a"][i]
        gb = graph_data["graph_b"][i]
        meta = graph_data["metadata"][i]
        pair_id = meta["pair_id"]

        # Get source code pair
        pair = pair_lookup.get(pair_id)
        if not pair:
            continue

        correct_code = pair.get("correct_code", "")
        buggy_code = pair.get("buggy_code", "")
        if not correct_code or not buggy_code:
            continue

        # Find ground-truth bug tokens
        bug_tokens = find_bug_tokens(correct_code, buggy_code, tokenizer)
        if not bug_tokens:
            continue

        # Add batch dimension
        ga_batch = Batch.from_data_list([ga])
        gb_batch = Batch.from_data_list([gb])

        try:
            importance_a, importance_b, pred = compute_node_importance(
                model, ga_batch, gb_batch, device)

            # Measure localization on the buggy code (graph_b)
            loc_metrics = localization_metrics(importance_b, bug_tokens)
            if loc_metrics.get("valid"):
                loc_metrics["pair_id"] = pair_id
                loc_metrics["dataset"] = meta["dataset"]
                loc_metrics["bug_type"] = meta.get("bug_type", "unknown")
                loc_metrics["correctly_classified"] = (pred == 0)  # 0 = A is correct
                all_metrics.append(loc_metrics)

                # Save case study (top 20 most interesting)
                if len(case_studies) < 20 and loc_metrics.get("top_1_hit"):
                    buggy_tokens = tokenizer.tokenize(buggy_code)[:254]
                    case_studies.append({
                        "pair_id": pair_id,
                        "dataset": meta["dataset"],
                        "bug_tokens": list(bug_tokens),
                        "top_5_nodes": np.argsort(-importance_b)[:5].tolist(),
                        "importance_scores": importance_b[:len(buggy_tokens)+2].tolist(),
                        "code_tokens": buggy_tokens[:50],
                        "mrr": loc_metrics["mrr"],
                    })

        except Exception as e:
            if i < 5:
                print(f"  Error on sample {i}: {e}")

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"BUG LOCALIZATION RESULTS ({len(all_metrics)} valid samples)")
    print(f"{'='*60}")

    if not all_metrics:
        print("  No valid results!")
        return

    # Filter to correctly classified only
    correct_only = [m for m in all_metrics if m.get("correctly_classified", False)]
    print(f"  Correctly classified: {len(correct_only)}/{len(all_metrics)}")

    for subset_name, subset in [("All", all_metrics), ("Correctly Classified", correct_only)]:
        if not subset:
            continue
        print(f"\n  [{subset_name}] (n={len(subset)})")
        for k in [1, 3, 5, 10]:
            key = f"top_{k}_hit"
            hits = sum(1 for m in subset if m.get(key, False))
            print(f"    Top-{k} Accuracy: {hits}/{len(subset)} = {hits/len(subset):.3f}")
        mrrs = [m["mrr"] for m in subset]
        print(f"    Mean Reciprocal Rank: {np.mean(mrrs):.4f}")
        entropies = [m["entropy"] for m in subset]
        print(f"    Mean Attention Entropy: {np.mean(entropies):.2f}")
        bug_ratios = [m["bug_attention_ratio"] for m in subset]
        print(f"    Mean Bug Attention Ratio: {np.mean(bug_ratios):.4f}")

    # Per-dataset
    ds_metrics = defaultdict(list)
    for m in correct_only:
        ds_metrics[m["dataset"]].append(m)
    print(f"\n  Per-dataset (correctly classified):")
    for ds, dm in ds_metrics.items():
        top1 = sum(1 for m in dm if m.get("top_1_hit")) / len(dm)
        top5 = sum(1 for m in dm if m.get("top_5_hit")) / len(dm)
        mrr = np.mean([m["mrr"] for m in dm])
        print(f"    {ds}: Top-1={top1:.3f}, Top-5={top5:.3f}, MRR={mrr:.3f} (n={len(dm)})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "localization_results.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)
    with open(os.path.join(output_dir, "case_studies.json"), "w") as f:
        json.dump(case_studies, f, indent=2, default=float)
    print(f"\nSaved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model .pt file")
    parser.add_argument("--data", default="./data/graphs/graph_data_all.pt")
    parser.add_argument("--pairs", default="./data/processed/pairs_all.json")
    parser.add_argument("--output-dir", default="./outputs/localization")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="auto")
    # Model config (must match training)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pooling", default="attention")
    args = parser.parse_args()

    if args.device == "auto":
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device = f"cuda:{n-1}" if n > 0 else "cpu"
    else:
        device = args.device

    config = {"hidden_dim": args.hidden_dim, "num_heads": args.num_heads,
              "num_layers": args.num_layers, "dropout": 0.3, "attention_dropout": 0.1,
              "mlp_hidden": args.hidden_dim, "mlp_dropout": 0.3, "num_classes": 2,
              "pooling": args.pooling, "embedding_dim": 768}

    graph_data = torch.load(args.data, weights_only=False)
    with open(args.pairs) as f:
        pairs = json.load(f)

    run_localization(args.model, graph_data, pairs, config, args.output_dir, device, args.max_samples)


if __name__ == "__main__":
    main()
