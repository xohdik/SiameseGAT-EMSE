"""
Enrich existing graph .pt files with token_ids for learnable embedding experiments.

This adds a `token_ids` tensor to each graph Data object:
  - Code token nodes (0..nc-1): GraphCodeBERT tokenizer vocab IDs
  - DFG nodes (nc..nc+nd-1): special ID = vocab_size (learned separately)

No GPU needed — only uses the tokenizer, not the model.

Usage:
    # Single language
    python scripts/enrich_token_ids.py --lang python

    # All languages
    python scripts/enrich_token_ids.py --all

    # Custom paths
    python scripts/enrich_token_ids.py \
        --graph-file data/graphs/graph_data_codenet_python.pt \
        --pairs-file data/processed/pairs_codenet_python.json
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm


def enrich_file(graph_path, pairs_path, tokenizer, max_code_length=256, dry_run=False):
    """Add token_ids to all graphs in a single .pt file."""
    print(f"\n{'─'*60}")
    print(f"Graph: {graph_path}")
    print(f"Pairs: {pairs_path}")

    if not os.path.exists(graph_path):
        print(f"  SKIP: graph file not found")
        return False
    if not os.path.exists(pairs_path):
        print(f"  SKIP: pairs file not found")
        return False

    gdata = torch.load(graph_path, weights_only=False, map_location="cpu")
    with open(pairs_path) as f:
        pairs = json.load(f)

    n_graphs = len(gdata["labels"])
    vocab_size = tokenizer.vocab_size  # 50265 for RoBERTa-based

    # Check if already enriched
    if n_graphs > 0 and hasattr(gdata["graph_a"][0], "token_ids"):
        print(f"  Already enriched ({n_graphs} graphs). Skipping.")
        return True

    # Build lookup by pair_id
    pair_lookup = {}
    for i, p in enumerate(pairs):
        pid = p.get("pair_id", f"pair_{i}")
        pair_lookup[pid] = p

    max_tok = min(max_code_length, 510)
    matched, mismatched, fallback = 0, 0, 0

    for gi in tqdm(range(n_graphs), desc="  Enriching", leave=False):
        ga = gdata["graph_a"][gi]
        gb = gdata["graph_b"][gi]
        meta = gdata["metadata"][gi]
        pid = meta.get("pair_id", "")

        pair = pair_lookup.get(pid)
        if pair is None:
            # Fallback: assign placeholder IDs (still trainable)
            ga.token_ids = torch.full((ga.num_nodes,), vocab_size, dtype=torch.long)
            gb.token_ids = torch.full((gb.num_nodes,), vocab_size, dtype=torch.long)
            fallback += 1
            continue

        for code_key, g in [("correct_code", ga), ("buggy_code", gb)]:
            code = pair.get(code_key, "")
            tokens = tokenizer.tokenize(code)[:max_tok - 2]
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            ids = tokenizer.convert_tokens_to_ids(tokens)

            nc = g.num_code_tokens
            total = g.num_nodes

            tid = torch.full((total,), vocab_size, dtype=torch.long)

            if len(ids) == nc:
                tid[:nc] = torch.tensor(ids, dtype=torch.long)
                matched += 1
            else:
                # Length mismatch (shouldn't happen if same tokenizer + max_length)
                n = min(len(ids), nc)
                tid[:n] = torch.tensor(ids[:n], dtype=torch.long)
                mismatched += 1

            g.token_ids = tid

    # Stats
    total_ops = matched + mismatched + fallback * 2
    print(f"  Graphs: {n_graphs}")
    print(f"  Token ID matches: {matched}, mismatches: {mismatched}, fallback: {fallback}")

    # Verify
    g0 = gdata["graph_a"][0]
    print(f"  Verification: token_ids shape={g0.token_ids.shape}, "
          f"nc={g0.num_code_tokens}, total={g0.num_nodes}")
    print(f"  First 10 token_ids: {g0.token_ids[:10].tolist()}")

    if not dry_run:
        torch.save(gdata, graph_path)
        print(f"  Saved: {graph_path}")
    else:
        print(f"  DRY RUN — not saved")

    del gdata
    import gc; gc.collect()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, help="Language to enrich (e.g. python)")
    ap.add_argument("--all", action="store_true", help="Enrich all languages")
    ap.add_argument("--graph-file", type=str, help="Single graph .pt file")
    ap.add_argument("--pairs-file", type=str, help="Corresponding pairs .json file")
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--pairs-dir", default="data/processed")
    ap.add_argument("--model", default="/data/workzone/local_models/graphcodebert-base",
                    help="Tokenizer model path (only tokenizer is loaded, no GPU)")
    ap.add_argument("--max-code-length", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Vocab size: {tokenizer.vocab_size}")

    if args.graph_file and args.pairs_file:
        enrich_file(args.graph_file, args.pairs_file, tokenizer,
                    args.max_code_length, args.dry_run)
    elif args.lang or args.all:
        LANGUAGES = ["python", "java", "cpp", "c", "ruby", "javascript"]
        langs = LANGUAGES if args.all else [args.lang.lower()]
        BENCHMARKS = ["codenet", "humanevalfix"]

        for lang in langs:
            for bench in BENCHMARKS:
                gf = os.path.join(args.graph_dir, f"graph_data_{bench}_{lang}.pt")
                pf = os.path.join(args.pairs_dir, f"pairs_{bench}_{lang}.json")
                enrich_file(gf, pf, tokenizer, args.max_code_length, args.dry_run)
    else:
        print("ERROR: Provide --lang, --all, or --graph-file + --pairs-file")
        return

    print("\nDone!")


if __name__ == "__main__":
    main()