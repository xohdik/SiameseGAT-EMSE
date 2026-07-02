"""
Build specification graphs from Input-Output examples.

For each coding problem, takes the sample IO pairs and creates a PyG graph
using GraphCodeBERT embeddings — same format as code graphs.

Spec graph structure:
- Nodes: GraphCodeBERT token embeddings (768-dim)
- Edge type 0: Sequential (token order)
- Edge type 4: IO-boundary (connects [INPUT] ↔ [OUTPUT] markers)
- Edge type 5: IO-pair (connects input_i → output_i within same IO pair)

Data sources:
- CodeNet: Project_CodeNet/data/{problem_id}/sample/input_*.txt + output_*.txt
- HumanEvalFix: test assertions parsed from pair JSON

Usage:
    # CodeNet (reads IO from sample directories)
    python scripts/build_spec_graphs.py \
        --pairs data/processed/pairs_codenet_python.json \
        --codenet-dir /data/workzone/PhD/Project_CodeNet \
        --device cuda:1

    # HumanEvalFix (parses IO from test code)
    python scripts/build_spec_graphs.py \
        --pairs data/processed/pairs_humanevalfix_python.json \
        --device cuda:1
"""
import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# ═══════════════════════════════════════
# IO EXTRACTION
# ═══════════════════════════════════════

def extract_codenet_io(problem_id: str, codenet_dir: str,
                        max_io_pairs: int = 5) -> List[Dict]:
    """
    Extract sample IO from CodeNet problem directory.
    
    CodeNet structure:
        Project_CodeNet/data/{problem_id}/
            sample/ or input/ or test/
                input_0.txt / 1.in / input0.txt
                output_0.txt / 1.out / output0.txt
    
    Also checks:
        Project_CodeNet/problem_descriptions/{problem_id}.html
        (for inline sample IO in HTML)
    """
    io_pairs = []
    
    if not codenet_dir:
        return io_pairs
    
    problem_dir = os.path.join(codenet_dir, "data", problem_id)
    
    # Strategy 1: Look for sample/ directory
    sample_dirs = ["sample", "samples", "test", "tests", "input"]
    for sd in sample_dirs:
        sample_path = os.path.join(problem_dir, sd)
        if not os.path.isdir(sample_path):
            continue
        
        # Find input/output file pairs
        files = os.listdir(sample_path)
        input_files = sorted([f for f in files
                              if re.match(r'(input|in)[_.]?\d*\.?(txt|in)?$', f, re.I)
                              or f.endswith('.in')])
        output_files = sorted([f for f in files
                               if re.match(r'(output|out|ans)[_.]?\d*\.?(txt|out)?$', f, re.I)
                               or f.endswith('.out') or f.endswith('.ans')])
        
        # Try numbered matching
        for i in range(min(max_io_pairs, 20)):
            inp_file = None
            out_file = None
            
            # Try various naming patterns
            for pattern_in in [f"input_{i}.txt", f"input{i}.txt", f"{i}.in",
                               f"in_{i}.txt", f"input_{i}"]:
                if pattern_in in files:
                    inp_file = pattern_in
                    break
            
            for pattern_out in [f"output_{i}.txt", f"output{i}.txt", f"{i}.out",
                                f"out_{i}.txt", f"output_{i}", f"{i}.ans"]:
                if pattern_out in files:
                    out_file = pattern_out
                    break
            
            if inp_file and out_file:
                try:
                    with open(os.path.join(sample_path, inp_file), 'r',
                              errors='ignore') as f:
                        inp_text = f.read().strip()
                    with open(os.path.join(sample_path, out_file), 'r',
                              errors='ignore') as f:
                        out_text = f.read().strip()
                    if inp_text and out_text:
                        io_pairs.append({"input": inp_text, "output": out_text})
                except:
                    continue
        
        if io_pairs:
            break
    
    # Strategy 2: If no sample dir, check problem_descriptions for inline IO
    if not io_pairs:
        desc_path = os.path.join(codenet_dir, "problem_descriptions",
                                  f"{problem_id}.html")
        if os.path.exists(desc_path):
            io_pairs = _parse_io_from_html(desc_path, max_io_pairs)
    
    return io_pairs[:max_io_pairs]


def _parse_io_from_html(html_path: str, max_pairs: int = 5) -> List[Dict]:
    """Parse sample IO from CodeNet problem description HTML."""
    io_pairs = []
    try:
        with open(html_path, 'r', errors='ignore') as f:
            html = f.read()
        
        # Common patterns in competitive programming problem pages
        # Pattern: <pre>input</pre> followed by <pre>output</pre>
        pre_blocks = re.findall(r'<pre[^>]*>(.*?)</pre>', html, re.DOTALL)
        
        # Try to pair consecutive pre blocks as input/output
        for i in range(0, len(pre_blocks) - 1, 2):
            inp = re.sub(r'<[^>]+>', '', pre_blocks[i]).strip()
            out = re.sub(r'<[^>]+>', '', pre_blocks[i + 1]).strip()
            if inp and out and len(inp) < 1000 and len(out) < 1000:
                io_pairs.append({"input": inp, "output": out})
                if len(io_pairs) >= max_pairs:
                    break
    except:
        pass
    return io_pairs


def extract_humanevalfix_io(pair_data: Dict,
                             max_io_pairs: int = 5) -> List[Dict]:
    """
    Extract IO from HumanEvalFix test assertions.
    
    HumanEvalFix pairs have:
    - "test": test function code with assert statements
    - "prompt": function signature + docstring
    - "docstring": natural language description
    
    Parses assertions like:
        assert candidate(2, 3) == 5
        assert candidate([1,2,3]) == [3,2,1]
    """
    io_pairs = []
    
    test_code = pair_data.get("test", "")
    if not test_code:
        return io_pairs
    
    for line in test_code.split("\n"):
        line = line.strip()
        if not line.startswith("assert"):
            continue
        
        # Remove 'assert' prefix
        assertion = line[6:].strip()
        
        # Pattern: candidate(...) == expected
        # or: func_name(...) == expected
        match = re.match(r'(\w+)\((.+?)\)\s*==\s*(.+?)(?:\s*$|\s*,)', assertion)
        if match:
            func_call = match.group(2).strip()
            expected = match.group(3).strip()
            # Clean up trailing comments
            expected = expected.split('#')[0].strip().rstrip(',')
            if func_call and expected:
                io_pairs.append({
                    "input": func_call,
                    "output": expected
                })
                if len(io_pairs) >= max_io_pairs:
                    break
    
    return io_pairs


def extract_io_for_pair(pair: Dict, codenet_dir: str = None,
                         max_io: int = 5) -> List[Dict]:
    """Unified IO extraction for any dataset."""
    source = pair.get("source", "")
    
    if source == "codenet":
        problem_id = pair.get("problem_id", "")
        return extract_codenet_io(problem_id, codenet_dir, max_io)
    elif source == "humanevalfix":
        return extract_humanevalfix_io(pair, max_io)
    else:
        # Try HumanEvalFix-style first, then empty
        io = extract_humanevalfix_io(pair, max_io)
        return io if io else []


# ═══════════════════════════════════════
# SPEC STRING FORMATTING
# ═══════════════════════════════════════

def format_spec_string(io_pairs: List[Dict], prompt: str = "",
                        max_io: int = 5, max_length: int = 300) -> str:
    """
    Format IO examples into a linearized spec string for tokenization.
    
    Format: "[SPEC] [PROMPT] {prompt} [IO] [IN] {input1} [OUT] {output1} [IN] {input2} [OUT] {output2}"
    
    The special markers help the model learn structural boundaries.
    GraphCodeBERT doesn't have these tokens, but it will learn subword
    representations for them.
    """
    parts = []
    
    # Optional: include prompt/docstring (truncated)
    if prompt:
        prompt_clean = prompt.strip()[:100]
        parts.append(f"[SPEC] {prompt_clean}")
    else:
        parts.append("[SPEC]")
    
    # IO examples
    for io in io_pairs[:max_io]:
        inp = io["input"].strip()[:80]  # Truncate long IO
        out = io["output"].strip()[:80]
        parts.append(f"[IN] {inp} [OUT] {out}")
    
    spec_string = " ".join(parts)
    
    # Hard truncate to max_length characters
    if len(spec_string) > max_length:
        spec_string = spec_string[:max_length]
    
    return spec_string


def format_empty_spec() -> str:
    """Fallback spec string for problems without IO examples."""
    return "[SPEC] [EMPTY]"


# ═══════════════════════════════════════
# SPEC GRAPH CONSTRUCTION
# ═══════════════════════════════════════

def spec_to_graph_data(spec_string: str, tokenizer, model,
                        device: str = "cuda:1",
                        max_length: int = 128) -> Optional[Data]:
    """
    Convert spec string → PyG Data graph.
    
    Same format as code graphs (768-dim GraphCodeBERT embeddings).
    
    Graph structure:
    - Nodes: token embeddings
    - Edge type 0: Sequential (adjacent tokens)
    - Edge type 4: IO-boundary (connects [IN] to corresponding [OUT])
    - Self-loops for stability
    """
    # Tokenize
    tokens = tokenizer.tokenize(spec_string)[:max_length - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Get embeddings from GraphCodeBERT
    ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        emb = model(input_ids=ids, attention_mask=mask).last_hidden_state[0]
    
    x = emb.cpu()  # [num_tokens, 768]
    nc = len(token_ids)
    
    edges, etypes = [], []
    
    # Edge type 0: Sequential (same as code graphs)
    for i in range(nc - 1):
        edges.extend([(i, i + 1), (i + 1, i)])
        etypes.extend([0, 0])
    
    # Find [IN] and [OUT] token positions for IO-boundary edges
    tokens_lower = [t.lower().replace("Ġ", "").replace("▁", "") for t in tokens]
    
    in_positions = []
    out_positions = []
    for i, t in enumerate(tokens_lower):
        if t in ("[in]", "in]", "[in", "in"):
            # Check context
            if i > 0 and tokens_lower[i-1] in ("[", ""):
                in_positions.append(i)
            elif t == "[in]":
                in_positions.append(i)
        if t in ("[out]", "out]", "[out", "out"):
            if i > 0 and tokens_lower[i-1] in ("[", ""):
                out_positions.append(i)
            elif t == "[out]":
                out_positions.append(i)
    
    # Edge type 4: IO-boundary edges (pair each [IN] with next [OUT])
    for j, in_pos in enumerate(in_positions):
        if j < len(out_positions):
            out_pos = out_positions[j]
            edges.extend([(in_pos, out_pos), (out_pos, in_pos)])
            etypes.extend([4, 4])
    
    # Edge type 5: Connect all IO pairs to [SPEC] token (CLS)
    for pos in in_positions + out_positions:
        edges.extend([(0, pos), (pos, 0)])
        etypes.extend([5, 5])
    
    # Ensure at least one edge
    if not edges:
        edges = [(0, 0)]
        etypes = [0]
    
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    et = torch.tensor(etypes, dtype=torch.long)
    
    return Data(
        x=x,
        edge_index=ei,
        edge_type=et,
        num_nodes=nc,
        node_types=torch.full((nc,), 2, dtype=torch.long),  # 2 = spec node
        num_code_tokens=nc,
        num_dfg_nodes=0,
        num_cfg_edges=0,
        is_spec=True,
    )


# ═══════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════

def build_spec_graphs(pairs_path: str, output_path: str,
                       codenet_dir: str = None,
                       model_name: str = "/data/workzone/local_models/graphcodebert-base",
                       max_io: int = 5, max_spec_length: int = 128,
                       device: str = "cuda:1"):
    """
    Build spec graphs for all pairs in a dataset.
    
    Output format (matches code graph format):
    {
        "graphs": [Data, Data, ...],     # One spec graph per unique problem
        "problem_ids": [str, ...],        # Problem IDs
        "pair_indices": [int, ...],       # Maps each pair to its spec graph index
        "io_coverage": {"with_io": N, "without_io": M},
    }
    """
    from transformers import AutoTokenizer, AutoModel
    
    print(f"Loading GraphCodeBERT → {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    with open(pairs_path) as f:
        pairs = json.load(f)
    print(f"Processing {len(pairs)} pairs...")
    
    # Group pairs by problem to avoid duplicate spec graphs
    problem_io = {}    # problem_id → io_pairs
    pair_to_prob = []  # pair_index → problem_id
    
    for i, pair in enumerate(tqdm(pairs, desc="Extracting IO")):
        prob_id = pair.get("problem_id", pair.get("pair_id", f"pair_{i}"))
        pair_to_prob.append(prob_id)
        
        if prob_id not in problem_io:
            io = extract_io_for_pair(pair, codenet_dir, max_io)
            prompt = pair.get("prompt", pair.get("docstring", ""))
            problem_io[prob_id] = {
                "io_pairs": io,
                "prompt": prompt,
                "has_io": len(io) > 0,
            }
    
    # Report IO coverage
    with_io = sum(1 for v in problem_io.values() if v["has_io"])
    without_io = sum(1 for v in problem_io.values() if not v["has_io"])
    print(f"\nIO coverage: {with_io} problems with IO, {without_io} without")
    print(f"Unique problems: {len(problem_io)}")
    
    # Build spec graphs
    spec_graphs = {}  # problem_id → Data
    
    for prob_id, info in tqdm(problem_io.items(), desc="Building spec graphs"):
        if info["has_io"]:
            spec_str = format_spec_string(info["io_pairs"], info["prompt"],
                                           max_io, max_spec_length * 4)
        else:
            spec_str = format_empty_spec()
        
        try:
            graph = spec_to_graph_data(spec_str, tokenizer, model,
                                        device, max_spec_length)
            spec_graphs[prob_id] = graph
        except Exception as e:
            print(f"  Error on {prob_id}: {e}")
            # Create minimal fallback graph
            spec_graphs[prob_id] = spec_to_graph_data(
                format_empty_spec(), tokenizer, model, device, max_spec_length
            )
    
    # Map each pair to its spec graph
    pair_spec_list = []
    for i, prob_id in enumerate(pair_to_prob):
        pair_spec_list.append(spec_graphs.get(prob_id))
    
    # Save
    output_data = {
        "graphs": pair_spec_list,
        "problem_ids": pair_to_prob,
        "io_coverage": {"with_io": with_io, "without_io": without_io,
                         "total_problems": len(problem_io)},
        "stats": {
            "avg_nodes": np.mean([g.num_nodes for g in pair_spec_list if g is not None]),
            "avg_edges": np.mean([g.edge_index.shape[1] for g in pair_spec_list if g is not None]),
        },
    }
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(output_data, output_path)
    
    print(f"\n{'='*60}")
    print(f"DONE: {len(pair_spec_list)} spec graphs")
    print(f"  Avg nodes: {output_data['stats']['avg_nodes']:.1f}")
    print(f"  Avg edges: {output_data['stats']['avg_edges']:.1f}")
    print(f"  IO coverage: {with_io}/{with_io+without_io} problems ({100*with_io/(with_io+without_io):.1f}%)")
    print(f"  Saved to: {output_path}")


# ═══════════════════════════════════════
# MERGE SPEC WITH CODE GRAPHS
# ═══════════════════════════════════════

def merge_spec_with_code(code_graph_path: str, spec_graph_path: str,
                          output_path: str):
    """
    Merge spec graphs with existing code graph pairs.
    
    Input:
        code_graph_path: graph_data_codenet_python.pt
            → {"graph_a": [...], "graph_b": [...], "labels": [...], "metadata": [...]}
        spec_graph_path: spec_data_codenet_python.pt
            → {"graphs": [...], "problem_ids": [...]}
    
    Output:
        spec_graph_data_codenet_python.pt
            → {"graph_a": [...], "graph_b": [...], "graph_spec": [...],
               "labels": [...], "metadata": [...]}
    """
    print(f"Loading code graphs: {code_graph_path}")
    code_data = torch.load(code_graph_path, weights_only=False)
    
    print(f"Loading spec graphs: {spec_graph_path}")
    spec_data = torch.load(spec_graph_path, weights_only=False)
    
    n_pairs = len(code_data["labels"])
    n_specs = len(spec_data["graphs"])
    
    assert n_pairs == n_specs, (
        f"Mismatch: {n_pairs} code pairs vs {n_specs} spec graphs. "
        f"They must be built from the same pairs JSON."
    )
    
    # Create merged dataset
    merged = {
        "graph_a": code_data["graph_a"],
        "graph_b": code_data["graph_b"],
        "graph_spec": spec_data["graphs"],
        "labels": code_data["labels"],
        "metadata": code_data["metadata"],
        "io_coverage": spec_data.get("io_coverage", {}),
    }
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(merged, output_path)
    
    print(f"\nMerged: {n_pairs} pairs with spec graphs")
    print(f"  Saved to: {output_path}")
    
    # Verify
    check = torch.load(output_path, weights_only=False)
    assert "graph_spec" in check
    assert len(check["graph_spec"]) == len(check["graph_a"])
    print("  ✓ Verification passed")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Build specification graphs from IO examples")
    ap.add_argument("--pairs", type=str, required=True,
                    help="Path to pairs JSON (e.g., data/processed/pairs_codenet_python.json)")
    ap.add_argument("--codenet-dir", type=str, default=None,
                    help="Path to Project_CodeNet (for IO extraction)")
    ap.add_argument("--output", type=str, default=None,
                    help="Output path (default: auto from pairs filename)")
    ap.add_argument("--model", default="/data/workzone/local_models/graphcodebert-base")
    ap.add_argument("--max-io", type=int, default=5,
                    help="Max IO pairs per problem")
    ap.add_argument("--max-spec-length", type=int, default=128,
                    help="Max tokens for spec graph")
    ap.add_argument("--device", default="auto")
    
    # Merge mode
    ap.add_argument("--merge", action="store_true",
                    help="Merge spec graphs with existing code graphs")
    ap.add_argument("--code-graphs", type=str,
                    help="Path to existing code graph file (for --merge)")
    ap.add_argument("--spec-graphs", type=str,
                    help="Path to spec graph file (for --merge)")
    
    args = ap.parse_args()
    
    if args.merge:
        if not args.code_graphs or not args.spec_graphs:
            print("ERROR: --merge requires --code-graphs and --spec-graphs")
            return
        output = args.output or args.code_graphs.replace("graph_data_", "spec_graph_data_")
        merge_spec_with_code(args.code_graphs, args.spec_graphs, output)
        return
    
    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            device = f"cuda:{n-1}"
        else:
            device = "cpu"
    else:
        device = args.device
    
    # Auto-generate output path
    if args.output is None:
        bn = os.path.basename(args.pairs).replace("pairs_", "spec_data_").replace(".json", ".pt")
        args.output = os.path.join("./data/graphs", bn)
    
    print(f"Input:  {args.pairs}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"CodeNet dir: {args.codenet_dir or 'None (HumanEvalFix mode)'}")
    print(f"Max IO: {args.max_io}, Max spec tokens: {args.max_spec_length}\n")
    
    # Auto-detect CodeNet dir
    if args.codenet_dir is None:
        for p in ["/data/workzone/PhD/Project_CodeNet",
                  os.path.expanduser("~/Tunde/PhD/Project_CodeNet")]:
            if os.path.isdir(p):
                args.codenet_dir = p
                print(f"Auto-found CodeNet: {p}")
                break
    
    build_spec_graphs(
        args.pairs, args.output, args.codenet_dir,
        args.model, args.max_io, args.max_spec_length, device
    )


if __name__ == "__main__":
    main()