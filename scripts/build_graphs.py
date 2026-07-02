"""
Step 3: Build heterogeneous code graphs with DFG + CFG edges.

Edge types: 0=sequential, 1=DFG-code, 2=DFG-DFG, 3=CFG

Uses parser_utils.py for tree-sitter (v0.23.2 compatible).
Uses official GraphCodeBERT DFG functions for python/java/ruby/go/javascript.
Uses tree-sitter AST for DFG on C/C++/Rust and CFG on ALL languages.

Usage:
    python scripts/build_graphs.py --pairs ./data/processed/pairs_codenet_python.json --device cuda:1
    python scripts/build_graphs.py --check-gpu
"""
import argparse
import json
import os
import re
import signal
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Add scripts dir to path for parser_utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ═══════════════════════════════════════
# TIMEOUT
# ═══════════════════════════════════════

class PairTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise PairTimeout()


# ═══════════════════════════════════════
# GPU
# ═══════════════════════════════════════

def detect_best_gpu():
    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    print(f"Available GPUs: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"  cuda:{i} = {name} ({mem:.1f} GB)")
    device = f"cuda:{n - 1}"
    print(f"Selected: {device}")
    return device


# ═══════════════════════════════════════
# PARSER SETUP (via parser_utils.py)
# ═══════════════════════════════════════

_parsers_ready = {}


def init_parsers():
    """Load parsers via parser_utils.py for all available languages."""
    global _parsers_ready
    try:
        from parser_utils import get_parser, get_supported_languages
        supported = get_supported_languages()
        for lang in supported:
            try:
                p = get_parser(lang)
                _parsers_ready[lang] = p
            except Exception as e:
                print(f"  {lang}: ✗ {e}")
        print(f"[Parsers] Loaded: {list(_parsers_ready.keys())}")
    except ImportError as e:
        print(f"[Parsers] parser_utils.py not found: {e}")
        print(f"[Parsers] Searched: {SCRIPT_DIR}")


def get_tree(code, language):
    """Parse code and return AST tree, or None."""
    if language not in _parsers_ready:
        return None
    try:
        p = _parsers_ready[language]
        return p.parse(bytes(code, 'utf8') if isinstance(code, str) else code)
    except:
        return None


# ═══════════════════════════════════════
# OFFICIAL GRAPHCODEBERT DFG FUNCTIONS
# ═══════════════════════════════════════

PARSER_DIR = "/data/workzone/code_retrieval/parser"
_official_dfg_fns = {}
_official_available = False


def init_official_dfg():
    """Load official DFG extraction functions from GraphCodeBERT parser module."""
    global _official_dfg_fns, _official_available
    try:
        if PARSER_DIR not in sys.path:
            sys.path.insert(0, PARSER_DIR)
        parent = os.path.dirname(PARSER_DIR)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        from parser import (DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_javascript)
        _official_dfg_fns = {
            'python': DFG_python, 'java': DFG_java, 'ruby': DFG_ruby,
            'go': DFG_go, 'javascript': DFG_javascript,
        }
        _official_available = True
        print(f"[DFG] Official functions: {list(_official_dfg_fns.keys())}")
    except Exception as e:
        print(f"[DFG] Official functions unavailable: {e}")


# ═══════════════════════════════════════
# DFG EXTRACTION
# ═══════════════════════════════════════

def extract_dfg_official(code, lang):
    """DFG via official GraphCodeBERT functions (python/java/ruby/go/js)."""
    from parser import (remove_comments_and_docstrings,
                       tree_to_token_index, index_to_code_token)
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass

    tree = get_tree(code, lang)
    if tree is None:
        return [], []

    try:
        root = tree.root_node
        tokens_index = tree_to_token_index(root)
        code_lines = code.split('\n')
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code_tok) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code_tok)
        try:
            DFG, _ = _official_dfg_fns[lang](root, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        return code_tokens, [d for d in DFG if d[1] in indexs]
    except:
        return [], []


def _get_ids(node, depth=0):
    """Extract identifier nodes from AST subtree."""
    if depth > 50:
        return []
    ids = []
    SKIP = {"int", "float", "double", "char", "void", "string", "bool",
            "true", "false", "null", "None", "True", "False", "return",
            "if", "else", "for", "while", "include", "using", "namespace",
            "std", "class", "public", "private", "static", "main",
            "printf", "scanf", "cout", "cin", "endl", "auto", "const",
            "unsigned", "long", "short", "signed", "size_t", "sizeof",
            "new", "delete", "this", "self", "super", "nil", "var", "let"}
    if node.type in ("identifier", "IDENTIFIER", "variable_name",
                     "simple_identifier", "field_identifier"):
        name = node.text.decode("utf8") if isinstance(node.text, bytes) else node.text
        if name not in SKIP and len(name) > 0:
            ids.append({"name": name, "start": node.start_byte})
    for child in node.children:
        ids.extend(_get_ids(child, depth + 1))
    return ids


def extract_dfg_treesitter(code, language):
    """DFG via tree-sitter AST (for C/C++/Rust/any language)."""
    tree = get_tree(code, language)
    if tree is None:
        return [], []

    dfg = []
    scope = {}

    def walk(node, depth=0):
        if depth > 80:
            return

        # Assignments: x = expr
        if node.type in ("assignment_expression", "assignment",
                         "init_declarator", "variable_declarator"):
            children = node.children
            eq_idx = None
            for i, ch in enumerate(children):
                txt = ch.text.decode("utf8") if isinstance(ch.text, bytes) else ch.text
                if ch.type in ("=", "assignment_operator") or txt == "=":
                    eq_idx = i
                    break
            if eq_idx is not None and eq_idx > 0:
                left = _get_ids(children[0])
                right = []
                for ch in children[eq_idx + 1:]:
                    right.extend(_get_ids(ch))
                for lv in left:
                    dfg.append((lv["name"], lv["start"], "comesFrom", right))
                    scope[lv["name"]] = lv["start"]

        # Compound assignment (+=, -=)
        elif node.type in ("augmented_assignment", "compound_assignment_expr",
                           "update_expression"):
            ids = _get_ids(node)
            if ids:
                dfg.append((ids[0]["name"], ids[0]["start"], "computedFrom", ids[1:]))
                scope[ids[0]["name"]] = ids[0]["start"]

        # Variable declarations: int x = 5;
        elif node.type in ("declaration", "local_variable_declaration",
                           "variable_declaration", "field_declaration"):
            for child in node.children:
                if child.type in ("init_declarator", "variable_declarator"):
                    name_node = child.child_by_field_name("declarator")
                    val_node = child.child_by_field_name("value")
                    if name_node is None:
                        for gc in child.children:
                            if gc.type in ("identifier", "IDENTIFIER"):
                                name_node = gc
                                break
                    if name_node:
                        names = _get_ids(name_node)
                        values = _get_ids(val_node) if val_node else []
                        for n in names:
                            dfg.append((n["name"], n["start"], "comesFrom", values))
                            scope[n["name"]] = n["start"]

        # Function parameters
        elif node.type in ("function_definition", "function_declarator",
                           "method_declaration", "function_declaration",
                           "lambda_expression", "arrow_function",
                           "function_item", "method_definition"):
            params = (node.child_by_field_name("parameters") or
                     node.child_by_field_name("formal_parameters"))
            if params is None and node.type == "function_definition":
                decl = node.child_by_field_name("declarator")
                if decl:
                    params = decl.child_by_field_name("parameters")
            if params:
                for v in _get_ids(params):
                    dfg.append((v["name"], v["start"], "parameter", []))
                    scope[v["name"]] = v["start"]

        # For loops
        elif node.type in ("for_statement", "for_in_statement",
                           "enhanced_for_statement", "for_range_statement"):
            init = node.child_by_field_name("initializer") or node.child_by_field_name("init")
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if init:
                ids = _get_ids(init)
                for v in ids[:1]:
                    dfg.append((v["name"], v["start"], "comesFrom", ids[1:]))
                    scope[v["name"]] = v["start"]
            if left and right:
                for lv in _get_ids(left):
                    dfg.append((lv["name"], lv["start"], "comesFrom", _get_ids(right)))
                    scope[lv["name"]] = lv["start"]

        # Return
        elif node.type in ("return_statement", "return_expression"):
            for v in _get_ids(node):
                if v["name"] in scope:
                    dfg.append((v["name"], v["start"], "comesFrom",
                               [{"name": v["name"], "start": scope[v["name"]]}]))

        for child in node.children:
            walk(child, depth + 1)

    walk(tree.root_node)
    return [], dfg


def extract_dfg(code, language):
    """Unified DFG: official functions if available, else tree-sitter AST."""
    if _official_available and language in _official_dfg_fns and language in _parsers_ready:
        toks, dfg = extract_dfg_official(code, language)
        if dfg:
            return toks, dfg
    # Fallback to tree-sitter AST
    return extract_dfg_treesitter(code, language)


# ═══════════════════════════════════════
# CFG EXTRACTION (via parser_utils parsers)
# ═══════════════════════════════════════

def extract_cfg(code, language):
    """CFG edges from tree-sitter AST."""
    tree = get_tree(code, language)
    if tree is None:
        return []
    edges = []
    _cfg_walk(tree.root_node, edges, set())
    return edges


def _cfg_walk(node, edges, visited, depth=0):
    if depth > 80 or id(node) in visited:
        return
    visited.add(id(node))

    # Handle Ruby specifically
    if node.type == "if":  # Ruby if statement
        cond = None
        then_block = None
        else_block = None
        
        for child in node.children:
            if child.type == "binary":
                cond = child
            elif child.type == "then":
                then_block = child
            elif child.type == "else":
                else_block = child
            elif child.type == "elsif":
                else_block = child  # treat elsif as else
        
        c = cond.start_byte if cond else node.start_byte
        if then_block:
            edges.append((c, then_block.start_byte))
        if else_block:
            edges.append((c, else_block.start_byte))

    elif node.type == "unless":  # Ruby unless
        cond = None
        then_block = None
        else_block = None
        
        for child in node.children:
            if child.type == "binary":
                cond = child
            elif child.type == "then":
                then_block = child
            elif child.type == "else":
                else_block = child
        
        c = cond.start_byte if cond else node.start_byte
        if then_block:
            edges.append((c, then_block.start_byte))
        if else_block:
            edges.append((c, else_block.start_byte))

    elif node.type == "while" or node.type == "until":  # Ruby loops
        cond = None
        body = None
        
        for child in node.children:
            if child.type == "binary":
                cond = child
            elif child.type == "do" or child.type == "then":
                body = child
        
        c = cond.start_byte if cond else node.start_byte
        if body:
            edges.append((c, body.start_byte))
            # Back edge for loops
            edges.append((body.end_byte, c))
    elif node.type == "block":  # Ruby blocks (9.times, [1,2,3].each, etc.)
        # Find block_body (the code inside {|x| ... })
        block_body = None
        for child in node.children:
            if child.type == "block_body":
                block_body = child
                break
        
        if block_body:
            # Find what this block is attached to (method call like "9.times" or "[1,2,3].each")
            parent = node.parent
            if parent:
                # Try to find the method call that owns this block
                method_start = node.start_byte
                # Look backward for the method call
                for i in range(node.start_byte - 1, max(0, node.start_byte - 50), -1):
                    if parent.start_byte <= i < parent.end_byte:
                        method_start = i
                        break
                
                # Add edge from method to block body
                edges.append((method_start, block_body.start_byte))
                # Add back edge from block end to after block
                edges.append((block_body.end_byte, node.end_byte))
    
    elif node.type == "for":  # Ruby for loops
        # Ruby: for i in 1..n do ... end
        in_node = None
        body = None
        
        for child in node.children:
            if child.type == "in":
                in_node = child
            elif child.type == "do" or child.type == "body":
                body = child
        
        if in_node and body:
            edges.append((in_node.start_byte, body.start_byte))
            edges.append((body.end_byte, in_node.start_byte))  # Back edge

    elif node.type == "case":  # Ruby case statement
        subject = None
        for child in node.children:
            if child.type == "case":
                subject = child
                break
        
        v = subject.start_byte if subject else node.start_byte
        for child in node.children:
            if child.type == "when":
                edges.append((v, child.start_byte))

    elif node.type == "begin":  # Ruby begin-rescue (try-catch)
        body = None
        for child in node.children:
            if child.type == "body":
                body = child
                break
        
        if body:
            edges.append((node.start_byte, body.start_byte))
        for child in node.children:
            if child.type == "rescue" or child.type == "ensure":
                edges.append((node.start_byte, child.start_byte))

    # Original logic for other languages
    elif node.type in ("if_statement", "if_expression"):
        cond = node.child_by_field_name("condition")
        cons = node.child_by_field_name("consequence") or node.child_by_field_name("body")
        alt = node.child_by_field_name("alternative")
        c = cond.start_byte if cond else node.start_byte
        if cons:
            edges.append((c, cons.start_byte))
        if alt:
            edges.append((c, alt.start_byte))

    elif node.type in ("for_statement", "while_statement", "do_statement",
                       "for_in_statement", "enhanced_for_statement",
                       "while_expression", "for_expression"):
        cond = (node.child_by_field_name("condition") or
                node.child_by_field_name("right") or
                node.child_by_field_name("value"))
        body = node.child_by_field_name("body")
        c = cond.start_byte if cond else node.start_byte
        if body:
            edges.append((c, body.start_byte))
            edges.append((body.end_byte, c))  # Back edge

    elif node.type == "try_statement":
        body = node.child_by_field_name("body")
        if body:
            edges.append((node.start_byte, body.start_byte))
        for ch in node.children:
            if ch.type in ("except_clause", "catch_clause", "finally_clause",
                          "rescue_clause", "ensure_clause"):
                edges.append((node.start_byte, ch.start_byte))

    elif node.type in ("switch_statement", "case_statement", "match_expression"):
        val = node.child_by_field_name("value") or node.child_by_field_name("subject")
        v = val.start_byte if val else node.start_byte
        for ch in node.children:
            if ch.type in ("case_clause", "switch_case", "match_arm", "when_clause"):
                edges.append((v, ch.start_byte))

    for ch in node.children:
        _cfg_walk(ch, edges, visited, depth + 1)
# ═══════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════

def code_to_graph_data(code, language, tokenizer, model,
                        max_code_length=256, max_dfg_length=64,
                        device="cuda:1", include_cfg=True):
    """Code → PyG Data with 4 edge types."""

    # 1. Tokenize (cap at 510 for RoBERTa 512 limit)
    max_tok = min(max_code_length, 510)
    tokens = tokenizer.tokenize(code)[:max_tok - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 2. DFG
    _, dfg = extract_dfg(code, language)
    toks_lower = [t.lower().replace("Ġ", "").replace("▁", "") for t in tokens]

    dfg_nodes = []
    for i, entry in enumerate(dfg[:max_dfg_length]):
        if len(entry) < 4:
            continue
        var = entry[0]
        deps = entry[-1]
        positions = [j for j, tok in enumerate(toks_lower) if tok == var.lower()]
        if positions:
            dfg_nodes.append({"name": var, "pos": positions, "deps": deps,
                             "idx": entry[1] if len(entry) > 1 else i})

    # 3. CFG
    cfg_raw = extract_cfg(code, language) if include_cfg else []

    # byte→token map for CFG
    b2t = {}
    if cfg_raw:
        cur = 0
        for ti, tok in enumerate(tokens[1:-1], 1):
            txt = tok.replace("Ġ", " ").replace("▁", " ").strip()
            if txt:
                pos = code.find(txt, max(0, cur - 5))
                if pos >= 0:
                    for b in range(pos, min(pos + len(txt), pos + 20)):
                        b2t[b] = ti
                    cur = pos + len(txt)

    # 4. Embeddings
    ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        emb = model(input_ids=ids, attention_mask=mask).last_hidden_state[0]

    # 5. Node features
    nc = len(token_ids)
    nd = len(dfg_nodes)
    total = nc + nd
    dim = emb.shape[-1]

    x = torch.zeros(total, dim)
    x[:nc] = emb[:nc].cpu()
    for i, dn in enumerate(dfg_nodes):
        if dn["pos"]:
            x[nc + i] = emb[dn["pos"]].mean(dim=0).cpu()

    # 6. Edges
    edges, etypes = [], []

    # Type 0: Sequential
    for i in range(nc - 1):
        edges.extend([(i, i + 1), (i + 1, i)])
        etypes.extend([0, 0])

    # Type 1: DFG-to-code
    for i, dn in enumerate(dfg_nodes):
        for p in dn["pos"]:
            edges.extend([(nc + i, p), (p, nc + i)])
            etypes.extend([1, 1])

    # Type 2: DFG-to-DFG
    idx_map = {d["idx"]: i for i, d in enumerate(dfg_nodes)}
    name_map = {d["name"]: i for i, d in enumerate(dfg_nodes)}
    for i, dn in enumerate(dfg_nodes):
        for dep in dn["deps"]:
            j = None
            if isinstance(dep, int):
                j = idx_map.get(dep)
            elif isinstance(dep, dict):
                j = name_map.get(dep.get("name"))
            elif isinstance(dep, str):
                j = name_map.get(dep)
            if j is not None and i != j:
                edges.extend([(nc + i, nc + j), (nc + j, nc + i)])
                etypes.extend([2, 2])

    # Type 3: CFG
    cfg_count = 0
    for sb, tb in cfg_raw:
        st = b2t.get(sb)
        tt = b2t.get(tb)
        if st is None:
            for off in range(-5, 6):
                st = b2t.get(sb + off)
                if st is not None:
                    break
        if tt is None:
            for off in range(-5, 6):
                tt = b2t.get(tb + off)
                if tt is not None:
                    break
        if st is not None and tt is not None and st != tt and 0 <= st < nc and 0 <= tt < nc:
            edges.extend([(st, tt), (tt, st)])
            etypes.extend([3, 3])
            cfg_count += 1

    if not edges:
        edges = [(0, 0)]
        etypes = [0]

    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    et = torch.tensor(etypes, dtype=torch.long)
    nt = torch.zeros(total, dtype=torch.long)
    nt[nc:] = 1

    return Data(x=x, edge_index=ei, edge_type=et, num_nodes=total,
                node_types=nt, num_code_tokens=nc, num_dfg_nodes=nd,
                num_cfg_edges=cfg_count)


# ═══════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════

def build_graph_dataset(pairs_path, output_path,
                         model_name="/data/workzone/local_models/graphcodebert-base",
                         max_code_length=256, max_dfg_length=64,
                         device="cuda:1", include_cfg=True, pair_timeout=30,
                         resume=True):
    from transformers import AutoTokenizer, AutoModel

    # Resume from checkpoint?
    ckpt = output_path + ".checkpoint"
    start_idx = 0
    gdata = {"graph_a": [], "graph_b": [], "labels": [], "metadata": []}
    if resume and os.path.exists(ckpt):
        print(f"Resuming from {ckpt}")
        gdata = torch.load(ckpt, map_location="cpu", weights_only=False)
        start_idx = len(gdata["labels"])
        print(f"  {start_idx} existing graphs")

    print(f"Loading GraphCodeBERT → {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    with open(pairs_path) as f:
        pairs = json.load(f)
    print(f"Processing {len(pairs)} pairs (from {start_idx}, CFG={'ON' if include_cfg else 'OFF'})")

    lc = defaultdict(int)
    for p in pairs:
        lc[p.get("language", "?")] += 1
    for lang, c in sorted(lc.items(), key=lambda x: -x[1]):
        dfg_method = "official" if (_official_available and lang in _official_dfg_fns) else "tree-sitter"
        cfg_method = "✓" if lang in _parsers_ready else "✗"
        parser_status = "✓" if lang in _parsers_ready else "✗"
        print(f"  {lang}: {c} pairs | parser:{parser_status} DFG:{dfg_method} CFG:{cfg_method}")

    failed = 0
    timeouts = 0

    for i in tqdm(range(start_idx, len(pairs)), desc="Building graphs",
                  initial=start_idx, total=len(pairs)):
        pair = pairs[i]
        correct = pair.get("correct_code", "")
        buggy = pair.get("buggy_code", "")
        language = pair.get("language", "python")

        if not correct.strip() or not buggy.strip():
            failed += 1
            continue

        try:
            old = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(pair_timeout)

            ga = code_to_graph_data(correct, language, tokenizer, model,
                                     max_code_length, max_dfg_length, device, include_cfg)
            gb = code_to_graph_data(buggy, language, tokenizer, model,
                                     max_code_length, max_dfg_length, device, include_cfg)

            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

            if ga is None or gb is None:
                failed += 1
                continue

            gdata["graph_a"].append(ga)
            gdata["graph_b"].append(gb)
            gdata["labels"].append(0)
            gdata["metadata"].append({
                "pair_id": pair.get("pair_id", f"pair_{i}"),
                "dataset": pair.get("dataset", "unknown"),
                "language": language,
                "bug_type": pair.get("bug_type", "unknown"),
            })
        except PairTimeout:
            timeouts += 1
            signal.alarm(0)
            if timeouts <= 5:
                tqdm.write(f"  ⏰ Timeout pair {i}")
        except Exception as e:
            failed += 1
            signal.alarm(0)
            if failed <= 5:
                tqdm.write(f"  Error pair {i}: {str(e)[:80]}")

        if len(gdata["labels"]) % 5000 == 0 and len(gdata["labels"]) > start_idx:
            tqdm.write(f"  Checkpoint: {len(gdata['labels'])} built, {failed} failed, {timeouts} timeouts")
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(gdata, ckpt)

    print(f"\nDONE: {len(gdata['labels'])} graphs ({failed} failed, {timeouts} timeouts)")

    if gdata["graph_a"]:
        langs = defaultdict(int)
        for m in gdata["metadata"]:
            langs[m["language"]] += 1
        print(f"\nBy language:")
        for l, c in sorted(langs.items(), key=lambda x: -x[1]):
            print(f"  {l}: {c}")

        avg_n = np.mean([g.num_nodes for g in gdata["graph_a"]])
        avg_e = np.mean([g.edge_index.shape[1] for g in gdata["graph_a"]])
        avg_dfg = np.mean([g.num_dfg_nodes for g in gdata["graph_a"]])
        avg_cfg = np.mean([g.num_cfg_edges for g in gdata["graph_a"]])

        tc = defaultdict(int)
        for g in gdata["graph_a"]:
            for t in g.edge_type.tolist():
                tc[t] += 1

        print(f"\nGraph stats:")
        print(f"  Avg nodes: {avg_n:.1f}, edges: {avg_e:.1f}, DFG: {avg_dfg:.1f}, CFG: {avg_cfg:.1f}")
        tn = {0: "sequential", 1: "dfg-code", 2: "dfg-dfg", 3: "cfg"}
        te = sum(tc.values())
        for t in sorted(tc):
            print(f"    {tn.get(t, f'type_{t}')}: {tc[t]} ({100*tc[t]/te:.1f}%)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(gdata, output_path)
    print(f"\nSaved to {output_path}")
    if os.path.exists(ckpt):
        os.remove(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--model", default="/data/workzone/local_models/graphcodebert-base")
    ap.add_argument("--max-code-length", type=int, default=256)
    ap.add_argument("--max-dfg-length", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-cfg", action="store_true")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--check-gpu", action="store_true")
    args = ap.parse_args()

    if args.check_gpu:
        detect_best_gpu()
        return
    if not args.pairs:
        print("ERROR: --pairs required")
        return
    device = detect_best_gpu() if args.device == "auto" else args.device
    if args.output is None:
        bn = os.path.basename(args.pairs).replace("pairs_", "graph_data_").replace(".json", ".pt")
        args.output = os.path.join("./data/graphs", bn)

    print(f"Input:  {args.pairs}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print(f"CFG: {'OFF' if args.no_cfg else 'ON'}, Timeout: {args.timeout}s\n")

    init_parsers()
    init_official_dfg()
    print()

    build_graph_dataset(args.pairs, args.output, args.model,
                         args.max_code_length, args.max_dfg_length, device,
                         not args.no_cfg, args.timeout, not args.no_resume)


if __name__ == "__main__":
    main()