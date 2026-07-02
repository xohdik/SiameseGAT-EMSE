"""
PROBE 3: (A) multilingual clean-bug count, (B) localization FLOOR.

Part A: run execution feasibility for all HumanEvalFix languages that have a
        runnable interpreter here (python definitely; others if available).
        => total clean, executable, located bug set size.

Part B: on the PYTHON clean set, compute how often DUMB baselines hit the true
        bug line(s). Establishes the floor any real localizer must beat.

  Baselines (all cheap, no training):
    - random_line         : uniform random code line
    - longest_line        : the longest (proxy: most tokens) code line
    - max_perplexity_line : line GraphCodeBERT finds most surprising
                            (per-token NLL averaged per line; needs GPU/CPU model)

  Ground truth bug lines = lines that differ between correct and buggy (diff).
  We localize on the BUGGY code (that's the realistic input).

  Metrics: top-1, top-3, MFR (mean first rank), plus n.

Reading:
  If max_perplexity (or longest_line) already gets HIGH top-3 (>~0.65)
      => trivial methods already localize; little room for a fancy method. Risky paper.
  If floor is modest (random ~ 1/n_lines, heuristics ~0.3-0.5)
      => real room for a method to beat the floor. Door for a METHOD paper open.

Usage:
  # Part A only (fast, no model):
  python probe3_floor.py --count-only
  # Full (loads GraphCodeBERT for perplexity baseline):
  python probe3_floor.py --model /data/workzone/local_models/graphcodebert-base --device cuda:1
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile, random
import difflib
from collections import Counter
import numpy as np

# ---------- execution (reused from probe 2) ----------
def entry_point(code):
    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", code, re.M)
    return m.group(1) if m else None

def run_python(code, test, timeout=10):
    fn = entry_point(code)
    if not fn: return "error:no_entry"
    tb = re.sub(r"\ncheck\([^\)]*\)\s*$", "\n", test.rstrip()) + "\n"
    script = f"{code}\n\n{tb}\ncheck({fn})\nprint('___PASS___')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script); path=f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        if "___PASS___" in r.stdout: return "pass"
        return "fail" if "AssertionError" in r.stderr else "fail"
    except subprocess.TimeoutExpired: return "error:timeout"
    except Exception as e: return f"error:{str(e)[:40]}"
    finally:
        try: os.unlink(path)
        except: pass

# ---------- ground-truth bug lines ----------
def bug_lines(correct, buggy):
    """1-indexed line numbers in BUGGY that differ from correct."""
    c = correct.split("\n"); b = buggy.split("\n")
    sm = difflib.SequenceMatcher(None, c, b)
    lines = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            for j in range(j1, j2): lines.add(j+1)        # buggy-side lines
        elif tag == "delete":
            lines.add(j1+1 if j1 < len(b) else max(1, len(b)))  # deletion point
    return lines or {1}

# ---------- candidate code lines (skip blank/comment) ----------
def code_line_indices(code):
    out=[]
    for i, l in enumerate(code.split("\n"), 1):
        s=l.strip()
        if s and not s.startswith("#"):
            out.append(i)
    return out

# ---------- Part A: multilingual clean-bug count ----------
def part_a(pairs_dir):
    print("="*74); print("PART A: clean executable bugs per language"); print("="*74)
    files = sorted(glob.glob(os.path.join(pairs_dir, "pairs_humanevalfix_*.json")))
    total_python_clean = []
    grand = 0
    for fp in files:
        lang = os.path.basename(fp).replace("pairs_humanevalfix_","").replace(".json","")
        data = json.load(open(fp))
        if lang != "python":
            # only python is executable here without extra toolchains
            print(f"  {lang:12s} {len(data):4d} pairs   (not executed: needs {lang} runtime)")
            continue
        clean=[]
        for i, r in enumerate(data):
            cc,bc,t = r.get("correct_code",""),r.get("buggy_code",""),r.get("test","")
            if not (cc and bc and t): continue
            if run_python(cc,t)=="pass" and run_python(bc,t)=="fail":
                clean.append(r)
        print(f"  {lang:12s} {len(data):4d} pairs   CLEAN executable bugs = {len(clean)}")
        total_python_clean = clean
        grand += len(clean)
    print(f"\n  Python clean executable bug set: {grand}")
    print(f"  (other languages have located-by-diff bugs too, just not run here)")
    return total_python_clean

# ---------- Part B: localization floor ----------
def perplexity_lines(code, tok, model, device):
    """Per-line average NLL under GraphCodeBERT MLM. Higher = more surprising."""
    import torch
    lines = code.split("\n")
    # map each token to its line by re-tokenizing line by line
    line_scores = {}
    for i, l in enumerate(lines, 1):
        s=l.strip()
        if not s or s.startswith("#"): continue
        ids = tok(l, return_tensors="pt", truncation=True, max_length=64)
        input_ids = ids["input_ids"].to(device)
        if input_ids.shape[1] < 3:
            line_scores[i]=0.0; continue
        with torch.no_grad():
            # mask each token, measure NLL (cheap approx: single forward, use loss vs self)
            out = model(input_ids=input_ids, attention_mask=ids["attention_mask"].to(device))
            logits = out.logits if hasattr(out,"logits") else out[0]
        import torch.nn.functional as F
        lp = F.log_softmax(logits[0], dim=-1)
        tok_nll = -lp[range(input_ids.shape[1]), input_ids[0]]
        line_scores[i] = float(tok_nll[1:-1].mean().item()) if input_ids.shape[1]>2 else 0.0
    return line_scores

def rank_of_truth(ranked_lines, truth):
    for rank, ln in enumerate(ranked_lines, 1):
        if ln in truth: return rank
    return len(ranked_lines)+1

def part_b(clean, model_path, device, seed=42):
    print("\n"+"="*74); print("PART B: localization FLOOR on python clean bugs"); print("="*74)
    if not clean:
        print("  no clean python bugs; skip"); return
    rng = random.Random(seed)

    use_model = model_path is not None
    if use_model:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            tok = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForMaskedLM.from_pretrained(model_path).to(device).eval()
            print(f"  loaded {model_path} on {device}")
        except Exception as e:
            print(f"  model load failed ({str(e)[:60]}); running heuristic baselines only")
            use_model=False

    res = {"random":[], "longest":[], "perplexity":[]}
    nlines_list=[]
    for r in clean:
        buggy = r["buggy_code"]; correct = r["correct_code"]
        truth = bug_lines(correct, buggy)
        cand = code_line_indices(buggy)
        if not cand: continue
        nlines_list.append(len(cand))

        # random
        ranked = cand[:]; rng.shuffle(ranked)
        res["random"].append(rank_of_truth(ranked, truth))

        # longest line (most tokens)
        toks = {i: len(re.findall(r"\w+|[^\s\w]", buggy.split(chr(10))[i-1])) for i in cand}
        ranked = sorted(cand, key=lambda i:-toks[i])
        res["longest"].append(rank_of_truth(ranked, truth))

        # perplexity
        if use_model:
            ps = perplexity_lines(buggy, tok, model, device)
            ranked = sorted(cand, key=lambda i:-ps.get(i,0.0))
            res["perplexity"].append(rank_of_truth(ranked, truth))

    n = len(res["random"])
    print(f"  n={n} clean python bugs, median code lines/func = {int(np.median(nlines_list))}")
    print(f"  random top-1 expectation ~= {np.mean([1/x for x in nlines_list]):.3f}\n")
    print(f"  {'method':14s} {'top1':>6} {'top3':>6} {'MFR':>6}")
    for name, ranks in res.items():
        if not ranks: continue
        ranks=np.array(ranks)
        top1=(ranks<=1).mean(); top3=(ranks<=3).mean(); mfr=ranks.mean()
        print(f"  {name:14s} {top1:6.3f} {top3:6.3f} {mfr:6.2f}")
    print("\n  READ: if perplexity/longest top-3 is already HIGH (>0.65), trivial")
    print("        methods localize well -> a fancy method has little headroom.")
    print("        if floor is modest -> real room for a method. ")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default="data/processed")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--count-only", action="store_true")
    args=ap.parse_args()
    clean = part_a(args.pairs_dir)
    if not args.count_only:
        part_b(clean, args.model, args.device)

if __name__=="__main__":
    main()