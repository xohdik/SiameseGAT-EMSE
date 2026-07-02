"""
Does the component decomposition survive its own construct-validity critique?

§9.1 showed CodeNet F1 falls from ~0.92 to ~0.67 on near-identical pairs (where
surface/style cues are absent). The reviewer's key question: were the ABLATION
conclusions (edges matter, attention matters) measuring bug localization, or the
same surface confound on the easy majority?

This script stratifies EACH ablation configuration by edit distance and reports
F1 per bin, so you can see whether the configuration ranking (e.g. full > seq_only)
holds on near-identical pairs. If full still beats seq_only when surface cues are
gone, the edge-structure finding is real bug-discrimination, not a confound.

------------------------------------------------------------------------------
STEP 1 -- dump per-pair predictions for each ablation config (reuses
dump_preds_from_ckpt.py with the matching --edge-filter). Full model is already
in outputs/cv_<lang>; add the edge ablations:

    for L in python cpp java c ruby javascript; do
      for E in seq_only cfg_only dfg_only; do
        python dump_preds_from_ckpt.py --lang $L \
          --ckpt-dir ../outputs/siamese_gat_${L}_${E} \
          --edge-filter $E \
          --out ../outputs/cvabl_${E}_${L}/siamesegat_preds.json
      done
    done

STEP 2 -- run this (quote the patterns so the shell doesn't expand them):

    python ablation_edit_distance.py \
      --pairs ../data/processed/pairs_all.json \
      --config 'full=../outputs/cv_*/siamesegat_preds.json' \
      --config 'seq_only=../outputs/cvabl_seq_only_*/siamesegat_preds.json' \
      --config 'cfg_only=../outputs/cvabl_cfg_only_*/siamesegat_preds.json' \
      --config 'dfg_only=../outputs/cvabl_dfg_only_*/siamesegat_preds.json' \
      --out ../outputs/ablation_cv
------------------------------------------------------------------------------
"""
import argparse, glob, json, os, re
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

try:
    import Levenshtein as _Lev
    _BK = "levenshtein"
except Exception:
    from difflib import SequenceMatcher as _SM
    _BK = "difflib"

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|==|!=|<=|>=|&&|\|\||[^\s\w]")

def tok(c): return TOKEN_RE.findall(c or "")

def ted(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0: return max(n, m)
    if _BK == "levenshtein":
        v = {}
        enc = lambda ts: "".join(v.setdefault(t, chr(len(v))) for t in ts)
        return _Lev.distance(enc(a), enc(b))
    return int(round((n + m) * (1.0 - _SM(None, a, b, autojunk=False).ratio()) / 2.0))

BINS = [(0.00, 0.10, "near-identical (<=0.10)"),
        (0.10, 0.25, "small (0.10-0.25)"),
        (0.25, 0.50, "medium (0.25-0.50)"),
        (0.50, 1.01, "large (>0.50)")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--config", action="append", required=True,
                    help="NAME=glob_pattern (repeatable). Quote the pattern.")
    ap.add_argument("--out", default="./ablation_cv")
    ap.add_argument("--max-tokens", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"edit-distance backend: {_BK}")

    pairs = {p["pair_id"]: p for p in json.load(open(args.pairs))
             if p.get("correct_code") and p.get("buggy_code")}

    # parse configs
    configs = {}
    for spec in args.config:
        name, pat = spec.split("=", 1)
        files = sorted(glob.glob(pat))
        recs = []
        for f in files:
            recs.extend(json.load(open(f)))
        configs[name] = recs
        print(f"  config '{name}': {len(files)} files, {len(recs)} records")

    # edit distance cache (codenet pairs only)
    ed_cache = {}
    def rel_ed(pid):
        if pid in ed_cache: return ed_cache[pid]
        p = pairs.get(pid)
        if p is None: ed_cache[pid] = None; return None
        ta, tb = tok(p["correct_code"]), tok(p["buggy_code"])
        if max(len(ta), len(tb)) > args.max_tokens:
            d = 1.0
        else:
            d = (ted(ta, tb) / max(len(ta), len(tb))) if (ta or tb) else 0.0
        ed_cache[pid] = d
        return d

    # per config, per bin: collect (label, pred)
    table = {}      # name -> bin_name -> (f1, acc, n)
    overall = {}    # name -> (f1, n)
    for name, recs in configs.items():
        binned = {b[2]: {"y": [], "p": []} for b in BINS}
        ally, allp = [], []
        miss = 0
        for r in recs:
            if not str(r.get("dataset", "")).startswith("codenet"): continue
            d = rel_ed(r["pair_id"])
            if d is None: miss += 1; continue
            for lo, hi, bn in BINS:
                if lo <= d < hi:
                    binned[bn]["y"].append(int(r["label"])); binned[bn]["p"].append(int(r["pred"]))
                    break
            ally.append(int(r["label"])); allp.append(int(r["pred"]))
        table[name] = {}
        for _, _, bn in BINS:
            y, p = binned[bn]["y"], binned[bn]["p"]
            table[name][bn] = (f1_score(y, p, average="macro"), accuracy_score(y, p), len(y)) if y else (None, None, 0)
        overall[name] = (f1_score(ally, allp, average="macro") if ally else None, len(ally))
        if miss: print(f"  ({name}: {miss} records unmatched to pairs)")

    # print table: rows=bins, cols=configs
    names = list(configs.keys())
    print("\n" + "=" * (26 + 13 * len(names)))
    hdr = f"{'Edit-distance bin':<26}" + "".join(f"{n:>13}" for n in names)
    print(hdr); print("-" * len(hdr))
    for _, _, bn in BINS:
        row = f"{bn:<26}"
        for n in names:
            f1, acc, k = table[n][bn]
            row += f"{(f'{f1:.3f}' if f1 is not None else '-'):>13}"
        print(row)
    print("-" * len(hdr))
    row = f"{'ALL CodeNet':<26}"
    for n in names:
        f1, k = overall[n]
        row += f"{(f'{f1:.3f}' if f1 is not None else '-'):>13}"
    print(row)
    # n per bin (from first config)
    print("\nn per bin:", {bn: table[names[0]][bn][2] for _, _, bn in BINS})
    print("=" * len(hdr))

    # verdict: does 'full' keep its edge over ablations on near-identical pairs?
    if "full" in table:
        ni = BINS[0][2]
        print(f"\nOn near-identical pairs ({ni}):")
        f_full = table["full"][ni][0]
        for n in names:
            if n == "full": continue
            f_abl = table[n][ni][0]
            if f_full is not None and f_abl is not None:
                gap = f_full - f_abl
                verdict = "edges still help (signal real)" if gap > 0.02 else \
                          "gap closes (was confound)" if gap < -0.02 else "no difference"
                print(f"  full ({f_full:.3f}) vs {n} ({f_abl:.3f}): delta {gap:+.3f}  => {verdict}")

    json.dump({"table": {n: {b: table[n][b][:2] for b in table[n]} for n in table},
               "overall": overall}, open(os.path.join(args.out, "ablation_cv_summary.json"), "w"), indent=2)

    if HAVE_PLT:
        binlabels = [b[2].split(" (")[0] for b in BINS]
        xpos = np.arange(len(BINS)); w = 0.8 / max(1, len(names))
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        palette = ["#4a9d6f", "#3b6ea5", "#e08a3c", "#8a8a8a", "#b23a3a"]
        for i, n in enumerate(names):
            vals = [table[n][b[2]][0] or 0 for b in BINS]
            ax.bar(xpos + (i - (len(names) - 1) / 2) * w, vals, w, label=n, color=palette[i % len(palette)])
        ax.set_xticks(xpos); ax.set_xticklabels(binlabels)
        ax.set_ylabel("F1 macro"); ax.set_ylim(0, 1.0)
        ax.set_title("Ablation F1 by edit-distance stratum (does the decomposition survive deconfounding?)", fontsize=9)
        ax.legend(frameon=False, fontsize=8, ncol=len(names))
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(os.path.join(args.out, "ablation_edit_distance.png"), dpi=200, bbox_inches="tight")
        print(f"\nFigure: {os.path.join(args.out, 'ablation_edit_distance.png')}")

if __name__ == "__main__":
    main()