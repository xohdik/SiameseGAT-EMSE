"""
Construct-validity check: does SiameseGAT's in-distribution accuracy survive on
NEAR-IDENTICAL CodeNet pairs, or does it ride on author/style/length priors?

CodeNet correct/incorrect pairs come from DIFFERENT authors, so two same-problem
submissions can differ in length and idiom, not just correctness. If accuracy holds
when the two programs are almost identical (small token edit distance), the model is
discriminating the bug. If it collapses as the programs become more similar, the
in-distribution number is partly a surface confound. This is the TOSEM reviewer's
question; this script answers it.

------------------------------------------------------------------------------
STEP 1 (in YOUR eval code) -- dump per-pair predictions for the CodeNet test folds.
Wherever you compute test predictions, collect one record per pair and save JSON:

    records = []   # accumulate across folds
    # inside the test loop, for each pair in the batch:
    #   records.append({
    #       "pair_id":  pair_id,            # MUST match pair_id in pairs_all.json
    #       "dataset":  dataset,           # e.g. "codenet_python"
    #       "label":    int(true_label),   # 1 if A is correct, else 0 (your convention)
    #       "pred":     int(pred_label),   # model's predicted label
    #   })
    import json; json.dump(records, open("siamesegat_preds.json","w"))

Only CodeNet records are needed (HumanEvalFix pairs are minimal-edit by construction,
so they already control for style -- that is the complementary control in the paper).

STEP 2 -- run this script:
    python edit_distance_analysis.py \
        --pairs ../data/processed/pairs_all.json \
        --preds siamesegat_preds.json \
        --out   ./outputs/construct_validity
------------------------------------------------------------------------------
"""
import argparse, json, os, re
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|==|!=|<=|>=|&&|\|\||[^\s\w]")

def tokenize(code: str):
    return TOKEN_RE.findall(code or "")

try:
    import Levenshtein as _Lev
    _BACKEND = "levenshtein"
except Exception:
    from difflib import SequenceMatcher as _SM
    _BACKEND = "difflib"

def token_edit_distance(a_toks, b_toks):
    """True token-level Levenshtein, C-accelerated.
    Maps each pair's tokens to chars (per-pair vocab is tiny) then runs the C
    Levenshtein. Falls back to difflib similarity if the library is absent."""
    n, m = len(a_toks), len(b_toks)
    if n == 0 or m == 0:
        return max(n, m)
    if _BACKEND == "levenshtein":
        vocab = {}
        def enc(toks):
            return "".join(vocab.setdefault(t, chr(len(vocab))) for t in toks)
        return _Lev.distance(enc(a_toks), enc(b_toks))
    # difflib fallback: ratio-based edit count (insert+delete), close proxy
    ratio = _SM(None, a_toks, b_toks, autojunk=False).ratio()
    return int(round((n + m) * (1.0 - ratio) / 2.0)) + abs(n - m) // 1 * 0

def rel_edit_distance(code_a, code_b):
    ta, tb = tokenize(code_a), tokenize(code_b)
    if not ta and not tb:
        return 0.0
    return token_edit_distance(ta, tb) / max(len(ta), len(tb))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="../data/processed/pairs_all.json")
    ap.add_argument("--preds", required=True, nargs="+",
                    help="one or more per-pair prediction JSONs (e.g. outputs/cv_*/siamesegat_preds.json)")
    ap.add_argument("--out", default="./outputs/construct_validity")
    ap.add_argument("--max-tokens", type=int, default=2000,
                    help="skip edit-distance DP above this length (kept, marked 'large')")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pairs = {p["pair_id"]: p for p in json.load(open(args.pairs))
             if p.get("correct_code") and p.get("buggy_code")}
    preds = []
    for pf in args.preds:
        recs = json.load(open(pf))
        preds.extend(recs)
        print(f"  + {pf}: {len(recs)} records")
    print(f"Loaded {len(pairs)} pairs, {len(preds)} prediction records from {len(args.preds)} file(s)")
    print(f"Edit-distance backend: {_BACKEND}")

    # join predictions with edit distance (CodeNet only)
    rows = []
    missing = 0
    for r in preds:
        if not str(r.get("dataset", "")).startswith("codenet"):
            continue
        p = pairs.get(r["pair_id"])
        if p is None:
            missing += 1
            continue
        ta, tb = tokenize(p["correct_code"]), tokenize(p["buggy_code"])
        if max(len(ta), len(tb)) > args.max_tokens:
            d = 1.0  # too long to DP cheaply; treat as 'large' difference
        else:
            d = (token_edit_distance(ta, tb) / max(len(ta), len(tb))) if (ta or tb) else 0.0
        rows.append({"rel_ed": d, "label": int(r["label"]), "pred": int(r["pred"]),
                     "lang": r["dataset"].split("_")[-1]})
    if missing:
        print(f"  ({missing} prediction records had no matching pair_id -- check id alignment)")
    if not rows:
        raise SystemExit("No CodeNet predictions joined. Check --preds dataset/pair_id fields.")

    eds = np.array([r["rel_ed"] for r in rows])
    print(f"\nRelative token edit distance over {len(rows)} CodeNet pairs: "
          f"min={eds.min():.3f} median={np.median(eds):.3f} max={eds.max():.3f}")

    # fixed, interpretable bins
    bins = [(0.00, 0.10, "near-identical (<=0.10)"),
            (0.10, 0.25, "small (0.10-0.25)"),
            (0.25, 0.50, "medium (0.25-0.50)"),
            (0.50, 1.01, "large (>0.50)")]
    def f1acc(rs):
        y = [r["label"] for r in rs]; yh = [r["pred"] for r in rs]
        return f1_score(y, yh, average="macro"), accuracy_score(y, yh)

    print("\n" + "=" * 64)
    print(f"{'Edit-distance bin':<26}{'n':>7}{'F1 macro':>11}{'Accuracy':>11}")
    print("-" * 64)
    summary = []
    for lo, hi, name in bins:
        rs = [r for r in rows if lo <= r["rel_ed"] < hi]
        if not rs:
            print(f"{name:<26}{0:>7}{'-':>11}{'-':>11}"); continue
        f1, acc = f1acc(rs)
        print(f"{name:<26}{len(rs):>7}{f1:>11.4f}{acc:>11.4f}")
        summary.append((name, len(rs), f1, acc))
    f1_all, acc_all = f1acc(rows)
    print("-" * 64)
    print(f"{'ALL CodeNet':<26}{len(rows):>7}{f1_all:>11.4f}{acc_all:>11.4f}")
    print("=" * 64)

    # verdict heuristic
    near = [r for r in rows if r["rel_ed"] <= 0.10]
    if near:
        f1_near, _ = f1acc(near)
        drop = f1_all - f1_near
        print(f"\nNear-identical F1 = {f1_near:.3f} vs overall {f1_all:.3f}  (delta {drop:+.3f})")
        if f1_near >= f1_all - 0.05:
            print("=> Signal PRESERVED on near-identical pairs: evidence for genuine bug")
            print("   discrimination rather than stylistic priors. Strong for construct validity.")
        else:
            print("=> Signal DROPS on near-identical pairs: in-distribution accuracy is partly")
            print("   attributable to surface/style confounds. Report this honestly.")

    json.dump(summary, open(os.path.join(args.out, "edit_distance_summary.json"), "w"), indent=2)

    if HAVE_PLT and summary:
        names = [s[0].split(" (")[0] for s in summary]
        f1s = [s[2] for s in summary]; ns = [s[1] for s in summary]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(range(len(names)), f1s, color="#3b6ea5")
        ax.axhline(f1_all, ls="--", color="#4a9d6f", lw=1.2, label=f"overall F1 = {f1_all:.3f}")
        for i, (b, n) in enumerate(zip(bars, ns)):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{f1s[i]:.3f}\n(n={n})",
                    ha="center", fontsize=8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("F1 macro"); ax.set_ylim(0, 1.0)
        ax.set_title("SiameseGAT accuracy vs. correct/buggy edit distance (CodeNet)", fontsize=10)
        ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(os.path.join(args.out, "edit_distance_f1.png"), dpi=200, bbox_inches="tight")
        print(f"\nFigure: {os.path.join(args.out, 'edit_distance_f1.png')}")

if __name__ == "__main__":
    main()