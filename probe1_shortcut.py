"""
PROBE 1: Is correct-vs-buggy separable from SURFACE FEATURES ALONE?

Runs separately per regime/dataset. For each pair we build *differential*
surface features in RANDOM order (so position is not a cue) and ask a dumb
classifier (logistic regression) to predict which side is buggy.

Honest design:
  - features are surface-only: length, lines, tokens, char classes, indentation,
    keyword counts. NO embeddings, NO semantics.
  - pairwise + order-randomized: feature vector = f(side0) - f(side1); label = which
    side is buggy. A symmetric/positional shortcut cannot win.
  - GroupKFold by problem_id: no problem leaks across folds.
  - reports F1-macro mean+/-std and AUC per dataset.

Reading:
  HIGH on CodeNet (cross-author)  + CHANCE on HumanEvalFix (minimal-edit)
      => the cross-author benchmark is SHORTCUT-SOLVABLE from surface form.
         That is a real, publishable measurement finding.
  CHANCE everywhere
      => no surface shortcut; collapse seen earlier is genuine difficulty.

Usage:
  python probe1_shortcut.py --pairs-dir /data/workzone/siamese_gat_journal/data/processed
"""
import argparse, glob, json, os, re, random
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

KW = ["if", "else", "elif", "for", "while", "return", "def", "class", "try",
      "except", "break", "continue", "and", "or", "not", "in", "is", "==",
      "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "%", "+=", "-="]

def feats(code):
    """Surface-only feature vector for one program string."""
    if not code:
        code = ""
    lines = code.split("\n")
    nonblank = [l for l in lines if l.strip()]
    toks = re.findall(r"\w+|[^\s\w]", code)
    f = [
        len(code),                                  # char length
        len(lines),                                 # total lines
        len(nonblank),                              # non-blank lines
        len(toks),                                  # token count
        len(set(toks)),                             # unique tokens
        np.mean([len(l) for l in lines]) if lines else 0,   # avg line len
        max([len(l) - len(l.lstrip()) for l in lines] + [0]),  # max indent
        code.count("("), code.count("["), code.count("{"),
        code.count("="), code.count(":"), code.count(","),
        sum(c.isdigit() for c in code),             # digit count
        len(re.findall(r"[A-Za-z_]\w*", code)),     # identifier-ish count
    ]
    low = code.lower()
    f += [low.count(k) for k in KW]                 # keyword/operator counts
    return np.array(f, dtype=np.float64)

def load_pairs(path):
    with open(path) as fh:
        data = json.load(fh)
    rows = []
    for p in data:
        c, b = p.get("correct_code", ""), p.get("buggy_code", "")
        if not c.strip() or not b.strip():
            continue
        rows.append((c, b, p.get("problem_id", p.get("pair_id", "unk"))))
    return rows

def build_xy(rows, seed=42):
    rng = random.Random(seed)
    X, y, groups = [], [], []
    for c, b, pid in rows:
        fc, fb = feats(c), feats(b)
        if rng.random() < 0.5:           # randomize side order
            X.append(fc - fb); y.append(0)   # side0=correct -> label 0 = "side0 correct"
        else:
            X.append(fb - fc); y.append(1)   # side0=buggy   -> label 1
        groups.append(pid)
    return np.array(X), np.array(y), np.array(groups)

def evaluate(path, n_folds=5, seed=42):
    name = os.path.basename(path).replace("pairs_", "").replace(".json", "")
    rows = load_pairs(path)
    if len(rows) < 30:
        print(f"{name:28s} SKIP (only {len(rows)} pairs)")
        return None
    X, y, groups = build_xy(rows, seed)
    ng = len(set(groups))
    if ng >= n_folds:
        splits = list(GroupKFold(n_splits=n_folds).split(X, y, groups))
    else:
        splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                      random_state=seed).split(X, y))
    f1s, aucs, accs = [], [], []
    for tr, te in splits:
        clf = Pipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(max_iter=2000, C=1.0))])
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        prob = clf.predict_proba(X[te])[:, 1]
        f1s.append(f1_score(y[te], pred, average="macro"))
        accs.append(accuracy_score(y[te], pred))
        try: aucs.append(roc_auc_score(y[te], prob))
        except: aucs.append(0.5)
    print(f"{name:28s} n={len(rows):6d} grp={ng:5d} | "
          f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
          f"ACC={np.mean(accs):.3f}  AUC={np.mean(aucs):.3f}")
    return {"name": name, "n": len(rows), "f1": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)), "auc": float(np.mean(aucs))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default="/data/workzone/siamese_gat_journal/data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pairs_dir, "pairs_*.json")))
    files = [f for f in files if "pairs_all" not in f]
    print("="*78)
    print("PROBE 1: can SURFACE FEATURES alone separate correct vs buggy?")
    print("  (chance F1 ~= 0.50; pairwise, order-randomized, GroupKFold by problem)")
    print("="*78)
    codenet, heval = [], []
    for f in files:
        r = evaluate(f, seed=args.seed)
        if not r: continue
        (codenet if "codenet" in r["name"] else heval).append(r)

    def avg(group):
        if not group: return None
        return np.mean([g["f1"] for g in group]), np.mean([g["auc"] for g in group])
    print("\n" + "-"*78)
    cn, he = avg(codenet), avg(heval)
    if cn: print(f"CodeNet (cross-author)   mean F1={cn[0]:.3f}  AUC={cn[1]:.3f}   "
                 f"<- shortcut if HIGH")
    if he: print(f"HumanEvalFix (min-edit)  mean F1={he[0]:.3f}  AUC={he[1]:.3f}   "
                 f"<- should be ~chance")
    print("-"*78)
    if cn and he:
        gap = cn[0] - he[0]
        print(f"GAP (CodeNet - HumanEvalFix) = {gap:+.3f}")
        if cn[0] > 0.62 and he[0] < 0.58:
            print(">> DOOR OPEN: cross-author benchmark is surface-shortcut-solvable.")
            print("   Minimal-edit is not. This is the measurement finding.")
        elif cn[0] < 0.58 and he[0] < 0.58:
            print(">> No surface shortcut anywhere; collapse is genuine difficulty.")
        else:
            print(">> Mixed; inspect per-language before concluding.")

if __name__ == "__main__":
    main()