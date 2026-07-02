"""
PROBE 2a: Inspect HumanEvalFix Python records so we know EXACTLY what fields
exist before trying to execute anything. Prints keys, and shows one full
correct/buggy/test triple so we can see if the code is executable as-is.
"""
import json, sys, os

CANDIDATES = [
    "data/processed/pairs_humanevalfix_python.json",
    "data/raw/humanevalfix/humanevalfix_python.json",
]

def find():
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find()
    if not path or not os.path.exists(path):
        print("Could not find HumanEvalFix python file. Pass path as arg.")
        print("Tried:", CANDIDATES)
        return
    print("FILE:", path)
    data = json.load(open(path))
    print("records:", len(data))
    r = data[0]
    print("\nKEYS:", list(r.keys()))
    print("\n--- field lengths/types (record 0) ---")
    for k, v in r.items():
        s = v if isinstance(v, str) else json.dumps(v)
        print(f"  {k:20s} len={len(str(s)):5d}  type={type(v).__name__}")

    print("\n" + "="*70)
    print("RECORD 0 — full dump of code-bearing fields")
    print("="*70)
    for k in ["pair_id", "task_id", "language", "bug_type", "entry_point"]:
        if k in r: print(f"{k}: {r[k]}")
    for k in ["prompt", "docstring", "declaration"]:
        if k in r and r[k]:
            print(f"\n----- {k} -----\n{r[k][:600]}")
    for k in ["correct_code", "canonical_solution"]:
        if k in r and r[k]:
            print(f"\n----- {k} -----\n{r[k][:600]}")
    for k in ["buggy_code", "buggy_solution"]:
        if k in r and r[k]:
            print(f"\n----- {k} -----\n{r[k][:600]}")
    for k in ["test"]:
        if k in r and r[k]:
            print(f"\n----- {k} (test harness) -----\n{r[k][:800]}")

    # Does any field look like a runnable test?
    print("\n" + "="*70)
    has_test = any(("test" in k.lower() or "check" in k.lower()) for k in r.keys())
    has_entry = "entry_point" in r or "task_id" in r
    print(f"has test-like field: {has_test}")
    print(f"has entry_point/task_id: {has_entry}")
    print("If has_test is False, we need the original HumanEvalPack 'test' +")
    print("'entry_point' fields to execute — tell me and I'll pull them from raw.")

if __name__ == "__main__":
    main()