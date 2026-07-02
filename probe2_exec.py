"""
PROBE 2: Execution feasibility for localization ground truth.

For each HumanEvalFix Python pair, run correct_code and buggy_code against the
`test` harness in a sandboxed subprocess. Reports:
  - correct PASS rate   (sanity: should be ~100%)
  - buggy FAIL rate     (the clean, executable bug set)
  - unusable count      (timeout / crash / no entry point)
  - diff-line stats      (how localized is the bug? lines changed per pair)

Reading:
  correct PASS ~100% AND buggy FAIL high (>85%)
      => clean executable ground truth exists. Localization is FEASIBLE.
         We can measure "does method X point at the changed line(s)".
  correct PASS low OR buggy FAIL low
      => tests don't run cleanly in this env; localization ground truth is
         unreliable. That door is (mostly) shut — report and reconsider.

No GPU, no LLM, no network. Pure subprocess execution.

Usage:
  python probe2_exec.py
  python probe2_exec.py --pairs data/processed/pairs_humanevalfix_python.json --timeout 10
"""
import argparse, json, os, re, subprocess, sys, tempfile, textwrap
import difflib
from collections import Counter

def entry_point(code):
    """First top-level def name."""
    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", code, re.M)
    return m.group(1) if m else None

def run_one(code, test, timeout=10):
    """Return 'pass' | 'fail' | 'error:<reason>'. Runs code+test in subprocess."""
    fn = entry_point(code)
    if not fn:
        return "error:no_entry_point"
    # Build a standalone script: the code, the test harness, then call check(fn).
    # The harness already defines check(...) and calls check(<name>) at the end,
    # but to be robust we strip a trailing bare check(...) call and re-issue it
    # with the detected entry point.
    test_body = test
    # remove a final "check(<something>)" line if present; we re-add our own
    test_body = re.sub(r"\ncheck\([^\)]*\)\s*$", "\n", test_body.rstrip()) + "\n"
    script = f"{code}\n\n{test_body}\ncheck({fn})\nprint('___PASS___')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script); path = f.name
    try:
        r = subprocess.run([sys.executable, path],
                           capture_output=True, text=True, timeout=timeout)
        if "___PASS___" in r.stdout:
            return "pass"
        # AssertionError or any exception => fail/error
        err = (r.stderr or "").strip().split("\n")[-1][:80]
        if "AssertionError" in r.stderr:
            return "fail"
        return f"error:{err}" if err else "fail"
    except subprocess.TimeoutExpired:
        return "error:timeout"
    except Exception as e:
        return f"error:{str(e)[:60]}"
    finally:
        try: os.unlink(path)
        except: pass

def changed_lines(correct, buggy):
    """Number of differing lines (rough bug-locality measure)."""
    c = correct.split("\n"); b = buggy.split("\n")
    sm = difflib.SequenceMatcher(None, c, b)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs_humanevalfix_python.json")
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()

    data = json.load(open(args.pairs))
    print("="*74)
    print(f"PROBE 2: execution feasibility on {len(data)} HumanEvalFix Python pairs")
    print("="*74)

    n = len(data)
    correct_pass = buggy_fail = clean = 0
    correct_status = Counter(); buggy_status = Counter()
    diffs = []
    clean_bug_ids = []

    for i, r in enumerate(data):
        cc, bc, test = r.get("correct_code",""), r.get("buggy_code",""), r.get("test","")
        if not cc or not bc or not test:
            correct_status["no_data"] += 1; continue
        cs = run_one(cc, test, args.timeout)
        bs = run_one(bc, test, args.timeout)
        correct_status[cs.split(":")[0]] += 1
        buggy_status[bs.split(":")[0]] += 1
        if cs == "pass": correct_pass += 1
        if bs == "fail": buggy_fail += 1
        # CLEAN bug = correct passes AND buggy fails (by assertion, not crash)
        if cs == "pass" and bs == "fail":
            clean += 1
            clean_bug_ids.append(r.get("pair_id", f"pair_{i}"))
            diffs.append(changed_lines(cc, bc))
        if (i+1) % 40 == 0:
            print(f"  ...{i+1}/{n}")

    print(f"\n--- correct_code status ---")
    for k,v in correct_status.most_common(): print(f"  {k:20s} {v}")
    print(f"--- buggy_code status ---")
    for k,v in buggy_status.most_common(): print(f"  {k:20s} {v}")

    print("\n" + "-"*74)
    print(f"correct PASS rate : {correct_pass}/{n} = {correct_pass/n:.1%}   (sanity ~100%)")
    print(f"buggy  FAIL rate  : {buggy_fail}/{n} = {buggy_fail/n:.1%}")
    print(f"CLEAN bug set     : {clean}/{n} = {clean/n:.1%}   (correct passes & buggy fails)")
    if diffs:
        import numpy as np
        print(f"bug locality      : changed lines per pair  "
              f"median={int(np.median(diffs))} mean={np.mean(diffs):.1f} "
              f"max={max(diffs)}  (1-2 = well localized)")
        single = sum(1 for d in diffs if d <= 2)
        print(f"                    {single}/{len(diffs)} = {single/len(diffs):.0%} change <=2 lines")
    print("-"*74)
    if clean/n >= 0.85 and correct_pass/n >= 0.9:
        print(">> DOOR OPEN: clean executable ground truth. Localization is FEASIBLE.")
        print(f"   {clean} pairs with known pass/fail + located bug lines.")
    elif clean/n >= 0.6:
        print(">> PARTIAL: usable but lossy ground truth; report attrition honestly.")
    else:
        print(">> DOOR (mostly) SHUT: too few cleanly executable bugs in this env.")

    # save the clean set for downstream localization work
    out = "outputs/probe2_clean_bugs.json"
    os.makedirs("outputs", exist_ok=True)
    json.dump(clean_bug_ids, open(out, "w"))
    print(f"\nclean bug ids saved -> {out}")

if __name__ == "__main__":
    main()