"""
Step 2 (FIXED): Build unified correct/buggy code pairs from all datasets.

Key fix: accepts explicit paths since data may be in different locations.

Usage:
    # Basic — point to where each dataset lives
    python scripts/build_pairs.py \
        --codenet-dir /data/workzone/PhD/Project_CodeNet \
        --humanevalfix-dir ./data/raw/humanevalfix \
        --apps-dir ./data/raw/apps \
        --mbpp-dir ./data/raw/mbpp \
        --output-dir ./data/processed

    # Or auto-search common locations
    python scripts/build_pairs.py --auto-find --output-dir ./data/processed

    # With LLM bug injection for APPS/MBPP
    python scripts/build_pairs.py --auto-find --inject-bugs --llm-provider openai
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional


# ═══════════════════════════════════════
# AUTO-FIND DATA LOCATIONS
# ═══════════════════════════════════════

def auto_find_paths():
    """Search common locations for each dataset."""
    found = {}

    # CodeNet
    for p in ["/data/workzone/PhD/Project_CodeNet",
              os.path.expanduser("~/Tunde/PhD/Project_CodeNet"),
              "./Project_CodeNet", "../Project_CodeNet",
              "/data/workzone/Project_CodeNet"]:
        if os.path.isdir(os.path.join(p, "data")) and os.path.isdir(os.path.join(p, "metadata")):
            found["codenet_dir"] = p
            break

    # HumanEvalFix
    for p in ["./data/raw/humanevalfix",
              os.path.expanduser("~/Tunde/PhD/siamese_gat_journal/data/raw/humanevalfix"),
              "/data/workzone/siamese_gat_journal/data/raw/humanevalfix"]:
        if os.path.isdir(p):
            found["humanevalfix_dir"] = p
            break

    # APPS
    for p in ["./data/raw/apps",
              os.path.expanduser("~/Tunde/PhD/siamese_gat_journal/data/raw/apps"),
              "/data/workzone/siamese_gat_journal/data/raw/apps"]:
        if os.path.isdir(p):
            found["apps_dir"] = p
            break

    # MBPP
    for p in ["./data/raw/mbpp",
              os.path.expanduser("~/Tunde/PhD/siamese_gat_journal/data/raw/mbpp"),
              "/data/workzone/siamese_gat_journal/data/raw/mbpp"]:
        if os.path.isdir(p):
            found["mbpp_dir"] = p
            break

    return found


# ═══════════════════════════════════════
# CODENET — DIRECT FROM EXTRACTED DIR
# ═══════════════════════════════════════

LANG_EXTENSIONS = {
    "Python": ".py", "Java": ".java", "C++": ".cpp", "C": ".c",
    "Ruby": ".rb", "C#": ".cs", "JavaScript": ".js", "Go": ".go",
}

TARGET_LANGUAGES = ["Python", "Java", "C++", "C", "Ruby", "JavaScript"]


def build_codenet_pairs(codenet_dir: str, languages=None, max_problems=None,
                         max_pairs_per_problem=5) -> List[Dict]:
    """
    Build correct/buggy pairs directly from extracted Project_CodeNet.

    Reads metadata CSVs to find Accepted vs Wrong Answer submissions,
    then reads actual source files to build pairs.
    """
    if not codenet_dir or not os.path.isdir(codenet_dir):
        print("[CodeNet] Directory not found, skipping")
        return []

    if languages is None:
        languages = TARGET_LANGUAGES

    data_dir = os.path.join(codenet_dir, "data")
    metadata_dir = os.path.join(codenet_dir, "metadata")

    if not os.path.isdir(metadata_dir):
        print(f"[CodeNet] ERROR: metadata dir not found at {metadata_dir}")
        return []

    csv_files = sorted([f for f in os.listdir(metadata_dir)
                        if f.endswith(".csv") and f != "problem_list.csv"])

    if max_problems:
        csv_files = csv_files[:max_problems]

    print(f"[CodeNet] Processing {len(csv_files)} problems for {languages}...")
    print(f"[CodeNet] Max {max_pairs_per_problem} wrong-answer pairs per problem per language")

    all_pairs = []
    problems_used = 0
    errors = 0

    for i, csv_file in enumerate(csv_files):
        if i % 500 == 0:
            print(f"  Progress: {i}/{len(csv_files)} problems, {len(all_pairs)} pairs so far...")

        pid = csv_file.replace(".csv", "")
        csv_path = os.path.join(metadata_dir, csv_file)
        problem_data_dir = os.path.join(data_dir, pid)

        if not os.path.isdir(problem_data_dir):
            continue

        try:
            # Read metadata to find accepted vs wrong-answer per language
            lang_subs = defaultdict(lambda: {"accepted": [], "wrong": []})

            with open(csv_path, "r", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get("status", "").strip()
                    lang = row.get("language", "").strip()
                    sub_id = row.get("submission_id", "").strip()
                    ext = row.get("filename_ext", "").strip()

                    if lang not in languages:
                        continue

                    if status == "Accepted":
                        lang_subs[lang]["accepted"].append((sub_id, ext))
                    elif status == "Wrong Answer":
                        lang_subs[lang]["wrong"].append((sub_id, ext))

            # Build pairs for each language
            problem_contributed = False
            for lang in languages:
                if lang not in lang_subs:
                    continue

                accepted_list = lang_subs[lang]["accepted"]
                wrong_list = lang_subs[lang]["wrong"]

                if not accepted_list or not wrong_list:
                    continue

                # Read one accepted solution
                correct_code = None
                correct_sub_id = None
                for sub_id, ext in accepted_list[:3]:
                    fpath = os.path.join(problem_data_dir, lang, f"{sub_id}.{ext}")
                    if os.path.exists(fpath):
                        try:
                            with open(fpath, "r", errors="ignore") as f:
                                code = f.read()
                            if len(code.strip()) >= 20:
                                correct_code = code
                                correct_sub_id = sub_id
                                break
                        except Exception:
                            continue

                if not correct_code:
                    continue

                # Read wrong-answer solutions
                lang_safe = lang.lower().replace("+", "p").replace("#", "sharp")
                for sub_id, ext in wrong_list[:max_pairs_per_problem]:
                    fpath = os.path.join(problem_data_dir, lang, f"{sub_id}.{ext}")
                    if not os.path.exists(fpath):
                        continue
                    try:
                        with open(fpath, "r", errors="ignore") as f:
                            buggy_code = f.read()
                        if len(buggy_code.strip()) < 20:
                            continue
                        # Skip if codes are identical
                        if buggy_code.strip() == correct_code.strip():
                            continue

                        all_pairs.append({
                            "pair_id": f"codenet_{pid}_{sub_id}",
                            "problem_id": pid,
                            "correct_code": correct_code,
                            "buggy_code": buggy_code,
                            "correct_sub_id": correct_sub_id,
                            "buggy_sub_id": sub_id,
                            "language": lang,
                            "dataset": f"codenet_{lang_safe}",
                            "source": "codenet",
                            "bug_type": "wrong_answer",
                        })
                        problem_contributed = True
                    except Exception:
                        errors += 1

            if problem_contributed:
                problems_used += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {csv_file}: {e}")

    # Summary
    lang_counts = defaultdict(int)
    for p in all_pairs:
        lang_counts[p["language"]] += 1

    print(f"\n[CodeNet] DONE: {len(all_pairs)} pairs from {problems_used} problems ({errors} errors)")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count} pairs")

    return all_pairs


# ═══════════════════════════════════════
# HUMANEVALFIX (MULTI-LANGUAGE)
# ═══════════════════════════════════════

def build_humanevalfix_pairs(heval_dir: str) -> List[Dict]:
    if not heval_dir or not os.path.isdir(heval_dir):
        print("[HumanEvalFix] Directory not found, skipping")
        return []

    pairs = []
    for fname in sorted(os.listdir(heval_dir)):
        if not fname.startswith("humanevalfix_") or not fname.endswith(".json"):
            continue
        if fname == "humanevalfix_all_langs.json":
            continue

        with open(os.path.join(heval_dir, fname)) as f:
            data = json.load(f)

        for row in data:
            correct = row.get("correct_code", "")
            buggy = row.get("buggy_code", "")
            if not correct or not buggy:
                decl = row.get("declaration", "")
                correct = decl + row.get("canonical_solution", "")
                buggy = decl + row.get("buggy_solution", "")
            if not correct.strip() or not buggy.strip():
                continue

            lang = row.get("language", "python")
            pairs.append({
                "pair_id": row.get("pair_id", row.get("task_id", "")),
                "correct_code": correct, "buggy_code": buggy,
                "prompt": row.get("prompt", ""), "docstring": row.get("docstring", ""),
                "bug_type": row.get("bug_type", "unknown"), "test": row.get("test", ""),
                "language": lang, "dataset": f"humanevalfix_{lang}", "source": "humanevalfix",
            })

    lang_counts = defaultdict(int)
    for p in pairs:
        lang_counts[p["language"]] += 1
    print(f"[HumanEvalFix] {len(pairs)} pairs: {dict(sorted(lang_counts.items()))}")
    return pairs


# ═══════════════════════════════════════
# APPS
# ═══════════════════════════════════════

def build_apps_pairs(apps_dir: str, inject_bugs=False, llm_fn=None) -> List[Dict]:
    if not apps_dir or not os.path.isdir(apps_dir):
        print("[APPS] Directory not found, skipping")
        return []

    path = os.path.join(apps_dir, "apps_solutions.json")
    if not os.path.exists(path):
        print("[APPS] apps_solutions.json not found, skipping")
        return []

    with open(path) as f:
        data = json.load(f)

    pairs = []
    if inject_bugs and llm_fn:
        print(f"[APPS] Injecting bugs for {len(data)} problems...")
        for i, row in enumerate(data):
            if i % 100 == 0: print(f"  {i}/{len(data)}...")
            buggy = llm_fn(row["correct_code"])
            if buggy and buggy.strip() != row["correct_code"].strip():
                pairs.append({
                    "pair_id": row["task_id"], "correct_code": row["correct_code"],
                    "buggy_code": buggy, "language": "python", "dataset": "apps",
                    "source": "apps", "bug_type": "llm_injected",
                    "prompt": "", "docstring": "",
                })
    else:
        print(f"[APPS] {len(data)} problems (need --inject-bugs for usable pairs)")

    print(f"[APPS] {len(pairs)} usable pairs")
    return pairs


# ═══════════════════════════════════════
# MBPP
# ═══════════════════════════════════════

def build_mbpp_pairs(mbpp_dir: str, inject_bugs=False, llm_fn=None) -> List[Dict]:
    if not mbpp_dir or not os.path.isdir(mbpp_dir):
        print("[MBPP] Directory not found, skipping")
        return []

    path = os.path.join(mbpp_dir, "mbpp_full.json")
    if not os.path.exists(path):
        print("[MBPP] mbpp_full.json not found, skipping")
        return []

    with open(path) as f:
        data = json.load(f)

    pairs = []
    if inject_bugs and llm_fn:
        print(f"[MBPP] Injecting bugs for {len(data)} problems...")
        for i, row in enumerate(data):
            if i % 100 == 0: print(f"  {i}/{len(data)}...")
            buggy = llm_fn(row["code"])
            if buggy and buggy.strip() != row["code"].strip():
                pairs.append({
                    "pair_id": row["task_id"], "correct_code": row["code"],
                    "buggy_code": buggy, "prompt": row["text"], "docstring": row["text"],
                    "language": "python", "dataset": "mbpp", "source": "mbpp",
                    "bug_type": "llm_injected",
                })
    else:
        print(f"[MBPP] {len(data)} problems (need --inject-bugs for usable pairs)")

    print(f"[MBPP] {len(pairs)} usable pairs")
    return pairs


# ═══════════════════════════════════════
# LLM BUG INJECTION
# ═══════════════════════════════════════

BUG_PROMPT = """Given this correct Python function, introduce EXACTLY ONE subtle bug that changes behavior (wrong operator, off-by-one, wrong variable, missing edge case). Keep it syntactically valid. Return ONLY the buggy code, no explanation:

```python
{code}
```"""


def create_llm_injector(provider="openai"):
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        def inject(code):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.7, max_tokens=1024,
                    messages=[{"role": "user", "content": BUG_PROMPT.format(code=code)}])
                t = r.choices[0].message.content.strip()
                if "```python" in t: t = t.split("```python")[1].split("```")[0].strip()
                elif "```" in t: t = t.split("```")[1].split("```")[0].strip()
                return t
            except: return None
        return inject
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        def inject(code):
            try:
                r = client.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=1024,
                    messages=[{"role": "user", "content": BUG_PROMPT.format(code=code)}])
                t = r.content[0].text.strip()
                if "```python" in t: t = t.split("```python")[1].split("```")[0].strip()
                elif "```" in t: t = t.split("```")[1].split("```")[0].strip()
                return t
            except: return None
        return inject


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build code pairs from all datasets")
    parser.add_argument("--codenet-dir", type=str, default=None,
                       help="Path to extracted Project_CodeNet directory")
    parser.add_argument("--humanevalfix-dir", type=str, default=None,
                       help="Path to humanevalfix data directory")
    parser.add_argument("--apps-dir", type=str, default=None,
                       help="Path to APPS data directory")
    parser.add_argument("--mbpp-dir", type=str, default=None,
                       help="Path to MBPP data directory")
    parser.add_argument("--raw-dir", type=str, default="./data/raw",
                       help="Base raw data dir (fallback)")
    parser.add_argument("--output-dir", type=str, default="./data/processed")
    parser.add_argument("--auto-find", action="store_true",
                       help="Auto-search common locations for data")
    parser.add_argument("--languages", nargs="+", default=None,
                       help="CodeNet languages to extract")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Max CodeNet problems (default: all)")
    parser.add_argument("--max-pairs-per-problem", type=int, default=5)
    parser.add_argument("--inject-bugs", action="store_true")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "anthropic"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve paths
    if args.auto_find:
        auto = auto_find_paths()
        print(f"Auto-found: {auto}")
    else:
        auto = {}

    codenet_dir = args.codenet_dir or auto.get("codenet_dir")
    heval_dir = args.humanevalfix_dir or auto.get("humanevalfix_dir") or os.path.join(args.raw_dir, "humanevalfix")
    apps_dir = args.apps_dir or auto.get("apps_dir") or os.path.join(args.raw_dir, "apps")
    mbpp_dir = args.mbpp_dir or auto.get("mbpp_dir") or os.path.join(args.raw_dir, "mbpp")

    print(f"\nData paths:")
    print(f"  CodeNet:      {codenet_dir or 'NOT FOUND'}")
    print(f"  HumanEvalFix: {heval_dir}")
    print(f"  APPS:         {apps_dir}")
    print(f"  MBPP:         {mbpp_dir}")
    print()

    llm_fn = None
    if args.inject_bugs:
        print(f"Initializing LLM injector ({args.llm_provider})...")
        llm_fn = create_llm_injector(args.llm_provider)

    # Build all pairs
    all_pairs = []

    # 1. HumanEvalFix (multi-language)
    all_pairs.extend(build_humanevalfix_pairs(heval_dir))

    # 2. CodeNet (multi-language, direct from extracted dir)
    all_pairs.extend(build_codenet_pairs(
        codenet_dir, languages=args.languages, max_problems=args.max_problems,
        max_pairs_per_problem=args.max_pairs_per_problem))

    # 3. APPS (Python, needs bug injection)
    all_pairs.extend(build_apps_pairs(apps_dir, args.inject_bugs, llm_fn))

    # 4. MBPP (Python, needs bug injection)
    all_pairs.extend(build_mbpp_pairs(mbpp_dir, args.inject_bugs, llm_fn))

    # Only keep pairs with both correct and buggy code
    usable = [p for p in all_pairs if p.get("correct_code", "").strip() and p.get("buggy_code", "").strip()]

    # Save per-dataset
    datasets = set(p["dataset"] for p in usable)
    for ds in datasets:
        subset = [p for p in usable if p["dataset"] == ds]
        with open(os.path.join(args.output_dir, f"pairs_{ds}.json"), "w") as f:
            json.dump(subset, f)  # No indent for large files — saves disk
        print(f"  Saved pairs_{ds}.json ({len(subset)} pairs)")

    # Save combined
    with open(os.path.join(args.output_dir, "pairs_all.json"), "w") as f:
        json.dump(usable, f)

    # ═══════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════
    print(f"\n{'='*70}")
    print(f"TOTAL USABLE PAIRS: {len(usable)}")
    print(f"{'='*70}")

    print(f"\nBy dataset:")
    ds_counts = defaultdict(int)
    for p in usable:
        ds_counts[p["dataset"]] += 1
    for ds, c in sorted(ds_counts.items(), key=lambda x: -x[1]):
        print(f"  {ds:<25} {c:>7} pairs")

    print(f"\nBy language:")
    lang_counts = defaultdict(int)
    for p in usable:
        lang_counts[p.get("language", "?")] += 1
    for lang, c in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang:<25} {c:>7} pairs")

    print(f"\nBy source:")
    src_counts = defaultdict(int)
    for p in usable:
        src_counts[p.get("source", "?")] += 1
    for src, c in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<25} {c:>7} pairs")

    print(f"\nSaved to: {args.output_dir}/")
    print(f"  pairs_all.json ({len(usable)} pairs)")
    for ds in sorted(datasets):
        print(f"  pairs_{ds}.json")


if __name__ == "__main__":
    main()
