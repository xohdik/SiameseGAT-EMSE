"""
Step 1b: Fix failed downloads + add multi-language support.

Fixes:
- APPS: remove trust_remote_code kwarg for older datasets lib
- CodeNet: process from locally downloaded tar.gz
- HumanEvalPack: download all 6 language splits
- CodeContests: fix trust_remote_code issue

Usage:
    # Fix APPS download
    python scripts/download_fix.py --fix-apps

    # Process CodeNet from local tar.gz (after manual download)
    python scripts/download_fix.py --codenet-tar /path/to/Project_CodeNet.tar.gz

    # Download all HumanEvalPack languages
    python scripts/download_fix.py --multilang

    # All fixes
    python scripts/download_fix.py --fix-apps --codenet-tar ~/Downloads/Project_CodeNet.tar.gz --multilang
"""
import argparse
import csv
import gzip
import json
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path


# ═══════════════════════════════════════
# 1. FIX APPS DOWNLOAD
# ═══════════════════════════════════════

def fix_apps_download(out_dir):
    """Download APPS with fix for trust_remote_code issue."""
    from datasets import load_dataset
    
    print("[APPS] Downloading with compatibility fix...")
    os.makedirs(out_dir, exist_ok=True)
    
    # Try without trust_remote_code first
    try:
        ds = load_dataset("codeparrot/apps", "all", split="test")
    except TypeError:
        # Older version of datasets lib
        try:
            ds = load_dataset("codeparrot/apps", split="test")
        except Exception:
            # Try the newer mirror
            try:
                ds = load_dataset("loubnabnl/apps", split="test")
            except Exception as e:
                print(f"[APPS] All download attempts failed: {e}")
                print("[APPS] Manual alternative: pip install datasets --upgrade")
                print("[APPS] Then retry: python scripts/download_datasets.py --dataset apps")
                return []
    
    print(f"[APPS] Loaded {len(ds)} problems")
    
    pairs = []
    for row in ds:
        solutions_raw = row.get("solutions", "")
        if not solutions_raw:
            continue
        try:
            solutions = json.loads(solutions_raw) if isinstance(solutions_raw, str) else solutions_raw
        except json.JSONDecodeError:
            continue
        
        if not solutions or len(solutions) < 1:
            continue
        
        pairs.append({
            "task_id": f"apps_{row.get('problem_id', len(pairs))}",
            "correct_code": solutions[0],
            "all_solutions": solutions[:5],
            "input_output": row.get("input_output", ""),
            "difficulty": row.get("difficulty", "unknown"),
            "source": "apps",
        })
        
        if len(pairs) >= 3000:
            break
    
    path = os.path.join(out_dir, "apps_solutions.json")
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[APPS] Saved {len(pairs)} problems to {path}")
    return pairs


# ═══════════════════════════════════════
# 2. PROCESS CODENET FROM LOCAL TAR.GZ
# ═══════════════════════════════════════

# Language extensions in CodeNet
LANG_EXTENSIONS = {
    "Python": [".py"],
    "Java": [".java"],
    "C++": [".cpp", ".cc", ".cxx"],
    "C": [".c"],
    "Ruby": [".rb"],
    "C#": [".cs"],
    "JavaScript": [".js"],
    "Go": [".go"],
}

# Which languages to extract (must have tree-sitter + GraphCodeBERT support)
TARGET_LANGUAGES = ["Python", "Java", "C++", "C", "Ruby"]


def detect_language(filename):
    """Detect language from file extension."""
    for lang, exts in LANG_EXTENSIONS.items():
        for ext in exts:
            if filename.endswith(ext):
                return lang
    return None


def process_codenet_tar(tar_path, out_dir, languages=None, max_problems=2000, 
                         max_pairs_per_problem=5):
    """
    Extract correct/buggy code pairs from Project_CodeNet.tar.gz.
    
    Structure inside tar:
        Project_CodeNet/
        ├── data/
        │   ├── p00000/
        │   │   ├── s123456.py  (submissions)
        │   │   ├── s234567.py
        │   │   └── ...
        │   ├── p00001/
        │   └── ...
        └── metadata/
            ├── p00000.csv  (submission metadata with status)
            └── ...
    
    For each problem:
    - Read metadata CSV to find accepted vs wrong-answer submissions
    - Pair: 1 accepted + N wrong-answer per target language
    """
    if languages is None:
        languages = TARGET_LANGUAGES
    
    print(f"[CodeNet] Processing {tar_path}...")
    print(f"[CodeNet] Target languages: {languages}")
    print(f"[CodeNet] Max problems: {max_problems}, Max pairs/problem: {max_pairs_per_problem}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Phase 1: Read all metadata CSVs to find good problems
    print("[CodeNet] Phase 1: Reading metadata...")
    problem_meta = {}  # problem_id → {lang → {"accepted": [sub_ids], "wrong": [sub_ids]}}
    
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        metadata_files = [m for m in members if "/metadata/" in m.name and m.name.endswith(".csv")]
        print(f"  Found {len(metadata_files)} metadata files")
        
        for i, member in enumerate(metadata_files):
            if i >= max_problems * 2:  # Read more than needed to get enough good ones
                break
            if i % 500 == 0:
                print(f"  Reading metadata: {i}/{len(metadata_files)}...")
            
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                
                problem_id = Path(member.name).stem  # e.g., "p00000"
                reader = csv.DictReader(f.read().decode("utf-8", errors="ignore").splitlines())
                
                submissions = defaultdict(lambda: {"accepted": [], "wrong": []})
                
                for row in reader:
                    status = row.get("status", "").strip()
                    sub_id = row.get("submission_id", "").strip()
                    filename = row.get("filename", "").strip()
                    lang = detect_language(filename)
                    
                    if lang not in languages:
                        continue
                    
                    if status.lower() == "accepted":
                        submissions[lang]["accepted"].append(sub_id)
                    elif status.lower() == "wrong answer":
                        submissions[lang]["wrong"].append(sub_id)
                
                # Only keep problems with both accepted and wrong for at least one language
                useful = {lang: subs for lang, subs in submissions.items() 
                         if subs["accepted"] and subs["wrong"]}
                if useful:
                    problem_meta[problem_id] = useful
                    
            except Exception as e:
                continue
    
    print(f"  Found {len(problem_meta)} problems with accepted+wrong pairs")
    
    # Print stats per language
    for lang in languages:
        count = sum(1 for p in problem_meta.values() if lang in p)
        total_pairs = sum(
            min(len(p[lang]["wrong"]), max_pairs_per_problem) 
            for p in problem_meta.values() if lang in p
        )
        print(f"  {lang}: {count} problems, ~{total_pairs} potential pairs")
    
    # Phase 2: Extract actual code files
    print("\n[CodeNet] Phase 2: Extracting code pairs...")
    
    # Select top problems (those with most pairs)
    selected_problems = sorted(
        problem_meta.keys(),
        key=lambda pid: sum(
            min(len(problem_meta[pid][l]["wrong"]), max_pairs_per_problem)
            for l in problem_meta[pid]
        ),
        reverse=True
    )[:max_problems]
    
    # Build a set of all submission files we need to extract
    needed_files = {}  # "data/pXXXXX/sYYYYYY.ext" → (problem_id, sub_id)
    for pid in selected_problems:
        for lang, subs in problem_meta[pid].items():
            # Need accepted submissions
            for sub_id in subs["accepted"][:1]:  # Just 1 accepted per lang
                # We don't know the exact filename, so we'll match by sub_id
                needed_files[sub_id] = (pid, lang, "accepted")
            # Need wrong submissions
            for sub_id in subs["wrong"][:max_pairs_per_problem]:
                needed_files[sub_id] = (pid, lang, "wrong")
    
    print(f"  Need to extract {len(needed_files)} submission files")
    
    # Extract code from tar
    code_store = defaultdict(lambda: defaultdict(lambda: {"accepted": [], "wrong": []}))
    # code_store[problem_id][language] = {"accepted": [(sub_id, code)], "wrong": [...]}
    
    extracted = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if "/data/" not in member.name:
                continue
            
            # Extract submission ID from path
            # Path format: Project_CodeNet/data/pXXXXX/sYYYYYY.ext
            parts = Path(member.name).parts
            if len(parts) < 4:
                continue
            
            filename = parts[-1]
            sub_id = Path(filename).stem  # e.g., "s123456"
            
            if sub_id not in needed_files:
                continue
            
            pid, lang, status = needed_files[sub_id]
            
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                code = f.read().decode("utf-8", errors="ignore")
                
                if len(code.strip()) < 20:  # Skip trivially short
                    continue
                
                code_store[pid][lang][status].append((sub_id, code))
                extracted += 1
                
                if extracted % 1000 == 0:
                    print(f"  Extracted {extracted} files...")
                    
            except Exception:
                continue
    
    print(f"  Extracted {extracted} code files total")
    
    # Phase 3: Build pairs
    print("\n[CodeNet] Phase 3: Building pairs...")
    
    all_pairs = defaultdict(list)  # language → list of pairs
    
    for pid in selected_problems:
        for lang in problem_meta[pid]:
            accepted = code_store[pid][lang]["accepted"]
            wrong = code_store[pid][lang]["wrong"]
            
            if not accepted or not wrong:
                continue
            
            correct_sub_id, correct_code = accepted[0]
            
            for buggy_sub_id, buggy_code in wrong[:max_pairs_per_problem]:
                all_pairs[lang].append({
                    "pair_id": f"codenet_{pid}_{buggy_sub_id}",
                    "problem_id": pid,
                    "correct_code": correct_code,
                    "buggy_code": buggy_code,
                    "correct_sub_id": correct_sub_id,
                    "buggy_sub_id": buggy_sub_id,
                    "language": lang,
                    "dataset": f"codenet_{lang.lower().replace('+', 'p')}",
                    "source": "codenet",
                    "bug_type": "wrong_answer",
                })
    
    # Save per-language and combined
    total = 0
    for lang, pairs in all_pairs.items():
        lang_safe = lang.lower().replace("+", "p").replace("#", "sharp")
        path = os.path.join(out_dir, f"codenet_{lang_safe}_pairs.json")
        with open(path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"  {lang}: {len(pairs)} pairs → {path}")
        total += len(pairs)
    
    # Combined file
    combined = []
    for pairs in all_pairs.values():
        combined.extend(pairs)
    
    path = os.path.join(out_dir, "codenet_pairs.json")
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  TOTAL: {total} pairs across {len(all_pairs)} languages → {path}")
    
    return combined


def process_codenet_extracted(codenet_dir, out_dir, languages=None, 
                               max_problems=2000, max_pairs_per_problem=5):
    """
    Process CodeNet from already-extracted directory.
    Use this if you extracted the tar.gz manually.
    
    Expected structure:
        codenet_dir/
        ├── data/
        │   ├── p00000/
        │   └── ...
        └── metadata/
            ├── p00000.csv
            └── ...
    """
    if languages is None:
        languages = TARGET_LANGUAGES
    
    metadata_dir = os.path.join(codenet_dir, "metadata")
    data_dir = os.path.join(codenet_dir, "data")
    
    if not os.path.isdir(metadata_dir):
        # Try one level deeper
        for d in os.listdir(codenet_dir):
            candidate = os.path.join(codenet_dir, d)
            if os.path.isdir(os.path.join(candidate, "metadata")):
                metadata_dir = os.path.join(candidate, "metadata")
                data_dir = os.path.join(candidate, "data")
                break
    
    if not os.path.isdir(metadata_dir):
        print(f"[CodeNet] ERROR: Cannot find metadata dir in {codenet_dir}")
        print(f"[CodeNet] Expected: {codenet_dir}/metadata/ or {codenet_dir}/Project_CodeNet/metadata/")
        return []
    
    print(f"[CodeNet] Processing extracted directory: {codenet_dir}")
    print(f"[CodeNet] Metadata: {metadata_dir}")
    print(f"[CodeNet] Data: {data_dir}")
    print(f"[CodeNet] Target languages: {languages}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Phase 1: Read metadata
    print("[CodeNet] Phase 1: Reading metadata...")
    csv_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith(".csv")])
    print(f"  Found {len(csv_files)} problem metadata files")
    
    problem_meta = {}
    for i, csv_file in enumerate(csv_files):
        if len(problem_meta) >= max_problems * 2:
            break
        if i % 500 == 0:
            print(f"  Reading: {i}/{len(csv_files)}...")
        
        pid = csv_file.replace(".csv", "")
        csv_path = os.path.join(metadata_dir, csv_file)
        
        try:
            submissions = defaultdict(lambda: {"accepted": [], "wrong": []})
            
            with open(csv_path, "r", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get("status", "").strip()
                    sub_id = row.get("submission_id", "").strip()
                    filename = row.get("filename", "").strip()
                    lang = detect_language(filename)
                    
                    if lang not in languages:
                        continue
                    
                    if status.lower() == "accepted":
                        submissions[lang]["accepted"].append((sub_id, filename))
                    elif status.lower() == "wrong answer":
                        submissions[lang]["wrong"].append((sub_id, filename))
            
            useful = {lang: subs for lang, subs in submissions.items()
                     if subs["accepted"] and subs["wrong"]}
            if useful:
                problem_meta[pid] = useful
                
        except Exception:
            continue
    
    print(f"  Found {len(problem_meta)} usable problems")
    for lang in languages:
        count = sum(1 for p in problem_meta.values() if lang in p)
        print(f"  {lang}: {count} problems")
    
    # Phase 2: Read code files and build pairs
    print("\n[CodeNet] Phase 2: Building pairs...")
    
    all_pairs = defaultdict(list)
    
    for pid_idx, (pid, lang_subs) in enumerate(problem_meta.items()):
        if pid_idx >= max_problems:
            break
        if pid_idx % 200 == 0:
            print(f"  Processing problem {pid_idx}/{min(len(problem_meta), max_problems)}...")
        
        problem_data_dir = os.path.join(data_dir, pid)
        if not os.path.isdir(problem_data_dir):
            continue
        
        for lang, subs in lang_subs.items():
            # Read one accepted solution
            correct_code = None
            correct_sub_id = None
            for sub_id, filename in subs["accepted"][:3]:
                fpath = os.path.join(problem_data_dir, filename)
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
            for sub_id, filename in subs["wrong"][:max_pairs_per_problem]:
                fpath = os.path.join(problem_data_dir, filename)
                if not os.path.exists(fpath):
                    continue
                try:
                    with open(fpath, "r", errors="ignore") as f:
                        buggy_code = f.read()
                    if len(buggy_code.strip()) < 20:
                        continue
                    
                    lang_safe = lang.lower().replace("+", "p").replace("#", "sharp")
                    all_pairs[lang].append({
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
                except Exception:
                    continue
    
    # Save
    total = 0
    for lang, pairs in all_pairs.items():
        lang_safe = lang.lower().replace("+", "p").replace("#", "sharp")
        path = os.path.join(out_dir, f"codenet_{lang_safe}_pairs.json")
        with open(path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"  {lang}: {len(pairs)} pairs → {path}")
        total += len(pairs)
    
    combined = []
    for pairs in all_pairs.values():
        combined.extend(pairs)
    path = os.path.join(out_dir, "codenet_pairs.json")
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  TOTAL: {total} pairs across {len(all_pairs)} languages")
    
    return combined


# ═══════════════════════════════════════
# 3. MULTI-LANGUAGE HUMANEVALPACK
# ═══════════════════════════════════════

HUMANEVALPACK_LANGS = ["python", "java", "js", "go", "cpp", "rust"]

def download_humanevalpack_multilang(out_dir):
    """Download all 6 language splits of HumanEvalPack."""
    from datasets import load_dataset
    
    os.makedirs(out_dir, exist_ok=True)
    all_pairs = []
    
    for lang in HUMANEVALPACK_LANGS:
        print(f"[HumanEvalPack] Downloading {lang}...")
        try:
            ds = load_dataset("bigcode/humanevalpack", lang, split="test")
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        
        pairs = []
        for row in ds:
            decl = row.get("declaration", "")
            correct = decl + row.get("canonical_solution", "")
            buggy = decl + row.get("buggy_solution", "")
            
            if not correct.strip() or not buggy.strip():
                continue
            
            pair = {
                "pair_id": f"heval_{lang}_{row['task_id']}",
                "correct_code": correct,
                "buggy_code": buggy,
                "prompt": row.get("prompt", ""),
                "docstring": row.get("docstring", ""),
                "bug_type": row.get("bug_type", "unknown"),
                "test": row.get("test", ""),
                "language": lang,
                "dataset": f"humanevalfix_{lang}",
                "source": "humanevalfix",
            }
            pairs.append(pair)
            all_pairs.append(pair)
        
        path = os.path.join(out_dir, f"humanevalfix_{lang}.json")
        with open(path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"  {lang}: {len(pairs)} pairs")
    
    # Combined
    path = os.path.join(out_dir, "humanevalfix_all_langs.json")
    with open(path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"\n  TOTAL: {len(all_pairs)} pairs across {len(HUMANEVALPACK_LANGS)} languages")
    
    return all_pairs


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fix downloads + multi-language support")
    parser.add_argument("--fix-apps", action="store_true", help="Fix APPS download")
    parser.add_argument("--codenet-tar", type=str, default=None,
                       help="Path to Project_CodeNet.tar.gz")
    parser.add_argument("--codenet-dir", type=str, default=None,
                       help="Path to extracted Project_CodeNet directory")
    parser.add_argument("--multilang", action="store_true",
                       help="Download all HumanEvalPack languages")
    parser.add_argument("--output-dir", type=str, default="./data/raw")
    parser.add_argument("--max-problems", type=int, default=2000)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--languages", nargs="+", default=None,
                       help="Languages to extract from CodeNet")
    args = parser.parse_args()
    
    languages = args.languages or TARGET_LANGUAGES
    
    if args.fix_apps:
        fix_apps_download(os.path.join(args.output_dir, "apps"))
    
    if args.codenet_tar:
        if not os.path.exists(args.codenet_tar):
            print(f"ERROR: {args.codenet_tar} not found!")
            sys.exit(1)
        process_codenet_tar(
            args.codenet_tar, 
            os.path.join(args.output_dir, "codenet"),
            languages=languages,
            max_problems=args.max_problems,
            max_pairs_per_problem=args.max_pairs,
        )
    
    if args.codenet_dir:
        if not os.path.isdir(args.codenet_dir):
            print(f"ERROR: {args.codenet_dir} not found!")
            sys.exit(1)
        process_codenet_extracted(
            args.codenet_dir,
            os.path.join(args.output_dir, "codenet"),
            languages=languages,
            max_problems=args.max_problems,
            max_pairs_per_problem=args.max_pairs,
        )
    
    if args.multilang:
        download_humanevalpack_multilang(os.path.join(args.output_dir, "humanevalfix"))
    
    if not any([args.fix_apps, args.codenet_tar, args.codenet_dir, args.multilang]):
        print("No action specified. Use --fix-apps, --codenet-tar, --codenet-dir, or --multilang")
        parser.print_help()


if __name__ == "__main__":
    main()
