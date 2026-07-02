"""
Step 1: Download all datasets for multi-benchmark evaluation.
Run this on your local machine with internet access.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset humanevalfix
    python scripts/download_datasets.py --dataset codenet --max-problems 1000
"""
import argparse
import json
import os
import sys
from pathlib import Path

def download_humanevalfix(out_dir):
    """Download HumanEvalFix from HumanEvalPack (bigcode)."""
    from datasets import load_dataset
    
    print("[HumanEvalFix] Downloading from bigcode/humanevalpack...")
    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    
    pairs = []
    for row in ds:
        pairs.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "declaration": row["declaration"],
            "docstring": row.get("docstring", ""),
            "canonical_solution": row["canonical_solution"],
            "buggy_solution": row["buggy_solution"],
            "test": row["test"],
            "bug_type": row.get("bug_type", "unknown"),
            "source": "humanevalfix",
        })
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "humanevalfix_python.json")
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[HumanEvalFix] Saved {len(pairs)} pairs to {path}")
    return pairs


def download_codenet(out_dir, max_problems=2000, max_pairs=5):
    """
    Download CodeNet Python submissions and construct correct/buggy pairs.
    
    For each problem:
    - Take 1 accepted Python submission
    - Take up to `max_pairs` wrong-answer Python submissions
    - Each (accepted, wrong) pair becomes a training sample
    """
    from datasets import load_dataset
    
    print("[CodeNet] Downloading from IBM Project_CodeNet...")
    print("[CodeNet] This uses the HuggingFace mirror: 'mhhmm/codenet-python'")
    print("[CodeNet] If unavailable, download manually from https://developer.ibm.com/exchanges/data/all/project-codenet/")
    
    # Try the HuggingFace mirror first
    try:
        ds = load_dataset("mhhmm/codenet-python", split="train")
        print(f"[CodeNet] Loaded {len(ds)} Python submissions from HF mirror")
    except Exception:
        print("[CodeNet] HF mirror not available. Trying alternative approach...")
        print("[CodeNet] Creating synthetic pairs from competitive programming data...")
        # Fallback: use code_contests dataset
        return download_codecontests_as_codenet(out_dir, max_problems, max_pairs)
    
    # Group by problem
    from collections import defaultdict
    problems = defaultdict(lambda: {"accepted": [], "wrong": []})
    
    for row in ds:
        pid = row.get("problem_id", row.get("id", "unknown"))
        status = row.get("status", row.get("verdict", ""))
        code = row.get("code", row.get("solution", ""))
        
        if not code or len(code.strip()) < 20:
            continue
        
        if status.lower() in ["accepted", "ac"]:
            problems[pid]["accepted"].append(code)
        elif status.lower() in ["wrong answer", "wa", "wrong_answer"]:
            problems[pid]["wrong"].append(code)
    
    pairs = []
    for pid, sub in list(problems.items())[:max_problems]:
        if not sub["accepted"] or not sub["wrong"]:
            continue
        correct = sub["accepted"][0]  # Take first accepted
        for buggy in sub["wrong"][:max_pairs]:
            pairs.append({
                "task_id": f"codenet_{pid}",
                "correct_code": correct,
                "buggy_code": buggy,
                "source": "codenet",
            })
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "codenet_pairs.json")
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[CodeNet] Saved {len(pairs)} pairs from {len(problems)} problems to {path}")
    return pairs


def download_codecontests_as_codenet(out_dir, max_problems=2000, max_pairs=5):
    """Fallback: Use deepmind/code_contests for correct/wrong pairs."""
    from datasets import load_dataset
    
    print("[CodeContests] Downloading from deepmind/code_contests...")
    pairs = []
    
    for split in ["test", "valid", "train"]:
        try:
            ds = load_dataset("deepmind/code_contests", split=split, trust_remote_code=True)
            print(f"[CodeContests] Loaded {len(ds)} problems from {split} split")
        except Exception as e:
            print(f"[CodeContests] Failed to load {split}: {e}")
            continue
        
        for row in ds:
            solutions = row.get("solutions", {})
            if not solutions:
                continue
            
            # Extract correct and incorrect Python solutions
            correct_sols = []
            wrong_sols = []
            
            langs = solutions.get("language", [])
            codes = solutions.get("solution", [])
            
            for lang, code in zip(langs, codes):
                if lang in [3]:  # 3 = Python3 in code_contests
                    correct_sols.append(code)
            
            incorrect = row.get("incorrect_solutions", {})
            if incorrect:
                i_langs = incorrect.get("language", [])
                i_codes = incorrect.get("solution", [])
                for lang, code in zip(i_langs, i_codes):
                    if lang in [3]:
                        wrong_sols.append(code)
            
            if not correct_sols or not wrong_sols:
                continue
            
            correct = correct_sols[0]
            name = row.get("name", f"cc_{len(pairs)}")
            
            for buggy in wrong_sols[:max_pairs]:
                pairs.append({
                    "task_id": f"codecontests_{name}",
                    "correct_code": correct,
                    "buggy_code": buggy,
                    "source": "codecontests",
                })
        
        if len(pairs) >= max_problems * max_pairs:
            break
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "codenet_pairs.json")  # Same output name
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[CodeContests] Saved {len(pairs)} pairs to {path}")
    return pairs


def download_apps(out_dir, max_pairs=3000):
    """
    Download APPS dataset and construct correct/buggy pairs.
    Uses problems that have both correct and incorrect solutions.
    """
    from datasets import load_dataset
    
    print("[APPS] Downloading from codeparrot/apps...")
    ds = load_dataset("codeparrot/apps", "all", split="test", trust_remote_code=True)
    print(f"[APPS] Loaded {len(ds)} problems")
    
    pairs = []
    for row in ds:
        solutions_raw = row.get("solutions", "")
        if not solutions_raw or solutions_raw == "":
            continue
        
        try:
            solutions = json.loads(solutions_raw) if isinstance(solutions_raw, str) else solutions_raw
        except json.JSONDecodeError:
            continue
        
        if not solutions or len(solutions) < 2:
            continue
        
        # APPS doesn't have explicit buggy solutions, but we can use:
        # - First solution as "correct" (verified by test cases)
        # - Generate buggy variants via LLM (done in step 2)
        # For now, store the correct solutions for later bug injection
        
        task_id = row.get("problem_id", f"apps_{len(pairs)}")
        pairs.append({
            "task_id": f"apps_{task_id}",
            "correct_code": solutions[0],
            "all_solutions": solutions[:5],  # Keep up to 5 solutions
            "input_output": row.get("input_output", ""),
            "difficulty": row.get("difficulty", "unknown"),
            "source": "apps",
        })
        
        if len(pairs) >= max_pairs:
            break
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "apps_solutions.json")
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[APPS] Saved {len(pairs)} problems (with solutions) to {path}")
    return pairs


def download_mbpp(out_dir):
    """Download MBPP dataset (correct solutions + test cases for bug injection)."""
    from datasets import load_dataset
    
    print("[MBPP] Downloading from google-research-datasets/mbpp...")
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    print(f"[MBPP] Loaded {len(ds)} problems")
    
    problems = []
    for row in ds:
        problems.append({
            "task_id": f"mbpp_{row['task_id']}",
            "text": row["text"],          # Problem description
            "code": row["code"],           # Correct solution
            "test_list": row["test_list"], # Test cases for verification
            "source": "mbpp",
        })
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mbpp_full.json")
    with open(path, "w") as f:
        json.dump(problems, f, indent=2)
    print(f"[MBPP] Saved {len(problems)} problems to {path}")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Siamese GAT journal paper")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["all", "humanevalfix", "codenet", "apps", "mbpp"],
                       help="Which dataset to download")
    parser.add_argument("--output-dir", type=str, default="./data/raw")
    parser.add_argument("--max-problems", type=int, default=2000)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets_to_download = {
        "humanevalfix": lambda: download_humanevalfix(f"{args.output_dir}/humanevalfix"),
        "codenet": lambda: download_codenet(f"{args.output_dir}/codenet", args.max_problems),
        "apps": lambda: download_apps(f"{args.output_dir}/apps"),
        "mbpp": lambda: download_mbpp(f"{args.output_dir}/mbpp"),
    }
    
    if args.dataset == "all":
        for name, fn in datasets_to_download.items():
            print(f"\n{'='*60}")
            try:
                fn()
            except Exception as e:
                print(f"[{name}] FAILED: {e}")
                print(f"[{name}] Skipping... you can retry with --dataset {name}")
    else:
        datasets_to_download[args.dataset]()
    
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    for root, dirs, files in os.walk(args.output_dir):
        for f in files:
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
