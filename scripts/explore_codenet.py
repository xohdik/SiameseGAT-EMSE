"""
Explore extracted CodeNet dataset - count everything before building pairs.

Usage:
    python scripts/explore_codenet.py --codenet-dir /data/workzone/PhD/Project_CodeNet
"""
import argparse
import csv
import os
from collections import defaultdict, Counter
from pathlib import Path


def explore_codenet(codenet_dir):
    """Full exploration of CodeNet directory structure."""
    
    # ═══════════════════════════════════════
    # 1. VERIFY DIRECTORY STRUCTURE
    # ═══════════════════════════════════════
    print("=" * 70)
    print(f"EXPLORING: {codenet_dir}")
    print("=" * 70)
    
    # Check top-level dirs
    top_dirs = sorted(os.listdir(codenet_dir))
    print(f"\nTop-level contents: {top_dirs}")
    
    # Find actual data/metadata dirs (might be nested)
    data_dir = None
    metadata_dir = None
    
    for candidate in [codenet_dir, os.path.join(codenet_dir, "Project_CodeNet")]:
        if os.path.isdir(os.path.join(candidate, "data")):
            data_dir = os.path.join(candidate, "data")
            metadata_dir = os.path.join(candidate, "metadata")
            break
    
    if not data_dir:
        print("ERROR: Cannot find 'data' directory!")
        print(f"Searched: {codenet_dir} and {codenet_dir}/Project_CodeNet")
        return
    
    print(f"\nData dir:     {data_dir}")
    print(f"Metadata dir: {metadata_dir}")
    print(f"Data dir exists:     {os.path.isdir(data_dir)}")
    print(f"Metadata dir exists: {os.path.isdir(metadata_dir)}")
    
    # ═══════════════════════════════════════
    # 2. COUNT PROBLEMS
    # ═══════════════════════════════════════
    print("\n" + "=" * 70)
    print("PROBLEMS")
    print("=" * 70)
    
    problem_dirs = sorted([d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("p")])
    print(f"Total problem directories: {len(problem_dirs)}")
    print(f"First 5: {problem_dirs[:5]}")
    print(f"Last 5:  {problem_dirs[-5:]}")
    
    # ═══════════════════════════════════════
    # 3. COUNT LANGUAGES FROM DATA DIRS
    # ═══════════════════════════════════════
    print("\n" + "=" * 70)
    print("LANGUAGES (from data directories)")
    print("=" * 70)
    
    lang_problem_count = Counter()   # language → how many problems have it
    lang_file_count = Counter()      # language → total submission files
    
    # Sample first 500 problems for speed
    sample_problems = problem_dirs[:500]
    print(f"Scanning {len(sample_problems)} problems (sample)...")
    
    for pid in sample_problems:
        pdir = os.path.join(data_dir, pid)
        if not os.path.isdir(pdir):
            continue
        for lang_dir in os.listdir(pdir):
            lang_path = os.path.join(pdir, lang_dir)
            if os.path.isdir(lang_path):
                n_files = len([f for f in os.listdir(lang_path) if os.path.isfile(os.path.join(lang_path, f))])
                lang_problem_count[lang_dir] += 1
                lang_file_count[lang_dir] += n_files
    
    print(f"\n{'Language':<20} {'Problems':<12} {'Files (sample)':<15}")
    print("-" * 50)
    for lang, count in lang_file_count.most_common(20):
        print(f"{lang:<20} {lang_problem_count[lang]:<12} {count:<15}")
    
    # ═══════════════════════════════════════
    # 4. METADATA ANALYSIS
    # ═══════════════════════════════════════
    print("\n" + "=" * 70)
    print("METADATA ANALYSIS")
    print("=" * 70)
    
    if not os.path.isdir(metadata_dir):
        print("Metadata dir not found — skipping")
    else:
        csv_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith(".csv") and f != "problem_list.csv"])
        print(f"Metadata CSV files: {len(csv_files)}")
        
        # Read problem_list.csv if exists
        problem_list_path = os.path.join(metadata_dir, "problem_list.csv")
        if os.path.exists(problem_list_path):
            with open(problem_list_path, "r", errors="ignore") as f:
                lines = f.readlines()
            print(f"problem_list.csv: {len(lines)-1} problems listed")
            print(f"  Header: {lines[0].strip()}")
            print(f"  Example: {lines[1].strip()}")
        
        # Sample metadata CSVs to count accepted/wrong per language
        print(f"\nSampling metadata from {min(len(csv_files), 500)} problems...")
        
        status_lang_count = defaultdict(Counter)  # lang → {status → count}
        problems_with_pairs = defaultdict(int)     # lang → n problems with accepted+wrong
        
        for i, csv_file in enumerate(csv_files[:500]):
            pid = csv_file.replace(".csv", "")
            csv_path = os.path.join(metadata_dir, csv_file)
            
            try:
                lang_status = defaultdict(lambda: {"accepted": 0, "wrong": 0, "other": 0})
                
                with open(csv_path, "r", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        status = row.get("status", "").strip()
                        lang = row.get("language", "").strip()
                        
                        if not lang:
                            continue
                        
                        if status == "Accepted":
                            lang_status[lang]["accepted"] += 1
                            status_lang_count[lang]["Accepted"] += 1
                        elif status == "Wrong Answer":
                            lang_status[lang]["wrong"] += 1
                            status_lang_count[lang]["Wrong Answer"] += 1
                        else:
                            lang_status[lang]["other"] += 1
                            status_lang_count[lang][status] += 1
                
                # Check which languages have both accepted and wrong
                for lang, counts in lang_status.items():
                    if counts["accepted"] > 0 and counts["wrong"] > 0:
                        problems_with_pairs[lang] += 1
                        
            except Exception as e:
                if i < 3:
                    print(f"  Error reading {csv_file}: {e}")
        
        # Print submission counts by language
        print(f"\n{'Language':<20} {'Accepted':<12} {'Wrong Ans':<12} {'Other':<12} {'Pairable Problems':<20}")
        print("-" * 80)
        
        # Sort by total submissions
        sorted_langs = sorted(status_lang_count.keys(), 
                             key=lambda l: sum(status_lang_count[l].values()), reverse=True)
        
        for lang in sorted_langs[:25]:
            accepted = status_lang_count[lang].get("Accepted", 0)
            wrong = status_lang_count[lang].get("Wrong Answer", 0)
            other = sum(v for k, v in status_lang_count[lang].items() if k not in ["Accepted", "Wrong Answer"])
            pairable = problems_with_pairs.get(lang, 0)
            print(f"{lang:<20} {accepted:<12} {wrong:<12} {other:<12} {pairable:<20}")
    
    # ═══════════════════════════════════════
    # 5. ESTIMATE PAIR COUNTS
    # ═══════════════════════════════════════
    print("\n" + "=" * 70)
    print("ESTIMATED PAIRS (if we use max 5 wrong per problem)")
    print("=" * 70)
    
    target_langs = ["Python", "Java", "C++", "C", "Ruby", "JavaScript", "Go", "C#"]
    
    for lang in target_langs:
        pairable = problems_with_pairs.get(lang, 0)
        # Extrapolate from 500 sample to full 4053 problems
        if len(csv_files) > 500:
            ratio = len(csv_files) / 500
            est_pairable = int(pairable * ratio)
        else:
            est_pairable = pairable
        est_pairs = est_pairable * 3  # Conservative: avg 3 wrong per problem
        
        if est_pairs > 0:
            print(f"  {lang:<15} ~{est_pairable:>5} problems → ~{est_pairs:>6} pairs")
    
    # ═══════════════════════════════════════
    # 6. SAMPLE: Show one problem structure
    # ═══════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAMPLE PROBLEM STRUCTURE")
    print("=" * 70)
    
    # Find a problem with multiple languages
    for pid in problem_dirs[:50]:
        pdir = os.path.join(data_dir, pid)
        langs = sorted([d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d))])
        if len(langs) >= 3:
            print(f"\nProblem: {pid}")
            print(f"Languages: {langs}")
            for lang in langs[:5]:
                ldir = os.path.join(pdir, lang)
                files = sorted(os.listdir(ldir))
                print(f"  {lang}: {len(files)} submissions (e.g., {files[0] if files else 'empty'})")
            
            # Show metadata
            csv_path = os.path.join(metadata_dir, f"{pid}.csv")
            if os.path.exists(csv_path):
                with open(csv_path, "r", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                print(f"  Metadata: {len(rows)} total submissions")
                # Count by status
                statuses = Counter(r.get("status", "?") for r in rows)
                for s, c in statuses.most_common():
                    print(f"    {s}: {c}")
            break
    
    print("\n" + "=" * 70)
    print("DONE! Now run download_fix.py to build pairs.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codenet-dir", type=str, required=True,
                       help="Path to extracted Project_CodeNet directory")
    args = parser.parse_args()
    explore_codenet(args.codenet_dir)


if __name__ == "__main__":
    main()
