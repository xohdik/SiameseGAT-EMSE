"""
Fix language name inconsistency across datasets before graph building.

Problem:
  CodeNet uses: "Python", "Java", "C++", "C", "Ruby", "JavaScript"
  HumanEvalFix uses: "python", "java", "cpp", "js", "go", "rust"

Solution: Normalize everything to lowercase canonical names.

Also fixes dataset names for consistency.

Usage:
    python scripts/fix_language_names.py --data-dir ./data/processed
"""
import argparse
import json
import os
from collections import defaultdict

# Canonical lowercase language names
LANG_MAP = {
    # CodeNet variants
    "Python": "python",
    "Java": "java",
    "C++": "cpp",
    "C": "c",
    "Ruby": "ruby",
    "JavaScript": "javascript",
    "C#": "csharp",
    "Go": "go",
    # HumanEvalFix variants (already lowercase mostly)
    "python": "python",
    "java": "java",
    "cpp": "cpp",
    "js": "javascript",
    "go": "go",
    "rust": "rust",
    "c": "c",
    "ruby": "ruby",
    "javascript": "javascript",
    "csharp": "csharp",
}

# Canonical dataset names
DATASET_MAP = {
    "codenet_cpp": "codenet_cpp",
    "codenet_python": "codenet_python",
    "codenet_java": "codenet_java",
    "codenet_c": "codenet_c",
    "codenet_ruby": "codenet_ruby",
    "codenet_javascript": "codenet_javascript",
    "codenet_csharp": "codenet_csharp",
    "humanevalfix_python": "humanevalfix_python",
    "humanevalfix_java": "humanevalfix_java",
    "humanevalfix_cpp": "humanevalfix_cpp",
    "humanevalfix_js": "humanevalfix_javascript",  # Normalize js → javascript
    "humanevalfix_javascript": "humanevalfix_javascript",
    "humanevalfix_go": "humanevalfix_go",
    "humanevalfix_rust": "humanevalfix_rust",
    "apps": "apps",
    "mbpp": "mbpp",
}


def fix_pair(pair):
    """Normalize language and dataset names in a single pair."""
    lang = pair.get("language", "")
    pair["language"] = LANG_MAP.get(lang, lang.lower())

    ds = pair.get("dataset", "")
    pair["dataset"] = DATASET_MAP.get(ds, ds)

    return pair


def fix_file(filepath):
    """Fix all pairs in a JSON file."""
    with open(filepath) as f:
        pairs = json.load(f)

    fixed = [fix_pair(p) for p in pairs]

    with open(filepath, "w") as f:
        json.dump(fixed, f)

    return len(fixed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/processed")
    args = parser.parse_args()

    print("=" * 70)
    print("FIXING LANGUAGE NAME INCONSISTENCIES")
    print("=" * 70)

    # Fix all JSON files in data dir
    json_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".json")])

    for fname in json_files:
        fpath = os.path.join(args.data_dir, fname)
        n = fix_file(fpath)
        print(f"  Fixed {fname} ({n} pairs)")

    # Rename files with inconsistent names
    renames = {
        "pairs_humanevalfix_js.json": "pairs_humanevalfix_javascript.json",
        "pairs_codenet_cpp.json": "pairs_codenet_cpp.json",  # Already correct
    }
    for old_name, new_name in renames.items():
        old_path = os.path.join(args.data_dir, old_name)
        new_path = os.path.join(args.data_dir, new_name)
        if old_name != new_name and os.path.exists(old_path):
            # Merge if target already exists
            if os.path.exists(new_path):
                with open(old_path) as f:
                    old_data = json.load(f)
                with open(new_path) as f:
                    new_data = json.load(f)
                # Deduplicate by pair_id
                seen = {p["pair_id"] for p in new_data}
                merged = new_data + [p for p in old_data if p["pair_id"] not in seen]
                with open(new_path, "w") as f:
                    json.dump(merged, f)
                os.remove(old_path)
                print(f"  Merged {old_name} → {new_name} ({len(merged)} pairs)")
            else:
                os.rename(old_path, new_path)
                print(f"  Renamed {old_name} → {new_name}")

    # Verify: reload pairs_all.json and print summary
    all_path = os.path.join(args.data_dir, "pairs_all.json")
    with open(all_path) as f:
        all_pairs = json.load(f)

    print(f"\n{'='*70}")
    print(f"VERIFIED: {len(all_pairs)} total pairs")
    print(f"{'='*70}")

    print(f"\nBy language (normalized):")
    lang_counts = defaultdict(int)
    for p in all_pairs:
        lang_counts[p["language"]] += 1
    for lang, c in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang:<20} {c:>7}")

    print(f"\nBy dataset (normalized):")
    ds_counts = defaultdict(int)
    for p in all_pairs:
        ds_counts[p["dataset"]] += 1
    for ds, c in sorted(ds_counts.items(), key=lambda x: -x[1]):
        print(f"  {ds:<30} {c:>7}")

    # Check for any unmapped languages
    all_langs = set(p["language"] for p in all_pairs)
    unknown = all_langs - set(LANG_MAP.values())
    if unknown:
        print(f"\n⚠ WARNING: Unmapped languages found: {unknown}")
    else:
        print(f"\n✅ All languages normalized correctly")

    # Check for any unmapped datasets
    all_ds = set(p["dataset"] for p in all_pairs)
    unknown_ds = all_ds - set(DATASET_MAP.values())
    if unknown_ds:
        print(f"⚠ WARNING: Unmapped datasets found: {unknown_ds}")
    else:
        print(f"✅ All dataset names normalized correctly")


if __name__ == "__main__":
    main()
