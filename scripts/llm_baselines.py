"""
Step 5: LLM Baselines for code correctness detection.

Evaluates LLMs on the same pairs as Siamese GAT. Providers:
- openai    : GPT-4o            (api.openai.com  - may be region-blocked)
- anthropic : Claude Sonnet     (api.anthropic.com - may be region-blocked)
- deepseek  : DeepSeek-V3/Coder (api.deepseek.com - reachable from China)
- local     : ollama / vLLM     (http://localhost:11434 - fully offline)

NOTE: this baseline is API-only and needs no GPU. If OpenAI/Anthropic are
region-blocked, either run this script from a machine/region with API access
(just copy the pairs_*.json files), set OPENAI_BASE_URL / ANTHROPIC_BASE_URL
to a proxy, or use --provider deepseek / local.

Key behaviours:
  * Prompts are LANGUAGE-AWARE ({lang}); the fenced block is not hardcoded to Python.
  * --languages restricts to the paper's six languages (drops Go/Rust HumanEvalFix).
  * Sampling is PER-DATASET: every HumanEvalFix (OOD) pair is kept; each CodeNet
    language is capped at --max-per-codenet.
  * The eval subset is drawn ONCE so every model/prompt is scored on the same pairs.
  * 403/401/auth errors fast-fail a provider after a few calls (no 1000x sleeps).
  * Final summary prints the CodeNet vs HumanEvalFix columns for the paper table.

Usage:
    # from a machine with OpenAI/Anthropic access:
    python llm_baselines.py --pairs ../data/processed/pairs_all.json --provider all --max-per-codenet 200
    # from the China server:
    export DEEPSEEK_API_KEY=sk-...
    python llm_baselines.py --pairs ../data/processed/pairs_all.json --provider deepseek --max-per-codenet 200
    python llm_baselines.py --pairs ../data/processed/pairs_all.json --provider local    --max-per-codenet 200
"""
import argparse, json, os, random, time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

# Paper languages (HumanEvalFix go/rust are excluded by default).
PAPER_LANGUAGES = ["python", "cpp", "java", "c", "ruby", "javascript"]

# ═══════════════════════════════════════
# PROMPTS  (language-aware)
# ═══════════════════════════════════════

ZERO_SHOT_PROMPT = """You are a code correctness detector. Given two versions of a {lang} function that should implement the same specification, determine which version is CORRECT and which is BUGGY.

Specification: {spec}

Version A:
```
{code_a}
```

Version B:
```
{code_b}
```

Which version is correct? Answer with ONLY "A" or "B", nothing else."""

COT_PROMPT = """You are a code correctness detector. Given two versions of a {lang} function, analyze them step by step to determine which is CORRECT.

Specification: {spec}

Version A:
```
{code_a}
```

Version B:
```
{code_b}
```

Analyze both versions step by step, checking for:
1. Boundary conditions
2. Operator correctness
3. Variable usage
4. Return values
5. Loop ranges

After your analysis, state your final answer on the LAST line as exactly: "ANSWER: A" or "ANSWER: B"."""


def parse_answer(text: str) -> Optional[str]:
    """Extract A or B from LLM response."""
    text = text.strip()
    lines = text.strip().split("\n")
    last = lines[-1].upper().strip()
    if "ANSWER: A" in last or "ANSWER:A" in last: return "A"
    if "ANSWER: B" in last or "ANSWER:B" in last: return "B"
    if text.upper() in ["A", "B"]: return text.upper()
    if text.upper().startswith("A"): return "A"
    if text.upper().startswith("B"): return "B"
    for line in reversed(lines):
        line = line.upper().strip()
        if "VERSION A" in line and "CORRECT" in line: return "A"
        if "VERSION B" in line and "CORRECT" in line: return "B"
    return None


def raw_lang(pair: Dict) -> str:
    """Raw lowercase language token, e.g. 'cpp', 'python', 'go'."""
    lang = pair.get("language")
    if not lang:
        ds = pair.get("dataset", "")
        lang = ds.split("_")[-1] if "_" in ds else ds
    return (lang or "").lower()


def lang_of(pair: Dict) -> str:
    """Human-readable language for the prompt."""
    pretty = {"cpp": "C++", "javascript": "JavaScript", "csharp": "C#"}
    lang = raw_lang(pair)
    return pretty.get(lang, lang.capitalize() if lang else "the given")


def is_auth_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(t in s for t in ("403", "401", "forbidden", "not allowed",
                                "permission", "unauthorized", "invalid_api_key",
                                "authentication"))


# ═══════════════════════════════════════
# LLM CALLERS  (base_url overridable via env for proxies)
# ═══════════════════════════════════════

def call_openai(prompt: str, model: str = "gpt-4o", temperature: float = 0.0) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=1024)
    return resp.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0) -> str:
    import anthropic
    client = anthropic.Anthropic(base_url=os.environ.get("ANTHROPIC_BASE_URL") or None)
    resp = client.messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text


def call_deepseek(prompt: str, model: str = "deepseek-chat", temperature: float = 0.0) -> str:
    """DeepSeek official API (OpenAI-compatible, reachable from China)."""
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    api_key=os.environ["DEEPSEEK_API_KEY"])
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=1024)
    return resp.choices[0].message.content


def call_local(prompt: str, model: str = "deepseek-coder-v2:latest", temperature: float = 0.0) -> str:
    """Local model via OpenAI-compatible API (ollama / vLLM). Fully offline."""
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("LOCAL_BASE_URL", "http://localhost:11434/v1"),
                    api_key="ollama")
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=1024)
    return resp.choices[0].message.content


LLM_CALLERS = {
    "openai":    {"fn": call_openai,    "models": ["gpt-4o"]},
    "anthropic": {"fn": call_anthropic, "models": ["claude-sonnet-4-20250514"]},
    "deepseek":  {"fn": call_deepseek,  "models": ["deepseek-chat"]},
    "local":     {"fn": call_local,     "models": ["qwen2.5-coder:7b"]},
}


# ═══════════════════════════════════════
# SAMPLING  (per-dataset, keeps all OOD, language-filtered)
# ═══════════════════════════════════════

def build_eval_set(pairs: List[Dict], max_per_codenet: int, seed: int,
                   allowed_langs: List[str]) -> List[Dict]:
    rng = random.Random(seed)
    allowed = set(allowed_langs)
    by_ds = defaultdict(list)
    for p in pairs:
        if raw_lang(p) in allowed:
            by_ds[p.get("dataset", "unknown")].append(p)

    eval_pairs = []
    print("Eval-set composition (languages: %s):" % ", ".join(allowed_langs))
    for ds in sorted(by_ds):
        ps = by_ds[ds]
        chosen = ps if ds.startswith("humanevalfix") else rng.sample(ps, min(max_per_codenet, len(ps)))
        eval_pairs.append(chosen)
        print(f"  {ds:<28} {len(chosen):>5} / {len(ps)}")
    flat = [p for chosen in eval_pairs for p in chosen]
    rng.shuffle(flat)
    print(f"  {'TOTAL':<28} {len(flat):>5}")
    return flat


# ═══════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════

def evaluate_llm(pairs: List[Dict], caller_fn, model_name: str,
                 prompt_type: str = "zero_shot", seed: int = 42) -> Dict:
    prompt_template = ZERO_SHOT_PROMPT if prompt_type == "zero_shot" else COT_PROMPT
    rng = random.Random(seed)

    results = []
    consec_auth_errors = 0
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  {model_name} [{prompt_type}]: {i}/{len(pairs)}...")

        swap = rng.random() < 0.5
        if swap:
            code_a, code_b, correct_answer = pair["buggy_code"], pair["correct_code"], "B"
        else:
            code_a, code_b, correct_answer = pair["correct_code"], pair["buggy_code"], "A"

        spec = pair.get("docstring", pair.get("prompt", "the given specification"))
        if not spec or not str(spec).strip():
            spec = "the given specification"

        prompt = prompt_template.format(lang=lang_of(pair), spec=spec, code_a=code_a, code_b=code_b)

        try:
            response = caller_fn(prompt, model_name)
            consec_auth_errors = 0
            predicted = parse_answer(response)
            results.append({
                "pair_id": pair.get("pair_id", f"pair_{i}"),
                "dataset": pair.get("dataset", "unknown"),
                "language": lang_of(pair),
                "correct_answer": correct_answer,
                "predicted": predicted,
                "is_correct": (predicted == correct_answer) if predicted else False,
                "swapped": swap,
                "response_preview": response[:200],
            })
        except Exception as e:
            if is_auth_error(e):
                consec_auth_errors += 1
                if consec_auth_errors >= 3:
                    raise RuntimeError(
                        f"{model_name}: provider blocked (403/auth) after {consec_auth_errors} calls. "
                        f"If region-blocked, set a proxy via *_BASE_URL, run from a machine with "
                        f"API access, or use --provider deepseek/local.")
            else:
                consec_auth_errors = 0
            print(f"  Error on pair {i}: {e}")
            results.append({
                "pair_id": pair.get("pair_id", f"pair_{i}"),
                "dataset": pair.get("dataset", "unknown"),
                "language": lang_of(pair),
                "correct_answer": correct_answer,
                "predicted": None, "is_correct": False, "error": str(e),
            })

        time.sleep(0.5)

    def f1_acc(rows):
        labels = [1 if r["correct_answer"] == "A" else 0 for r in rows]
        preds = [1 if r["predicted"] == "A" else 0 for r in rows]
        return (float(f1_score(labels, preds, average="macro")),
                float(accuracy_score(labels, preds)))

    valid = [r for r in results if r["predicted"] is not None]
    labels = [1 if r["correct_answer"] == "A" else 0 for r in valid]
    preds = [1 if r["predicted"] == "A" else 0 for r in valid]
    metrics = {
        "model": model_name, "prompt_type": prompt_type,
        "total": len(pairs), "valid": len(valid), "invalid": len(pairs) - len(valid),
        "accuracy": float(accuracy_score(labels, preds)) if valid else 0,
        "f1_macro": float(f1_score(labels, preds, average="macro")) if valid else 0,
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)) if valid else 0,
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)) if valid else 0,
    }
    ds_results = defaultdict(list)
    for r in valid:
        ds_results[r["dataset"]].append(r)
    metrics["per_dataset"] = {}
    for ds, ds_r in ds_results.items():
        f1, acc = f1_acc(ds_r)
        metrics["per_dataset"][ds] = {"n": len(ds_r), "f1_macro": f1, "accuracy": acc}

    cn = [r for r in valid if r["dataset"].startswith("codenet")]
    he = [r for r in valid if r["dataset"].startswith("humanevalfix")]
    metrics["codenet"] = (dict(zip(("f1_macro", "accuracy"), f1_acc(cn))) | {"n": len(cn)}) if cn else {}
    metrics["humanevalfix"] = (dict(zip(("f1_macro", "accuracy"), f1_acc(he))) | {"n": len(he)}) if he else {}
    return metrics, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="../data/processed/pairs_all.json")
    ap.add_argument("--output-dir", default="./outputs/llm_baselines")
    ap.add_argument("--provider", default="deepseek",
                    choices=["openai", "anthropic", "deepseek", "local", "all"])
    ap.add_argument("--max-per-codenet", type=int, default=200,
                    help="Max CodeNet pairs PER LANGUAGE (all HumanEvalFix pairs are kept).")
    ap.add_argument("--languages", default=",".join(PAPER_LANGUAGES),
                    help="Comma-separated languages to include (default: paper's six; excludes go/rust).")
    ap.add_argument("--prompt-types", nargs="+", default=["zero_shot", "cot"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    allowed_langs = [l.strip().lower() for l in args.languages.split(",") if l.strip()]

    with open(args.pairs) as f:
        pairs = json.load(f)
    pairs = [p for p in pairs if p.get("correct_code") and p.get("buggy_code")]
    print(f"Loaded {len(pairs)} valid pairs")

    eval_pairs = build_eval_set(pairs, args.max_per_codenet, args.seed, allowed_langs)

    providers = (["openai", "anthropic", "deepseek", "local"]
                 if args.provider == "all" else [args.provider])
    all_metrics = []
    for provider in providers:
        caller = LLM_CALLERS[provider]
        for model_name in caller["models"]:
            for prompt_type in args.prompt_types:
                print(f"\n{'='*60}\nEvaluating: {model_name} ({prompt_type})\n{'='*60}")
                try:
                    metrics, results = evaluate_llm(eval_pairs, caller["fn"], model_name, prompt_type, args.seed)
                    cn, he = metrics.get("codenet", {}), metrics.get("humanevalfix", {})
                    print(f"  Overall  F1={metrics['f1_macro']:.4f} Acc={metrics['accuracy']:.4f} "
                          f"(valid {metrics['valid']}/{metrics['total']})")
                    if cn: print(f"  CodeNet      F1={cn['f1_macro']:.4f} Acc={cn['accuracy']:.4f} (n={cn['n']})")
                    if he: print(f"  HumanEvalFix F1={he['f1_macro']:.4f} Acc={he['accuracy']:.4f} (n={he['n']})")
                    all_metrics.append(metrics)
                    fname = f"{model_name.replace('/', '_').replace(':', '_')}_{prompt_type}"
                    with open(os.path.join(args.output_dir, f"{fname}_results.json"), "w") as fp:
                        json.dump(results, fp, indent=2)
                except Exception as e:
                    print(f"  SKIPPED: {e}")

    with open(os.path.join(args.output_dir, "llm_summary.json"), "w") as fp:
        json.dump(all_metrics, fp, indent=2)

    print(f"\n{'='*78}\nLLM BASELINE SUMMARY  (maps to the paper's section 6.7 table)\n{'='*78}")
    print(f"{'Model':<26}{'Prompt':<11}{'CodeNet F1':>12}{'CN Acc':>9}{'HEFix F1':>10}{'HEFix Acc':>11}")
    print("-" * 78)
    for m in all_metrics:
        cn, he = m.get("codenet", {}), m.get("humanevalfix", {})
        print(f"{m['model']:<26}{m['prompt_type']:<11}"
              f"{cn.get('f1_macro', float('nan')):>12.4f}{cn.get('accuracy', float('nan')):>9.4f}"
              f"{he.get('f1_macro', float('nan')):>10.4f}{he.get('accuracy', float('nan')):>11.4f}")
    print("\nSiameseGAT (ours) reference:  CodeNet F1=0.922   HumanEvalFix F1=0.593")


if __name__ == "__main__":
    main()