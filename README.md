# Siamese GAT for Code Correctness Verification

**Graph Attention Networks for Verifying Correctness of AI-Generated Code: A Multi-Benchmark Empirical Study**

Tunde (UESTC) — Supervised by Prof. Bo Chen

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."        # For LLM baselines
export ANTHROPIC_API_KEY="sk-ant-..." # Optional
bash scripts/run_all.sh               # Run full pipeline
```

## Pipeline Steps

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `download_datasets.py` | Download HumanEvalFix, CodeNet, APPS, MBPP |
| 2 | `build_pairs.py` | Construct correct/buggy pairs + LLM bug injection |
| 3 | `build_graphs.py` | Extract DFGs + GraphCodeBERT embeddings → PyG Data |
| 4 | `train.py` | 5-fold CV training + cross-benchmark transfer |
| 5 | `llm_baselines.py` | Evaluate GPT-4, Claude, DeepSeek baselines |
| 6 | `bug_localization.py` | Attention-based bug localization (unique contribution) |

## Architecture

```
Code_A → GraphCodeBERT → DFG → [2-layer GAT] → Attn Pool → z_a ─┐
                                 (shared)                          ├→ [z_a‖z_b‖diff‖prod] → MLP → ŷ
Code_B → GraphCodeBERT → DFG → [2-layer GAT] → Attn Pool → z_b ─┘
```

## Target: ACM TOSEM Fast-Impact Track (90-day review, IF 6.6)
