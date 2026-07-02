#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Spec-Grounded Graph Neural Network for Code Verification
# NO pretrained models — fully independent
# ═══════════════════════════════════════════════════════════
# Run from: /data/workzone/siamese_gat_journal
# GPU: cuda:1
# ═══════════════════════════════════════════════════════════

cd /data/workzone/siamese_gat_journal
export PYTHONPATH="${PYTHONPATH}:scripts"

# ── Step 1: Build exec dataset with IO specs ──
echo "━━━ Step 1: Building exec dataset (generating IO specs) ━━━"
python scripts/build_exec_data.py \
    --pairs data/processed/pairs_codenet_python.json \
    --lang python --gen-io --max-problems 500 \
    --output data/exec/exec_dataset_python.pt

# ── Step 2: Full model (Code Graph + IO Spec + Cross-Attention) ──
echo ""
echo "━━━ Step 2: Full model training ━━━"
python scripts/train_exec.py \
    --data data/exec/exec_dataset_python.pt \
    --device cuda:1 --lr 5e-3 --batch-size 32 \
    --gat-layers 4 --patience 20

# ── Step 3: Ablation - Code only (no IO spec) ──
echo ""
echo "━━━ Step 3: Ablation - Code only (no spec) ━━━"
python scripts/train_exec.py \
    --data data/exec/exec_dataset_python.pt \
    --device cuda:1 --no-spec --lr 5e-3 --batch-size 32

# ── Step 4: Ablation - Seq-only edges (prove DFG/CFG matter) ──
echo ""
echo "━━━ Step 4: Ablation - Seq-only edges ━━━"
python scripts/train_exec.py \
    --data data/exec/exec_dataset_python.pt \
    --device cuda:1 --edge-filter seq_only --lr 5e-3 --batch-size 32

# ── Step 5: Ablation - DFG-only edges ──
echo ""
echo "━━━ Step 5: Ablation - DFG-only edges ━━━"
python scripts/train_exec.py \
    --data data/exec/exec_dataset_python.pt \
    --device cuda:1 --edge-filter dfg_only --lr 5e-3 --batch-size 32

# ── Step 6: Ablation - CFG-only edges ──
echo ""
echo "━━━ Step 6: Ablation - CFG-only edges ━━━"
python scripts/train_exec.py \
    --data data/exec/exec_dataset_python.pt \
    --device cuda:1 --edge-filter cfg_only --lr 5e-3 --batch-size 32

echo ""
echo "━━━ ALL DONE ━━━"
echo ""
echo "Expected ablation matrix:"
echo "┌─────────────────────────┬────────┐"
echo "│ Configuration           │ F1     │"
echo "├─────────────────────────┼────────┤"
echo "│ Full (graph+spec+xattn) │ ???    │"
echo "│ Code only (no spec)     │ ???    │"
echo "│ Full + seq-only edges   │ ???    │"
echo "│ Full + DFG-only edges   │ ???    │"
echo "│ Full + CFG-only edges   │ ???    │"
echo "└─────────────────────────┴────────┘"
echo ""
echo "If Full >> Seq-only: DFG/CFG carry execution signal"
echo "If Full >> Code-only: IO specs ground the semantics"
echo ""
echo "Results in outputs/exec*/results.json"