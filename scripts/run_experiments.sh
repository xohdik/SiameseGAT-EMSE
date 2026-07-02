#!/bin/bash
# Run all experiments: CLS baseline, edge ablations, cross-lingual transfer
#
# Usage: bash scripts/run_experiments.sh cuda:1

DEVICE=${1:-cuda:1}

echo "════════════════════════════════════════════════════"
echo "  EXPERIMENT SUITE"
echo "  Device: $DEVICE"
echo "════════════════════════════════════════════════════"

# ─── Step 1: CLS Baseline (no GPU needed, ~2 min total) ───
echo ""
echo "━━━ Step 1: CLS Cosine Baseline ━━━"
python scripts/baseline_cls.py --all 2>&1 | tee outputs/cls_baseline_all.log

# ─── Step 2: Create edge ablation files (CPU only, ~15 min) ───
echo ""
echo "━━━ Step 2: Creating edge ablation variants ━━━"
python scripts/ablation_edges.py --all 2>&1 | tee outputs/ablation_edges.log

# ─── Step 3: Train ablation variants (GPU, ~30 min each) ───
echo ""
echo "━━━ Step 3: Training ablation variants ━━━"
for suffix in dfg_only cfg_only seq_only; do
    for lang in python cpp java c ruby javascript; do
        echo ""
        echo "--- $lang / $suffix ---"
        OUTDIR="./outputs/siamese_gat_${lang}_${suffix}"
        mkdir -p "$OUTDIR"
        python scripts/train_spec.py \
            --lang $lang \
            --suffix $suffix \
            --model-type siamese_gat \
            --device $DEVICE \
            --batch-size 32 \
            2>&1 | tee "$OUTDIR/train.log"
    done
done

# ─── Step 4: Cross-lingual transfer (GPU, ~6 hours) ───
echo ""
echo "━━━ Step 4: Cross-lingual transfer matrix ━━━"
python scripts/cross_lingual.py --device $DEVICE 2>&1 | tee outputs/cross_lingual.log

echo ""
echo "════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "════════════════════════════════════════════════════"