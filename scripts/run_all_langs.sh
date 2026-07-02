#!/bin/bash
# Run 5-fold baseline training per language
# Each language fits in RAM (~8-27GB), runs ~230s/epoch
#
# Usage: bash scripts/run_all_langs.sh cuda:1

DEVICE=${1:-cuda:1}
LANGS="python cpp java c ruby javascript"

echo "════════════════════════════════════════"
echo "  Per-Language Baseline Training"
echo "  Device: $DEVICE"
echo "  Languages: $LANGS"
echo "════════════════════════════════════════"

for lang in $LANGS; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Starting: $lang"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    OUTDIR="./outputs/siamese_gat_${lang}"
    mkdir -p "$OUTDIR"
    
    python scripts/train_spec.py \
        --lang $lang \
        --model-type siamese_gat \
        --device $DEVICE \
        --batch-size 32 \
        2>&1 | tee "$OUTDIR/train.log"
    
    echo "  ✓ Finished: $lang (log: $OUTDIR/train.log)"
done

echo ""
echo "════════════════════════════════════════"
echo "  All languages complete!"
echo "  Results in ./outputs/siamese_gat_*/"
echo "════════════════════════════════════════"

# Print summary
echo ""
echo "Summary:"
for lang in $LANGS; do
    dir="./outputs/siamese_gat_${lang}"
    if [ -f "$dir/results.json" ]; then
        f1=$(python -c "import json; r=json.load(open('$dir/results.json')); print(f\"{r['summary']['f1_macro']['mean']:.4f} ± {r['summary']['f1_macro']['std']:.4f}\")")
        auc=$(python -c "import json; r=json.load(open('$dir/results.json')); print(f\"{r['summary']['auc']['mean']:.4f} ± {r['summary']['auc']['std']:.4f}\")")
        echo "  $lang: F1=$f1  AUC=$auc"
    else
        echo "  $lang: (no results)"
    fi
done