#!/bin/bash
cd /data/workzone/siamese_gat_journal

echo "━━━ Step 1: Enriching graphs with token_ids ━━━"
python scripts/enrich_token_ids.py --lang python

echo ""
echo "━━━ Step 2: Learnable embeddings + Full edges ━━━"
python scripts/train_spec.py \
    --lang python --device cuda:1 --no-pretrained --batch-size 32

echo ""
echo "━━━ Step 3: Learnable embeddings + Seq-only edges ━━━"
python scripts/train_spec.py \
    --lang python --device cuda:1 --no-pretrained \
    --edge-filter seq_only --batch-size 32

echo ""
echo "━━━ DONE ━━━"
cat outputs/siamese_gat_python_no_pretrained/results.json 2>/dev/null
cat outputs/siamese_gat_python_seq_only_no_pretrained/results.json 2>/dev/null
