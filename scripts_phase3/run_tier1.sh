#!/bin/bash
# Bio2NL Tier-1: Continuation matched-compute design, multi-seed.
#   warmup: gpt2 --(NL 300M)--> nl_base            [shared start, 1x]
#   then from nl_base, per seed, two arms of equal CONT_TOKENS:
#     NL-arm : continue on NL      (control: extra steps contain NL)
#     Bio-arm: continue on protein+dna (treatment: extra steps contain biology)
# Only the continuation content differs => any NL-eval delta = "what the extra steps contained".
# Multi-seed because the effect is expected small; we need mean±sd across seeds.
#
# Cards: warmup on GPU0; then each (seed,arm) on its own GPU in parallel.
set -uo pipefail
cd /root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/scripts
PY=/root/autodl-tmp/miniconda3/bin/python
ROOT=/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2
export POOL_DIR=$ROOT/pools
LOG=$ROOT/logs; MODELS=$ROOT/models; mkdir -p "$LOG" "$MODELS"

WARM_TOKENS=${WARM_TOKENS:-300000000}
CONT_TOKENS=${CONT_TOKENS:-200000000}
SEEDS=(0 1 2)

# ---- 1. warmup (once) ----
if [ ! -f "$MODELS/nl_base/train_meta.json" ]; then
  echo "=== warmup gpt2 -> nl_base (NL $WARM_TOKENS tok) ==="
  CUDA_VISIBLE_DEVICES=0 STAGE=warmup INIT=gpt2 MIX='{"nl":1.0}' \
    TOKENS=$WARM_TOKENS OUT=$MODELS/nl_base \
    $PY train.py 2>&1 | tee "$LOG/warmup.log"
else
  echo "[skip] nl_base exists"
fi

# ---- 2. two arms x seeds, parallel across cards ----
g=0
for s in "${SEEDS[@]}"; do
  for arm in nl bio; do
    tag="${arm}_seed${s}"
    [ -f "$MODELS/$tag/train_meta.json" ] && { echo "[skip] $tag"; g=$((g+1)); continue; }
    mix='{"nl":1.0}'; [ "$arm" = "bio" ] && mix='{"protein":0.5,"dna":0.5}'
    echo "  GPU $g -> $tag ($mix)"
    CUDA_VISIBLE_DEVICES=$g STAGE=continue INIT=$MODELS/nl_base MIX="$mix" \
      TOKENS=$CONT_TOKENS OUT=$MODELS/$tag SEED_OFFSET=$s \
      nohup $PY train.py > "$LOG/$tag.log" 2>&1 &
    g=$((g+1))
  done
done
wait
echo "TIER1 TRAINING DONE"
