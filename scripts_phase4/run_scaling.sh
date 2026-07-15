#!/bin/bash
# Scaling experiment (review2.md "必做2"): mirror Phase-2's M0-base (text100)
# vs M3-bio50 (bio50) contrast at two smaller DENSE Gemma-4 sizes (E2B ~2.3B,
# E4B ~4.5B), to see whether the capability-shaping curve (general flat, bio
# rises) holds, grows, or reverses below 26B. Same 100M-token budget, same
# bio50 mixture (text50/protein35/dna10/biolit5) as Phase-2's M0/M3.
set -uo pipefail
source /etc/network_turbo 2>/dev/null || true
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
PY=/root/autodl-tmp/miniconda3/bin/python
export POOL_DIR=/autodl-fs/data/bt2_phase2/gemma4_small_pools
export OUT_ROOT=/autodl-fs/data/bt2_phase2/gemma4_small_models
export MIX_TOKENS=100000000
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
mkdir -p "$LOG"

BASE=/autodl-fs/data/bt2_phase2/gemma4_small
BIO50_MIX='{"text":0.5,"protein":0.35,"dna":0.1,"biolit":0.05}'
TEXT_MIX='{"text":1.0}'

run () {  # TAG  MODEL_DIR  MIX_JSON
  local tag="$1"; local model="$2"; local mix="$3"
  if [ -f "$OUT_ROOT/$tag/cpt_meta.json" ]; then
    echo "[skip] $tag already done"; return
  fi
  echo "=================== $tag  model=$model  $mix ==================="
  TAG="$tag" MODEL_DIR="$model" MIX="$mix" $PY run_scaling.py 2>&1 | tee "$LOG/$tag.log"
}

run E2B-text100 "$BASE/gemma-4-E2B-it" "$TEXT_MIX"
run E2B-bio50   "$BASE/gemma-4-E2B-it" "$BIO50_MIX"
run E4B-text100 "$BASE/gemma-4-E4B-it" "$TEXT_MIX"
run E4B-bio50   "$BASE/gemma-4-E4B-it" "$BIO50_MIX"

echo "SCALING EXPERIMENT DONE"
