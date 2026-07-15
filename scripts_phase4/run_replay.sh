#!/bin/bash
# Replay-control experiment (review2.md "必做3"): does an explicit end-of-
# training text replay block recover general capability better than uniform
# shuffling at the SAME total tokens and SAME overall bio/text ratio?
#
# Baseline (already exists from Phase-2, REPLAY_TAIL_FRAC=0): M3-bio50
#   text50/protein35/dna10/biolit5, 100M tokens, fully shuffled.
# New arms (same mix, same 100M budget, only the TAIL SCHEDULE differs):
#   R1-bio50-replaytail10: last 10% of tokens = pure text block at the end
#   R2-bio50-replaytail20: last 20% of tokens = pure text block at the end
#
# One GPU, two models serial (or pass GPU via CUDA_VISIBLE_DEVICES externally
# to run in parallel with another job on a second card).
set -uo pipefail
source /etc/network_turbo 2>/dev/null || true
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
PY=/root/autodl-tmp/miniconda3/bin/python
export POOL_DIR=/autodl-fs/data/bt2_phase2/pools
export OUT_ROOT=/autodl-fs/data/bt2_phase2/models
export MODEL_DIR=/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio
export MIX_TOKENS=100000000
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
mkdir -p "$LOG"

BIO50_MIX='{"text":0.5,"protein":0.35,"dna":0.1,"biolit":0.05}'

run () {  # TAG  TAIL_FRAC
  local tag="$1"; local tail="$2"
  if [ -f "$OUT_ROOT/$tag/cpt_meta.json" ]; then
    echo "[skip] $tag already done"; return
  fi
  echo "=================== $tag  replay_tail_frac=$tail ==================="
  TAG="$tag" MIX="$BIO50_MIX" REPLAY_TAIL_FRAC="$tail" REPLAY_TEXT_SOURCE=text \
    $PY run_cpt_mix.py 2>&1 | tee "$LOG/$tag.log"
}

run R1-bio50-replaytail10 0.10
run R2-bio50-replaytail20 0.20

echo "REPLAY CONTROL DONE"
