#!/bin/bash
# One GPU, a list of model TAGs, run them serially on that card.
# Usage:  CUDA_VISIBLE_DEVICES=<gpu> bash run_shard.sh "TAG1 TAG2 ..."
# Mix definitions are looked up from the MIXES table below (single source of truth).
set -e
source /etc/network_turbo 2>/dev/null || true
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
PY=/root/autodl-tmp/miniconda3/bin/python

export POOL_DIR=/autodl-fs/data/bt2_phase2/pools
export OUT_ROOT=/autodl-fs/data/bt2_phase2/models
export MODEL_DIR=/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio
export MIX_TOKENS=${MIX_TOKENS:-100000000}
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
mkdir -p "$LOG"

declare -A MIXES=(
  [M0-base]='{"text":1.0}'
  [M1-bio5]='{"text":0.95,"protein":0.035,"dna":0.010,"biolit":0.005}'
  [M2-bio20]='{"text":0.80,"protein":0.140,"dna":0.040,"biolit":0.020}'
  [M3-bio50]='{"text":0.50,"protein":0.350,"dna":0.100,"biolit":0.050}'
  [C1-protein]='{"text":0.80,"protein":0.20}'
  [C2-dna]='{"text":0.80,"dna":0.20}'
  [C3-protdna]='{"text":0.80,"protein":0.10,"dna":0.10}'
)

for tag in $1; do
  mix="${MIXES[$tag]}"
  if [ -z "$mix" ]; then echo "[shard] unknown tag $tag, skip"; continue; fi
  if [ -f "$OUT_ROOT/$tag/cpt_meta.json" ]; then
    echo "[shard gpu=$CUDA_VISIBLE_DEVICES] $tag already done, skip"; continue
  fi
  echo "[shard gpu=$CUDA_VISIBLE_DEVICES] === $tag $mix ==="
  TAG="$tag" MIX="$mix" $PY run_cpt_mix.py 2>&1 | tee "$LOG/$tag.log"
done
echo "[shard gpu=$CUDA_VISIBLE_DEVICES] DONE: $1"
