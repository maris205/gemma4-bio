#!/bin/bash
# Auto-distribute the 7 Phase-II models across all visible GPUs, one process
# per card (each card runs its shard serially). Re-runnable: models with a
# cpt_meta.json are skipped, so this also resumes a partial sweep.
#
# Usage:  bash launch_parallel.sh
set -e
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
mkdir -p "$LOG"

ALL=(M0-base M1-bio5 M2-bio20 M3-bio50 C1-protein C2-dna C3-protdna)
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected $NGPU GPU(s); distributing ${#ALL[@]} models."

# round-robin assign models to GPUs
declare -A SHARD
for i in "${!ALL[@]}"; do
  g=$(( i % NGPU ))
  SHARD[$g]="${SHARD[$g]} ${ALL[$i]}"
done

for g in $(seq 0 $((NGPU-1))); do
  list="${SHARD[$g]## }"
  echo "  GPU $g -> $list"
  CUDA_VISIBLE_DEVICES=$g nohup bash run_shard.sh "$list" \
    > "$LOG/shard_gpu$g.log" 2>&1 &
  echo "    pid $!"
done
echo "All shards launched. Tail logs in $LOG/shard_gpu*.log"
