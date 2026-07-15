#!/bin/bash
# Eval all 4 scaling models (E2B/E4B x text100/bio50) on ONE GPU, serial.
# MMLU needs the Clash proxy for dataset fetch (cais/mmlu already cached from
# Phase-2's MMLU run, so this should be fast / offline-safe if cache holds).
set -uo pipefail
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_HUB_DISABLE_XET=1
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
RES=/root/autodl-tmp/dnagpt/bio-trans2/results_scaling
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
PY=/root/autodl-tmp/miniconda3/bin/python
mkdir -p "$RES" "$LOG"

TAGS=(E2B-text100 E2B-bio50 E4B-text100 E4B-bio50)
for t in "${TAGS[@]}"; do
  echo "=================== eval: $t (general) ==================="
  $PY run_eval_scaling.py --tag "$t" --tasks mmlu gsm8k --out-dir "$RES" 2>&1 | tee "$LOG/evalS_$t.log"
  echo "=================== eval: $t (bio) ==================="
  $PY run_eval_bio_scaling.py --tag "$t" --out-dir "$RES" 2>&1 | tee -a "$LOG/evalS_$t.log"
done
echo "SCALING EVAL DONE -> $RES"
