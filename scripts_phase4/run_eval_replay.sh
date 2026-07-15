#!/bin/bash
set -uo pipefail
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_HUB_DISABLE_XET=1
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
RES=/root/autodl-tmp/dnagpt/bio-trans2/results_replay
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
PY=/root/autodl-tmp/miniconda3/bin/python
mkdir -p "$RES" "$LOG"
TAGS=(R1-bio50-replaytail10 R2-bio50-replaytail20)
for t in "${TAGS[@]}"; do
  echo "=================== eval: $t (general) ==================="
  $PY run_eval.py --tag "$t" --tasks mmlu gsm8k --out-dir "$RES" 2>&1 | tee "$LOG/evalR_$t.log"
  echo "=================== eval: $t (bio) ==================="
  $PY run_eval_bio.py --tag "$t" --out-dir "$RES" 2>&1 | tee -a "$LOG/evalR_$t.log"
done
echo "REPLAY EVAL DONE -> $RES"
