#!/bin/bash
# Re-run MMLU (+ missing truthfulqa for C1) via Clash proxy, one card each.
# Datasets are cached; Clash handles the small metadata requests.
export OUT_ROOT=/autodl-fs/data/bt2_phase2/models
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
RES=/root/autodl-tmp/dnagpt/bio-trans2/results
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
PY=/root/autodl-tmp/miniconda3/bin/python
ALL=(M0-base M1-bio5 M2-bio20 M3-bio50 C1-protein C2-dna C3-protdna)
for i in "${!ALL[@]}"; do
  t="${ALL[$i]}"
  tasks="mmlu"; [ "$t" = "C1-protein" ] && tasks="mmlu truthfulqa_mc2"
  CUDA_VISIBLE_DEVICES=$i nohup $PY run_eval.py --tag "$t" --tasks $tasks \
    --out-dir "$RES" > "$LOG/mmlu4_$t.log" 2>&1 &
done
wait
echo "MMLU RERUN DONE"
