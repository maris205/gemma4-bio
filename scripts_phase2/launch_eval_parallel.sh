#!/bin/bash
# After training: eval all 7 CPT models in parallel, one card each.
# Only runs models whose cpt_meta.json exists (i.e. training finished).
# Idempotent: skips a (tag,task) whose result json already exists.
set -e
source /etc/network_turbo 2>/dev/null || true
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
PY=/root/autodl-tmp/miniconda3/bin/python
OUT_ROOT=/autodl-fs/data/bt2_phase2/models
RES=/root/autodl-tmp/dnagpt/bio-trans2/results
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
mkdir -p "$RES" "$LOG"
export OUT_ROOT
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

ALL=(M0-base M1-bio5 M2-bio20 M3-bio50 C1-protein C2-dna C3-protdna)
TASKS="mmlu gsm8k truthfulqa_mc2"

# only ready (trained) models
READY=()
for t in "${ALL[@]}"; do
  [ -f "$OUT_ROOT/$t/cpt_meta.json" ] && READY+=("$t")
done
echo "Ready to eval: ${READY[*]:-<none>}"
[ ${#READY[@]} -eq 0 ] && { echo "No trained models yet."; exit 0; }

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
# each card runs BOTH general and bio eval for its assigned model, serially
for i in "${!READY[@]}"; do
  g=$(( i % NGPU )); t="${READY[$i]}"
  echo "  GPU $g -> eval $t (general + bio)"
  CUDA_VISIBLE_DEVICES=$g nohup bash -c "
    $PY run_eval.py     --tag '$t' --tasks $TASKS --out-dir '$RES'
    $PY run_eval_bio.py --tag '$t' --out-dir '$RES'
  " > "$LOG/eval_$t.log" 2>&1 &
done
wait
echo "ALL EVAL DONE -> $RES"
