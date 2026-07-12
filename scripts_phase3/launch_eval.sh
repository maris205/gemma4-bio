#!/bin/bash
# Evaluate all 6 arm models (3 seeds x {nl,bio}) in parallel, one per GPU.
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/scripts
PY=/root/autodl-tmp/miniconda3/bin/python
RES=/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/results
LOG=/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/logs
mkdir -p "$RES" "$LOG"

TASKS="paws_en hellaswag lambada_openai anagrams1_local anagrams2_local cycle_letters_local random_insertion_local reversed_words_local"
ARMS=(nl_seed0 bio_seed0 nl_seed1 bio_seed1 nl_seed2 bio_seed2)

for i in "${!ARMS[@]}"; do
  t="${ARMS[$i]}"
  echo "  GPU $i -> $t"
  CUDA_VISIBLE_DEVICES=$i setsid $PY run_eval.py --tag "$t" --tasks $TASKS \
    --out-dir "$RES" > "$LOG/eval_$t.log" 2>&1 < /dev/null &
done
wait
echo "TIER1 EVAL DONE -> $RES"
