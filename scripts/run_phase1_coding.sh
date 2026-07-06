#!/bin/bash
# Phase-1 ③Coding: HumanEval(0s)+MBPP(3s) across 4 checkpoints, single GPU serial.
# Coding as a "structural language" probe (idea.md ③). Generative+execution tasks.
set -uo pipefail
source /etc/network_turbo 2>/dev/null
export HF_DATASETS_TRUST_REMOTE_CODE=1 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/dnagpt/bio-trans/paper3
mkdir -p results logs

TASKS="humaneval mbpp"
declare -A CKPTS=(
  ["base_it_orig"]="/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it"
  ["base_it-bio"]="/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio"
  ["biocpt"]="/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-bio"
  ["biocpt_sft"]="/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
)
for tag in base_it_orig base_it-bio biocpt biocpt_sft; do
  echo "===== $(date '+%H:%M:%S') START $tag ====="
  python scripts/run_coding.py \
    --ckpt "${CKPTS[$tag]}" --tag "$tag" --tasks $TASKS \
    --out-dir results 2>&1 | tee "logs/coding_${tag}.log" | grep -a "\[code:"
  echo "===== $(date '+%H:%M:%S') END $tag ====="
done
echo "CODING ALL DONE"
