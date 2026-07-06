#!/bin/bash
# Phase-1 ①General: 3 checkpoints x 4 benchmarks, standard few-shot, single GPU serial.
# Core question: does Base -> BioCPT drop general ability? (catastrophic forgetting check)
set -uo pipefail
source /etc/network_turbo 2>/dev/null
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false

cd /root/autodl-tmp/dnagpt/bio-trans/paper3
mkdir -p results logs

TASKS="mmlu arc_challenge hellaswag truthfulqa_mc2"

# tag : checkpoint path
declare -A CKPTS=(
  ["base_it-bio"]="/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio"
  ["biocpt"]="/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-bio"
  ["biocpt_sft"]="/root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged"
)

for tag in base_it-bio biocpt biocpt_sft; do
  echo "===== $(date '+%H:%M:%S') START $tag ====="
  python scripts/run_general.py \
    --ckpt "${CKPTS[$tag]}" --tag "$tag" --tasks $TASKS \
    --out-dir results 2>&1 | tee "logs/${tag}.log" | grep -a "\[eval:"
  echo "===== $(date '+%H:%M:%S') END $tag ====="
done

echo "ALL DONE"
