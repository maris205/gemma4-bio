#!/bin/bash
# Supplementary baseline: original Instruct (vocab 262144, no vocab-expansion perturbation).
# This is the CLEAN "Base" for CPT attribution. it-bio (expanded-but-untrained) stays as
# a vocab-expansion ablation. Run AFTER the main 3-ckpt sweep frees the GPU.
set -uo pipefail
source /etc/network_turbo 2>/dev/null
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/dnagpt/bio-trans/paper3
mkdir -p results logs

python scripts/run_general.py \
  --ckpt /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it \
  --tag base_it_orig --tasks mmlu arc_challenge hellaswag truthfulqa_mc2 \
  --out-dir results 2>&1 | tee logs/base_it_orig.log | grep -a "\[eval:"
echo "IT-ORIG DONE"
