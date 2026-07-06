#!/bin/bash
# Phase-1 ④CoT: reasoning-behavior diagnostics on GSM8K, 200 questions x k=5 samples.
# Measures HOW models reason (length / self-correction / hedging / self-consistency),
# not just accuracy. Shared 8-shot CoT scaffold across all ckpts to limit format confound.
set -uo pipefail
source /etc/network_turbo 2>/dev/null
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
PY=/root/autodl-tmp/miniconda3/bin/python
cd /root/autodl-tmp/dnagpt/bio-trans/paper3
mkdir -p results logs

run() {
  echo "===== $(date '+%H:%M:%S') START $1 ====="
  $PY scripts/run_cot.py --ckpt "$2" --tag "$1" \
    --n-questions 200 --k-samples 5 --max-new-tokens 320 \
    --out-dir results 2>&1 | tee "logs/cot_$1.log" | grep -a "\[cot:"
  echo "===== $(date '+%H:%M:%S') END $1 ====="
}
run base_it_orig /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it
run base_it-bio  /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio
run biocpt       /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-bio
run biocpt_sft   /root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged
echo "COT ALL DONE"
