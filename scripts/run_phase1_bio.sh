#!/bin/bash
# Phase-1 ②Biology: Mode-A zero-shot QA on 3 bio tasks x 4 checkpoints.
# Draws the general<->biology transfer curve (does bio ability co-rise with general?).
# homology_std is 90% Yes -> MCC is the honest metric; remote (balanced) is the discriminator.
# Instruct ckpts (it/it-bio/v5) use chat template; BioCPT (no template) uses Alpaca completion.
set -uo pipefail
source /etc/network_turbo 2>/dev/null
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/dnagpt/bio-trans/paper3
mkdir -p results logs
D=/root/autodl-tmp/dnagpt/biopaws2/data
TASKS="$D/protein_homology_std.jsonl $D/protein_homology_remote.jsonl $D/f8_bixbench_tf.jsonl"

run() {  # tag ckpt style
  echo "===== $(date '+%H:%M:%S') START $1 ($3) ====="
  python scripts/run_bio.py --ckpt "$2" --tag "$1" --task-files $TASKS \
    --prompt-style "$3" --max-new-tokens 8 --batch-size 16 \
    --out-dir results 2>&1 | tee "logs/bio_$1.log" | grep -a "\[bio:"
  echo "===== $(date '+%H:%M:%S') END $1 ====="
}

run base_it_orig /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it        chat
run base_it-bio  /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio    chat
run biocpt       /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-bio       alpaca
run biocpt_sft   /root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged         alpaca
echo "BIO ALL DONE"
