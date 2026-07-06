#!/bin/bash
# Phase-1 ②Biology (format-robust loglikelihood): 4 ckpts x 3 bio tasks.
# Yes/No chosen by argmax LL under one neutral prompt -> pure discrimination,
# no chat-vs-alpaca / instruction-following confound. Comparable to ①General.
set -uo pipefail
source /etc/network_turbo 2>/dev/null
export HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp/dnagpt/bio-trans/paper3
mkdir -p results logs
D=/root/autodl-tmp/dnagpt/biopaws2/data
TASKS="$D/protein_homology_std.jsonl $D/protein_homology_remote.jsonl $D/f8_bixbench_tf.jsonl"

run() {
  echo "===== $(date '+%H:%M:%S') START $1 ====="
  python scripts/run_bio_loglik.py --ckpt "$2" --tag "$1" --task-files $TASKS \
    --out-dir results 2>&1 | tee "logs/bioll_$1.log" | grep -a "\[bioLL:"
  echo "===== $(date '+%H:%M:%S') END $1 ====="
}
run base_it_orig /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it
run base_it-bio  /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio
run biocpt       /autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-bio
run biocpt_sft   /root/autodl-tmp/dnagpt/outputs/OmniGene-4-v5-merged
echo "BIOLL ALL DONE"
