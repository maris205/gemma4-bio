#!/bin/bash
# review2.md 必做1: router analysis across all 10 checkpoints, one GPU, serial.
# Pure inference (~120 prompts/model), should take a few minutes each.
set -uo pipefail
cd /root/autodl-tmp/dnagpt/bio-trans2/scripts
PY=/root/autodl-tmp/miniconda3/bin/python
LOG=/root/autodl-tmp/dnagpt/bio-trans2/logs
mkdir -p "$LOG" /root/autodl-tmp/dnagpt/bio-trans2/results_router

TAGS=(it it-bio bio M0-base M1-bio5 M2-bio20 M3-bio50 C1-protein C2-dna C3-protdna)
for t in "${TAGS[@]}"; do
  if [ -f "/root/autodl-tmp/dnagpt/bio-trans2/results_router/routing_$t.json" ]; then
    echo "[skip] $t already done"; continue
  fi
  echo "=================== router: $t ==================="
  $PY run_router_analysis.py --tag "$t" 2>&1 | tee "$LOG/router_$t.log"
done
echo "ROUTER SWEEP DONE"
