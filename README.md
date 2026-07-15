# gemma4-bio

Code and per-example results for the paper:

**Scientific Data Composition as a Capability-Shaping Mechanism for Foundation Models:
Evidence from Biological Continued Pretraining** — Liang Wang (HUST).

The study has two parts on a 26B-parameter Mixture-of-Experts model (Gemma-4-26B-A4B):

- **Part I — training-free re-analysis** of one checkpoint lineage (instruction-tuned base →
  biological CPT → SFT) across four capability axes.
- **Part II — a controlled seven-model experiment**: continue pretraining the same base under
  one fixed recipe, varying *only* the data mixture (0/5/20/50% biological share, plus a
  protein/DNA composition ablation), turning the observational claim into a causal one.
- **Part III — a small-model, matched-compute causal probe** (GPT-2, 3 seeds): asks the reverse
  question — does biological CPT feed anything back into natural-language understanding — in a
  regime with capability headroom the 26B model lacks.
- **Part IV — robustness and mechanism**: does the Part II pattern hold across a 10× model-scale
  range (2.3B–26B)? does explicit replay recover anything uniform mixing might have cost? does
  mixture ratio operate through the same router-reorganization mechanism as full CPT?

## Key finding (Part I)

Biological CPT does **not** cause catastrophic forgetting; it **lifts** the model on axes
unrelated to biology, and SFT then **narrows** it back — a consistent CPT-lifts / SFT-narrows
division of labor.

| axis | metric | Base (it) | **BioCPT** | BioCPT+SFT |
|---|---|---|---|---|
| General | MMLU (5-shot) | 0.646 | **0.776** | 0.635 |
| General | ARC-C acc (25-shot) | 0.367 | **0.700** | 0.382 |
| General | HellaSwag norm (10-shot) | 0.488 | **0.855** | 0.486 |
| General | TruthfulQA-mc2 | 0.533 | 0.445 | 0.562 |
| Coding | MBPP pass@1 (3-shot) | 0.332 | **0.630** | 0.348 |
| Biology | BixBench-TF (MCC) | 0.232 | **0.924** | 0.361 |
| Reasoning | CoT chain length (tok) | 108.8 | **64.4** | 125.8 |
| Reasoning | backtracks / gen | 0.370 | **0.006** | 0.396 |

Vocabulary expansion alone (it → it-bio) is free (< 0.4 pt on every general metric), so the
gains are attributable to CPT, not the tokenizer.

## Key finding (Part II — controlled mixture experiment)

Varying only the CPT data mixture (100M tokens each, one fixed recipe, from `it-bio`):

| model | bio% | MMLU | GSM8K | homology-std (MCC) | homology-remote (MCC) | BixBench (MCC) |
|---|---|---|---|---|---|---|
| M0-base | 0 | **0.802** | **0.873** | 0.228 | −0.023 | 0.850 |
| M1-bio5 | 5 | 0.801 | 0.867 | 0.270 | −0.055 | 0.882 |
| M2-bio20 | 20 | 0.800 | 0.867 | 0.217 | −0.074 | **1.000** |
| M3-bio50 | 50 | 0.796 | 0.861 | **0.375** | **0.088** | **1.000** |

Raising biological share to 50% preserves the general axis (MMLU −0.6 pt, GSM8K −1.2 pt) while
the biology axis rises **monotonically** — the two do not trade off. A composition ablation at
fixed 20% share shows **DNA** most improves remote homology while **protein** most improves
BixBench: the *type* of scientific data selects which capability is amplified. **Data mixture is
a controllable, general-preserving lever on the capability profile.**

## Key finding (Part III — small-model matched-compute causal probe)

GPT-2 (124M): shared warmup (`nl_base`, 300M NL tokens), then two arms continued for an
*identical* 200M tokens / 3,051 steps / optimizer schedule — only content differs (NL-only vs.
50/50 protein+DNA). Mean over 3 seeds:

| family | task | NL-arm | Bio-arm | Δ (Bio−NL), mean±sd |
|---|---|---|---|---|
| Structural | **PAWS-en** (paraphrase detection) | 0.498 | **0.529** | **+0.031 ± 0.009** |
| Structural | Anagrams / cycle-letters / insertion / reversed (character-level) | 0.000 | 0.000 | floor at this scale (uninformative) |
| Knowledge | HellaSwag | 0.291 | 0.285 | −0.006 ± 0.000 |
| Knowledge | LAMBADA (long-range discourse) | 0.333 | 0.188 | **−0.146 ± 0.004** |

A real, reproducible transfer (all 3 seeds agree in sign) to the direct NL counterpart of
protein-homology detection — but a genuine cost to long-range discourse, and the character-level
battery is simply unmeasurable at 124M (both arms floor at 0, consistent with the original
GPT-3 paper's own report of this floor below tens-of-billions of parameters). Reported as a
partial, not uniform, dissociation.

## Key finding (Part IV — robustness and mechanism)

**Scale robustness.** Repeating the Part II text-only-vs-bio50 contrast at two smaller dense
Gemma-4 sizes (E2B 2.3B, E4B 4.5B) alongside the 26B MoE result:

| model | MMLU (text→bio50) | BixBench MCC (text→bio50) |
|---|---|---|
| E2B (2.3B) | 0.576 → 0.578 (+0.2 pt) | 0.479 → **0.623** (+14.4 pt) |
| E4B (4.5B) | 0.694 → 0.695 (+0.1 pt) | 0.680 → **0.781** (+10.1 pt) |
| 26B-A4B | 0.802 → 0.796 (−0.6 pt) | 0.850 → **1.000** (+15.0 pt) |

General-preservation and the BixBench lift both hold across a **10× parameter range**. Homology
(harder, lower-sample-count) is noisier below 26B — likely a capability-floor effect on weaker
dense models, reported honestly rather than smoothed over.

**Replay control.** At fixed 50% biological share, 100M tokens (26B), explicit end-of-training
text replay (10%/20% of tokens) vs. the uniformly-shuffled baseline:

| model | replay | MMLU | homology-std MCC | BixBench MCC |
|---|---|---|---|---|
| M3-bio50 | 0% (uniform) | 0.796 | **0.375** | **1.000** |
| R1 | 10% (tail) | 0.800 | 0.318 | **1.000** |
| R2 | 20% (tail) | 0.795 | 0.266 | 0.873 |

The general axis is already flat without replay — **there was nothing to recover**. Replay does
not help and mildly *hurts* the biology axis (fewer effective bio tokens as replay fraction
rises). A clean negative result.

**Router mechanism.** Forward-hook analysis of all 30 MoE routers across every checkpoint: the
full-CPT lineage (it-bio→BioCPT, 8.7B tokens) shows a large increase in biology-vs-language
routing divergence (0.219→0.451), but the Phase-2 LoRA mixture sweep (M0→M3, 100M tokens) shows
**no routing dose-response** (0.376–0.385, flat) despite a clear *capability* dose-response —
capability shaping and routing reorganization are **dissociable mechanisms** that converge only
at large (full-parameter) training budgets.

## Checkpoints & data

- **BioCPT** (merged): [`dnagpt/OmniGene-4-CPT-v2-merged`](https://huggingface.co/dnagpt/OmniGene-4-CPT-v2-merged)
- **Data / per-example outputs**: [`dnagpt/OmniGene-4-bio`](https://huggingface.co/datasets/dnagpt/OmniGene-4-bio)
- Base is the public Gemma-4-26B-A4B instruction-tuned model.

## Repository layout

```
scripts/   evaluation harness (all reproduce on a single 96GB GPU, sm_120 grouped-mm guard)
  run_general.py         MMLU / ARC / HellaSwag / TruthfulQA (loglikelihood MC)
  run_coding.py          HumanEval / MBPP (generative + execution)
  run_bio_loglik.py      BixBench / homology (format-robust loglikelihood)
  run_cot.py             GSM8K chain-of-thought behavior diagnostics
  make_figs.py           paper figures
  run_phase1_*.sh        full sweeps over the 4 checkpoints
results/   per-(checkpoint, task) JSON + summary tables (PHASE1_*.md)
scripts_phase2/  controlled mixture-CPT experiment (Part II)
  prepare_pools.py       tokenize each source into a reusable pool
  run_cpt_mix.py         QLoRA CPT at a given data mixture (TAG, MIX, MIX_TOKENS)
  launch_parallel.sh     train the 7 models, one per GPU (auto-detects card count)
  bt2_load.py            reconstruct a CPT model (base + LoRA + trained embedding)
  run_eval.py / run_eval_bio.py   general + biology eval by model tag
  collect_results.py / make_figs.py   summary table + figures
  sm120_guard.py         dtype-safe MoE fallback for Blackwell (sm_120) training
results_phase2/  per-(model, task) JSON + phase2_summary.csv + figures
scripts_phase3/  small-model matched-compute causal probe (Part III)
  prepare_pools.py       tokenize NL/protein/DNA sources into GPT-2 pools
  train.py               warmup or matched-compute continuation (STAGE, MIX, TOKENS)
  run_tier1.sh            warmup -> 3 seeds x {NL,bio} arms, one GPU each
  run_eval.py / launch_eval.sh   NL-probe battery by model tag
  collect_results.py      per-seed delta + mean/sd, dissociation verdict
  *_local.yaml             custom lm-eval tasks for the GPT-3-paper character battery
                            (mirrors EleutherAI/unscramble, whose HF loading script is
                            no longer supported by current `datasets`)
results_phase3/  per-(model, task) JSON + tier1_summary.json
scripts_phase4/  robustness + mechanism follow-ups (Part IV)
  prepare_pools_gemma_small.py   tokenize sources w/ E2B/E4B native tokenizer
  run_scaling.py / run_scaling.sh   LoRA CPT on E2B/E4B (dense, no router)
  scaling_load.py                reconstruct an E2B/E4B CPT model for eval
  run_eval_scaling.py / run_eval_bio_scaling.py / launch_eval_scaling.sh
  run_replay.sh                  replay-control sweep (reuses run_cpt_mix.py
                                  with REPLAY_TAIL_FRAC/REPLAY_TEXT_SOURCE)
  run_eval_replay.sh              eval the replay-control models
  run_router_analysis.py / run_router_sweep.sh   30-router forward-hook JS
                                  divergence analysis across every checkpoint
  make_figs_part4.py              scaling / replay / router figures
results_phase4/  scaling + replay eval JSON, router/ = per-checkpoint routing reports
paper/     LaTeX source (Part I--IV sections), figures, compiled PDF
```

## Reproducing

All benchmarks are public (MMLU, ARC, HellaSwag, TruthfulQA, MBPP, HumanEval, GSM8K, BixBench);
no proprietary data. General/coding use `lm-evaluation-harness` 0.4.12 at community-standard
few-shot. Example:

```bash
python scripts/run_general.py --ckpt <checkpoint> --tag mymodel \
  --tasks mmlu arc_challenge hellaswag truthfulqa_mc2 --out-dir results
```

Note: on Blackwell (sm_120) the scripts install a grouped-mm guard
(`transformers.integrations.moe._can_use_grouped_mm = lambda *a, **k: False`) before any forward.

## Citation

```bibtex
@article{wang2026biocpt,
  title  = {Scientific Data Composition as a Capability-Shaping Mechanism for
            Foundation Models: Evidence from Biological Continued Pretraining},
  author = {Wang, Liang},
  year   = {2026},
  note   = {preprint}
}
```
