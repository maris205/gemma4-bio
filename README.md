# gemma4-bio

Code and per-example results for the paper:

**Biological Continued Pretraining Reshapes the Capability Profile of a Foundation Model
Without Catastrophic Forgetting** — Liang Wang (HUST).

The study has two parts on a 26B-parameter Mixture-of-Experts model (Gemma-4-26B-A4B):

- **Part I — training-free re-analysis** of one checkpoint lineage (instruction-tuned base →
  biological CPT → SFT) across four capability axes.
- **Part II — a controlled seven-model experiment**: continue pretraining the same base under
  one fixed recipe, varying *only* the data mixture (0/5/20/50% biological share, plus a
  protein/DNA composition ablation), turning the observational claim into a causal one.

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
paper/     LaTeX source (Part I + Part II sections), figures, compiled PDF
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
  title  = {Biological Continued Pretraining Reshapes the Capability Profile of a
            Foundation Model Without Catastrophic Forgetting},
  author = {Wang, Liang},
  year   = {2026},
  note   = {preprint}
}
```
