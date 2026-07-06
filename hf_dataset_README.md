---
license: apache-2.0
task_categories:
  - question-answering
tags:
  - foundation-models
  - continued-pretraining
  - catastrophic-forgetting
  - data-centric-ai
  - mixture-of-experts
  - bioinformatics
pretty_name: OmniGene-4-bio — Phase-1 re-analysis artifacts
---

# OmniGene-4-bio: capability re-analysis artifacts

Per-example evaluation outputs and summary tables for the paper:

**Biological Continued Pretraining Reshapes the Capability Profile of a Foundation Model
Without Catastrophic Forgetting** (Liang Wang, HUST).

A training-free re-analysis of one Gemma-4-26B-A4B (MoE) lineage across four capability axes:
instruction-tuned base → biological CPT (BioCPT) → supervised fine-tuning (SFT).

Code: https://github.com/maris205/gemma4-bio
BioCPT checkpoint: https://huggingface.co/dnagpt/OmniGene-4-CPT-v2-merged

## Contents

- `results/*.json` — per-(checkpoint, task) outputs for every table in the paper:
  - `<tag>__<task>.json` — General (MMLU/ARC/HellaSwag/TruthfulQA) and Coding (HumanEval/MBPP)
    via lm-evaluation-harness (loglikelihood MC / generative pass@1).
  - `bioLL_<tag>__<task>.json` — BixBench and protein-homology via format-robust loglikelihood.
  - `cot_<tag>.json` — GSM8K chain-of-thought behavior diagnostics, with per-generation records
    (chain length, backtracks, hedges, final answer, correctness).
  - Tags: `base_it_orig`, `base_it-bio`, `biocpt`, `biocpt_sft`.
- `PHASE1_*.md` — human-readable summary tables per axis.
- `PHASE1_SUMMARY.md` — the unified findings.
- `paper/` — LaTeX source and compiled PDF.

## Headline

| axis | Base (it) | **BioCPT** | BioCPT+SFT |
|---|---|---|---|
| MMLU (5s) | 0.646 | **0.776** | 0.635 |
| MBPP pass@1 (3s) | 0.332 | **0.630** | 0.348 |
| BixBench-TF MCC | 0.232 | **0.924** | 0.361 |
| CoT chain length | 108.8 | **64.4** | 125.8 |

All source benchmarks are public; no proprietary data are redistributed here — only model
outputs and scores.
