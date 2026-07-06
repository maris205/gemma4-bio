# gemma4-bio

Code and per-example results for the paper:

**Biological Continued Pretraining Reshapes the Capability Profile of a Foundation Model
Without Catastrophic Forgetting** — Liang Wang (HUST).

A **training-free re-analysis** of one lineage of a 26B-parameter Mixture-of-Experts model
(Gemma-4-26B-A4B). We compare three checkpoints — instruction-tuned base, after biological
continued pretraining (BioCPT, 8.7B tokens of DNA/protein/biomedical text), and after
subsequent supervised fine-tuning (SFT) — across four capability axes.

## Key finding

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
paper/     LaTeX source, figures, compiled PDF
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
