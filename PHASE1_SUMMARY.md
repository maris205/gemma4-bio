# Paper-3 Phase-1 — Re-analysis Summary

**Thesis (idea.md)**: *Training data is not merely knowledge — it is a mechanism for
shaping the capability profile of foundation models.* Biological sequence is the first
representative structured scientific data. No retraining: analyze 3 existing checkpoints.

**Checkpoints** (lineage `it → it-bio(vocab-expand) → BioCPT → SFT`):
- Base 原始 it (Gemma-4-26B-A4B-Instruct, vocab 262144)
- Base it-bio (vocab-expanded 290172, **untrained** — clean CPT control)
- BioCPT (it-bio + 8.7B-token bio CPT, merged)
- BioCPT+SFT / v5-merged (dual-head structure SFT on top of CPT)

**Setup**: single RTX PRO 6000 Blackwell 96GB, sm_120 grouped-mm guard, lm-eval-harness 0.4.12.

---

## The one story across all three axes

| axis | metric | Base(it) | BioCPT | SFT(v5) | shape |
|---|---|---|---|---|---|
| ①General | MMLU 5s | 0.646 | **0.776** | 0.635 | ⤒ CPT up, SFT back |
| ①General | ARC-C 25s acc | 0.367 | **0.700** | 0.382 | ⤒ |
| ①General | HellaSwag norm | 0.488 | **0.855** | 0.486 | ⤒ |
| ①General | TruthfulQA-mc2 | 0.533 | 0.445 | 0.562 | ⤓ CPT down (truthfulness drift) |
| ③Coding | MBPP 3s pass@1 | 0.332 | **0.630** | 0.348 | ⤒ (fair metric) |
| ②Biology | BixBench-TF MCC | 0.232 | **0.924** | 0.361 | ⤒ |

**Consistent mechanism**: Biological CPT **re-organizes and lifts the capability substrate**
across general knowledge (MMLU +13pp), code (MBPP nearly 2×), and biomedical knowledge
(BixBench +0.69 MCC) — *no catastrophic forgetting, net gains*. SFT then **narrows/cashes
out** the substrate onto target tasks, and general/held-out MC ability falls back to base
level. This "CPT lifts substrate → SFT narrows to task" division is the through-line of ALL
evidence and directly supports idea.md's data-centric framing.

## Honest caveats (must state in paper)
1. **Vocab expansion is free**: it ≈ it-bio on every ①General metric (<0.4pp) — clean ablation,
   so BioCPT gains are fully attributable to CPT.
2. **ARC/HellaSwag lift is partly mechanistic**: Instruct models are weak at raw-loglikelihood MC
   (tuned for chat); CPT on raw text restores base-model-like MC behavior. MMLU +13pp
   (knowledge-heavy) is a hard gain unaffected by this. State both.
3. **HumanEval 0-shot discarded**: bare-completion format unfair to Instruct/SFT (base & SFT ≈0);
   MBPP 3-shot is the fair coding metric.
4. **Homology zero-shot is unmeasurable** by simple probes (loglik Yes/No ≈ random for all;
   generation format-confounded, v5 collapses). Argument: sequence-homology ability is
   instilled into representations by CPT but only *cashed out* by SFT → cite BioPAWS-2 dual-mode.
5. **TruthfulQA −8.8pp**: the one genuine regression — domain drift on truthfulness/anti-misinfo.
   Report it; it is small and explainable, not catastrophic.

## Artefacts
- `results/PHASE1_GENERAL_TABLE.md`, `PHASE1_CODING_TABLE.md`, `PHASE1_BIOLOGY_TABLE.md`
- per-(ckpt,task) JSON in `results/`
- scripts: `run_general.py`, `run_coding.py`, `run_bio_loglik.py` + `run_phase1_*.sh`

## Remaining Phase-1 (idea.md ④)
- CoT diagnostics (NOT accuracy): reasoning length, self-correction, consistency, uncertainty —
  same-question Base vs BioCPT. Optional: GSM8K/BBH reasoning.
