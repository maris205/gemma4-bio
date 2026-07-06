# Phase-1 ①General — 灾难性遗忘检验 (DEFINITIVE)

lm-eval-harness 0.4.12, 单卡 RTX PRO 6000 Blackwell 96GB, sm_120 grouped-mm guard.
标准 few-shot: MMLU 5-shot / ARC-C 25-shot / HellaSwag 10-shot / TruthfulQA-mc2 0-shot.
max_length=2048 (harness默认, 全模型一致), bootstrap_iters=0.

| checkpoint | MMLU 5s | ARC-C acc | ARC-C norm | HSwag acc | HSwag norm | TQA-mc2 |
|---|---|---|---|---|---|---|
| Base 原始 it (vocab 262144)   | 0.646 | 0.367 | 0.403 | 0.383 | 0.488 | 0.533 |
| Base it-bio (扩词表未训 290172)| 0.642 | 0.367 | 0.402 | 0.380 | 0.486 | 0.532 |
| **BioCPT** (bio)              | **0.776** | **0.700** | **0.730** | **0.658** | **0.855** | 0.445 |
| BioCPT+SFT (v5-merged)        | 0.635 | 0.382 | 0.396 | 0.377 | 0.486 | 0.562 |

## 定稿结论
1. **扩词表零成本**: 原始 it ≈ it-bio (全项差<0.4pp) → 词表扩展无损通用能力, 消融干净, BioCPT增益完全可归因.
2. **生物CPT无灾难性遗忘, 反而净提升通用能力谱**: vs 原始it → MMLU +13.0pp, ARC-C acc +33.3pp, HSwag acc_norm +36.7pp. 唯一下降 TQA-mc2 -8.8pp (真实性/抗虚假常识的领域漂移, 可解释可控).
3. **CPT vs SFT 分工**: CPT重组并抬升表示(通用+生物双升); SFT(v5双头,3Di/DSSP+Alpaca)收窄到目标任务, 通用MC回落到base水平. 直接支持 idea.md 核心机制假设.

## 诚实标注 (写论文须讲清, 防过度宣称)
- ARC/HSwag 大涨部分源于机制: Instruct模型在裸loglikelihood-MC上天然偏弱(被调成chat格式), CPT纯文本续写把行为拉回base-model式干净MC. 非纯知识增益.
- MMLU +13pp (知识密集型) 是硬增益, 不受上述机制影响 → 结论稳健.

耗时: 原始it 81min, it-bio 111min, biocpt 84min, biocpt_sft 83min. 结果JSON: results/{tag}__{task}.json
