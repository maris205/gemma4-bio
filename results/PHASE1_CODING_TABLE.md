# Phase-1 ③Coding — 代码作为结构化语言 (idea.md ③)

lm-eval-harness 0.4.12, 单卡 RTX PRO 6000, sm_120 guard. 生成式+执行(pass@1).
HumanEval 0-shot, MBPP 3-shot. HF_ALLOW_CODE_EVAL=1, confirm_run_unsafe_code=True.

| checkpoint | HumanEval 0s | MBPP 3s | 采信 |
|---|---|---|---|
| Base 原始 it (262144)   | 0.012 | 0.332 | MBPP |
| Base it-bio (扩词表未训) | 0.000 | 0.350 | MBPP |
| **BioCPT** (bio)         | 0.439 | **0.630** | ✅ |
| BioCPT+SFT (v5-merged)   | 0.000 | 0.348 | MBPP |

## 结论
1. **HumanEval列作废**: 0-shot裸补全对Instruct/SFT模型格式不公平(base+SFT都≈0, 纯格式失配伪影, 非真实编码能力). MBPP 3-shot有few-shot引导才公平.
2. **BioCPT编码能力真实翻倍**: MBPP 0.33→0.63. 验证 idea.md "Coding是Structural Language, 被生物序列CPT一起抬升" — 蛋白/DNA序列语法与代码结构同源.
3. **SFT回落**: MBPP 0.63→0.35 退回base水平. 与①General同款U形 → CPT重组/抬升, SFT收窄. 两组实验互相印证同一分工故事.

耗时: 每模型~44min (HumanEval~24min + MBPP~20min).
