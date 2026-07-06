# Phase-1 ④CoT — 推理行为诊断 (GSM8K, 200题 × k=5采样, 8-shot CoT脚手架)

不看accuracy本身, 看CPT是否改变"推理方式". 指标: reasoning_len(思维链token数) /
backtrack_per_gen(自我修正标记密度 wait/actually/however..) / hedge_per_gen(不确定标记) /
self_consistency(k次采样答案自洽率).

| checkpoint | reasoning_len | backtrack/gen | hedge/gen | self_consistency | acc(次要) |
|---|---|---|---|---|---|
| Base 原始it   | 108.8 | 0.370 | 0.001 | 0.749 | 0.667 |
| Base it-bio   | 107.5 | 0.324 | 0.000 | 0.728 | 0.652 |
| **BioCPT**    | **64.4** | **0.006** | 0.000 | 0.745 | 0.689 |
| BioCPT+SFT    | 125.8 | 0.396 | 0.018 | 0.757 | 0.703 |

## 结论 (新机制维度, 与①②③自洽)
1. **CPT把推理压缩得又短又笃定**: 思维链缩短41%(108→64tok), 回溯近乎归零(0.37→0.006), 但acc反升(0.667→0.689)、自洽率持平(0.745). 生物序列CPT(确定性、无冗余的序列续写)训出"更直接、更少自我怀疑"的推理风格.
2. **SFT把推理拉回冗长多回溯**: 125.8tok、backtrack 0.40、hedge上升 → 回到甚至超过base啰嗦程度. 又一次"SFT改变/收窄行为", 与①②③的U形一致.
3. **it≈it-bio**(len 108.8/107.5, backtrack 0.37/0.32) 再证扩词表消融干净.
4. **核心增量**: CPT不只改知识(①②③), 还改**推理风格**(简洁化). 这是accuracy指标看不到的机制发现.

每模型~52min. 结果JSON: results/cot_{tag}.json (含per_gen逐条: q/len/backtrack/hedge/final/correct).
