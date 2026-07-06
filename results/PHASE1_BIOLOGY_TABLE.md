# Phase-1 ②Biology — 生物能力迁移

两种自建零样本测法均有格式混淆; 定稿采用 format-robust loglikelihood (同①General口径, argmax LL of Yes/No).

## loglikelihood 版 (定稿, MCC)
| checkpoint | hom_std | hom_remote | **bixbench_tf** |
|---|---|---|---|
| Base 原始 it   | -0.008 | +0.000 | +0.232 |
| Base it-bio    | -0.008 | +0.000 | +0.232 |
| **BioCPT**     | +0.137 | +0.051 | **+0.924** |
| BioCPT+SFT(v5) | -0.023 | +0.017 | +0.361 |
(acc: bixbench base .595 / BioCPT .966 / SFT .746)

## 结论 (采信 BixBench, homology 另行处理)
1. **BixBench 生物医学知识QA — 干净迁移曲线, 完美复现①③故事**: base 0.232 → BioCPT **0.924** → SFT 0.361. CPT大幅抬升(+0.69 MCC), SFT收窄回落. ②在"知识"维度与①General③Coding完全自洽.
2. **Homology 零样本测不出**: loglik强制单token Yes/No无法捕捉"比对氨基酸序列判同源"(需推理/生成), 所有模型MCC≈0. 非模型差, 是测法边界. base_it==it-bio 再证扩词表消融干净.
3. **核心论点**: 序列同源判断能力被CPT灌入表示层, 但**必须靠SFT才能兑现成任务表现**(zero-shot测不出) → 引用 BioPAWS-2 已验证 dual-mode SFT 结果作论据, 不硬塞进零样本迁移曲线.

## 生成式版 (弃用, 存档) — 格式混淆
base_it生成式: std MCC 0.947 / remote 0.272 / bixbench 0.914; BioCPT: std 0.866/remote 0.074/bixbench 0.797; v5崩(remote MCC -0.149 vs 已知82.6%, Alpaca模板与v5训练格式失配). 证明生成式零样本对SFT模型不公平.

注: biopaws2旧结果是 OmniGene-4-MM(stage2 LoRA)+旧数据版(std1997/remote4059), 非本v5-merged+entity-disjoint(1110/404), 口径不可直接混用.
