## Outlier-Resilient Quantizer (离群值鲁棒量化器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Outlier-Resilient Quantizer 是 LOGART 的离群值自适应裁剪量化器。传统对数 PTQ 用 max(|W|) 定量化范围，单个 outlier 撑大范围降低整体精度。LogART 引入可搜索缩放因子 s_of ∈ (0,1] 缩放 scale S：Q_W = clamp(⌊-log_B(|W|/(s_of·S))⌋ + σ(R), l_a, U)。s_of < 1 时范围缩小，outlier 被裁剪到边界码字；s_of = 1 时退化为无裁剪。

从算法pipeline角度拆解术语：
```
# SFS (Scaling Factor Search) - block-wise, calibration-based
for each block:
    best_s_of = argmin_s_of ||(W_block - Ŵ_block(s_of))·X||_F²
# s_of 为 per-channel 参数，与 DBS 联合搜索
argmin_{s_of, n1, n2} E[||ΔW·X||_F²]
```

术语一般如何实现？如何使用？
SFS 是 LOGART 中最有影响力的组件之一：LLaMA2-7B 上加 SFS 后 PPL 从 9.74→6.24（LLR baseline）。s_of 为 per-channel 参数，block-wise 搜索提供更好的离群值感知。仅需 32 段校准数据，额外耗时数分钟（LLM）至数十秒（CNN）。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
