## Intra-modality / Inter-modality Attention Decomposition in VLM KV Cache Pruning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Intra-modality（模态内）和 Inter-modality（跨模态）注意力分解是将多模态 VLM 的注意力矩阵按 token 模态归属进行结构化拆分的技术。给定包含 L_t 个文本 token 和 L_v 个视觉 token 的序列，注意力矩阵 A ∈ R^{L×L} 分解为：

- **Intra-modality**：A^{st}（text→text）捕捉文本内语义关系；A^{sv}（visual→visual）捕捉图像内空间关系
- **Inter-modality**：A^{ct}（visual→text）表示视觉信息对文本理解的影响；A^{cv}（text→visual）表示文本查询对图像区域的聚焦

CSP 通过 Kernel Density Estimation (KDE) 和 Jensen-Shannon (JS) Divergence 定量发现：self-attention 和 cross-attention 分布在 VLM 中显著不同且不重叠，不同层间 JS divergence 大幅变化。统一使用原始 self-attention scores 做剪枝会导致文本 token（通常 self-attention score 更大）被系统性地保留更多，而关键视觉 token 被过度剪枝。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 矩阵分解
A_st = A[0:L_t, 0:L_t]          // text→text [L_t, L_t]
A_sv = A[L_t:L, L_t:L]          // visual→visual [L_v, L_v]
A_ct = A[L_t:L, 0:L_t]          // visual→text [L_v, L_t]
A_cv = A[0:L_t, L_t:L]          // text→visual [L_t, L_v]

// 沿 query 轴求和 → 各 key token 的重要性
A_s = sum(A_st, q) ⊕ sum(A_sv, q)    // intra-importance [L]
A_c = sum(A_ct, q) ⊕ sum(A_cv, q)    // inter-importance [L]

// K^s/K^c 比率根据数据集调整
// 多数数据集: cross_ratio=0.5 (平衡)
// IR/Spot-the-Diff: cross_ratio=0.9 (偏 cross-attention)
// ActionPrediction: cross_ratio=0.0 (仅 self-attention)
```

**KDE/JS 分布分析揭示的数据集特异性**：
- CLEVR-Change: cross-attention 峰值集中且主导 → 强跨模态依赖
- DocVQA: self-attention 更分散 → 强模态内依赖
- ActionPrediction: 高 JS divergence → 完全依赖 intra-modal attention

术语一般如何实现？如何使用？

该分解是 CSP 的预处理步骤，在每次剪枝时对 multi-head 平均后的 A 矩阵执行。K^s/K^c 比率是仅有的可调超参数。代码开源：https://github.com/TerryPei/CSP。

涉及论文标题：
- Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

---
