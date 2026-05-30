## Expert Redundancy in MoE (MoE 专家冗余)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Redundancy 指 MoE 架构中多个 expert 学习到相似或重叠的表示，导致 expert 之间无法有效分工，浪费参数和计算预算。Chen et al. (2023) 发现 Sparse MoE 可作为 Dropout 的替代，暗示许多 expert 确实冗余；Zoph et al. (2022, ST-MoE) 报告 routing policy overfitting 导致少数 expert 被过度使用。MoLA 论文给出 quantitative definition: Expert Redundancy measures the layer-wise difference between expert modules, 通过 Frobenius Norm of pairwise expert weight differences 量化：值越小 → expert 越相似 → 冗余越高。

从算法pipeline角度拆解术语（MoLA 的 Expert Redundancy 分析流程）：
```
# 对每层 j 的每个 attention module
for layer_j in 1..m:
    for module in [Wq, Wk, Wv, Wo]:
        # 1. 合并 LoRA: W_full = B_e @ A_e
        # 2. 计算 pairwise Frobenius Norm
        norms = []
        for (p, q) in combinations(range(N_j), 2):
            diff = W_p - W_q                        # [d_p, d_q]
            norms.append(sqrt(sum(diff ** 2)))      # Frobenius Norm
        redundancy[layer_j] = mean(norms)           # 该层平均冗余
```

数值示例（LLaMA-2-7B MoLA-□ 8888, instruction tuning 后）：
- Layers 1-8（底层）: Frobenius Norm ~0.1-0.2 → 高冗余
- Layers 9-24（中层）: ~0.3-0.4 → 中等冗余
- Layers 25-32（高层）: ~0.5-0.6 → 低冗余（expert 差异化大）

所有 MoLA 配置（▽, △, ▷◁, □ 各种变体）均呈现底层→高层 Frobenius Norm 单调递增，证实底层冗余是普遍现象而非特定配置导致。

极端配置实验验证：
- MoLA (10-2-2-2): Expert 集中在底层 → AVG 83.0%（LLaMA-2）
- MoLA (2-2-2-10): Expert 集中在高层 → AVG 84.2%（LLaMA-2）
底层 10 expert 不如高层 10 expert 有效，证明底层冗余 > 高层。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 定量工具：Frobenius Norm（MoLA）、Cosine Similarity、CKA (Centered Kernel Alignment)、SVCCA 均可度量 expert 相似度。
- 实际应用：(1) Expert Pruning — 剪除冗余 expert 减参数；(2) Layer-wise Allocation — 在冗余高的层减 expert（MoLA）；(3) Expert Merging — 合并相似 expert。
- Router 层面分析补充：大部分 expert 被选中的平均融合权重 ~0.5（重要性相近），大部分 expert 被选择频率较高且均匀 → 冗余主要来自 expert 表示而非 routing collapse。

涉及论文标题：
- MoLA: MoE LoRA with Layer-wise Expert Allocation
- Sparse MoE as the New Dropout (Chen et al. 2023, ICLR)
- ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al. 2022)

---
