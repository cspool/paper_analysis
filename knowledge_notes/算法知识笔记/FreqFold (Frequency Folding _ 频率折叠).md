## FreqFold (Frequency Folding / 频率折叠)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FreqFold 是 TransMLA 论文提出的增强 RoRoPE 的技术：利用 RoPE 中相邻频率 θ_l = 10000^{-2(l-1)/d} 的数值接近性，将多个频率维度的 key 段合并后做联合 PCA，使 K_rope 能占用更多维度（而非仅第一 head 的 d 维），保留更丰富的位置信息。核心原理（Proposition 2）：将 M 个原始 RoPE 频率组的 d' 维数据段拼接为 M·d' 维后进行 PCA，保留 M 个主成分所捕获的方差 V_2 严格大于分别对各组做 PCA 各保留 1 个主成分的方差之和 V_1。即合并后做 PCA 的方差保留效率更高。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**FreqFold 计算流程（以 4D FreqFold 为例，g=2 heads, d_head=8）**：
```
// 原始 RoRoPE（无 FreqFold）：4 个频率索引，各自独立 PCA
// φ₁: dims {1,2} head1 + {9,10} head2 → 4D vectors → PCA → keep 1PC
// φ₂: dims {3,4} head1 + {11,12} head2 → 4D vectors → PCA → keep 1PC
// φ₃: dims {5,6} head1 + {13,14} head2 → 4D vectors → PCA → keep 1PC
// φ₄: dims {7,8} head1 + {15,16} head2 → 4D vectors → PCA → keep 1PC
// → 4 个独立 PCA，各保留 1 个主成分

// FreqFold (M=2)：将 φ₁≈φ₂, φ₃≈φ₄ 合并
// 合并组 Φ_A (φ₁+φ₂): dims {1,2,3,4} head1 + {9,10,11,12} head2
//   → 8D vectors → PCA → keep 2 PCs（对应 M=2 个主成分）
// 合并组 Φ_B (φ₃+φ₄): dims {5,6,7,8} head1 + {13,14,15,16} head2
//   → 8D vectors → PCA → keep 2 PCs
// → K_rope 现在可使用 4×d 维（而非 d 维）保留位置信息
```

**Trade-off 分析**：
- M 越大 → PCA 方差保留越多（Proposition 2） → 位置信息损失越小
- M 越大 → 频率近似的偏差越大 → RoPE 位置编码精度损失越大
- 存在 sweet spot：TransMLA 在 LLaMA-3-8B 上发现 4D FreqFold 最优（Figure 3b）

术语一般如何实现？如何使用？

FreqFold 在校准数据集上离线执行，作为 RoRoPE 的可选增强步骤。实现：对 key 激活按 RoPE 频率索引分组 → 将频率值相近（如 |θ_a - θ_b|/θ_a < threshold）的组拼接 → 在拼接后的高维向量上做 PCA → 保留 M 个主成分作为 K_rope 的多维位置表示。保留的主成分数 M = 合并的原始频率组数（如 4D FreqFold 中 M=4）。需权衡 M（更好的位置信息保留 vs 更大的近似误差）。TransMLA 实验报告：4D FreqFold 在 extreme RoPE removal (90%) 下显著优于无 FreqFold 的 RoRoPE，但过度 FreqFold（如更大 M）可能因近似误差累积而退化。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need

---
