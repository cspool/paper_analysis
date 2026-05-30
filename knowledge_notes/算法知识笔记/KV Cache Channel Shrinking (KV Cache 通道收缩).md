## KV Cache Channel Shrinking (KV Cache 通道收缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Channel Shrinking for KV Cache 是一种从通道维度（channel/head dimension）而非 token 维度压缩 KV Cache 的方法。核心思路：对 Key 投影矩阵 $W^K \in \mathbb{R}^{h_{in} \times h_{out}}$ 和 Value 投影矩阵 $W^V$ 做低秩分解 $W^K \approx A^K B^K$，其中 $A^K \in \mathbb{R}^{h_{in} \times h_{comp}}$，$B^K \in \mathbb{R}^{h_{comp} \times h_{out}}$，$h_{comp} \ll h_{out}$。推理时存储压缩特征 $X A^K \in \mathbb{R}^{n \times h_{comp}}$（而非完整 Key $X W^K \in \mathbb{R}^{n \times h_{out}}$），内存从 $O(n \times h_{out})$ 降至 $O(n \times h_{comp})$。

与 Token Pruning（从 token 维度丢弃 token）不同，Channel Shrinking 从通道/特征维度压缩，保留所有 token 的信息但以低维近似表示。CSKV 论文通过 SVD 分析发现 KV Cache 的奇异值呈显著长尾分布——仅保留最大的 50% 奇异值导致 MMLU 精度下降 <1%，证明通道维度存在大量冗余。

从算法pipeline角度拆解术语。

```
// Channel Shrinking 核心流程
// 低秩分解
A_K ∈ R^{hin × hcomp}  // 降维投影矩阵
B_K ∈ R^{hcomp × hout} // 升维重建矩阵
// 压缩率 = (hout - hcomp) / hout

// Prefilling: 输入序列 X ∈ R^{n×hin}
K_full = X @ W_K          // 完整 Key (n, hout)，用于 attention
K_compressed = X @ A_K    // 压缩 Key (n, hcomp)，存入 KV Cache

// Decoding: 需要完整 Key 时重建
K_reconstructed = K_compressed @ B_K  // (n, hcomp)@(hcomp, hout) → (n, hout)
```

术语一般如何实现？如何使用？

CSKV 在 HuggingFace Transformers 中实现：修改 attention 层的 key/value 投影逻辑，增加 A_K/B_K/A_V/B_V 低秩权重。训练通过 ASVD 初始化 + 逐层 MSE 重建损失微调（单 A100 90 分钟）。适用于长上下文 32K+ tokens，可与量化叠加达到 95% 总压缩率。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios
