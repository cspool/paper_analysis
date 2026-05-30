## Softmax-Free Attention Importance Scoring (无 Softmax 的注意力重要性评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Softmax-Free Attention Importance Scoring 是 KVzip 附录 C.3 提出的优化变体，通过移除重要性评分中的 Softmax 归一化步骤，将评分嵌入 FlashAttention fused kernel 内部。标准 KVzip 评分需要：FlashAttention 前向（含 Softmax）→ 读取注意力矩阵 → 沿 query 维度取 max。Softmax-Free 变体直接使用未归一化的 QK^T logits 作为重要性得分，消除了读取 Softmax 后注意力矩阵的冗余步骤。

从kernel调度角度拆解术语，给出具体例子。

**标准 FlashAttention block-wise 计算与评分的冲突**：

标准 FlashAttention 的 online softmax 算法在 SRAM 中逐块计算 QK^T → rescale → Softmax → ×V，中间注意力矩阵不写回 HBM。KVzip 需要在 Softmax 之后沿 query 维度取 max，这与 online softmax 的逐块计算模式冲突——每个 block 的 softmax 依赖于全局 max 做 rescaling，而 KVzip 需要的 max 本身就是跨 query 的全局统计量。

**Softmax-Free 变体**：

```
// === Standard KVzip (requires post-Softmax attention) ===
for each flash block:
    S_ij = Q_i @ K_j^T                // on-chip
    m_ij = rowmax(S_ij)               // online softmax rescale
    P_ij = exp(S_ij - m_ij)           // Softmax
    O_i += P_ij @ V_j
    // KVzip 需要额外: max_score = max(max_score, max(P_ij, dim=query))
    // P_ij 需要额外读回 HBM 或重新计算 → 10% overhead

// === Softmax-Free KVzip ===
for each flash block:
    S_ij = Q_i @ K_j^T                // on-chip
    // 直接使用 logits 作为得分，无需 Softmax
    score_chunk = max(S_ij[:,:m], dim=query)  // H×m
    // 继续正常 FlashAttention:
    m_ij = rowmax(S_ij)
    P_ij = exp(S_ij - m_ij)
    O_i += P_ij @ V_j
// 10% forward overhead 消除
```

**Triton Custom CUDA Kernel 实现**：在 Triton 编写的 FlashAttention kernel 中，在 QK^T 计算后、Softmax 之前插入 score max 操作，利用 on-chip SRAM 中的中间结果，避免额外的 HBM 读写。Kernel 输出同时包含 attention output 和 importance scores。

术语一般如何实现？如何使用？

通过 Triton DSL 实现自定义 FlashAttention CUDA kernel，在 forward pass 内部完成评分。消除约 10% 的评分开销（原来占 forward 时间的 10%），但压缩比下降约 10%（Figure 15），因为未归一化的 logits 不能准确反映注意力权重分布——高 logit 值的 KV pair 不一定是 Softmax 归一化后的高注意力 pair。论文建议在延迟敏感场景使用，未明确说明开源仓库中是否独立提供该 kernel。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
