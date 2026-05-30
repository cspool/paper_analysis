## Sparse MLA (Sparse Multi-Head Latent Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sparse MLA 是 DeepSeek-V3.2 中采用的稀疏注意力算子，是 MLA (Multi-Head Latent Attention, DeepSeek-V2, 2024) 的稀疏变体。MLA 的核心设计是将传统 MHA 的 Key 和 Value 投影压缩为低维 latent 向量 c_s ∈ R^{d_c}（d_c ≪ d，通过低秩分解 W^{KV} = W^{UK} W^{DK} 实现），大幅减少 KV cache 内存占用。

Sparse MLA 采用 MQA (Multi-Query Attention) mode：每个 token 存储单个 latent key-value entry c_s，由所有 query heads 共享。给定 indexer 选择的 token 集 T_t（|T_t| = k），Sparse MLA 仅在这些 token 上计算 attention：

$$\mathbf{u}_t = \operatorname{Attn}(\mathbf{h}_t, \{\mathbf{c}_s \mid s \in \mathcal{T}_t\})$$

从算法pipeline角度拆解术语。

```
// Sparse MLA 计算流程（per layer, per token t）
输入: h_t ∈ R^d（当前 token hidden state）
      latent KV cache C = {c_s ∈ R^{d_c} | s = 1..L}
      selected indices T_t = {i_1, ..., i_k}

// Step 1: Query projection（标准流程）
q_t = h_t @ W_Q                              // [d] → [H, d_head]
q_t = RoPE(q_t)                              // 位置编码

// Step 2: 从 latent cache 中 gather 选中的 KV
C_selected = gather(C, T_t)                  // [k, d_c]

// Step 3: Up-projection 到完整维度（MLA 特有）
K_selected = C_selected @ W_UK               // [k, d_c] → [k, d]
V_selected = C_selected @ W_UV               // [k, d_c] → [k, d]

// Step 4: Sparse attention（仅 k 个 token）
scores = q_t @ K_selected^T / sqrt(d_head)   // [H, k]
weights = softmax(scores, dim=-1)            // [H, k]
o_t = weights @ V_selected                   // [H, d_head]
u_t = o_t @ W_O                               // [H·d_head] → [d]
```

与 Dense MLA 的区别：Dense MLA 对所有 L 个 token 计算 attention（O(L)），Sparse MLA 仅在 T_t 中的 k 个 token 上计算（O(k)）。当 k ≪ L 时（如 k=2048 vs L=64K），Sparse MLA 大幅节省 attention 计算。

术语一般如何实现？如何使用？

Sparse MLA 是 DeepSeek-V3.2 和 GLM-5 等模型的标准 attention 机制。在 vLLM 等 serving 框架中，MLA 的 latent KV cache 以 FP8 精度存储以减少内存占用。Sparse MLA 的输出接口是 token 索引集 T_t——只要 indexer 产生正确格式的 T_t，Sparse MLA 无需任何修改。这是 HISA 能够作为"即插即用" indexer 替代品的关键：HISA 产生的 T_t 与 DSA indexer 完全同构，下游 Sparse MLA 保持不变。

涉及论文标题：
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

---
