## Token-wise Sparse Indexer (DSA Indexer / DeepSeek Sparse Attention Indexer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token-wise Sparse Indexer 是 DeepSeek Sparse Attention (DSA) 中的核心组件，用于在 long-context 场景下为每个 query token 从全前缀中选择 top-k 个最相关的 key tokens。DSA 首次在 DeepSeek-V3.2 (DeepSeek-AI, 2025) 中被采用，后续也被 GLM-5 等模型采用。

Indexer 维持轻量级 indexing keys k_s^I ∈ R^d、indexing queries q_{t,j}^I ∈ R^d（共 H^I 个 indexing heads）和 per-head gating weights w_{t,j}^I。对于 query position t 和 key position s，relevance score 定义：

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\left(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I\right)$$

Indexer 对全前缀 L 个 token 逐一打分后，取 top-k token 索引集 T_t = TopK(I_{t,:}, k)，送入下游 Sparse MLA 执行稀疏注意力计算。

核心矛盾：下游 Sparse MLA 仅需在 k 个 token 上计算 attention（O(Lk)），但 indexer 需扫描全前缀 L 个 token 打分（per-query O(L)，per-layer O(L²)）。在超长上下文（128K-1M tokens）下，indexer 从可忽略开销变为主导瓶颈。

从算法pipeline角度拆解术语。

**DSA Indexer + Sparse MLA 完整 pipeline**：

```
输入: 第 l 层 hidden states h ∈ R^{L×d}
      轻量 indexing weights W_Q^I, W_K^I, w_gate

// Step 1: 计算 indexing queries 和 keys
q_{t,j}^I = h_t @ W_Q^I[:, j, :]          // [L, H^I, d_head]
k_s^I = h_s @ W_K^I                        // [L, d_head]（所有 query heads 共享）

// Step 2: Token-wise scoring（瓶颈）
for t = 1 to L:                            // 每个 query position
    for s = 1 to t:                        // causal: 仅前向 token
        I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
    T_t = TopK(I_{t,:}, k)                 // 选出 top-k token 索引

// Step 3: Sparse MLA（仅在 T_t 中的 token 上计算）
for t = 1 to L:
    c_selected = {c_s | s ∈ T_t}           // 从 KV latent cache 中 gather
    u_t = Attn(h_t, c_selected)            // 稀疏注意力
```

复杂度：Step 2 per-layer O(L²)，Step 3 per-layer O(Lk)。当 L=64K, k=2048 时，indexer 占主导（~5.6 ms vs Sparse MLA ~1.6 ms at A100）。

术语一般如何实现？如何使用？

Indexer 使用独立的轻量 indexing heads（通常 H^I 远小于主 attention heads），通过 TileLang 或 CUDA kernel 实现高效的 token-level matmul + ReLU + gating + TopK。DeepSeek-V3.2 的开源参考实现在 TileLang 仓库（https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32），高性能 CUDA kernel 在 DeepGEMM 和 FlashMLA 中。

在 HISA 论文中，Token-wise Sparse Indexer 被 HISA 层级索引器替换——HISA 将 flat token scan 改写为 block-level 粗过滤 + token-level 精筛两阶段，保留相同的 token-level scoring 公式但仅在候选 block 内执行。

涉及论文标题：
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

---
