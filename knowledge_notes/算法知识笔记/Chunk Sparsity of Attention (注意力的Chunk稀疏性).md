## Chunk Sparsity of Attention (注意力的Chunk稀疏性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Chunk Sparsity of Attention 是 InfiniteHiP 发现并利用的 LLM 注意力分布规律：在长上下文中，top-k 高注意力 token 高度集中在极少数 chunk 中，而非均匀分布在整个序列上。具体观察（Llama 3.1 8B, 128K context）：(1) 不到 2% 的 chunk 包含了超过 12.5% 的 top-2K token；(2) 约 75% 的 64-token chunk 不包含任何 top-2K token。这一观察构成了 InfiniteHiP 模块化剪枝算法的设计基础——通过选择包含 top-k token 的少数 chunk 而非逐个选择 top-k token，可在极低成本下获得良好的 top-k 近似。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Chunk Sparsity 驱动的 token 选择策略**：

```
// 传统 top-k 选择：O(T_kv) 逐个比较
topk_indices = argtop_k(attn_scores[:, :])  // 对每个 query 评估所有 key

// Chunk Sparsity 驱动的选择：O(n_chunks + k log n_chunks)
n_chunks = T_kv / l_c  // 如 128K/256 = 500 chunks
For each chunk j:
  r_j = SelectRep(q, chunk_j)  // 每 chunk 仅 1 次 O(log l_c) 操作
  s_j = estimate_score(q, k[r_j])  // 估计该 chunk 的最高注意力分数
top_chunks = argtop_K(s)  // 仅对 n_chunks 个分数排序
selected_keys = ∪_{j∈top_chunks} chunk_j  // 展开 chunk 得到约 K×l_c 个 key
```

与 InfLLM 的预选固定代表 token 不同，InfiniteHiP 每个 query block 都动态重选代表 token，使 select 精度更高（recall 比 InfLLM 高 1.57%、比 HiP 高 4.72%）。

术语一般如何实现？如何使用？

Chunk Sparsity 的分析方法：(1) 在给定上下文中运行 dense attention 获取完整 attention matrix；(2) 对每个 query position 取 top-k key indices；(3) 将 key 序列划分为固定大小的 chunk；(4) 统计每个 chunk 包含的 top-k key 数量；(5) 绘制直方图（chunk 频率 vs 包含的 top-k key 百分比）。这一方法可推广到其他模型以评估剪枝方法的适用性。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU
