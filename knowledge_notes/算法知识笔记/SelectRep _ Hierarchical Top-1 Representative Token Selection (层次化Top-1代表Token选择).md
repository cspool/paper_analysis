## SelectRep / Hierarchical Top-1 Representative Token Selection (层次化Top-1代表Token选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

SelectRep 是 InfiniteHiP 和 HiP Attention 中用于在 key chunk 内快速估计 top-1 注意力 token 位置的层次化二分搜索算法。给定 query block q 和 key chunk C（大小 l_c），SelectRep 通过 O(log₂ l_c) 次点积操作收敛到 chunk 内与 q 注意力分数最高的 token 的近似位置，而无需评估 chunk 内所有 l_c 个 token。这是实现高效 chunk-level 剪枝的关键组件。

从算法pipeline角度拆解术语：

```
Input: Query block q ∈ R^(b_q×d), key chunk indices C ∈ N^(l_c), Keys K
Output: Representative token index r ∈ C

1: q̃ = ApplyRopeQ_l(q)
2: n_iter = ⌈log₂(l_c)⌉  // 如 l_c=256 → 8 次迭代
3: (n_first, n_last) = (1, l_c)
4: For i = 1 .. n_iter:
5:   m = ⌊(n_first + n_last) / 2⌋  // 二分中点
6:   B₁ = [n_first, m-1], B₂ = [m, n_last]  // 左右分支
7:   For j ∈ {1, 2}:
8:     r_j = B_j[0]  // 取分支首 token 作为代表
9:     k̃_j = ApplyRopeK_lj(K[r_j])  // 位置编码
10:    σ_j = max_t (q̃_t^T · k̃_j)  // 分支分数
11:  t = argmax_j σ_j  // 选择高分分支
12:  (n_first, n_last) = B_t  // 更新搜索范围
13: r = n_first  // 收敛到单个 token
```

关键性质：(1) 每次迭代仅需 2 次 token-level 点积（与 chunk size l_c 无关），因此整个 SelectRep 仅需 2·log₂(l_c) 次点积；(2) 层次化二分搜索避免了 HiP Attention 原始实现中的全局 top-k 同步；(3) 利用 attention locality（邻近 token 的注意力分数相似）保证估计质量。

术语一般如何实现？如何使用？

InfiniteHiP 将 SelectRep 实现为单个 Triton kernel 的一部分（与 chunk score estimation + top-K chunk selection 融合），利用 GPU 的 key sequence dimension 并行度（类似 FlashDecoding 的 split-KV）。SelectRep 的左右分支分别使用不同的 RoPE position offset（j=1 偏移 n_stream+1，j=2 偏移 n_stream），实现层次化相对位置编码。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU
- HiP Attention: A Training-free Sub-quadratic Cost Transformer Model Serving Framework With Hierarchically Pruned Attention
