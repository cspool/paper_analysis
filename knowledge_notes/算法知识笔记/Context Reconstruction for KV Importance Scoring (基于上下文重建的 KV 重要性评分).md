## Context Reconstruction for KV Importance Scoring (基于上下文重建的 KV 重要性评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Context Reconstruction for KV Importance Scoring 是 KVzip 提出的 KV pair 重要性评估机制。核心思想：让 LLM 模拟重建原始上下文（teacher-forced decoding），观察每个 KV pair 在此过程中接收到的 attention 大小，以此判断其重要性——被重建"需要"的 KV pairs 为关键，应保留；几乎不被关注的为冗余，可淘汰。

该机制的关键发现：(1) 重建过程的 cross-attention 比 prefill self-attention 显著稀疏（Figure 5 直方图：大部分 KV pairs 收到极低 attention），因为模型可以高效利用 KV_c 中的高层表示+自身权重中的知识；(2) 重建所需的 KV pairs 与 QA、摘要、推理等下游任务的注意力模式高度重叠（Figure 6 2D histogram 下三角区域集中），证明重建作为 proxy task 能泛化到多种下游任务（类似 BERT/MAE 的自监督学习范式）。

从算法pipeline角度拆解术语：

**评分计算流程**：

```
输入: f_LM (LLM), context c (n_c tokens), chunk_size m=2048
输出: importance scores S ∈ R^{L×H×n_c}

1. KV_c = Prefill(c)

2. 将 c 分为 T = ceil(n_c/m) 个 chunk

3. for t = 1..T:
     if t == 1:
         input = "Repeat the previous context:" + c_1
     else:
         input = "Repeat the previous context starting with "
                 + c_{t-1}[-8:] + ":" + c_t
     
     通过 f_LM forward，使用 KV_c 作为 cache
     for l = 1..L, h = 1..H:
         Q = query_proj(hidden)           // G×n_in×d
         K_sub = subsample(KV_c, chunk_t) // (m+n_in)×d
         A = Softmax(Q @ K_sub^T)         // G×n_in×(m+n_in)
         A_sliced = A[:,:,:m]             // KV_c 部分
         S_chunk = max_{g,i} A_sliced     // H×m

4. S = concat([S_chunk_1, ..., S_chunk_T])
5. 淘汰: keep_indices = topk(S, r × n_c) across all heads
```

术语一般如何实现？如何使用？

评分使用标准 FlashAttention-2 forward pass，无需修改 attention kernel（除 softmax-free 变体外）。chunked scoring 使复杂度 O(m·n_c)，峰值内存恒定 O(m²)。repeat prompt 具体措辞影响极小（Table 2: 原始/改写/无 prompt 准确率差异 <0.2%）。评分开销约 2x 标准 prefill，但仅执行一次可被多查询摊还。代码开源：https://github.com/snu-mllab/KVzip。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

---
