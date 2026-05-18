## Chunk Parallelism in Recurrent Attention（递归注意力中的块并行）

术语是什么？通过联网搜索让回答具体和精准。

Chunk Parallelism 是一种用于 recurrent/linear attention 的并行优化技术：将长序列沿 sequence 维度切为多个 chunk，chunk 内使用并行矩阵运算（intra-chunk，可并行化），chunk 间以 recurrent 方式传递压缩 state（inter-chunk，需顺序执行）。该技术使 recurrent attention（如 Mamba2 SSM、RetNet Recurrent、Gated Retention）能在训练和长序列 prefill 时利用 GPU 并行性，避免完全的 token-by-token 串行执行。MetaAttention 在 Recurrent Pattern kernel template 中集成了 chunk parallelism，由 scheduler 根据序列长度和硬件资源自动确定 chunk size。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Chunk Parallelism 在 Recurrent Attention 中的执行流程：

```
// 假设序列长度 seq_len，chunk size = C，共 num_chunks = seq_len / C 个 chunk
// 全局 hidden state H 的 chunk-parallel 更新

H_initial = zeros[dimqk, dimv]         // 初始 state

for chunk_id in range(num_chunks):     // inter-chunk: 顺序传递 state
    chunk_start = chunk_id * C
    chunk_end = chunk_start + C

    // 以下 intra-chunk: GPU 上并行执行（matrix operations）
    Q_chunk = Q[chunk_start:chunk_end] // [C, dimqk]
    K_chunk = K[chunk_start:chunk_end] // [C, dimqk]
    V_chunk = V[chunk_start:chunk_end] // [C, dimv]

    // 并行 relevance scoring: 用当前 state 输出所有 chunk token
    output_chunk = matmul(Q_chunk, H_initial)  // [C, dimv]

    // 并行 state 累积: K 和 V 的 chunk 内累积
    H_update = matmul(K_chunk^T, V_chunk)      // [dimqk, dimv] via outer product sum
    H_initial = H_initial + H_update           // 更新 state 传给下一个 chunk

    // customizable Mod/RowNorm 在 chunk 内 elementwise+reduction 融合
    output[chunk_start:chunk_end] = output_chunk
```

与 Parallel Pattern (FlashAttention-style) 的对比：
- Parallel Pattern: O(seq_len²) memory（需完整 score matrix），通过 online tiling 避免物化
- Recurrent Pattern + Chunk Parallelism: O(chunk_size² + dimqk×dimv) memory，通过 chunk 内并行 + chunk 间 state 传递实现 O(seq_len) 计算复杂度

Chunk size 的 trade-off：大 chunk → 更高 intra-chunk 并行度 + 更高算术强度，但 O(C²) memory + O(C²) intra-chunk 计算；小 chunk → 更多 inter-chunk 串行步骤，但 memory 占用低。MetaAttention scheduler 通过 IntermediateTensor-based scheduling 权衡确定 chunk size（考虑 shared memory 容量、register 数量等硬件约束）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Chunk Parallelism 在多个框架中有实现：(1) Mamba2 chunk kernel（Mamba2 论文的 Triton 实现）——用 chunk-based scan 替代 sequential scan；(2) Flash-Linear-Attention (FLA v0.2.0) ——提供 chunkwise parallel operators for DeltaNet、GatedDeltaNet、RetNet Recurrent 等；(3) MetaAttention Recurrent Pattern runtime——自动为 recurrent attention 应用 chunk parallelism。在 MetaAttention 中，用户选择 Recurrent Pattern 后无需手动配置 chunk 参数，scheduler 自动确定 chunk size，runtime 生成 chunk-parallel kernel（将 elementwise/reduction 逻辑融合到 recurrent kernel 中，避免额外 kernel launch）。chunk 大小通常为 64-256，受 shared memory 容量约束。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

---

