## Attention Parallel Pattern / Recurrent Pattern

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention Parallel Pattern 和 Recurrent Pattern 是 MetaAttention 从统一 attention 抽象中推导出的两种计算模式，分别对应需要**全局上下文**和可以**压缩为固定大小状态**的 attention 机制。

**Parallel Pattern**（并行模式）：attention 需要在完整 KV sequence 上计算全局上下文。Relevance scoring 实现为并行矩阵乘法 `scores = matmul(Q, K)`（Q 的每个 token query 与所有 K token 做内积，O(N²) 复杂度）。Aggregation 实现为 `output = matmul(weights, V)`（每个 query 聚合所有 V token 的信息）。适用于 Softmax Attention、Sigmoid Attention、RetNet Parallel、MLA、Sliding Window Attention、Sparse Attention 等。关键优化：online block-wise normalization（online softmax/sigmoid 等）避免物化 N×N score matrix。

**Recurrent Pattern**（循环模式）：attention 将上下文压缩为固定大小的 hidden state h，迭代遍历 sequence。Relevance scoring 实现为 `output = matmul(Q, h)`（仅需与压缩 state 做 matmul，O(d²) 复杂度）。Aggregation 实现为 `h = h + matmul(K[i]^T, V[i])`（增量更新 hidden state）。适用于 Mamba2 SSM、RetNet Recurrent、Gated Retention 等 state space model 类 attention。关键优化：chunk parallelism——将 sequence 分块并行处理，块内用 recurrent 更新 state，块间传递 state。

从算法pipeline角度拆解：

Parallel Pattern 伪代码（以在线归一化为例）：
```
def parallel_attention(Q, K, V):
    O = zeros(B, H, S, d_v)
    for q_block in Q.split(B_r):            # 沿 seq_len 并行
        m, l, O_acc = -inf, 0, 0
        for kv_block in (K, V).split(B_c):  # 沿 KV seq_len 串行迭代
            S = q_block @ kv_block.K^T       # [B_r, B_c] relevance scoring
            S = scores_Mod(S)                # mask/scale
            m_new = max(m, rowmax(S))
            l = l * exp(m - m_new) + rowsum(exp(S - m_new))
            O_acc = O_acc * exp(m - m_new) + softmax(S) @ kv_block.V
            m = m_new
        O[q_block] = O_acc / l
    return O
```

Recurrent Pattern 伪代码（以 Mamba2 chunk parallelism 为例）：
```
def recurrent_attention_chunked(Q, K, V, chunk_size):
    h = zeros(B, H, d_state)          # 初始压缩 state
    for chunk in sequence.split(chunk_size):
        # Chunk 内并行计算 relevance + aggregation
        O[chunk] = matmul(Q[chunk], h)   # relevance scoring from state
        # 块内并行更新 state（每个位置增量贡献）
        h = h + sum(matmul(K[i]^T, V[i]) for i in chunk)
    return O
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention 将两种 pattern 实现为固定的 kernel 模板。Parallel pattern 模板包含 online normalization mainloop（外层沿 KV seq_len 分 tile，内层 TMA load + wgmma QK^T + SIMT customizable functions + wgmma PV + rescale）；Recurrent pattern 模板包含 chunk-based mainloop（外层沿 chunk 迭代，内层并行 matmul + state update）。用户选择 pattern 后，customizable functions 通过 code inlining 注入模板固定位置，无需修改 scheduling logic。同一 pattern 可应用于多个 attention 变体（如 Parallel pattern 支持 Softmax/Sigmoid/ReLU/MLA/RetNetParallel/Sparse GQA）。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends
