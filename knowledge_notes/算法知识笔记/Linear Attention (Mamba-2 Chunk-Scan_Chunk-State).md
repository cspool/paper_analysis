## Linear Attention (Mamba-2 Chunk-Scan/Chunk-State)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Linear Attention 是一类通过解耦 softmax 将标准注意力从 O(N²d) 降为 O(Nd²) 的方法族。引入 feature map φ(·)，用 φ(Q)φ(K)^T 替代 softmax(QK^T)，利用矩阵结合律重排：先算 φ(K)^T V ∈ R^{d×d}，再乘 φ(Q) 得 O ∈ R^{N×d}。TileLang 论文中的 Linear Attention 特指 Mamba-2 模型的 chunk-scan 和 chunk-state 函数——这些是 State Space Model (SSM) 中的 recurrent computation kernel，数学上等价于线性注意力形式。TileLang 将这两个函数作为算子级 benchmark，在 H100 上对比 Triton 实现：chunk-scan 平均 1.77× speedup，chunk-state 平均 2.10× speedup。

从算法 pipeline 角度拆解术语，比如术语所在 pipeline 的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mamba-2 Linear Attention 的 chunk 分解：
```
// Mamba-2 的核心递归:
h_t = A_t ⊙ h_{t-1} + B_t ⊙ x_t       // state update
y_t = C_t^T h_t                         // output

// 序列切为 chunks 实现并行化:
for each chunk_i in range(seq_len / chunk_size):
  // Step 1: Intra-chunk (chunk 内部并行)
  for j in chunk_i:
    h_j = A_j ⊙ h_{j-1} + B_j ⊙ x_j
    y_j = C_j^T h_j
  // Step 2: Inter-chunk (chunk 间顺序传递 compressed state)
  h_chunk = A_chunk ⊙ h_prev + B_chunk  // chunk-state 函数

// chunk-scan: 对 chunks 做 parallel scan (类似 prefix sum)
// chunk-state: 计算单 chunk 的初始→最终 state 映射矩阵
```

TileLang 的 benchmark 使用 Table 4 的 12 种 shape 配置（chunk-scan CC0-CC5 和 chunk-state CT0-CT5），覆盖 batch=1/64, nheads=64, seq_len=1024/2048/8192, head_dim=64, d_state=128。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Mamba-2 的 Linear Attention 用 Triton 实现（参考代码在 mamba 仓库）。TileLang 用约 50 行 Python 实现等价功能（使用 T.Pipelined + T.gemm + T.reduce），在 H100 上获得 1.77-2.10× speedup。加速原因：TileLang 自动利用 TMA + wgmma.mma_async + warp specialization，而 Triton 在 H100 上未充分利用这些 Hopper 特性。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems
