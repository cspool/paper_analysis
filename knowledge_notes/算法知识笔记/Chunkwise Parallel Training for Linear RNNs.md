## Chunkwise Parallel Training for Linear RNNs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chunkwise Parallel Training 是将线性 RNN 序列计算分块并行的训练算法（Hua et al., 2022; Sun et al., 2023; Yang et al., 2024a,b）。将长度 L 的序列分为大小为 C 的 chunk，chunk 内使用 dense matmul（利用 Tensor Core），chunk 间通过 recurrent state 传递。实现 O(L) 时间复杂度和 O(C²L) 空间复杂度。Mamba2 的 SSD 分解等价于这种算法。Gated DeltaNet 扩展了 DeltaNet 的 chunkwise 算法：在 WY 表示中加入 chunk-local decay mask Γ（(Γ)_{ij} = γ^i / γ^j），通过修改 T = (I + strictLower(diag(β)(Γ ⊙ K K^T)))^{-1} diag(β) 实现。xLSTM 7B 使用该算法的 Tiled Flash Linear Attention (TFLA) 版本（基于 mlstm_kernels, Anonymous 2025）：引入第二层 tile 级并行（在 chunk 内的矩阵计算也进行 tiling），使 chunk size 可任意大而不再受限于 GPU SRAM，在 H100 上比 Flash Attention 和 Mamba kernel 更快。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 通用 Chunkwise 伪代码
For each chunk t (size C, head dim d):
  // chunk 内并行计算（dense matmul）
  Q, K, V = projections(X_chunk)
  // chunk 间通过 recurrent state S 传递
  S_chunk = f(S_prev, K, V)  // recurrent update
  O_chunk = g(Q, S_chunk, K, V)  // output computation
  
// xLSTM mLSTM 的 chunkwise 特有公式
// Eq. 2-9 的 chunk 形式：
// 1. chunk 内 gate 矩阵：用 cumsum 计算 chunk 内的 f_t 累积
// 2. 分子 C_t 更新：分 block-diagonal 和低秩两部分分别并行
// 3. 分母 n_t：向量递归，可用 parallel scan
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：Flash Linear Attention (FLA) https://github.com/fla-org/flash-linear-attention；xLSTM 专用 kernel 库 mlstm_kernels https://github.com/NX-AI/mlstm_kernels。PyTorch + Triton kernel 实现。Chunk size 通常 64-256。Gated DeltaNet 论文中该算法仅比 Mamba2 慢 2-3K tokens/sec（H100）。xLSTM 7B 训练基于 TFLA kernel，在 chunk 内对 outer product C_t 和 dot product 做 tiled 矩阵乘法以提升 arithmetic intensity 并降低 IO。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---
