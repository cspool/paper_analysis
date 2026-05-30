## Tiled Matrix Multiplication for Mixed Precision (分块混合精度矩阵乘法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tiled Matrix Multiplication（分块矩阵乘法）在 KIVI 中指将 grouped quantized 部分和 residual FP16 部分的矩阵乘法分块独立计算后拼接的策略。KIVI 的 KV Cache 分为两部分：grouped 量化部分 `Q(X_K_g)` (2bit) 和 residual 全精度部分 `X_K_r` (FP16)。Attention score 计算 `A = t_Q X_K^T` 无法直接执行因为两部分精度和布局不同。Tiled matmul 将 X_K 视为两个 tile，分别用不同 kernel 计算后 Concat。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI 的 tiled mixed-precision attention：

```
// Attention Score = Concat(tiled sub-results)
A_g = t_Q @ Dequant(Q(X_K_g))^T   // Tile 1: Q_MatMul kernel (fused dequant+matmul)
A_r = t_Q @ X_K_r^T               // Tile 2: Standard FP16 matmul
A = Concat([A_g, A_r], dim=token) // Concatenate along token dimension

// Attention Output = sum of tiled sub-results
Softmax_split:
    A_g_sm = Softmax(A)[:, :-R]   // normalized weights for grouped part
    A_r_sm = Softmax(A)[:, -R:]   // normalized weights for residual part

t_O_g = A_g_sm @ Dequant(Q(X_V_g))   // Tile 1: Q_MatMul for grouped value
t_O_r = A_r_sm @ X_V_r               // Tile 2: Standard matmul for residual value
t_O = t_O_g + t_O_r                  // Sum (not concat)
```

关键设计点：
- **Token 维度拆分**：both key 和 value 沿 token 维度拆分为 grouped + residual，tiled matmul 在 token 维度分块
- **Softmax 跨 tile 归一化**：softmax 必须跨全部 token 执行，因此先在拼接后的 A 上 softmax，再按 grouped/residual 分割
- **Output 为 Sum 而非 Concat**：attention output 在 hidden dim 上未拆分，两个 tile 的结果直接相加

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KIVI 在 CUDA 层面实现了两种 kernel 的无缝调度：(1) Q_MatMul 处理 grouped 量化 tile；(2) cuBLAS GEMM 处理 residual FP16 tile。tile 拆分和拼接在 PyTorch 层面通过 tensor slicing 完成。类似的分块策略也被 FlashAttention 用于处理长序列（分块 softmax），以及 vLLM 的 PagedAttention（分页处理 KV cache block）。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---
