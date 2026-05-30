## Block Masking (Sparsity-aware Block Scheduling for Sparse Attention Kernels)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Block Masking 是 AdaSplash 提出的 sparsity-aware kernel 调度技术，动态跳过不产生非零注意力权重的 Q-K block pair。在 α-entmax τ 求解的最后迭代中，检测每对 Q_i-K_j block 是否产生非零 P，构造 binary mask M ∈ {0,1}^{T_r×T_c}；基于 M 构造 pointer-increment lookup tables K_j = {i | M_{ij}=1} 和 Q_i = {j | M_{ij}=1}，使后续前向/反向 kernel 仅迭代有效 block 对。

从kernel调度角度拆解术语。

```
// Block Mask 生成（在 Halley-Bisection 最后迭代）
for i in 1..T_r:
    for j in 1..T_c:
        S_i^{(j)} = Q_i @ K_j^T                     // [B_r, B_c], on SRAM
        M[i][j] = any(S_i^{(j)} > τ_i)              // 1 bit per block

// 构造 lookup tables
K_j = {i | M[i][j] == 1}   // K_j block 需迭代的有效 Q_i 行
Q_i = {j | M[i][j] == 1}   // Q_i block 需迭代的有效 K_j 列

// 前向：仅迭代 j ∈ Q_i
for i in 1..T_r:
    for j in Q_i:  // skip null blocks!
        Load K_j, V_j; P = [(α-1)S-τ]_+^{1/(α-1)}; O += P@V

// 反向 dK/dV：仅迭代 i ∈ K_j
for j in 1..T_c:
    for i in K_j:  // skip null blocks!
        // gradient computation...
```

术语一般如何实现？如何使用？

Triton 实现：用 `torch.argwhere(M)` 在 GPU 上提取 (i,j) 索引对，构造 per-row/col 索引列表。Triton pointer-increment 语义使 kernel 循环自动跳过无效 block。内存开销：M 仅 T_r×T_c bits（如 n=8192, B_r=B_c=64 → M 仅 2KB），可跨 attention 层共享。当 α=1.5 产生 ~95% 稀疏时，block masking 可跳过大量 HBM 读写和 GEMM，使 ADASPLASH 超越 FlashAttention-2。

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention
