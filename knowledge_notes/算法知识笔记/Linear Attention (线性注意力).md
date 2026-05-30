## Linear Attention (线性注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Linear Attention 是一种将标准 softmax attention 的 O(N²d) 计算复杂度降至 O(Nd²) 的注意力机制变体。其核心思想是：(1) 移除 softmax 操作（或用 kernel feature map 替代），(2) 利用矩阵乘法结合律，将 (Q K^T) V 的计算顺序改为 Q (K^T V)（即 right-product kernel trick）。

标准 softmax attention 计算 O = Softmax(Q K^T) V，必须先物化 N×N 的 attention score 矩阵（O(N²d) 空间和时间）。Linear Attention 改为 O = Q (K^T V)，先计算 K^T V（d×d 矩阵），再乘以 Q。由于 K^T V 的大小 d×d 与序列长度 N 无关，复杂度由 O(N²d) 降至 O(Nd²)。

Linear Attention 的递推形式揭示了其本质：
```
M_s = M_{s-1} + k_s^T v_s     // 递推更新 memory state
o_s = q_s M_s                  // 查询 memory state
```
其中 M_s ∈ R^{d×d} 是累积的 memory state。该形式等同于带有矩阵值隐藏状态的线性 RNN，因此线性注意力支持常量内存推理（无需 KV cache）和线性时间训练。

实际应用中，线性注意力有多种变体：Basic Linear Attention（恒等 kernel）、Lightning Attention（IO 优化）、Retention（chunk-wise recurrent）、GLA（Gated Linear Attention，带门控）、Based（混合 linear + sliding window attention）、Rebased（可学习 kernel 函数）。注意：线性注意力在 recall-intensive 任务（如 in-context learning、Needle-in-a-Haystack）上通常弱于标准 softmax attention，因此 hybrid 架构（混合 linear + standard）是常见折中。

从算法pipeline角度拆解术语。

**Linear Attention 的两种计算模式**：

Parallel form（训练，无 causal mask）:
```
Q, K, V = X @ W_Q, X @ W_K, X @ W_V    # 全部 [N, d]
M = K^T @ V                              # [d, d] — right-product first
O = Q @ M                                # [N, d]
```

Recurrent form（推理，逐 token）:
```
M_0 = zeros(d, d)
for s in 1..N:
    q_s, k_s, v_s = x_s @ W_Q, x_s @ W_K, x_s @ W_V
    M_s = M_{s-1} + k_s^T @ v_s         # O(d²) per token, constant memory
    o_s = q_s @ M_s                     # O(d²)
```

术语一般如何实现？如何使用？

Linear Attention 通过 Triton kernel 实现（如 LASP-2 使用 Triton 2.3.1），也可通过 Lightning Attention-2 的 left-product GPU kernel 优化。在分布式训练中，LASP-2 利用 memory state M_t ∈ R^{d×d} 与序列长度无关的特点，通过 AllGather M_t 实现高效序列并行。开源实现见 https://github.com/OpenSparseLLMs/Linear-MoE。

**Linear Attention 的根本局限——全局上下文坍缩（Global Context Collapse）**：Zhang et al. (MHLA, ICLR 2026) 系统揭示了 Linear Attention 的一个内在瓶颈：所有 token 被压缩进一个共享的全局 KV summary G = Σ_j φ(K_j)^T V_j ∈ R^{d×d}，导致：(1) **Rank 受限**：attention 矩阵 A_lin = Q̃ K̃^T 的 rank ≤ min(rank(Q̃), rank(K̃)) ≤ d，无论序列长度 N 多大，表达能力被严格限制在 head dimension d_h（通常 ≤ 72）；(2) **稀疏性丧失**：随 N 增长，每个 token 对全局 summary 的贡献趋于微不足道，注意力分布趋向均匀（高熵），模型无法选择性聚焦于信息量高的 token。这两点共同构成"全局上下文坍缩"，是 Linear Attention 在长序列任务上性能严重下降的根源。缓解方案包括：Focused Linear Attention（加 DW-Conv）、GLA（门控）、Mamba2（SSM）、以及本文的 MHLA（token 维度多头分组 + 可学习混合）。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---
