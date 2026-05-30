## Computation Decomposition in Linear Attention (线性注意力中的计算分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Computation Decomposition 是将带 causal mask 的线性注意力计算分解为 intra-chunk（块内）和 inter-chunk（块间）两部分的技术。当存在 causal mask Ψ（下三角矩阵）时，right-product kernel trick 无法直接全局应用（token s 不能 attend token s+1..N）。将输出分解为：
- **Intra-chunk**: O_{t,intra} = [(Q_t K_t^T) ⊙ Ψ] V_t — 仅涉及 chunk 内部，使用 left-product（quadratic），可跨设备并行
- **Inter-chunk**: O_{t,inter} = Q_t M_{1:t-1} — 与之前所有 chunk 的 attention，使用 right-product（linear），PrefixSum 累积 memory states

这种分解使 intra-chunk 计算各设备完全并行（无通信依赖），inter-chunk 通信仅传输 memory state M_t（d×d，与序列长度无关），且 AllGather 可与 intra-chunk 计算 overlap。

从算法pipeline角度拆解术语。

**LASP-2 with Masking 流程**：

```
// 并行阶段
for chunk t in 1..T in parallel:
    Q_t, K_t, V_t = X_t @ W_Q, X_t @ W_K, X_t @ W_V
    M_t = K_t^T @ V_t                           // [d, d]

    // AllGather 与 intra 计算 overlap（不同 CUDA stream）
    [M_1, ..., M_T] = AllGather([M_1, ..., M_T])  ||  O_{t,intra} = [(Q_t @ K_t^T) ⊙ Ψ_t] @ V_t

    // Inter-chunk: recursive PrefixSum
    M_{1:t-1} = M_{1:t-2} + M_{t-1}             // 缓存到 HBM
    O_{t,inter} = Q_t @ M_{1:t-1}

    O_t = O_{t,intra} + O_{t,inter}
```

术语一般如何实现？如何使用？

Computation Decomposition 最早由 Yang et al. (2023) 在 GLA 中提出，Sun et al. (2024a) 在 LASP-1 中将其应用于分布式 SP。LASP-2 在此基础上将 ring P2P 改为单次 AllGather，利用 CUDA stream 实现通信-计算 overlap。Intra-chunk left-product 使用 Triton kernel。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

---
