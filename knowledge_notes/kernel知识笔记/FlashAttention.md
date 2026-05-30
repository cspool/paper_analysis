## FlashAttention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashAttention（Dao et al., 2022, 2023）是一种 IO-aware 的精确注意力计算 fused kernel。传统 attention 计算需要物化完整的 N×N attention matrix（O(N²) 内存），而 FlashAttention 通过分 tile（tiling）和在线 softmax（online softmax）技术，在不物化完整 attention matrix 的情况下计算精确的 attention 输出，将内存访问从 O(N²) 降低到 O(N)。FlashAttention-2 进一步优化了 work partitioning，减少非矩阵乘法的 FLOPs。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashAttention 的核心算法（简化伪代码）：

```
def flash_attention(Q, K, V):  # Q,K,V: [N, d]
    # 将 Q 分为块大小 Br 的 tile, K/V 分为块大小 Bc 的 tile
    O = zeros(N, d)
    l = zeros(N, 1)    # 归一化常数 (log-sum-exp)
    m = zeros(N, 1)    # running max

    for i in 0..Tr-1:                           # Q tiles (外循环)
        Q_i = Q[i*Br : (i+1)*Br]
        O_i = zeros(Br, d), l_i = zeros(Br, 1), m_i = -inf

        for j in 0..Tc-1:                       # K/V tiles (内循环)
            K_j = K[j*Bc : (j+1)*Bc]
            V_j = V[j*Bc : (j+1)*Bc]
            S_ij = Q_i @ K_j^T                  # [Br, Bc] on-chip
            m_ij = rowmax(S_ij)                 # running max update
            P_ij = exp(S_ij - m_ij)             # softmax numerator
            l_ij = rowsum(P_ij)                  # softmax denominator
            # 在线更新 (避免存储完整 attention matrix)
            O_i = diag(exp(m_i - m_ij)) @ O_i + P_ij @ V_j
            l_i = exp(m_i - m_ij) * l_i + l_ij
            m_i = m_ij

        O_i = diag(1/l_i) @ O_i                 # 最终归一化
        O[i*Br : (i+1)*Br] = O_i

    return O
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- FlashAttention GitHub: https://github.com/Dao-AILab/flash-attention
- 在 FOLDMOE 中，FlashAttention 用于每个 attention micro-batch 的计算，因为 micro-batch causal attention 与全序列 causal attention 产生相同的 mask pattern 和输出
- 内存节省：N×d 而非 N×N（对于 32K seqlen 节省 ~1000×）
- 速度：通常 2-4× 加速 vs 标准 attention（尤其长序列）

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
