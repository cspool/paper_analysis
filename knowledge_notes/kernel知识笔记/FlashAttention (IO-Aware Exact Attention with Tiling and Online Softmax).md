## FlashAttention (IO-Aware Exact Attention with Tiling and Online Softmax)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FlashAttention（Dao et al., 2022; Dao, 2024）是一种 IO-aware exact attention 算法，通过 tiling 和 online softmax 将注意力计算融合为单 pass 操作，避免将中间 attention matrix（$S = QK^T$，大小为 $O(L^2)$）写回 HBM。标准 attention 需要三次 HBM 往返，FlashAttention 将 Q、K、V 分 tile 加载到 on-chip SRAM，在 SRAM 内完成 $S$ 计算、online softmax 和输出累加，仅将最终输出写回 HBM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

**FlashAttention tiling + online softmax 伪代码**：
```
# Q[seqlen, d], K[seqlen, d], V[seqlen, d] 在 HBM
# tiles: Q按B_r切, KV按B_c切

for i in 0..T_r:                              # Q tiles
    Q_i = HBM→SRAM(Q[i*B_r : (i+1)*B_r])       # [B_r, d]
    O_i = zeros([B_r, d]); l_i = zeros([B_r])
    m_i = -inf * ones([B_r])

    for j in 0..T_c:                           # KV tiles
        K_j = HBM→SRAM(K[j*B_c : (j+1)*B_c])   # [B_c, d]
        V_j = HBM→SRAM(V[j*B_c : (j+1)*B_c])

        S_ij = Q_i @ K_j^T                      # [B_r, B_c], SRAM内
        m_new = max(m_i, rowmax(S_ij))          # online softmax
        l_new = exp(m_i - m_new)*l_i + rowsum(exp(S_ij - m_new))
        P_ij = exp(S_ij - m_new)
        O_i = diag(exp(m_i-m_new)) @ O_i + P_ij @ V_j
        m_i, l_i = m_new, l_new

    O_i = diag(1/l_i) @ O_i
    SRAM→HBM(O_i)                               # 仅一次HBM写
```

**Annotations**: $B_r, B_c$ 由 SRAM 大小决定（典型128 for A100 192KB/SM）。Online softmax 用 running max $m_i$ 保持数值稳定。关键事实：FlashAttention 不保存中间 $S = QK^T$——这正是 sparsity-based KV cache eviction（如 H2O）需要 $S$ 来计算 importance metric 的原因，因此二者天然不兼容。

术语一般如何实现？如何使用？

开源：https://github.com/Dao-AILab/flash-attention。`pip install flash-attn` 安装，调用 `flash_attn_func(q, k, v)`。论文 "Rethinking KV Cache Compression" 使用 FlashAttention v2.5.6 在 LMDeploy 中评估压缩算法，发现 FlashAttention 本身已大幅减少 KV cache memory access overhead，压缩算法的相对加速比在此框架下显著缩水。FlashAttention 的不保存 attention scores 特性与基于 attention scores 的 eviction policy 存在根本性冲突，需要额外 passes 重新计算 $S$。

LightTransfer 发现 FlashAttention 的 `return_lse=True` 参数可返回 log-sum-exp 值（即 softmax 分母），该值可作为"免费"的注意力分布代理。利用 LSE 值计算 lazy ratio（流式 attention score 的 logsumexp - LSE），仅需一次 O(w_last × (w_sink + w_recent)) 的小矩阵乘法，避免 $O(n^2)$ 完整 attention 矩阵重算。这使得在 prefilling 阶段实时分析每层注意力分布模式成为可能，额外开销在长序列下可忽略（相对吞吐仅下降 0.0014-0.0058×）。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation
