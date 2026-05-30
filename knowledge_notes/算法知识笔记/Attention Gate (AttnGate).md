## Attention Gate (AttnGate)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention Gate（AttnGate）是 SeerAttention/SeerAttention-R 提出的轻量级可学习门控模块，插入预训练 Transformer 注意力层之前，用于预测哪些 KV blocks 对当前 query token（或 query 序列）最重要，从而实现 block-level 稀疏注意力。AttnGate 受到 MoE gating 机制启发，但用于注意力稀疏性预测而非专家路由。

AttnGate 的核心计算流程（decode 阶段，SeerAttention-R）：
1. **Q 分支**：取 pre-RoPE 的 Q tensor（multi-head），通过线性层 W_q_gate 将 GQA group 内 query heads 聚合为 KV-head 数量，再应用 RoPE → Q_gate ∈ R^{1, num_kv_heads, d_gate}
2. **K 分支**：取 pre-RoPE 的 K tensor，进行 Max/Min/Avg 三种非重叠块级 pooling（pooling kernel size = block_size），concat 后通过线性层 W_k_gate，再应用 RoPE → K_gate ∈ R^{num_blocks, num_kv_heads, d_gate}
3. **块级注意力分数**：S = softmax(Q_gate @ K_gate^T / sqrt(d_gate))，输出每块的激活分数
4. **稀疏化**：通过 Top-K（token budget）或阈值过滤将软分数转换为二进制块掩码/块索引

AttnGate 的参数量极小：对 8B 模型约 66MB，仅原始模型参数的 ~0.8%。训练时仅更新 AttnGate 参数（冻结原始模型权重），使用 KL 散度损失将 AttnGate 输出对齐到原始模型注意力分布的 block-level ground truth。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# AttnGate 在 decode 阶段的推理流程（SeerAttention-R）
# 输入: 单 token decode, Q ∈ R^{1, num_heads, d_head}, K_cache ∈ R^{seq_len, num_kv_heads, d_head}

# === AttnGate 前向: 稀疏块选择 ===
def attngate_forward(Q, K_compression_cache, block_size, token_budget):
    # Step 1: Q 分支 — GQA head aggregation
    Q_nope, Q_pe = split_rope(Q)                    # 分离 RoPE 部分
    Q_reshaped = reshape(Q_nope, [num_kv_heads, g*d_head])  # 按 GQA group 重组
    Q_gate_ = W_q_gate @ Q_reshaped                 # [num_kv_heads, d_gate]
    Q_gate = RoPE(Q_gate_, Q_pe[0])                 # 重新应用 RoPE
    
    # Step 2: K 分支 — 使用 K Compression Cache
    # K_compression_cache 已存储压缩后的 K 表示
    K_gate = K_compression_cache                    # [num_blocks, num_kv_heads, d_gate]
    
    # Step 3: 块级注意力分数
    S = softmax((Q_gate @ K_gate.T) / sqrt(d_gate))  # [1, num_kv_heads, num_blocks]
    
    # Step 4: Top-K 选择
    block_budget = token_budget // block_size
    selected_blocks = topk(S, k=block_budget, dim=-1)  # [1, num_kv_heads, block_budget]
    selected_blocks = selected_blocks ∪ {last_incomplete_block}  # 始终包含最后不完整块
    
    return selected_blocks

# === 块稀疏 Attention ===
def block_sparse_attention(Q, K_cache, V_cache, selected_blocks, block_size):
    O = zeros_like(Q)
    m_prev = -inf
    for block_idx in selected_blocks:
        K_block = K_cache[block_idx*block_size : (block_idx+1)*block_size]
        V_block = V_cache[block_idx*block_size : (block_idx+1)*block_size]
        S_block = Q @ K_block.T / sqrt(d_head)
        # FlashAttention online softmax rescaling
        m_new = max(m_prev, rowmax(S_block))
        O = diag(exp(m_prev - m_new)) * O + exp(S_block - m_new) @ V_block
        m_prev = m_new
    return O
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AttnGate 通过 HuggingFace Transformers 的模型修改实现：在原始 attention 层的 forward 函数中插入 AttnGate 模块。训练时使用修改版 FlashAttention-2 kernel 同时计算 attention output 和 block-level ground truth（column-wise 1D maxpooled attention scores），用 KL divergence 训练。推理时，AttnGate 的 K 分支利用 K Compression Cache 避免重复计算历史 token 的压缩表示。预训练好的 AttnGate 权重已发布在 HuggingFace（https://huggingface.co/SeerAttention）。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning
- SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs (NeurIPS 2025)

---
