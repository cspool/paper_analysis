## Multi-head Temporal Latent Attention (MTLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-head Temporal Latent Attention (MTLA) 是在 MLA 的低秩 latent KV 压缩基础上，进一步沿 temporal（时间）维度压缩 KV cache 的注意力机制。MTLA 是首个在时序维度压缩 KV cache 的自注意力变体。核心流程：

(1) **低秩 latent 压缩**（继承 MLA）：输入 X ∈ R^{T×d} 经 W_r ∈ R^{d×r} 投影为低维 latent vector C ∈ R^{T×r}（r ≪ d），LayerNorm 稳定训练；(2) **Temporal 压缩 via hyper-network**：以 hyper-network 对每 s 个相邻 latent vector 动态生成 merge weight w_i = Sigmoid(Linear(c_i) · Linear(pe_j))，合并为 ĉ_j = Σ w_i·c_i，将 KV cache 序列长度从 T 降为 t = ⌈T/s⌉；(3) **Stride-aware causal mask**（训练时）：解决 training 时 compressed cache 长度与 sequence length 不匹配的问题；(4) **Absorbed attention**：利用矩阵乘法结合律将 W_K 吸收进 W_Q、W_V 吸收进 W_O，避免显式计算完整 K/V 矩阵；(5) **Decoupled RoPE temporal compression**：RoPE key 同样沿 temporal 维压缩，每 s 个 token 仅保留最新的 RoPE key。

MTLA 的 per-token KV cache 大小 = 9d_h·l/(2s)（s=2 时为 2.25d_h·l，接近 MQA 的 2d_h·l）。Per-token 解码复杂度从 O(T) 降至 O(T/s)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MTLA 训练前向（s=2, r=256, d=512, n_h=8）**：

```
输入: X ∈ R^{T×d}

# Step 1: Query（标准 MHA）
Q = X @ W_Q               # W_Q ∈ R^{d×(n_h·d_h)}, Q ∈ R^{T×512}

# Step 2: Low-rank latent
C = LayerNorm(X @ W_r)   # W_r ∈ R^{d×r}, C ∈ R^{T×256}

# Step 3: Hyper-network 批量生成 merge weights
PE = (pe_1,...,pe_1, ..., pe_t,...,pe_t)  # 每个 pe_j 重复 s 次, 总长 T
W = Sigmoid(Linear(PE) × Linear(C))       # W ∈ R^{T×T}
W = chunk_mask(W)        # 仅保留对角线附近 chunk
Ĉ' = W @ C               # extended compressed sequence ∈ R^{T×256}

# Step 4: Absorbed attention with stride-aware mask
scores = (X @ (W_Q @ W_K^T)) @ Ĉ'^T / sqrt(d_h)   # W_K absorbed into W_Q
# Stride-aware mask: mask[n,m] = 0 if n==m or (m<n and m%2==0) else -∞
attn = softmax(scores + mask)
output = attn @ (Ĉ' @ (W_V @ W_O))                 # W_V absorbed into W_O
```

**MTLA 推理前向（s=2，incremental decoding）**：

```
输入: x_i ∈ R^{1×d}

# Step 1: Query
q_i = x_i @ W_Q

# Step 2: Latent vector
c_i = LayerNorm(x_i @ W_r)  # c_i ∈ R^{1×256}

# Step 3: Hyper-network 生成 merge weight
j = ceil(i/2)
pe_j: positional embedding at step j
w_i = Sigmoid(Linear(c_i) · Linear(pe_j))

# Step 4: 更新 compressed KV cache
if i % 2 == 1:   # 新 slot
    Ĉ = Concat(Ĉ, w_i ⊙ c_i)
else:             # 合并到当前 slot
    Ĉ_j = Ĉ_j + w_i ⊙ c_i  # 动态融合（可覆盖之前临时值 Ĉ_j'）

# Step 5: Absorbed attention
output = softmax((x_i @ W_Q_absorbed) @ Ĉ^T / sqrt(d_h)) @ (Ĉ @ W_V_absorbed)
```

术语一般如何实现？如何使用？

MTLA 在 Fairseq 框架上实现，作为可替换的 self-attention 模块。无需修改模型其他组件（FFN、LayerNorm 等）。开源代码（含 extended FlashAttention-2 CUDA kernel）：https://github.com/D-Keqi/mtla。

MTLA 特别适用于长序列任务（speech translation/recognition/understanding、text summarisation），因为 temporal compression 在长序列场景下收益最大。s=2 时已接近 MQA 的 KV cache 水平（144l vs 128l per-token elements），同时保持 MHA 级别的质量。s=4 时 per-token cache 降至 72l elements。但 s 过大导致性能下降（s=4 BLEU 23.05 vs s=2 23.28）。

涉及论文标题：
- Multi-head_Temporal_Latent_Attention

---
