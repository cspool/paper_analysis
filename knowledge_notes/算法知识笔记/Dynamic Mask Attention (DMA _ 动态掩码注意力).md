## Dynamic Mask Attention (DMA / 动态掩码注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Mask Attention (DMA) 是一种可训练的 content-position 双感知稀疏注意力机制，由港科大(广州)与智源(BAAI)于 2025 年提出（arXiv: 2508.02124，提交 NeurIPS 2025）。DMA 将稀疏注意力的计算分解为两个解耦的阶段：(1) Content-Aware Dynamic Mask Generation：从 value 向量表示中采样生成 per-head 动态 mask，使模型能自适应识别关键 token；(2) Position-Aware Sparse Weight Computation：利用动态 mask 对 scaled dot-product attention 做稀疏化，mask 值为 −∞ 的位置直接跳过计算。整个过程完全可微，支持端到端训练。DMA 的核心创新在于将"哪些 token 需要关注"（内容感知）和"如何高效计算这些关注"（位置感知/硬件友好）解耦，使有效复杂度从 O(n²d_h) 降为 O(nwd_h)，内存从 O(n²) 降为 O(nw)。

注意区分：DMA（本术语）与 DAM (Dynamic Attention Mask, Zhang et al., 2025) 是不同的方法。DAM 是一种针对推理阶段的动态注意力掩码方法（被 DMA 论文列为 baseline），而 DMA 是端到端可训练的稀疏注意力机制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DMA 算法 pipeline 分为两个阶段：

**Phase 1 — Content-Aware Dynamic Mask Generation**：
```
Input: V ∈ R^{b, n_h, n, d_h}  (value 矩阵)
Params: Δ ∈ R^{n_h, d_h, n_h}  (采样权重), A ∈ R^{n_h} (门控系数)
Hyperparams: w (per-head window size), τ(·) (非负激活, 如 softplus)

1. V_flat = V.transpose(1,2).reshape(b, n, n_h*d_h)   # 展平 head 维度
2. dt = V_flat @ W_dt                                  # 线性投影 → [b, n, n_h]
3. dt = exp(A * τ(dt))                                 # 门控 + 激活 + exp → 非负分数
4. dt = dt.transpose(-1, -2)                          # → [b, n_h, n]
5. m_t = dt.expand(-1, -1, q_len, -1)                 # broadcast 到 query 维度
6. m_t = m_t.masked_fill(causal_mask, -inf)           # 施加 causal mask
7. topk_indices = topk(m_t, w, dim=-1).indices        # per-head 选择 top-w
8. m_t = m_t.masked_fill(not in topk_indices, -inf)   # 非 top-w 位置置 −∞
Output: m_t ∈ R^{b, n_h, q_len, n}  (dynamic mask)
```
A 可设计为 query-dependent：A = f(q_t)，使 gating coefficient 随输入自适应。

**Phase 2 — Position-Aware Sparse Attention**：
```
Input: Q, K, V, m_t, topk_indices
Output: O_t

for each (batch, head, query_pos):
    indices = topk_indices[b, h, q]           # top-w 位置索引, shape [w]
    K_sel = K[b, h, indices, :]               # [w, d_h]
    V_sel = V[b, h, indices, :]               # [w, d_h]
    m_sel = m_t[b, h, q, indices]             # [w]
    scores = (Q[b,h,q,:] @ K_sel^T) / sqrt(d_h) + m_sel  # [w]
    attn_w = softmax(scores)                   # [w]
    O[b,h,q,:] = attn_w @ V_sel               # [d_h]
```
Kernel 实现中，若 mask block 全为 −∞，则直接跳过整个 block 的加载和矩阵乘。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现：CUDA kernel 在 https://github.com/HKUSTDial/flash-sparse-attention（亦为 flash-algo/flash-sparse-attention），Triton 参考实现在 https://github.com/SmallDoges/flash-dmattn。

使用方式：替换标准 Transformer 的 self-attention 模块。训练配置：AdamW + WSD LR scheduler，RoPE 位置编码（base freq 从 10K 调整到 100K 用于长上下文适应），NeoX tokenizer。关键超参数 w（per-head 保留 key 数量）——在 Scaling Laws 实验中 w=1024（80M-680M）和 w=2048（1.7B）。DMA 可近似 full attention：当 n_h × w ≥ n 时，所有 token 都可能被某些 head 选中。

局限：(1) 固定 window size w 无法自适应任务复杂度变化；(2) RoPE 位置编码仍是外推瓶颈；(3) 目前仅在 text domain 验证，多模态扩展尚未实现；(4) 实验规模最大 1.7B 参数，更大模型（7B+）效果待验证。

涉及论文标题：
- Trainable_Dynamic_Mask_Sparse_Attention
