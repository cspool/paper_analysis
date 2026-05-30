## Group Query Attention (GQA)

术语解释
GQA (Group Query Attention) 是多查询注意力 (MQA) 和多头注意力 (MHA) 之间的折中方案，将 Query heads 分组共享 K、V heads，在减少 KV cache 大小和保持模型质量间取平衡。首次由 Ainslie et al. (2023) 在 "GQA: Training Generalized Multi-Query Transformer Models" 中提出。

术语是什么？
MHA 每 head 有独立 Q、K、V 投影 → KV cache = num_heads × d_head × 2。MQA 所有 head 共享 K、V → KV cache = 1 × d_head × 2。GQA 折中：将 heads 分成 groups，每组内共享 K、V。例如 40 heads + 8 KV groups = 5 Q heads per KV head。KV cache = num_groups × d_head × 2。

AquilaMoE 的模型配置演变：1.8B/7B 的 KV group=32 (实际是 MHA)，16B/MoE 的 KV group=8 (8 KV groups for 40 heads = GQA)。AKI-Pro 的 GQA 兼容性改造基于此设计：保持源和目标模型的 group 数一致，将每个 group 作为独立 MHA block 进行扩展。

从算法pipeline角度拆解术语：
```
# GQA Attention (40 heads, 8 groups, heads_per_group=5)
Q = x @ W_Q  # [seq, 40*d_head] — 每个 head 独立 Q
K = x @ W_K  # [seq, 8*d_head]  — 每组共享 K
V = x @ W_V  # [seq, 8*d_head]  — 每组共享 V

# Q: reshape → [seq, 8, 5, d_head]
# K, V: reshape → [seq, 8, 1, d_head] → expand → [seq, 8, 5, d_head]

for group in range(8):
    Q_group = Q[:, group, :, :]  # [seq, 5, d_head]
    K_group = K[:, group, :, :]  # [seq, 5, d_head] — 原为 [seq, 1, d_head] 扩展
    V_group = V[:, group, :, :]
    attn_group = Softmax(Q_group @ K_group^T / sqrt(d_head)) @ V_group

output = concat([attn_group for group in range(8)], dim=-1) @ W_O
```

术语一般如何实现？如何使用？
- 典型配置：Llama 2 70B (8 KV groups for 64 heads, i.e., MQA), Llama 3 70B (GQA, 8 groups)
- 相较于 MHA，GQA 的 KV cache 减少 num_heads/num_groups 倍
- GQA 在推理中降低 KV cache 显存占用，对长 context 场景尤为关键
- 训练中 GQA 略微降低收敛速度但仍能收敛到相似质量
- PyTorch 实现：SDPA 的 `torch.nn.functional.scaled_dot_product_attention` 支持 GQA 广播机制

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

**DS-MoE 中的 GQA 使用**：DS-MoE 在 MoA (Mixture of Attention Head) 中使用 GQA 机制：每个 MoA expert 计算 N_head 个 query vectors，但 K、V 由所有 expert 共享（通过 GQA 的 shared KV heads）。1B 模型使用 2 shared KV heads，3B/6B 模型使用 4 shared KV heads，与 N_head（2/4）数量相等，退化为 MHA 的 KV pattern。

---
