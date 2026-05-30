## HMoE (Heterogeneous Mixture of Experts)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

HMoE 是一种 MoE 变体，核心区别在于同一 MoE 层中的不同 expert 具有**不同的参数量/容量**（不同的 FFN hidden dimension），而非传统 MoE 中所有 expert 大小相同。每个 expert 仍沿用标准 LLaMA-style FFN 设计：$e_i(\mathbf{x}) = \mathbf{W}_{o,i} \cdot (\text{SiLU}(\mathbf{W}_{g,i} \cdot \mathbf{x}) \odot (\mathbf{W}_{p,i} \cdot \mathbf{x}))$，但关键差异在于不同 expert 的 $\mathbf{W}_{g,i} \in \mathbb{R}^{h_{\text{input}} \times h_{\text{ffn},i}}$ 中的 hidden dim $h_{\text{ffn},i}$ 各不相同。例如在 HMoE-3B 主实验中，8 个 expert 的 hidden dim 按 arithmetic progression 设置为 {2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888}，large expert (5888 dim) 的参数量约为 small expert (2304 dim) 的 2.5×。异构设计使不同 expert 天然具有不同的表示容量和处理能力——大 expert 处理复杂 semantic token（如需要深度推理的后缀词），小 expert 处理简单 token（如冠词、介词）。

HMoE 面临的核心挑战：(1) 训练中 router 自然偏好激活大 expert（容量更强），导致小 expert 被闲置，总激活参数量不降反升；(2) 异构 expert 的不规则形状（不同 GEMM dim）给批量计算带来工程挑战，需使用 Megablocks 等 block-sparse kernel 替代传统统一形状 GEMM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

HMoE 层的 forward pass 伪代码（与 homogeneous MoE 的前向传播结构相同，差异在于各 expert FFN 的 hidden dim 不同 + 额外的 P-Penalty loss）：

```python
# HMoE Layer Forward (Top-P routing, arithmetic distribution)
# x: [B, S, h_input], 如 h_input=4096
# W_r: [h_input, N], N=8 experts
# expert_dims: [2304, 2816, ..., 5888]  # 异构 hidden dim

# Step 1: Router
P = softmax(x @ W_r)  # [B, S, N]

# Step 2: Top-P Routing (adaptive expert selection)
P_sorted, indices = sort(P, descending=True, dim=-1)
if P_sorted[0] >= p_threshold:  # p=0.6
    n_selected = 1
else:
    n_selected = min_k_where(cumsum(P_sorted) >= p_threshold)
selected_experts = indices[:, :, :n_selected]

# Step 3: Heterogeneous Expert Computation
output = zeros([B, S, h_input])
for e_idx in selected_experts:  # 每个 expert 的 hidden dim 不同
    mask = (tokens routed to expert e_idx)
    x_e = x[mask]                    # [n_e, h_input]
    gate_e = P[mask, e_idx]          # [n_e], 归一化后

    W_g = expert_weights[e_idx].W_g  # [h_input, h_ffn,e]
    W_p = expert_weights[e_idx].W_p  # [h_input, h_ffn,e]
    W_o = expert_weights[e_idx].W_o  # [h_ffn,e, h_input]

    # LLaMA-style SiLU-gated FFN (各 expert 的 h_ffn 不同)
    gate_out = SiLU(x_e @ W_g)       # [n_e, h_ffn,e]
    up_out = x_e @ W_p               # [n_e, h_ffn,e]
    hidden = gate_out * up_out       # element-wise
    expert_out = hidden @ W_o        # [n_e, h_input]

    output[mask] += gate_e.unsqueeze(-1) * expert_out

# Step 4: P-Penalty Loss (训练时)
# 对比传统 load balancing loss
L_pp = N * sum_i(M_i * P_hat_i)
# M_i = (1/T) * sum_t(indicator(e_i activated for t) * h_ffn,i)
# P_hat_i = (1/T) * sum_t(P_i,t)
# 激活大 expert 时 M_i 更大 → loss 更高 → 引导使用小 expert
```

异构 expert 的 token 分配行为：实验表明 smaller experts (2304 dim) 最常被激活的 top tokens 为简单冠词/代词 (the, such, your, these, most)，medium experts (3328-3840 dim) 处理具象语义词汇 (tables, valley, sun, day, war, water)，large experts (5376-5888 dim) 处理后缀/模糊 token (_ly, _zen, _icker, _decom, _inf)，验证了异构容量驱动的差异化 token 分配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HMoE 的实现基于 PyTorch，训练使用 DeepSpeed Zero2 + gradient checkpointing。关键实现考量：(1) 异构 expert 的不规则 GEMM shape 使用 Megablocks block-sparse kernel 高效批量计算；(2) P-Penalty loss 在每个 MoE 层的 forward 中计算 M_i（expert 激活次数 × hidden dim），系数设为 0.1；(3) 三种 expert 大小分布策略可通过配置 expert_dims 列表切换：arithmetic {9,11,13,15,17,19,21,23}（归一化比例）性能最优，geometric {1,2,4,8,16,32,64,128} 因过大的容量差距导致小 expert 训练不足，hybrid {1,1,1,1,2,2,4,4} 次优。代码尚未开源（论文声明 "Codes will be released upon acceptance"）。HMoE 使用 A800/H800 (80GB) GPU，AdamW optimizer (β1=0.9, β2=0.999), LR=1e-4 with 1000-step warmup, context=4096, batch=640, seed=12345。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
