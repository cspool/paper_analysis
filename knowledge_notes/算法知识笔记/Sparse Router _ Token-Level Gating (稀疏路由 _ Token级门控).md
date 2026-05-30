## Sparse Router / Token-Level Gating (稀疏路由 / Token级门控)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse Router（稀疏路由器）是 MoE 架构中决定每个 token 分配给哪些专家的组件。它是一个可学习的线性函数 f(x) = W · x（W ∈ ℝ^{d×M}，d 为 hidden state 维度，M 为专家数），输出每个专家对当前 token 的分配概率（通过 softmax 归一化），然后通过 top-k 选择激活概率最高的 k 个专家。在 Uni-MoE 中，每个 MoE 层有独立的 Router，对每个 token 独立进行 top-2 专家选择，router 参数在阶段三与 LoRA 参数一起训练。

与传统 dense FFN 每个 token 激活所有参数不同，sparse router 实现了 token-level conditional computation——不同 token（不同模态、不同语义）激活不同的专家子集。其在多模态 MoE 中的特殊意义在于：router 可以学习到"模态感知"的路由策略，将图像 tokens 路由到图像专长专家，音频 tokens 路由到音频专长专家。

从算法pipeline角度拆解术语：

Sparse Router 在每个 MoE 层的计算流程（Uni-MoE 式 16-17）：

```
输入: X_l^s ∈ ℝ^{T×d}   # 经过 self-attention 后的 hidden states (T 个 tokens)

# Step 1: Router 计算每个 token 的专家分配概率
logits = X_l^s @ W_router              # W_router ∈ ℝ^{d×M}, logits ∈ ℝ^{T×M}
P = softmax(logits, dim=-1)            # P ∈ ℝ^{T×M}, 每行和为1

# Step 2: Top-K 选择
P_topk, indices = top_k(P, k=2, dim=-1)  # 每 token 选 top-2 专家

# Step 3: 归一化选中概率（可选）
P_topk = P_topk / sum(P_topk, dim=-1)    # 使选中概率和为 1

# Step 4: 加权累加专家输出
output = zeros_like(X_l^s)
for each token t:
    for each selected expert e_i (i=1..k):
        output[t] += P_topk[t, i] * Expert_FFN_{e_i}(X_l^s[t])   # 式(17)
```

Uni-MoE 可视化分析（Figure 4-5）揭示的 router 行为：
- 在 text-audio 输入下，专家 2 和 4 几乎主导所有 token 分配
- 在 text-image 输入下，专家 2（图像预训练）在初始层大幅领先
- 在 video 输入下（含音频+视觉），负载在各层更均衡
- 专家 1（原始 LLaVA MLP）在各场景下参与度最低——暗示预训练对专家专业化至关重要

术语一般如何实现？如何使用？

典型实现：Router 是一个简单的 `nn.Linear(hidden_size, num_experts)`，输出经 softmax 后 top-k。训练时与 LoRA 参数共同更新。在 Uni-MoE 中，router 在阶段三中与 LoRA 参数一起训练（学习率 4e-5），不使用 auxiliary balancing loss 时仍能学到有效的 routing 模式（因为 modality-specific 预训练专家提供了自然的路由信号）。关键权衡：k 值增大（如从 1→2）提升模型表达能力但增加计算量；Uni-MoE 消融实验（Table 7a）显示 top-2 在各 benchmark 上优于 top-1。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models
- Upcycling Large Language Models into Mixture of Experts
