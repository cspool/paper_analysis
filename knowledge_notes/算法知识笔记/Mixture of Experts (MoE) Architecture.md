## Mixture of Experts (MoE) Architecture

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixture of Experts (MoE) 是一种神经网络架构模式，将传统 Transformer 中的单一 FFN（Feed-Forward Network）替换为多个（E 个）规模相同的 FFN（称为 "experts"），并通过一个可训练的 gating network 动态选择每个输入 token 对应的 top-k 个 experts（通常 k=1 或 2）进行稀疏激活。MoE 的核心价值是"条件计算"（conditional computation）——增加模型总参数量（更多 experts）而不等比增加计算量（每个 token 仅激活少量 experts），从而实现 sublinear scaling of compute cost with model size。MoE 架构最早由 Shazeer et al. (2017, "Outrageously Large Neural Networks") 引入深度学习，后在 GShard (Lepikhin et al., 2021)、Switch Transformer (Fedus et al., 2022)、DeepSeek-V3 (2024)、Mixtral 8x7B (2024) 等模型中广泛采用。

MoE layer 执行流程：
1. **Gate**: 对输入 token x，gate function G(x) = softmax(x·W_g) 计算所有 E 个 experts 的 affinity scores
2. **Top-K selection**: 从 affinity scores 中选 top-k (k=2)，得 selected_experts 和对应 weights g
3. **Dispatch**: 将 token 送到选中的 experts 所在的设备（本地或远端 GPU）
4. **Expert FFN**: 各 expert 对被路由到的 token 执行 FFN(x) = W_2·φ(xW_1 + b_1) + b_2，其中 φ 为 GELU/ReLU/SiLU 等激活函数
5. **Combine**: 若 k>1，将多个 expert 的输出按 gating weights 加权合并：h_i = Σ_j (g_{i,e_j}/C_i)·h_i^j，其中 C_i = Σ_j g_{i,e_j} 为归一化因子

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# ===== MoE Layer Forward Pass (Algorithm) =====
# Input:  x ∈ R^{S×H}  (S tokens, H hidden dim)
# Output: y ∈ R^{S×H}
# Experts: E FFNs, each W1[e]∈R^{H×D}, W2[e]∈R^{D×H}
# Gate: W_g ∈ R^{H×E}

def moe_forward(x, experts, gate, k=2, capacity_factor=1.0):
    S, H = x.shape
    E = len(experts)
    C = int(S * k * capacity_factor / E)  # expert capacity
    
    # Step 1: Gate — 计算所有 token 对所有 expert 的 affinity
    gate_logits = x @ W_g  # [S, E]  (或 softmax(x·W_g))
    gate_scores = softmax(gate_logits, dim=-1)
    
    # Step 2: Top-K selection — 每个 token 选 top-2 experts
    topk_weights, topk_indices = topk(gate_scores, k, dim=-1)
    # topk_weights: [S, k], topk_indices: [S, k]  (哪些 expert)
    
    # Step 3: Dispatch — 按 expert 分组 token
    expert_inputs = {e: [] for e in range(E)}
    expert_weights = {e: [] for e in range(E)}
    for i in range(S):
        for j in range(k):
            e = topk_indices[i, j]
            if len(expert_inputs[e]) < C:  # capacity check
                expert_inputs[e].append(x[i])
                expert_weights[e].append(topk_weights[i, j])
    
    # Step 4: Expert FFN — 各 expert 独立处理
    expert_outputs = {}
    for e in range(E):
        if len(expert_inputs[e]) > 0:
            batch_e = stack(expert_inputs[e])  # [n_e, H]
            # FFN: two linear layers with activation
            h1 = batch_e @ W1[e]  # [n_e, D]
            h1 = gelu(h1)         # activation
            h2 = h1 @ W2[e]       # [n_e, H]
            expert_outputs[e] = h2
        else:
            expert_outputs[e] = None
    
    # Step 5: Combine — weighted sum of expert outputs
    y = zeros(S, H)
    combine_norm = zeros(S)
    for i in range(S):
        for j in range(k):
            e = topk_indices[i, j]
            w = topk_weights[i, j]
            if expert_outputs[e] is not None:
                # need to find which row in expert_outputs[e] is token i
                y[i] += w * expert_outputs[e][token_idx_in_expert[i, j]]
        y[i] /= combine_norm[i] if combine_norm[i] > 0 else 1.0
    
    return y
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MoE 实现涉及几个关键设计选择：(1) **Expert capacity C**——通过 capacity factor 控制，C = (tokens × top_k × cf) / num_experts，cf < 1 时可能丢弃 token（需 auxiliary load balancing loss 鼓励 uniform routing）；(2) **Load balancing**——MoE 需要 auxiliary loss 鼓励 token 均匀分布到各 experts，常见公式: L_aux = E·Σ_i f_i·P_i，其中 f_i = expert i 处理的 token 比例，P_i = gate 分配给 expert i 的平均概率；(3) **分布式部署**——experts 分布在多 GPU 上时需 cross-GPU AlltoAll 通信（dispatch + combine），通信开销可占总运行时间 68%；(4) **Auxiliary loss**——除 load balancing loss 外，还有 z-loss（防止 logits 过大导致数值不稳定）。

代表模型: GShard (E=2048, k=2), Switch Transformer (E=2048, k=1), DeepSeek-V3 (E=256, k=8, 685B total params), Mixtral 8x7B (E=8, k=2), Qwen3-235B-A22B (E=128)。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
