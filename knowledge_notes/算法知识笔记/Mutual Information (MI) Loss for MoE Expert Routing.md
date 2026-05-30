## Mutual Information (MI) Loss for MoE Expert Routing

术语解释
由 Shen et al. (2023, ModuleFormer) 提出并由 Pan et al. (2024, DS-MoE) 用于 Dense Training MoE 的核心损失函数。基于信息论中 Mutual Information 的概念，通过最大化 expert 分布的边际熵 H(e) 和最小化条件熵 H(e|X) 来实现 expert 负载均衡和专家集中。

术语是什么？
MI Loss 的数学形式：L_MI = -H(e) + (1/|X|) Σ_{x∈X} H(e|x)，其中：
- H(e) = -Σ_{i=1..N} p(e_i) log p(e_i)：expert 的边际熵。p(e_i) = mean_{batch}(S_i) 为 batch 内 expert i 的平均 Router 概率。最大化 H(e) → 所有 N 个 expert 被平均使用（负载均衡）。
- H(e|x) = -Σ_{i=1..N} S_i log S_i：给定 token x 条件下 expert 的条件熵。最小化 H(e|x) → Router 对每个 token 产生集中的概率分布（expert concentration / sparsity）。
两项形成"对抗平衡"：负载均衡（maximize H(e)）vs. 专家集中（minimize H(e|x)）。

与标准 switch loss 的区别：
| 维度 | Switch Loss (Fedus 2022) | MI Loss (Shen 2023 / Pan 2024) |
|------|--------------------------|-------------------------------|
| 形式 | L = α·N·Σ f_i·P_i (双线性乘积) | L = -H(e) + (1/|X|)·Σ H(e|x) (信息熵差) |
| 所需信息 | f_i (路由频率) + P_i (平均概率) | 仅 Router scores S |
| K 固定 | 通常需 fixed K | 支持 flexible/inference-time K |
| Sparsity 控制 | 隐式（通过 α 和 K） | 显式（H 项间平衡） |

从算法pipeline角度拆解术语：
```
# MI Loss computation (per micro-batch)
def mi_loss(router_scores, alpha, N_experts):
    # router_scores: [B, N] softmax outputs per token
    p_e = router_scores.mean(dim=0)              # [N], marginal P(e)
    H_e = -sum(p_e * log(p_e))                   # expert entropy
    H_cond = -sum(router_scores * log(router_scores), dim=-1).mean()  # conditional entropy
    L_mi = -H_e + H_cond                         # MI Loss
    return alpha * L_mi

# Total loss
L_total = cross_entropy(logits, labels) + mi_loss(router_scores, alpha, N)
```

术语一般如何实现？如何使用？
- **α 调参**：α 控制 sparsity 程度。DS-MoE 验证 α 越大 → 模型在高 sparsity 下性能保持更好，但可能在低 sparsity 下性能略差。需要 α 在"稀疏度"和"整体性能"间平衡。
- **DS-MoE α 值**：MoA 层 3.5e-4 (1B) / 2e-4 (3B/6B), MLP 层 6.3e-4 (1B) / 4e-4 (3B) / 2e-4 (6B)
- **训练后 sparsity 调整**：MI Loss 训练出的 Router 可灵活切换推理 sparsity 级别（调整 K 或 ε），无需重新训练。DS-MoE-6B 在 24% active hidden (vs default 29%) 时仍可通过调大训练 α 保持性能。
- **源工作**：ModuleFormer (Shen et al. 2023) 使用 MI Loss 训练模块化 LLM；DS-MoE 将其应用于 MoE 大规模预训练场景

涉及论文标题：
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

---
