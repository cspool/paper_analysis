## Inter-Expert Pruning / Intra-Expert Pruning in MoE（MoE 的专家间剪枝 / 专家内剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE 剪枝是针对 Mixture of Experts 模型的两类结构化压缩策略：(a) **Inter-Expert Pruning（专家间剪枝）**：移除整个 expert 子网络及其对应的 router 权重行，减少总 expert 数量但保持 active experts 数量不变。例如，Mixtral-8x7B 有 32 层 × 每层 8 experts = 256 个 expert 实例，12.5% inter-expert pruning 移除每层 1 个 expert（即 32 个 expert 实例）。(b) **Intra-Expert Pruning（专家内剪枝）**：在不改变 expert 数量的前提下，缩减每个 expert 内部的 FFN dimension（intermediate/hidden dimension）。例如，25% intra-expert pruning 将每个 expert 的 FFN intermediate dimension 从 14336 缩减至约 10752，降低每 expert 的计算量但不减少 expert 数量。这两种剪枝策略有不同的内存-计算 trade-off：inter-expert pruning 直接减少模型总参数量（移除整列参数），intra-expert pruning 减少 per-expert 计算量。MoE-I²（Yang et al., EMNLP 2024）是该方向代表性工作，结合 inter-expert pruning（移除低重要性 expert）和 intra-expert low-rank decomposition（对保留 expert 进行低秩分解压缩）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE 剪枝后的推理 pipeline（以 Mixtral-8x7B, 50% intra-expert pruning, FFN dim: 14336→7168）：

```
# Pruned MoE Layer Forward Pass (pseudocode)
# Pruning applied offline; inference uses pruned dimensions directly

# Pre-processing (offline): Pruning decision
# inter-expert: compute importance_score[i] = mean(|W_gate[i,:]|) + mean(|W_expert_i|)
#    → remove expert j if importance_score[j] < threshold (e.g., bottom 50%)
# intra-expert: compute per-channel importance of FFN weight columns
#    → remove columns with lowest importance_score (e.g., bottom 50%)

hidden_states = input  # [B, S, 4096]

# Router: #experts reduced if inter-pruning applied
router_logits = hidden_states @ W_gate_reduced   # [B, S, num_experts_surviving]
topk_weights, topk_indices = topk(softmax(router_logits), k=2)

for expert_id in surviving_expert_ids:  # fewer experts if inter-pruning
    tokens = hidden_states[topk_indices == expert_id]

    # FFN with REDUCED intermediate dim (if intra-pruning)
    # W_gate: [4096, ffn_dim_reduced=7168]  (originally 14336)
    gate_out = silu(tokens @ W_gate[expert_id])   # [n_tokens, 7168]
    up_out   = tokens @ W_up[expert_id]            # [n_tokens, 7168]
    hidden   = gate_out * up_out
    expert_out = hidden @ W_down[expert_id]        # [n_tokens, 4096]

    output[topk_indices == expert_id] += topk_weights[:, expert_id] * expert_out
```

MoE-Inference-Bench 的核心发现（Section 6.2）：(a) **50% aggressive pruning 反而显著提高吞吐量**（因为减少的总参数和计算量超过负载不平衡带来的损失）；(b) **12.5%/25% 低比例剪枝可能降低吞吐量**——因为剪枝引入了负载不均衡（某些 expert 成为瓶颈），但节省的计算量不足以补偿；(c) **OLMoE-1B-7B 对 intra-expert 剪枝容忍度高**（结构设计使其计算分布更均匀）；**Qwen1.5-MoE-A2.7B 更敏感**（高剪枝比例在低 TopK 时吞吐量显著退化）。这表明剪枝策略需要模型特定的调优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MoE 剪枝的实现方法：
- **重要性评估**：常见方法包括 weight magnitude（权重大小）、activation-based importance（激活值感知）、Taylor expansion（梯度×权重近似精度损失）、以及 expert 使用频率（inference 时激活次数）。MoE-Inference-Bench 具体使用的方法论文未详细说明，但引用了 [29]（Lu et al., 2024, "Not all experts are equal"）和 [48]（Yang et al., 2024, "MoE-I²"）。
- **剪枝时机**：post-training one-shot pruning（无需重新训练，直接基于预训练模型的权重/激活统计做剪枝决策）。
- **框架集成**：剪枝在模型加载前完成（weight matrix 直接缩减），推理框架无需感知剪枝过程——只需加载剪枝后的较小权重矩阵。
- 局限：(a) inter-expert pruning 移除 expert 后 router 也要同步更新（移除对应输出维度），可能改变 routing 行为；(b) 剪枝比例的选择是模型和任务特定的；(c) 激进的剪枝可能导致某些 token 的 routing 选择范围受限（所有被路由到的 expert 都被剪枝）。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
