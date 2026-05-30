## Gate Probability in Mixture-of-Experts

术语解释
Gate probability（门控概率）是 MoE router 经过 softmax 归一化后对每个 expert 的分配概率，反映 router 认为每个 expert 对当前 token 的有用程度。Gate probability 是理解 MoE expert 利用率和知识分布的核心指标。

术语是什么？
给定 token x，router 先计算 gate logits H(x)（含噪声），经 softmax 得到 gate probability：
$$G_i(x) = \frac{\exp(H(x)_i)}{\sum_{j=1}^N \exp(H(x)_j)}$$

Top-k routing 仅激活 gate probability 最高的 k 个 expert，其余 expert 的 gate probability 被遮蔽（masked to 0 after KeepTopK）。

关键发现 (Kim et al., 2025)：在 MoE KD 过程中，所有层中 activated experts 的 gate probability 之和通常低于 50%。这意味着超过一半的"router 信心"分配给了 non-activated experts，但这些专家的知识未被传统 KD 利用。

从算法pipeline角度拆解术语：
```
# Gate Probability Analysis
for each training sample during KD:
    for each MoE layer l:
        gate_probs = softmax(router(x_l))             # [N]
        activated_probs = gate_probs[topk_indices]    # [K]
        non_activated_probs = gate_probs[~topk_indices]  # [N-K]
        
        sum_activated = sum(activated_probs)          # < 0.5 in most layers
        sum_non_activated = sum(non_activated_probs)  # > 0.5
        
        # Observation: non-activated experts collectively 
        # have higher router confidence than activated ones
```

Gate probability 的两重角色：
1. **Expert 选择**：决定哪些 expert 被激活（top-k selection）
2. **输出加权**：聚合 expert 输出时的权重（weighted sum）

术语一般如何实现？如何使用？
- Softmax 归一化确保所有 expert 概率和为 1
- 负载均衡辅助损失最小化 gate probability 分布的 CV（变异系数），使分布更均匀
- 在 KD 中，SAR 通过 student feedback 微调 gate probability，使知识传递更有效
- Gate probability 是 MoE 知识分布的可视化和诊断工具

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
