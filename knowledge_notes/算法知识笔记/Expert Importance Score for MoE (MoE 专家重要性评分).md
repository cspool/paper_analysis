## Expert Importance Score for MoE (MoE 专家重要性评分)

术语解释
Expert Importance Score 是基于 prefill 阶段 expert 激活统计计算的重要性度量，用于驱动 GPU-NDP 系统中的 expert placement 和 bitwidth allocation。定义为归一化的激活频率和路由评分的加权混合。

术语是什么？
对于 MoE 模型第 l 层的 expert e，在 prefill 阶段收集两个统计量：
- P_{l,e}：expert e 在所有 prefill tokens 中被选中的次数（激活频率）
- W_{l,e}：expert e 被选中时的累计 routing score（Softmax 后的门控权重之和，反映激活的"置信度"）

重要性分数：S_{l,e} = α · P̃_{l,e} + (1-α) · W̃_{l,e}
其中 P̃ 和 W̃ 是归一化后的值（除以该层所有 experts 的总和），α ∈ [0,1] 控制两个信号的权重。

设计动机：激活频率反映 expert 的"热度"（被用得越多越重要），routing score 反映激活的"质量"（高 score 表示 router 对该选择的置信度高）。两个信号互补——有时高频 expert 的 routing score 不高（被广泛但不强烈需要），有时低频但有高置信度 score（对特定上下文重要）。

从算法pipeline角度拆解术语：

```
=== Prefill 阶段统计收集 ===
def prefill_with_importance_stats(tokens):
    for each MoE layer l:
        P[l] = zeros(E)    # activation counts
        W[l] = zeros(E)    # cumulative routing scores
        
        for each token x in tokens:
            scores = Softmax(W_gate[l] @ x)  # [E]
            top_k = TopK(scores, k=2)
            
            for e in top_k:
                P[l][e] += 1
                W[l][e] += scores[e]
    
    return P, W

=== Importance Score 计算 ===
def compute_importance_scores(P, W, alpha=0.5):
    S = {}
    for layer l:
        # 归一化到 [0, 1] (per-layer)
        P_tilde = P[l] / sum(P[l])
        W_tilde = W[l] / sum(W[l])
        S[l] = alpha * P_tilde + (1-alpha) * W_tilde
    return S
```

```
=== 基于 Importance 的 Expert Placement ===
K = GPU_expert_budget  # 由 GPU HBM 容量决定

for each layer l:
    ranked_experts = argsort(S[l], descending=True)
    H[l] = ranked_experts[:K]   # GPU (FP16, hot)
    C[l] = ranked_experts[K:]   # NDP (quantized, cold)
```

关键性质：
- Prefill-decode similarity (cosine sim ~0.89) → prefill importance 可预测 decoding 行为
- Per-sequence 计算 → 捕捉 context-dependent expert 重要性
- Once-per-sequence → decoding 期间零额外迁移

术语一般如何实现？如何使用？
- 收集开销：每层 E 个计数器（如 Mixtral 8 experts），2 个指标 × 32 layers = 512 values，metadata 开销可忽略
- 归一化：per-layer normalization 保证层间可比，适应不同层 expert 分布的差异
- α 选择：论文使用 α=0.5，可通过 minimal grid search 调优
- 应用：(1) expert placement to GPU/NDP；(2) bitwidth allocation ordering；(3) 可扩展到 expert pruning/caching 决策
- 替代方案：仅用 frequency (simple but ignores routing confidence)、仅用 routing score (captures confidence but misses volume)

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework（使用更简单的 Expert Drop 重要性评分：S(E_i) = (1/|X|) * Σ_{x∈X} G_i(x)，即批数据上的平均路由分，用于 Expert Drop 的 layer-wise 和 global dropping 策略——layer-wise 每层保留相同数量 experts，global 跨层全局选择 Top experts；发现 score distribution 影响 MoE 对 Expert Drop 的鲁棒性：DeepSeek-MoE-16B 左偏分布（多数 expert 低分→可 drop 更多），Mixtral-8×7B 右偏分布（仅少数不重要→drop 代价高））

---
