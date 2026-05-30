## P-Warm Strategy (Top-P Warm Start for MoE Routing)

术语解释
P-Warm 是 Ada-K 提出的 allocator 预训练策略，利用 nucleus sampling (Top-P) 从原始 router 的专家概率分布中生成伪标签，warm-start 训练 allocator，避免随机初始化导致 RL 训练初期的采样不稳定。

术语是什么？
P-Warm 的核心思想：
1. **Top-P Nucleus Sampling**: 对每个 token x_i，按 router 输出概率降序排列专家，选择最小子集使其累积概率 ≥ p，该子集大小 n_i(p) 作为"应激活专家数量"的伪标签
2. **最优 p 选择**: n_i(p) = argmin_{k} Σ_{j≤k} P_{i,j}^↓ ≥ p。在所有可能的 p 值中，选择使平均 n_i(p) 最接近默认 Top-K 值 k 的 p*：
   $$p^* = \operatorname{argmin}_p \left|\frac{1}{T}\sum_{i=1}^{T} n_i(p) - k\right|$$
3. **伪标签监督训练**: 使用 n_i(p*) 作为 cross-entropy loss 的目标，预训练 allocator 使其输出分布接近 Top-P 导出的伪标签

从算法pipeline角度拆解术语。
```
# P-Warm 伪代码
def p_warm_pretrain(router, allocator, tokens, k_default):
    # Step 1: 选择最优 p
    best_p, best_diff = None, inf
    for p in [0.1, 0.2, 0.3, ..., 0.9]:
        labels = []
        for token in tokens:
            P_router = Softmax(router(token))
            P_sorted = sort_descending(P_router)
            n = argmin_k(cumsum(P_sorted)[:k] >= p)
            labels.append(n)
        avg_n = mean(labels)
        diff = abs(avg_n - k_default)
        if diff < best_diff:
            best_p, best_diff = p, diff
    
    # Step 2: 使用伪标签训练 allocator
    for epoch in warm_start_epochs:
        for token in tokens:
            P_router = Softmax(router(token))
            P_sorted = sort_descending(P_router)
            pseudo_label = argmin_k(cumsum(P_sorted)[:k] >= best_p)
            
            P_alloc = Softmax(allocator(token))
            L = CrossEntropy(P_alloc, pseudo_label)
            optimizer.step(L)
    
    # Step 3: 初始化完成，进入 PPO 训练
```

术语一般如何实现？如何使用？
- 对每个 baseline model，独立计算最优 p* 值（论文中使用的 threshold p=0.3）
- 在 moderate amount of tokens (T) 上计算 n_i(p) 的平均值来确定 p*
- P-Warm 仅需少量 warm-start epochs
- 消融实验显示：P-Warm (Acc=55.13) > K-Warm (Acc=54.97) > Random (Acc=54.18)，证明灵活的 Top-P 伪标签优于固定 K 值伪标签
- 核心优势：让 allocator 初始输出接近"自然"的专家分布（router 自身的专家概率累积），加速 PPO 训练收敛

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs

---
