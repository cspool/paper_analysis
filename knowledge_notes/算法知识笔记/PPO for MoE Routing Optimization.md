## PPO for MoE Routing Optimization

术语解释
将 Proximal Policy Optimization (PPO) 强化学习算法应用于 MoE 路由策略优化，通过端到端训练路由 agent（allocator）来最大化语言模型预测质量与计算效率的加权目标，绕过路由决策的非可微分问题。

术语是什么？
在 MoE 路由场景中，allocator 的采样操作 k* ~ P_alloc(x_i) 是不可微分的，无法通过标准反向传播优化。PPO 将路由建模为 sequential decision-making 问题：
- **Agent**: 每层的 allocator (policy π_θ_l)
- **State s_l**: token 在第 l 层的 hidden state x_i^(l)
- **Action ĉ_l**: 采样得到的专家激活数量 k*
- **Reward**: 仅最后一层 (l=L) 的 agent 接收 reward = log P(x_i|x_1,...,x_{i-1})（即语言模型的对数似然，等价于 NLP caption loss 的负值）
- **Discount factor γ**: 在训练中控制远期 reward 的折扣

关键设计：
1. **Reward 仅分配给最后一层**：因为语言模型的预测质量仅在最终 output token 中体现，中间层的"贡献"通过 advantage 传播
2. **Advantage = reinforce with baseline**：A_l = γ^{L-l}[R(ĉ_L, s_L) - R(c*_L, s*_L)]，其中 baseline 为默认 Top-K 路由的 reward。baseline 减除了方差，使训练更稳定
3. **无需额外的 value network**：与标准 PPO 不同，Ada-K 不需要 value function 来估计 advantage，直接使用 reinforce with baseline 形式
4. **仅需 2 PPO epochs**：因为 action space 较小（最多 N 个选择），训练快速收敛

从算法pipeline角度拆解术语。
```
# PPO Training for Ada-K Allocators
# 仅优化 allocator 参数 θ = {θ_1, ..., θ_L}，LLM 主干冻结

for epoch in 1..2:  # 2 PPO epochs
    for batch in dataloader:
        # === Forward Pass (收集 experience) ===
        for layer l in 1..L:
            for token x_i:
                P_alloc = Softmax(W_alloc[l] @ x_i)
                k* ~ Categorical(P_alloc)           # action
                save_old_prob(π_θ_old(k* | x_i))    # old policy log prob
                # Router (frozen) -> TopK(k*) -> Expert FFN -> hidden state
        
        # === Reward 计算 ===
        R = cross_entropy_loss(logits, labels)      # 负的 language modeling loss
        R_baseline = compute_baseline_reward(x, labels)  # 默认 Top-K 输出
        
        # === Advantage (reinforce with baseline) ===
        for layer l in L..1:  # 从后往前
            A[l] = γ^{L-l} * (R[l] - R_baseline[l])
        
        # === PPO Loss + Regularization ===
        for layer l in 1..L:
            π_θ_new = Softmax(W_alloc[l] @ x_i)[k*]
            r = π_θ_new / π_θ_old                # importance sampling ratio
            L_clip = min(r * A[l], clip(r, 1-ε, 1+ε) * A[l])
            
            # Activation Regularization (期望 k 最小化)
            L_reg = Σ_n n * P_alloc[n]           # 每层期望专家数
            
            L = -L_clip + λ * L_reg
        
        # === Update ===
        θ = AdamW(L)  # 仅更新 allocator 参数
```

术语一般如何实现？如何使用？
- 使用 AdamW optimizer，learning rate = 1e-3，batch size = 64
- PPO clip 参数 ε 通常设 0.2（论文未明确说明具体值）
- 训练 1 epoch over 10k 样本，仅 2 PPO epochs
- λ = 3e-3 作为性能与效率的平衡参数
- 与 RLHF 中的 PPO 有根本区别：RLHF 需要一个独立的 reward model (RM) 评估生成质量；而 Ada-K 的 reward 直接从 language modeling cross-entropy loss 派生，无需外部 RM
- 硬件：Mixtral-8x22B 使用 16×A800-80G，其他模型 8×A800-80G

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs

---
