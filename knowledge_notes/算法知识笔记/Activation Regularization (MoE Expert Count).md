## Activation Regularization (MoE Expert Count)

术语解释
Activation Regularization 是 Ada-K 训练中用于控制专家激活数量的正则化损失，通过最小化所有层 allocator 输出分布的期望值，直接减少平均激活专家数量，在训练中与 PPO loss 共同优化。

术语是什么？
Activation Regularization Loss 的公式：
$$\mathcal{L}^{reg}(\theta) = \frac{1}{L}\sum_{l=1}^{L}\sum_{n=1}^{N} n \cdot \mathcal{P}_{\theta_l}(n)$$

其中：
- L 为 MoE layer 数量
- N 为每层的专家总数
- P_θ_l(n) 为第 l 层 allocator 输出激活 n 个专家的概率
- n · P_θ_l(n) 为第 l 层激活专家数量的期望值

该损失项**可微分**，因为它直接优化 allocator 输出的概率分布期望，而非通过采样操作。因此可以同时参与标准梯度反向传播（"As Loss"模式），或作为 reward 项合并到 PPO objective 中（"As Reward"模式）。

从算法pipeline角度拆解术语。
```
# Activation Regularization: "As Loss"模式 (default)
# 直接计算 allocator 输出分布的期望值并反向传播

def activation_regularization(allocator_outputs, L):
    # allocator_outputs: list of [batch, seq, N] for each layer
    # L: number of MoE layers
    total_reg = 0
    for l in range(L):
        P = allocator_outputs[l]           # [batch, seq, N]
        n_range = arange(1, N+1)            # [1, 2, ..., N]
        expectation = sum(n_range * P, dim=-1)  # [batch, seq]
        total_reg += expectation.mean()    # average over batch & seq
    return total_reg / L

# 总损失
L_total = L_RL + λ * L_reg
# λ = 3e-3 as default trade-off coefficient
```

术语一般如何实现？如何使用？
- "As Loss"模式（直接优化期望）比"As Reward"模式（将期望纳入 reward）表现略优：Acc=55.13 vs 54.64
- λ 控制性能与效率的 trade-off：更大的 λ → 更强的激活压缩 → 更高的效率但可能降低性能
- 在 activation reduction rate 达 44% 前，Ada-K 性能始终高于 baseline（见图 2 trade-off curve）
- λ 的扫描过程：为每个模型单独扫描 λ 值生成 trade-off curve，选择最优平衡点（论文统一使用 λ=3e-3）
- 该正则化项的梯度直接通过 allocator 输出概率反向传播，不经过采样操作，因此与 PPO loss 互补：PPO 优化采样决策质量，regularization 优化期望激活数量

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs
- AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models

---
