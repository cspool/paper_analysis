## Attention Router (注意力路由器)

术语是什么？

Attention Router 是 Elastic Attention 提出的轻量级模块，以 MoE 风格的 gating 在推理时动态决定每个 attention head 使用 FA 还是 SA 计算模式，实现 test-time 自适应稀疏度分配。每层仅 0.27M 参数（head_dim=128），由 Task MLP + Router MLP 组成。输入 Key hidden states，输出 head-wise 二值路由决策。

从算法pipeline角度拆解术语。

```
# Attention Router 前向（per layer）
Input: x_K ∈ R^{s×H×d'}  # Key hidden states

# Step 1: Boundary Pooling（首100 + 尾100 tokens）
x_K' = Pool(x_K[:100] ∪ x_K[-100:])  # [H, d']

# Step 2: Task MLP → task-specific features
z_task = SiLU(W_task1 @ x_K')        # intermediate = 4×d'
z_task = W_task2 @ z_task            # [H, d']

# Step 3: Router MLP → routing logits
z = SiLU(W_router1 @ z_task)
z = W_router2 @ z                    # [H, 2]

# Step 4: Gumbel-Sigmoid → hard routing
g = -log(-log(u + ε) + ε)           # Gumbel noise
r_soft[:, 1] = σ((z[:, 1] - z[:, 0] + g) / τ)  # SA prob
r_hard = argmax(r_soft, dim=-1)      # 0=FA, 1=SA
r_hard = r_hard + (r_soft - detach(r_soft))  # STE
```

术语一般如何实现？如何使用？

训练：backbone 冻结，仅优化 Router（12h on 8×A800，300 steps，seq_len=65536），decoupled LR（router=5e-4, λ=1e-3），训练数据 0.74B tokens 五源混合。推理：Router 仅 ~0.196ms/call，延迟不随 seq_len 增长（Boundary Pooling 固定 200 tokens）。SA 模式可选 SSA（sink+local window）或 XA（block sparse），训练时同时学习。代码：https://github.com/LCM-Lab/Elastic-Attention。

涉及论文标题：
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers
