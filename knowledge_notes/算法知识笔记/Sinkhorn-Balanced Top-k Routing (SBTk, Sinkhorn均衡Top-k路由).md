## Sinkhorn-Balanced Top-k Routing (SBTk, Sinkhorn均衡Top-k路由)

术语解释
Sinkhorn-Balanced Top-k (SBTk) Routing 是将 MoE 的 token-to-expert 分配建模为最优传输（Optimal Transport）问题并通过 Sinkhorn-Knopp 算法迭代求解的路由策略。与 PBTk 的"事后惩罚"不同，SBTk 在路由决策阶段显式调整 routing probabilities 以达到负载均衡。首次由 Clark et al. (2022) 引入 MoE 路由，后由 Anthony et al. (2024) 通过有利初始条件加速收敛。

术语是什么？
SBTk 将路由建模为带熵正则的 Kantorovich 最优传输问题：在满足每 token 概率和为 1、每 expert 接收等量 token 的约束下，最小化路由代价。Sinkhorn-Knopp 算法通过迭代行归一化和列归一化给出近似解：
```
K = exp(logits / ε)          # 将 logits 转为正矩阵，ε 控制熵正则强度
repeat until convergence:
    K = row_normalize(K)     # 每 token 的概率和为 1
    K = col_normalize(K)     # 每 expert 接收等量 token
probs = K                    # 最终的均衡路由概率
```
然后从调整后的 probs 中选 top-k experts。Anthony et al. (2024) 使用分组均值作为初始条件加速收敛。

关键区别：SBTk 的均衡步骤在推理时不可用（Sinkhorn 需要整 batch 的 token 信息，与自回归生成逐 token 解码不兼容），因此推理时必须移除 Sinkhorn 步骤（fall back to greedy top-k），导致推理时的 MRI 高于训练时。

从算法pipeline角度拆解术语：
```python
def sinkhorn_routing(logits, n_iters=5, tol=0.01):
    # logits: [T, E]  T=num_tokens, E=num_experts
    # Anthony et al. (2024) initialization: group-mean of logits
    K = exp(logits)  # exponentiate
    for _ in range(n_iters):
        K = K / K.sum(dim=1, keepdim=True)   # row normalize (token sum=1)
        K = K / K.sum(dim=0, keepdim=True)   # col normalize (expert receives 1/E)
    return K  # doubly stochastic routing matrix

# During training:
probs = sinkhorn_routing(logits)
topk_probs, topk_indices = topk(probs, k=K)

# During inference (auto-regressive):
probs = softmax(logits)      # NO Sinkhorn step
topk_probs, topk_indices = topk(probs, k=K)
```

术语一般如何实现？如何使用？
- **收敛性**：Sinkhorn 迭代收敛到满足双随机约束的矩阵，tolerance 0.01（本文设置）
- **CPT 中的行为**：SBTk 对分布偏移具有"固有鲁棒性"——分布偏移时 MRI 几乎不变（因为显式均衡步骤强制保证），但稳定状态的 MRI 高于 PBTk（均衡不如 PBTk 精细）
- **推理不兼容**：Sinkhorn 需要 batch-level 统计信息，与自回归生成不兼容。推理时 MRI 高于训练时 MRI
- **计算开销**：SBTk 的 forward 和 backward 时间均高于 PBTk（本文 Table 2：SB Granular MoE ~1789ms/step vs PB Granular MoE ~1680ms/step）

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---
