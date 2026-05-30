## Ada-K Routing

术语解释
Ada-K 是一种基于强化学习的动态 MoE 路由策略，通过可学习的轻量级 allocator 模块为每个 token 动态决定激活的专家数量，替代传统固定 Top-K 路由。

术语是什么？
Ada-K Routing 的核心架构：
1. **Allocator 模块**：在每个 MoE layer 插入一个与 router 同结构的轻量级线性层 W_alloc ∈ R^{C×N}，输入 token embedding x_i，输出专家数量概率分布 P_alloc(x_i) = Softmax(W_alloc · x_i)，通过采样 k* ~ P_alloc(x_i) 动态决定该 token 应激活的专家数量。
2. **PPO 训练框架**：由于采样不可微分，使用 Proximal Policy Optimization 端到端训练 allocator。每个 MoE layer 的 allocator 作为 agent (policy π_θ)，token 的 hidden state 作为 state，采样的专家数量作为 action，仅最后一层接收 reward = log P(x_i|x_1,...,x_{i-1})（语言模型对数似然）。Advantage 函数使用 reinforce with baseline 形式，以默认 Top-K 路由输出为 baseline。
3. **Warm-Start (P-Warm)**：使用 Top-P nucleus sampling 生成伪标签预训练 allocator，避免随机初始化导致的训练不稳定。
4. **可插拔设计**：allocator 与原始 router 独立，LLM 主干参数完全冻结，训练仅更新 allocator 参数。

从算法pipeline角度拆解术语。
```
# Ada-K Forward Pass (per token x_i, per layer l)
def adak_forward(x_i, W_router, W_alloc, experts):
    # x_i: [d_model], W_alloc: [d_model, N], W_router: [d_model, N]
    
    # Step 1: Allocator 决定专家数量
    P_alloc = Softmax(W_alloc @ x_i)        # [N] 各 k 值的概率
    k_star = Categorical(P_alloc).sample()   # 采样 (不可微分!)
    
    # Step 2: Router 选择 top-k* 专家
    P_router = Softmax(W_router @ x_i)      # [N] 专家概率
    top_indices = TopK(P_router, k_star)
    top_weights = Softmax(P_router[top_indices])
    
    # Step 3: 加权聚合专家输出
    output = sum(w * expert_j(x_i) for w, expert_j in zip(top_weights, top_indices))
    return output, k_star, log_prob

# Ada-K PPO Training
def ppo_training_step(token_batch, allocators, baseline_model):
    # Forward: 收集 actions 和 log_probs
    for layer in layers:
        for token in token_batch:
            out, k_star, log_prob_old = adak_forward(token, ...)
            save(k_star, log_prob_old)
    
    # Reward: 仅最后一层 (L = total layers)
    R = log P(token | context)                    # LM 对数似然 (Ada-K)
    R_baseline = log P_baseline(token | context)  # 默认 Top-K 输出
    
    # Advantage (reinforce with baseline)
    for layer l in 1..L:
        A_l = gamma^{L-l} * (R - R_baseline)
    
    # PPO Loss (2 PPO epochs)
    for layer l in 1..L:
        r = pi_theta(k_star | x_i) / pi_theta_old(k_star | x_i)
        L_RL = -min(r * A_l, clip(r, 1-eps, 1+eps) * A_l)
        
        # Regularization: 最小化期望专家数量
        L_reg = (1/L) * sum(n * P_theta_l(n) for n in 1..N)
        
        L_total = L_RL + lambda * L_reg
        theta_l = AdamW(L_total, lr=1e-3)
```

术语一般如何实现？如何使用？
- Allocator 等价于一个线性层 + SoftMax（与 router 同规模，约 C×N 参数），训练仅需 1M-3M 参数（vs 140B+ 总参数）
- 训练数据仅需 10k 样本，1 epoch，16 GPU (A800) 最慢 8 小时
- λ=3e-3 作为性能与效率的平衡点；调整 λ 可灵活控制 trade-off
- 完全可插拔：allocator 独立于原始 router 和 LLM 主干，可应用到任何 routing-based MoE 模型（包括 shared expert 架构）
- 保持负载均衡：router 冻结确保专家负载分布不变
- Allocator 可选择性部署（如仅 50% 层），以进一步减少训练开销
- 代码和 checkpoint 将发布于 https://github.com/ivattyue/Ada-K

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs

---
