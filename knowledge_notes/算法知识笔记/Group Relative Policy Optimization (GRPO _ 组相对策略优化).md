## Group Relative Policy Optimization (GRPO / 组相对策略优化)

术语解释
Group Relative Policy Optimization (GRPO) 是 DeepSeek 系列使用的 LLM 强化学习对齐方法（Shao et al. 2024, DeepSeekMath），无需训练与 policy model 同规模的 critic model。对每个 question q，从旧策略采样一组 G 个输出 {o_i}，以组内奖励的均值和标准差归一化得到 advantage A_i = (r_i - mean(r))/std(r)，使用 PPO 风格的 clipped objective 优化。

术语是什么？
GRPO 的 objective：J(θ) = E[ (1/G) Σ_i min(ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i) - β * D_KL(π_θ||π_ref) ]，其中 D_KL = π_ref/π_θ - log(π_ref/π_θ) - 1（unbiased estimator）。

从算法pipeline角度拆解术语：
```
=== GRPO Training Step ===

for each batch of questions Q:
    for each question q in Q:
        {o_1, ..., o_G} ~ π_θ_old(·|q)         // sample G outputs
        
        for each o_i:
            r_i = RM(q, o_i)                     // rule-based or model-based RM
        
        A_i = (r_i - mean({r})) / std({r})       // group-relative advantage
    
    θ = θ + η * ∇_θ J_GRPO(θ)                   // PPO-style update without critic
```

术语一般如何实现？如何使用？
DeepSeek-V3 使用 GRPO 进行 post-training RL 对齐，结合 rule-based RM（数学确定性答案、LeetCode compiler feedback）和 model-based RM（从 DeepSeek-V3 SFT checkpoints 训练，含 chain-of-thought reward reasoning）。RL 数据覆盖 coding、math、writing、role-playing、QA 等多 domain。GRPO 优势：(1) 消除 critic model 的显存和训练开销（critic 通常与 policy 同规模）；(2) group-relative advantage 自动归一化奖励尺度，无需额外 reward normalization。DeepSeek-V2 和 DeepSeekMath 也使用 GRPO。

涉及论文标题：
- DeepSeek-V3 Technical Report
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
