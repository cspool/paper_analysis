## Stochastic Routing Warmup

术语解释
一种 MoE 训练早期的路由稳定性机制。在训练初期以线性衰减的权重将受控随机噪声混入路由器 logits，迫使 token 均匀分布到所有 expert，防止路由初始化导致的个别 expert 过载或崩溃。由 Ling 团队提出。

术语是什么？
MoE 训练早期（尤其是 fine-grained experts 场景）路由器随机初始化会导致 token 分布极度不均衡——某些 expert 接收远多于容量的 token，造成 OOM 或 expert 负载崩溃。该机制通过混合随机噪声和 learned logits 渐进过渡解决此问题。

公式：ŝ_t = α · s_t + (1 - α) · (μ_s + σ_s · ε)，α = min(i/W, 1.0)，ε ~ N(0, I)。s_t 为路由器线性投影的原始 logits，μ_s/σ_s 为运行时均值和标准差，W 为预热步数。

从算法pipeline角度拆解术语：
```
def moe_warmup_forward(x, step, W):
    s_t = router_linear(x)              # [batch, N_experts]
    if step <= W:
        alpha = step / W
        mu_s, sigma_s = running_mean(s_t), running_std(s_t)
        s_t = alpha * s_t + (1-alpha) * (mu_s + sigma_s * randn_like(s_t))
    return TopK(SoftMax(s_t), k=top_k)
```
α=0 时路由完全随机（所有 expert 等概率），α=1 时完全由学习到的分布控制。

术语一般如何实现？如何使用？
- 预热步数 W 为超参数；μ_s/σ_s 通过 EMA 维护
- 与 dropless routing + load balance loss + z-loss 联合使用
- 可在训练早期完全消除 expert 崩溃问题

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---
