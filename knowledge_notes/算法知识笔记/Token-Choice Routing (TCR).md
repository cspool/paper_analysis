## Token-Choice Routing (TCR)

术语解释
Token-Choice Routing 是 MoE 中最基础的路由策略：每个 token 独立选择 top-k 个 expert 进行处理，由 Shazeer et al. (2017) 在 Sparsely-Gated MoE 中首次提出。

术语是什么？
TCR 流程: (1) Router 对每个 token x_t 计算 gate scores; (2) 取 top-k (通常 k=1 或 2) 最高分 expert; (3) token 被 dispatch 至选中 expert 进行 FFN 计算。TCR 优势：token 有充分自由选择最适合的 expert。缺陷：load imbalance——某些 expert 收到远多于 capacity 的 token。ETR 论文证明 (Theorem 5)：早期训练阶段 (class-irrelevant token 呈各向同性分布，q_i = Θ(1))，TCR 训练成功率为 Θ(C·Σp_i/s)，显著优于 ECR 的指数衰减率 e^{-s}。

从算法pipeline角度拆解：
```
def TCR_route(scores, k, C):
    # scores: s×n, k: top-k, C: expert capacity
    topk_val, topk_idx = TopK(scores, k)  # s×k
    capacity = zeros(n)
    dispatch = {i: [] for i in range(n)}
    for t in range(s):
        for expert_id in topk_idx[t]:
            if capacity[expert_id] < C:
                dispatch[expert_id].append(t)
                capacity[expert_id] += 1
            # else: token dropped → residual bypass
    return dispatch
```

术语一般如何实现？如何使用？
TCR 在现代 MoE 框架 (Megatron-LM、DeepSpeed-MoE、MindSpeed-LLM) 中广泛实现。使用 All-to-All 通信在 Expert Parallelism 维度上完成 token dispatch/combine。是 GShard、Switch Transformer、Mixtral、DeepSeek-V3 的默认路由策略。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection
