## Bidirectional Routing (TCR+ECR Hybrid) / Affinity-Driven Active Selection

术语解释
Bidirectional Routing 是 ETR 的核心创新：在 MoE 路由中同时使用 TCR 和 ECR，让 token 和 expert 双向主动选择，形成"共振效应"。整个流程由 cosine similarity 亲和力分数统一驱动 (Affinity-Driven Active Selection)。

术语是什么？
ETR 双向路由分两阶段: (1) TCR: 每个 token 按 GrAP 亲和力分数 δ_{t,i} = cos(x_t, w_i) 选 top-ℓ experts; (2) ECR: 每个 expert i 从已分配 token 中按其 δ 选 top-C tokens (Bottom-C 保留最高分数)。动态过渡：早期训练 TCR 更优 (q_i ≈ Θ(1))，后期 ECR 更优 (q_i << 1, 接近 100% 成功率)。Theorem 5 提供了全程最大化成功率的最优过渡策略的理论依据。

从算法pipeline角度拆解：
```
def ETR_bidirectional(x, W_aff, k, C):
    # Step 1: 亲和力分数
    delta = cosine_similarity(x, W_aff)  # s×n

    # Step 2: TCR — token 选 top-k experts
    tcr_assign = defaultdict(list)
    for t in range(s):
        for expert_id in TopK(delta[t, :], k):
            tcr_assign[expert_id].append(t)

    # Step 3: ECR — expert 选 top-C tokens
    ecr_assign = {}
    for i in range(n):
        candidates = tcr_assign[i]
        if len(candidates) <= C:
            ecr_assign[i] = candidates
        else:
            scores = [delta[t, i] for t in candidates]
            ecr_assign[i] = BottomC(candidates, scores, c=C)

    return ecr_assign
```

术语一般如何实现？如何使用？
在 Ascend NPU 上通过 MindSpeed-LLM 实现。TCR 阶段用 TopK 算子，ECR 阶段用 BottomC/IndexPutV2 做 token rearrangement。引入的 TopK/IndexPutV2 开销较小，但使 FFN MatMul 获 17× 加速 (相对 baseline)，因为只计算高亲和力 token-expert 对。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection
