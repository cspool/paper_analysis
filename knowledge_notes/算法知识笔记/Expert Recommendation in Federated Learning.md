## Expert Recommendation in Federated Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Recommendation 是 FedMoE Stage 2 中的动态子模型结构调整机制。当客户端连续多轮性能无提升（达到瓶颈），云端利用其他客户端作为参考，推荐增加高效 expert 或裁剪低效 expert。

核心流程：(1) 基于 expert 激活概率向量的 cosine similarity 找到 top-K 最相似客户端；(2) 若参考组平均 expert 数多于当前客户端，按参考组加权激活概率（Eq. 6）排序推荐引入外部 expert；否则推荐裁剪低效 expert；(3) 调整后验证，不改善则回退并固定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert Recommendation
for client u_k:
    if acc[u_k] stalled:
        # Cosine similarity (Eq. 4)
        sim(u_k, u_a) = Σ_i Σ_j p_{i,j}(u_k)·p_{i,j}(u_a) / (||p(u_k)||·||p(u_a)||)
        S' = top_K_by_similarity(sim, K)
        n = AVG(n_expert(S')) - n_expert(u_k)

        if n > 0:  # 增加 expert
            for expert e NOT in w_k:
                p_hat(e) = Σ_{a∈S'} sim(u_k,u_a)·p_e(u_a) / Σ_{a∈S'} sim(u_k,u_a)
            add top_n experts sorted by p_hat descending
        elif n < 0:  # 裁剪 expert
            for expert e in w_k:
                p_hat(e) = Σ_{a∈S'} sim(u_k,u_a)·p_e(u_a) / Σ_{a∈S'} sim(u_k,u_a)
            remove top_|n| experts sorted by p_hat ascending

        if adjusted model worse:
            revert and fix structure
```

Expert Recommendation 利用群体智慧指导个体结构调整，是一种"试探-验证-回退"的安全机制。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts
