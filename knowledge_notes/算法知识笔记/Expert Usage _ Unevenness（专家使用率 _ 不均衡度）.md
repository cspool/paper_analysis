## Expert Usage / Unevenness（专家使用率 / 不均衡度）

术语是什么？
Expert Usage 和 Unevenness 是 PKM (Lample et al., 2019) 提出的 MoE 专家利用评估指标，PEER 沿用这些指标评估百万级 expert 的利用效率。给定验证集上所有 token 的路由分数累积：z_i = Σ_x g_i(x)（expert i 在所有 token 上的 router score 之和），定义：(1) Expert Usage = 被至少一个 token 选中的 expert 比例：#{i | z_i ≠ 0} / N；(2) Unevenness = z 分布与均匀分布之间的 KL 散度：log(N) + Σ_i (z_i / ||z||₁) log(z_i / ||z||₁)。Unevenness 越小表示 expert 使用越均衡，0 表示完全均匀。

从算法pipeline角度拆解术语：
评估计算过程：
```
# 遍历验证集所有 token，累积每个 expert 的 router score
z = zeros(N)                     # 初始化
for each token x in validation_set:
    indices, scores = peer_forward.get_indices_and_scores(x)
    for (i, s) in zip(indices, scores):
        z[i] += s                # 累积 router score

p = z / sum(z)                   # 归一化为概率分布

# Expert Usage: 被使用的 expert 比例
usage = count(p > 0) / N

# Unevenness: KL(p || uniform)
uniform = ones(N) / N
unevenness = KL(p || uniform) = sum(p_i * log(p_i / (1/N)))
           = log(N) + sum(p_i * log(p_i))
```

术语一般如何实现？
PEER 在 C4 验证集上评估所有 expert usage 指标。实验表明：即使 N=1M，expert usage 接近 100%（使用 query BN 时 100%，不使用 BN 时 99.8%）。Unevenness 随 N 增大而上升（16K: 0.30→1M: 1.06 with BN），表明更大 expert 池中负载均衡更具挑战性，但 BN 可将 unevenness 控制在可接受范围。这些指标仅评估覆盖度（哪些 expert 被使用），不评估 expert 是否学到了有意义的专业化功能。

涉及论文标题：
- Mixture of A Million Experts
