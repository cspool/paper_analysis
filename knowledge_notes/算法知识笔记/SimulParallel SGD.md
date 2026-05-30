## SimulParallel SGD

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SimulParallel SGD（Simultaneous Parallel SGD，Zinkevich et al., 2011）是一种分布式训练优化框架，在多个计算节点上独立训练模型副本，每个副本使用互不相交的数据子集，训练完成后通过参数平均聚合。MoE-DisCo 受此启发，将其视为 MoE 的极端情况（uniform gating + All-K averaged output），并据此设计 expert 级分块训练。两个关键洞见被采用：(1) 最大化数据子集间分布差异可加速收敛并提升集成效果——通过 K-Means 聚类实现；(2) 数据均衡时简单参数平均可逼近全局最优，不均衡时需加权平均（即 WP-SGD）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SimulParallel SGD 核心流程
for k in 1..K:                      # K 个 worker，完全并行
    Θ^(k) = Θ_init                  # 复制初始参数
    for batch in D_k:               # 互不相交的数据子集
        Θ^(k) = Θ^(k) - η · ∇L(Θ^(k), batch)

# 聚合：简单平均（数据均衡）
Θ_final = (1/K) · Σ_{k=1}^{K} Θ^(k)

# 聚合：WP-SGD 加权（数据不均衡）
Θ_final = Σ_{k=1}^{K} (|D_k|/|D|) · Θ^(k)
```

MoE-DisCo 将此框架映射到 MoE：每个 worker 对应一个 expert 子模型 + 其 K-Means 数据子集。共享参数 θ_shared 按 WP-SGD 加权平均融合，expert 参数直接拼接（保持专业化差异）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SimulParallel SGD 为 MoE-DisCo 的两阶段训练设计提供理论保证：不同的 K-Means 数据簇天然最大化分布差异，实现 expert 专业化。消融实验确认聚类必要性——随机数据分配使 fine-tune 性能退化至 Full-Parameter 水平。框架的完全去中心化特性确保子模型训练期间零通信开销，仅需本地操作。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models
