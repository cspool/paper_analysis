## Segment Routing Best Performance (SRP / 分段路由最优性能)

术语是什么？
SRP 是衡量 MoE 模型中 expert 或 expert group 局部路由一致性的无参数量化指标 (ICLR 2026)。定义：segment-based estimator R_e^m 对长度 m 的 segment 统一预测（全激活或不激活），在所有可能 segment 上的最大 F1 分数。数学证明 F1 最大化当且仅当 estimator 对所有 f ≥ α_e^m 的 segment 给出激活预测，α_e^m ∈ [0,m] 是 expert e 和 m 的唯一函数——因此 SRP 是 expert 的固有属性，与具体路由方法无关。辅助指标 ρ̂（segment routing size ratio）= 最佳预测所需激活 expert 数 / 原始激活数，越小说明局部一致性越强。

从算法pipeline角度拆解术语：
```
# Single-expert SRP (Eq.4)
for α in [0..m]:
    TP = Σ f  for {f >= α};  FP = Σ (m-f) for {f >= α}
    FN = Σ f  for {f < α}
    F1 = 2*TP / (2*TP + FP + FN)
SRP(e, m) = max_α F1

# Expert group SRP (Eq.5-6, 联合优化所有 expert)
# 决策空间: ∀e ∈ E, 对 f[e,T,p,m] >= α_e^m 的 segment 激活 expert e
# α_e^m 由 group E 和 m 联合决定
```

术语一般如何实现？如何使用？
论文实现：收集 20 个 MoE 模型在 22,528 样本上的路由决策，统计 per-expert per-segment 激活频率，搜索 α 计算 SRP。由于不同位置 segment 的 SRP 几乎恒定 (Appendix E.2)，所有位置统一计算。代码开源 https://github.com/ljcleo/moe-lrc。

涉及论文标题：
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models
