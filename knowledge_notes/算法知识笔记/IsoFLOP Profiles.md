## IsoFLOP Profiles

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

IsoFLOP Profiles（等计算量曲线）是 Scaling Laws 研究中用于可视化和分析模型性能的核心工具。在固定的训练 FLOPs budget F 下，绘制 loss L 随 model size N_act 的变化曲线——曲线上每一点的 (N_act, D) 满足 6·N_act·D = F。曲线的最低点对应 compute-optimal 配置 (N_act_opt, D_opt)。多条曲线的包络线展示 scaling 的整体趋势。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Joint MoE Scaling Laws 中 IsoFLOP Profiles 的生成：

```
给定: FLOPs budgets F ∈ {10^19, 10^20, 10^21, 10^22}
      scaling law L(N_act, D, E) = m(E)·N_act^μ(E) + n(E)·D^ν(E) + c

for each F:
    for N_act in reasonable_range:
        D = F / (6·N_act)  # token 数自动确定
        L = m(E)·N_act^μ(E) + n(E)·D^ν(E) + c
        plot(N_act, L)
    # compute-optimal point:
    G = (μ·m/(ν·n))^(1/(μ+ν))
    N_act_opt = G·(F/6)^(ν/(μ+ν)), D_opt = G^(-1)·(F/6)^(μ/(μ+ν))
```

关键观察（Fig.2a）：E=1 在较大 N_act 处最优，E=8 在较小 N_act 处最优——反映 ν(E) 更负时数据需求更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- IsoFLOP Profiles 可跨 E 值比较，揭示 FLOPs savings（Fig.2b: E=4 在 10^20 FLOPS 节省 40% vs dense）
- 曲线形状受 μ(E) 和 ν(E) 影响：μ 决定 small-model 侧陡峭度（underfitting），ν 决定 large-model 侧陡峭度（overtraining）
- 在 memory-constrained 场景中被 N_total ≤ M 截断，截断后最低点对应 memory-optimal 配置

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
