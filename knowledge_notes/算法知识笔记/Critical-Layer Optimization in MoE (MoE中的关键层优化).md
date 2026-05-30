## Critical-Layer Optimization in MoE (MoE中的关键层优化)

术语解释
Critical-Layer Optimization 是 C3PO 的优化策略之一：在 pathway 优化时只修改 MoE 模型中部分"关键层"的 routing weights，而非全部层。实验发现只优化最后 5 层的 routing weights 不仅节省计算，而且性能反超全 16 层优化（OLMoE: L5 79.2% vs All16 77.7%）。

术语是什么？
C3PO 的层重要性分析揭示三层 hierarchy（OLMoE 16 层）：
- **深层 (Late/L)**: 最重要，负责任务特定的高层语义理解
- **浅层 (Early/F)**: 次重要，编码基础特征表示
- **中层 (Middle/M)**: 过渡角色，对最终预测影响最小

规律：M1 < F1 < L1, M2 < F2 < L2, M5 < F5 < L5

从算法pipeline角度拆解术语：
```
def extract_critical_layers(all_routing_weights, strategy="last_5"):
    critical_layers = {"last_5": [12,13,14,15,16], "first_2_last_3": [1,2,14,15,16]}
    ω_opt = {l: all_routing_weights[l] for l in critical_layers[strategy]}
    ω_frozen = {l: all_routing_weights[l] for l not in critical_layers[strategy]}
    return ω_opt, ω_frozen
```

术语一般如何实现？如何使用？
- 层选择策略需在目标 MoE 模型上通过 ablation 验证
- OLMoE 最优: 最后 5/16 层；DeepSeekMoE (28层) 论文未单独报告
- 原则：深层 > 浅层 > 中层，组合时应优先包含深层

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
