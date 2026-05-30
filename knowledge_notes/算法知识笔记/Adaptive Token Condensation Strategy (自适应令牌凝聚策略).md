## Adaptive Token Condensation Strategy (自适应令牌凝聚策略)

术语解释
Adaptive Token Condensation Strategy 是 LUFFY 中用于动态调整 Token Condensation 相似度阈值 h_t 的策略。它根据训练收敛状态自动平衡通信效率与训练稳定性：训练早期使用高阈值保留更多 token 以保证收敛，训练后期降低阈值以最大化通信节省。

术语是什么？
固定阈值的问题：h=0.3 时 MoE-BERT-Large 的 F1 从 90.82 降至 85.41（显著精度损失），h=0.8 时 F1 为 88.29（仍低于 Vanilla 的 90.82）。自适应策略使用 sigmoid 函数将归一化 loss 下降量映射为阈值：

$$h_t = \frac{1}{1 + \exp(l_{norm})}, \quad l_{norm} = \frac{l_{ini} - l_{t-1}}{l_{ini}}$$

其中 l_{ini} 是第一个 training iteration 的 loss，l_{t-1} 是前一个 iteration 的 loss。

从算法pipeline角度拆解术语：
```
Algorithm: Adaptive Threshold Computation

在每个 training iteration t:
    l_norm = (loss[0] - loss[t-1]) / loss[0]
    h_t = 1.0 / (1.0 + exp(l_norm))
    
    # 行为分析:
    # t=0:    l_norm ≈ 0    → h_t ≈ 0.73  (保留 ~73% token)
    # t=mid:  l_norm ≈ 0.5  → h_t ≈ 0.38  (凝聚更多)
    # t=late: l_norm ≈ 2.0  → h_t ≈ 0.12  (大量凝聚)

    # 在 Token Condensation 中使用:
    for each subgraph in G:
        # 删除 weight < h_t 的边
        keep edges where weight >= h_t
        # 连通分量分析
        for each component:
            rep = token with max degree
            condense all others into rep
```

术语一般如何实现？如何使用？
- 使用指数函数（sigmoid）确保当 loss 下降趋于平缓（训练稳定期）时阈值变化也是平滑的
- 无需额外超参数调优，完全由 loss 信号驱动
- 可与任何基于相似度阈值的 token 选择/剪枝策略结合
- 适用于训练阶段；推理阶段无 loss 信号，需替代策略

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---
