## Mode Finding / Meanshift for MoE Routing (MoE路由的模式发现/均值漂移)

术语解释
Mode Finding (Meanshift) 是 C3PO 中的梯度自由 pathway 优化方法。它借鉴均值漂移（Mean Shift）聚类算法的思想：在 expert pathway 的权重空间（ω-space）中，将测试样本的 pathway 向邻居样本 pathway 的最密集区域（mode）迭代移动。不需要反向传播，计算开销最小。

术语是什么？
Mode Finding 的更新公式：

$$\bar{\omega} = \frac{\sum_{i \in \mathcal{N}(\omega)} K(\omega_i, \omega) \cdot \omega_i}{\sum_{i \in \mathcal{N}(\omega)} K(\omega_i, \omega)}$$

$$\omega \leftarrow \alpha \cdot \omega + (1-\alpha) \cdot \bar{\omega}$$

关键区别：NGD 在 x-space 中定义邻居并在 loss space 中优化；Mode Finding 在 ω-space 中定义邻居并在 ω-space 中做 meanshift。Kernel Regression 在 x-space 中定义邻居并在 ω-space 中做加权平均。

从算法pipeline角度拆解术语：
```
def mode_finding(ω_curr, ref_pathways, bandwidth, alpha=0.5, max_iter=5):
    ω = ω_curr.clone()
    for t in range(max_iter):
        distances = [||ω - ω_i||^2 for ω_i in ref_pathways]
        K_vals = [exp(-d^2 / (2 * bandwidth^2)) for d in distances]
        ω_bar = sum(K_i * ω_i for K_i, ω_i in zip(K_vals, ref_pathways)) / sum(K_vals)
        ω = alpha * ω + (1 - alpha) * ω_bar
    return ω
```

| 特性 | Mode Finding | Kernel Regression | NGD |
|------|-------------|-------------------|-----|
| 需要梯度 | 否 | 否 | 是 |
| 邻居空间 | ω-space | x-space | x-space |
| 优化空间 | ω-space | ω-space | loss space |
| 性能 (OLMoE avg) | 72.4% | 76.9% | 79.2% |

术语一般如何实现？如何使用？
- 梯度自由，适合资源受限场景或快速原型验证
- 需为参考集中每个样本存储其成功的 pathway 矩阵（存储开销：|D_ref| × L × E × 4 bytes）
- bandwidth 参数控制 ω-space 中邻居的影响范围

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
