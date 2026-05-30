## Kernel Regression for MoE Pathway Estimation (MoE路径的核回归估计)

术语解释
Kernel Regression for Pathway 是 C3PO 中的梯度自由 pathway 优化方法。它在样本嵌入空间（x-space）中用 Gaussian kernel 加权平均邻居的 pathway 矩阵，得到目标 pathway 的估计值，然后通过最优插值系数 α* 平衡估计值与原始 pathway。

术语是什么：

$$\hat{\omega} = \frac{\sum_{i \in \mathcal{N}(x)} K(x_i, x) \cdot \omega_i}{\sum_{i \in \mathcal{N}(x)} K(x_i, x)}$$

$$\omega \leftarrow \alpha^* \cdot \omega + (1-\alpha^*) \cdot \hat{\omega}$$

其中 α* 通过在邻居样本上最小化 surrogate loss 搜索得到：

$$\alpha^* = \arg\min_{\alpha} L(\alpha \cdot \omega + (1-\alpha) \cdot \hat{\omega})$$

核心思想：如果测试样本 x 与参考样本 x_i 在语义上相似，那么它们的 optimal pathway 也应该相似。

从算法pipeline角度拆解术语：
```
def kernel_regression_pathway(x, model, ref_set, k=3):
    emb_x = embedding_model(x)
    neighbors = ref_set.knn(emb_x, k=3)
    K_vals = [exp(-||emb_x - emb_xi||^2 / (2*h^2)) for xi in neighbors]
    
    # 核加权平均
    ω_hat = sum(K_i * ω_i for K_i, ω_i in zip(K_vals, neighbors.omegas)) / sum(K_vals)
    
    # 搜索最优 α
    ω_curr = model.get_routing_weights(x)
    best_alpha = min(range(0, 11), key=lambda a:
        sum(K_i * loss(model(xi, a*0.1*ω_curr + (1-a*0.1)*ω_hat), yi) 
            for xi, yi, K_i in zip(neighbors.x, neighbors.y, K_vals)))
    
    return best_alpha*0.1 * ω_curr + (1-best_alpha*0.1) * ω_hat
```

Kernel 选择的影响（OLMoE）：Linear 69.95%, Polynomial 73.33%, Matern 76.28%, Gaussian 79.20%。

术语一般如何实现？如何使用？
- 梯度自由，计算成本低于 NGD
- 性能介于 Mode Finding (72.4%) 和 NGD (79.2%) 之间
- α* 搜索粒度 0.1，影响最终精度

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
