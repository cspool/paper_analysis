## Neighborhood Gradient Descent for MoE Pathway (邻域梯度下降MoE路径优化)

术语解释
Neighborhood Gradient Descent (NGD) 是 C3PO 中性能最强的 pathway 优化方法。它不直接使用测试样本的 ground truth（未知），而是用参考集中 kNN 邻居样本的 loss 加权平均作为 surrogate objective，对 routing weights 做梯度下降。NGD 达到 Oracle 性能的 85-95%，是 C3PO 三种方法中唯一需要反向传播的方法。

术语是什么？
NGD 的核心公式：

$$L(\omega) = \frac{\sum_{i \in \mathcal{N}(x)} K(x_i, x) \cdot \ell(f(x_i, \omega), y_i)}{\sum_{i \in \mathcal{N}(x)} K(x_i, x)}$$

其中 N(x) 是 x 的 k=3 个最近邻（基于 embedding 相似度），K(x_i, x) = exp(-||E(x_i) - E(x)||^2 / (2σ^2)) 是 Gaussian kernel，ℓ(f(x_i, ω), y_i) 是邻居样本在当前优化中的 ω 下的 cross-entropy loss。

关键洞察：虽然测试样本 x 的 ground truth 未知，但邻居样本的 ground truth 已知。如果 ω 能让邻居样本的 loss 降低，那么它很可能也能让 x 的输出变好。

从算法pipeline角度拆解术语：
```
def NGD_optimize(x, model, ref_set, k=3, steps=10):
    emb_x = embedding_model(x)
    neighbors = ref_set.knn(emb_x, k=3)
    
    ω = model.get_routing_weights(x)[last_5_layers][:, top_20_experts]
    K_vals = [exp(-||emb_x - emb_xi||^2 / (2*h^2)) for xi in neighbors]
    K_sum = sum(K_vals)
    
    optimizer = SGD([ω], lr=1e-2)
    scheduler = CosineAnnealing(optimizer, T_max=10, eta_min=1e-5)
    
    for step in range(steps):
        total_loss = 0
        for (xi, yi), K_val in zip(neighbors, K_vals):
            logits_i = model.forward(xi, routing_override=ω)
            total_loss += (K_val / K_sum) * cross_entropy(logits_i, yi)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return ω
```

术语一般如何实现？如何使用？
- 需要对 MoE 模型的 routing weights 可微（梯度可以从 loss 回传到 ω）
- 在 HuggingFace 实现中，需修改 MoE 层的 forward 使 routing weights 成为可优化的 nn.Parameter
- 每个测试样本独立运行 NGD，不共享状态
- FLOPs 开销：主要在邻居样本的前向传播（k=3 × steps=10 = 30 次前向）

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
