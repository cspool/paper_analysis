## Layer-wise Reconstruction Loss for KV Cache (逐层重建损失)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

逐层重建损失是 CSKV 用于训练低秩 KV Cache 压缩矩阵的损失函数。传统端到端 fine-tuning 需在整个模型上做前向/反向传播；逐层重建损失在每层内部独立最小化 MSE：$L_K = \operatorname{MSE}(K, \hat{K})$，其中 $K = X W^K$，$\hat{K} = X A^K B^K$。总损失 $\mathcal{L}_{all} = \sum_{j=1}^{n_l} (L_{K,j} + L_{V,j})$。各层独立训练，可并行。仅需 90 分钟/单 A100-80G（AdamW, lr=5e-5, epoch=1, batch_size=1）。

从算法pipeline角度拆解术语。

```
for layer in model.layers:
    W_K, W_V = layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight
    A_K, B_K = ASVD_init(W_K, calib)  // SVD-based init
    A_V, B_V = ASVD_init(W_V, calib)
    for X in train_loader:
        K = X @ W_K.T; K_hat = X @ A_K.T @ B_K.T
        V = X @ W_V.T; V_hat = X @ A_V.T @ B_V.T
        loss = MSELoss(K, K_hat) + MSELoss(V, V_hat)
        loss.backward()  // 仅更新 A_K, B_K, A_V, B_V
        optimizer.step()
```

术语一般如何实现？如何使用？

适用于需压缩 KV Cache 但无法承受完整 retraining 的场景。逐层训练避免梯度跨层传播开销。数据量小（scaled-down Pile, epoch=1），泛化性好。适用于 LLaMA、Mistral 等标准 Transformer 架构。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

---
