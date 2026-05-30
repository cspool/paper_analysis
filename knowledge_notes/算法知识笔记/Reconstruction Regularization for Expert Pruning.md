## Reconstruction Regularization for Expert Pruning

术语解释
重建正则化（Reconstruction Regularization）是一种用于 MoE 专家剪枝的训练目标组件，定义为 Φ(α, β) = ∥F'(x; α, β) − F(x)∥_F，其中 F' 为应用 continuous relaxation 后的 pruned model 输出，F 为完整原始模型输出，∥·∥_F 为 Frobenius 范数。该正则项鼓励剪枝后模型的 token-level hidden states 与原始完整模型保持一致，相当于一种知识蒸馏的形式（无需单独的 teacher forward）。

术语是什么？
在 DiEP 中，Reconstruction Regularization 是与 Cross-Entropy Loss 共同优化的一项，总目标为 L = L_ce + λ·Φ。其作用机制：
1. 原始模型 F(x) 在每个 MoE 层使用全部专家的 FFN 输出加权求和
2. Pruned model F'(x) 使用 ᾱ_i^(l)（softmax 后的可学习重要性）和 β^(l) 对各专家输出加权
3. Φ 计算两者在 hidden state 空间的 Frobenius 距离
4. 梯度反向传播更新 α 和 β，使 F' 的中间表示逼近 F

λ 控制重建正则化的强度。论文在 Mixtral 架构上使用 λ=0.01。消融实验（Figure 8a）显示 λ∈{0.005, 0.01, 0.015, 0.02, 0.03} 中 λ=0.01 最优。

从算法pipeline角度拆解术语：
```
# Reconstruction Regularization computation
def reconstruction_loss(model_full, model_pruned, x, alpha, beta):
    """
    x: input hidden states [batch, seq_len, d_model]
    alpha: intra-layer scores [L, N]
    beta: inter-layer scores [L]
    """
    h_full = x
    h_pruned = x
    total_loss = 0.0
    
    for layer l in range(L):
        # Full model forward (all experts, uniform routing)
        h_full = full_moe_layer(h_full, l)
        
        # Pruned model forward (α-weighted experts)
        h_pruned_layer = 0
        for expert i in range(N):
            expert_out = FFN_i(h_pruned)
            h_pruned_layer += softmax(alpha[l])[i] * expert_out
        h_pruned = beta[l] * h_pruned_layer + h_pruned  # residual
        
        # Layer-wise or end-to-end
        total_loss += frobenius_norm(h_pruned - h_full)
    
    return total_loss

# Full objective
loss_total = cross_entropy(y_pred, y_true) + lambda_reg * total_loss
```

术语一般如何实现？如何使用？
- 计算开销：需额外完整模型前向传播一次（或预先缓存完整模型的 hidden states）
- 校准数据仅需 128 samples，正则化使得在小样本下也不会过拟合
- 类似于 Knowledge Distillation 的 feature-level alignment，但不依赖 teacher soft labels
- 可视为一种 self-distillation：完整模型自身作为 teacher，pruned version 作为 student
- 论文附录验证：即使只用 32 calibration samples，DiEP 也能避免性能崩溃（归因于 reconstruction regularization 的约束作用）

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

---
