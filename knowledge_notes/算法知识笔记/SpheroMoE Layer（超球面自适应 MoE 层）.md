## SpheroMoE Layer（超球面自适应 MoE 层）

术语是什么？
SpheroMoE（Hyperspherical Adaptive MoE）Layer 是 MoE Jetpack 框架中的 MoE 层架构，专为 fine-tuning 从 dense checkpoint 初始化的 MoE 模型而设计。它由三个子组件构成：(1) SpheroMoE Routing：基于 cross-attention 的超球面路由；(2) Expert Regularization：防止 expert 过度特化的正则化策略；(3) Adaptive Dual-path MoE：双路径 expert 结构（core experts + universal experts）。

从算法pipeline角度拆解术语：
SpheroMoE 层的前向传播流程（对应论文 Algorithm 1）：

```
def spheromoe_layer(X, Q, T, core_experts, univ_experts):
    # X: (b, n, d) 输入 token
    # Q: (e*s, d) 随机初始化的查询向量
    
    # 1. 继承 dense checkpoint 的 LayerNorm，保证分布一致性
    X_norm = inherit_layer_norm(X, dim=-1)           # (b, n, d)
    Q_norm = l2_norm(inherit_layer_norm(Q, dim=-1))   # (e*s, d), 超球面投影
    
    # 2. Key 投影
    K = linear(X_norm, W_k)                           # (b, n, d)
    
    # 3. 超球面相似度（L2-norm Q 与 K 做点积 = cosine similarity）
    S = einsum(K, Q_norm, "b n d, e s d -> b n e s")  # (b, n, e, s)
    
    # 4. Expert Regularization
    S = S + normal_noise(S) * noise_mult               # 加噪声
    dispatch = softmax(S / T, dim=1)                   # token→slot 分配
    combine = softmax(S / T, dim=[-1,-2])               # slot→token 重组
    
    # 5. Token 分发
    X_hat = einsum(dispatch, X_norm, "b n e s, b n d -> b e s d")
    X_core = X_hat[:, :core_num, :, :]                 # core expert slot
    X_univ = X_hat[:, core_num:, :, :]                 # universal expert slot
    
    # 6. 并行 Expert 前向（合并所有权重为单一大矩阵，一次 einsum）
    Y_core = parallel_expert_forward(X_core, core_experts)
    Y_univ = parallel_expert_forward(X_univ, univ_experts)
    Y_hat = concat([Y_core, Y_univ], dim=1)
    
    # 7. Stochastic Expert Dropout + 输出重组
    Y_hat = expert_dropout(Y_hat, p)
    Y = einsum(combine, Y_hat, "b n e s, b e s d -> b n d")
    return Y
```

术语一般如何实现？如何使用？
在 PyTorch 中实现为替换 ViT/ConvNeXt 后半层 MLP 的 MoE 模块。Q 随机初始化，与模型参数共同训练。LayerNorm 直接从 dense checkpoint 继承且固定。并行 expert 前向通过将 e 个 expert 的权重矩阵在 batch 维度合并（shape e×d2×d1），单次 einsum 完成所有 slot 的并行计算。使用 AdamW fine-tuning，总体 FLOPs 与原始 dense 模型相当。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks
