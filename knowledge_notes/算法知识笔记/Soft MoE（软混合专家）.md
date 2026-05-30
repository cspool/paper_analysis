## Soft MoE（软混合专家）

术语是什么？
Soft MoE（Soft Mixture of Experts）是一种隐式软分配的 MoE routing 方法，由 Puigcerver et al. (ICLR 2023) 提出。与 Top-K routing 的硬选择不同，Soft MoE 将每个 expert 处理一个 "slot"——所有输入 token 的加权平均，每个 slot 是一个 learnable 的 token 组合。具体地，通过 learnable parameters Φ ∈ R^{d×(e·s)} 将 m 个输入 token 映射到 e×s 个 slot：X̃ = softmax(XΦ)^T X。每个 expert 处理 s 个 slot，输出通过 softmax(XΦ) 的转置重新组合回 m 个 token。这种方法避免了 token dropping 和 load imbalance 问题。

从算法pipeline角度拆解术语：
Soft MoE 前向传播：
```
# X: (m, d) 输入 token
# Phi: (d, e*s) learnable routing parameters
# experts: e 个 MLP, 每个处理 s 个 slot

# 1. Token → Slot 映射
slots = X @ Phi                                      # (m, e*s)
dispatch_weights = softmax(slots, dim=1)              # (m, e*s)
X_tilde = dispatch_weights.T @ X                      # (e*s, d) 加权 slot

# 2. 每个 expert 独立处理其 slot
for i in range(e):
    Y_tilde[i*s:(i+1)*s] = expert_i(X_tilde[i*s:(i+1)*s])

# 3. Slot → Token 重组
combine_weights = softmax(slots, dim=0)               # (m, e*s)
Y = combine_weights @ Y_tilde                         # (m, d)
```

术语一般如何实现？如何使用？
作为 ViT 或 ConvNeXt 中 MLP 层的替代。在 timm/MMPretrain 框架中使用。每个 MoE 层的 Φ 与 expert MLP 权重共同训练。slot 数 s 通常设为 1（每个 expert 1 个 slot）。作为 MoE Jetpack 论文的 baseline，直接用 Soft MoE 替换 dense ViT 的后半 MLP 层进行 from-scratch 训练。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks
