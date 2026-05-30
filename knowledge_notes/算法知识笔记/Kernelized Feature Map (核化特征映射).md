## Kernelized Feature Map (核化特征映射)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Kernelized Feature Map φ(·) 是 linear attention 中将 softmax kernel 替换为正定特征映射的数学工具。Softmax attention 中 Sim(Q_i, K_j) = exp(Q_i K_j^T/√d) 不可分解为独立 feature 的内积。Linear attention 通过选取合适的 φ(·) 使得 Sim(Q_i, K_j) ≈ φ(Q_i) φ(K_j)^T，从而可用结合律将计算顺序从 (Q K^T) V 改为 Q (K^T V)，将复杂度从 O(N²d) 降至 O(Nd²)。

常用的 φ(·) 包括：
- φ(x) = elu(x) + 1（保证正值，最常用）
- φ(x) = ReLU(x)（MHLA 论文中在图像任务上使用）
- φ(x) = 1 + tanh(x)（某些变体）
- 可学习 kernel（如 Rebased attention）

从算法pipeline角度拆解术语。

```
// Kernelized feature map 在 linear attention 中的作用
Q, K = X @ W_Q, X @ W_K          // [N, d]
Q_tilde = phi(Q)                  // 例如: Q_tilde = ReLU(Q)
K_tilde = phi(K)                  // K_tilde = ReLU(K)

// 此时: Q_tilde @ K_tilde^T 近似 exp(Q @ K^T/sqrt(d))

// Right-product trick:
// O = (Q_tilde @ K_tilde^T) @ V   [N, N] @ [N, d]  ← O(N²d)
//   = Q_tilde @ (K_tilde^T @ V)   [N, d] @ [d, d]  ← O(Nd²)
```

关键约束：φ(·) 必须输出正值以保证注意力权重的非负性，避免除零问题。MHLA 论文中，NLP 任务上推荐省略 normalizer 项（q̃^T z̃_i）以提高长序列下的训练稳定性。

术语一般如何实现？如何使用？

纯 PyTorch 操作，直接应用激活函数于 Q 和 K 张量。在 GPU 上无额外开销。选择具体 φ(·) 影响模型性能，一般推荐 elu(x)+1（平衡正值性和梯度流）。代码中通常实现为：`Q_tilde = F.elu(Q) + 1; K_tilde = F.elu(K) + 1` 或 `Q_tilde = F.relu(Q); K_tilde = F.relu(K)`。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---
