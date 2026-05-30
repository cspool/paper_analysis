## Pre-quantization Transformation（预量化变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pre-quantization Transformation（预量化变换）是在量化之前对权重和/或激活应用的数学变换，目的是消除离群值（outliers）、平滑分布、降低量化误差。变换必须满足**等价性**：变换后的矩阵乘法输出与原输出相同，即 Y = XW^T = (XT)(T^{-1}W^T)，其中 T 为可逆变换矩阵。变换矩阵 T 作用于激活侧的 TX 可在线计算，T^{-1}W^T 作用于权重侧可离线预计算并融合到量化权重中。常见的预量化变换包括：(1) Per-channel Scaling（对角变换，T=diag(c)）；(2) Hadamard Transformation（T=H，H∈{+1,-1}^{n×n}，正交矩阵）；(3) 可学习正交旋转（SpinQuant）；(4) Kronecker 仿射变换（FlatQuant，T=P₁⊗P₂）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 FlatQuant 的预量化变换为例，在线性层 Y=XW^T 中：

```
// 权重侧（离线预计算，融合到量化权重中）
W̃ = reshape(W, [m, n₁, n₂])              // n = n₁ × n₂
W' = P₁^{-1} ×₁ W̃ ×₂ (P₂^{-1})^T         // 逆变换，维度 [m, n₁, n₂]
W_q = Q(W')                               // 量化到 INT4

// 激活侧（在线推理）
X̃ = reshape(X, [k, n₁, n₂])
X' = P₁^T ×₁ X̃ ×₂ P₂                      // 仿射变换，平坦化分布
X_q = Q(X')                               // 量化到 INT4

// 矩阵乘法（等价性保证）
Y = X_q W_q^T ≈ XW^T                      // 量化近似
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
预量化变换的核心在于选择变换矩阵 T。简单方法使用固定的 Hadamard 矩阵（QuaRot），高级方法通过梯度下降学习最优 T（FlatQuant、SpinQuant）。T 的构造需要在"表达力"和"推理开销"之间权衡：全尺寸矩阵 T∈R^{n×n} 提升表达力但带来 O(n²) 在线计算开销；Kronecker 分解 T=P₁⊗P₂ 将开销降至 O(n√n)。变换矩阵的逆 T^{-1} 通过 SVD 分解（P^{-1}=VΣ^{-1}U^T）稳定计算，并与权重离线融合，不增加推理时的额外存储。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

---
