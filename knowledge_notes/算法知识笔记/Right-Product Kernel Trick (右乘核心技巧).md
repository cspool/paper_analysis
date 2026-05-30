## Right-Product Kernel Trick (右乘核心技巧)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Right-product kernel trick 是线性注意力实现线性复杂度的核心计算技巧。利用矩阵乘法结合律，将标准 attention 的 (Q K^T) V 计算顺序改为 Q (K^T V)，避免物化 N×N 的中间矩阵。

标准 attention（left-product）: O = (Q K^T) V
- Step 1: S = Q K^T → [N, N] 矩阵，O(N²d)
- Step 2: O = S V → [N, d]

Linear attention（right-product）: O = Q (K^T V)
- Step 1: M = K^T V → [d, d] 矩阵，O(Nd²)
- Step 2: O = Q M → [N, d]，O(Nd²)

关键差异：right-product 的中间结果 M 是 d×d 矩阵，其大小与序列长度 N 无关。这使得：(1) 训练复杂度从 O(N²d) 降至 O(Nd²)；(2) 分布式通信量（传输 M）与序列长度无关，对长序列 SP 极其有利。

从算法pipeline角度拆解术语：

```
// Standard Attention (left-product)
Q, K, V = [N, d] each
S = Q @ K^T           // [N, N] — 必须物化，O(N²d)
A = Softmax(S)        // [N, N]
O = A @ V             // [N, d]

// Linear Attention (right-product kernel trick)
Q, K, V = [N, d] each
M = K^T @ V           // [d, d] — 与 N 无关！ O(Nd²)
O = Q @ M             // [N, d], O(Nd²)
```

术语一般如何实现？如何使用？

Right-product kernel trick 通过 GPU kernel 实现，可用 Triton 或 CUDA。当 d 较小时（如 d=64 per head），d²=4096，K^T V 计算量很小。当模型使用大 hidden dim 时（如 d=2048），需考虑 TP 沿 d 维度切分。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

---
