## Expert Output Merging (Output-Merging View)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Output Merging 是 MergeMoE 提出的对 expert merging 的新理论视角。传统"参数合并"视角将 expert 参数加权平均，而输出合并视角将 merging 重新表述为对 expert 输出空间的优化：压缩后 expert E'_i 的输出应逼近原始 experts 输出的线性组合 E'_i(X) ≈ Σ_j B_{ji} E_j(X)，而非简单平均参数。该视角将压缩建模为在前向计算中插入矩阵的线性优化：原始 Y · mask_top_K(...) → 压缩后 Y · B · A · mask_top_K(...)，最小化 ||YBA - Y|| 的 Frobenius 误差。在 MergeMoE 框架中，merged expert 被形式化为 E'_i(X) = W'_Di T1 (σ(T2 W'_Gi X) ⊙ (T3 W'_Ui X))，其中 T1/T2/T3 为维度缩减矩阵。传统 M-SMoE 等价于 T1=[I;I;...;I]（不做优化）、T2/T3 做加权平均的特例，而输出合并视角允许分别优化三个矩阵。

从算法pipeline角度拆解术语：
```
// 矩阵 A (路由求和): A_{ij}=1 if expert j→cluster i, else 0
// 矩阵 B (输出组合): B_{ji}=f_j/Σf_k if j∈C_i, else 0
// 原始 forward: Y · mask_top_K(softmax(W_r X))^T
// 压缩后 forward: Y · B · A · mask_top_K(softmax(W_r X))^T
// 目标: min ||Y(BA - I_N) · mask_top_K(...)||_F^2
// 在独立假设下，问题分解为每个 cluster 的二次优化，使用频率为最优解
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 优势：将 merging 从启发式转变为可量化优化问题；T1 通过采样输入+最小二乘法求解（QP†）直接降低输出误差；解释了 M-SMoE 为何有效以及改进空间。
- 局限：依赖采样输入代表真实分布；T2/T3 仍为启发式（非线性 σ/⊙ 无法联合求解）；样本数 <32 时性能崩溃。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---
