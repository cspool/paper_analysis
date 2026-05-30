## Box-Cox Transformation for Attention Feature Amplification

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Box-Cox 变换是一种幂变换统计方法，在 DAM 用于 attention map 的特征放大。原始 attention 分布高度偏斜（少数大值主导，大量中小值被淹没），使直接阈值化难以区分重要的中等连接。Box-Cox 变换 $B(x) = (x^{\lambda} - 1) / \lambda$ 在 λ=0.5 时既放大中小 attention 值又不改变大值的相对顺序。DAM 比较了 9 种变换方法，Box-Cox 产生最紧凑的值范围（max≈2.0, mean≈0.27, std≈0.35），而 Square Root 产生极端值（max≈150），不利于阈值化。

从算法pipeline角度拆解术语：

```
// DAMPipeline_BoxCox_Step:
// Input: mean_attention A_mean (accumulated over dataset)
// Output: normalized_attention A_tilde (ready for thresholding)

ε = 1e-8
X = max(A_mean, ε)                // ensure positive
B = (X^{0.5} - 1) / 0.5           // λ=0.5: 放大小值, 保留大值尺度
A_tilde = B - min_all_layers_heads(B)  // shift to non-negative

// 对比: Square Root 变换 A_tilde_sqrt = sqrt(X) 产生 max≈150, std≈22
```

术语一般如何实现？如何使用？

λ=0.5 是经验选择且在 DAM 中固定。变换后全局减去最小值确保非负，然后直接进入 τ 阈值化步骤。跨 Multi-News 数据集累积的 mean attention 作为输入。论文对比 9 种方法后选择 Box-Cox，因其紧凑分布最便于统一阈值化。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration
