## True Mask Generation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

True mask $M_{\ell,h} \in \{0,1\}^{L \times L}$ 是 DAM Stage 1 中对归一化 attention map $\tilde{A}_{\ell,h}$ 以阈值 τ 二值化生成的二进制矩阵。每个元素 $m_{i,j}$ 通过比较 $\tilde{A}_{\ell,h,i,j}$ 与 τ 决定：≥τ=1（保留连接），<τ=0（丢弃连接）。τ=0.3 通过 attention 稀疏性分析确定。True mask 为后续结构模式匹配提供输入，用于识别可外推的规律性 pattern。

从算法pipeline角度拆解术语：

```
for each (ℓ,h), for each (i,j) in [0..L-1]×[0..L-1]:
    M_true[ℓ,h,i,j] = (A_tilde[ℓ,h,i,j] >= τ) ? 1 : 0
// τ=0.3: 仅保留归一化 attention 前 ~70% 的有效连接
```

术语一般如何实现？如何使用？

seq_len ≤ PCL 时直接使用 true mask 做推理；seq_len > PCL 时 true mask 覆盖前 L×L 区域，其余由 extended mask 补充。τ 控制 mask 密度——更高值产生更稀疏的 mask，减少更多计算但也可能丢弃重要连接。论文在 LongEval 上验证 τ=0.3 时 DAM 平均精度 0.7966 vs Full Attention 0.8011。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration
