## Sinkhorn-Knopp Algorithm

术语是什么？

Sinkhorn-Knopp 算法将正矩阵迭代投影为双随机矩阵（行和=列和=1）。给定初始正矩阵 $\mathbf{M}^{(0)}$（通过 $\exp(\cdot)$ 保证正性），交替行归一化 $\mathcal{T}_r$ 和列归一化 $\mathcal{T}_c$：$\mathbf{M}^{(t)} = \mathcal{T}_r(\mathcal{T}_c(\mathbf{M}^{(t-1)}))$。当 $t \to \infty$ 时收敛到唯一双随机矩阵，形式为 $\mathbf{M} = \text{diag}(\mathbf{u}) \cdot \mathbf{M}^{(0)} \cdot \text{diag}(\mathbf{v})$。在 mHC 中 $t_{\text{max}} = 20$ 为实际近似值。

从算法pipeline角度拆解：

```
M = exp(H_res_raw)            # element-wise exp for positivity
for t in 1..20:
    M = M / sum(M, axis=1, keepdim=True)   # row normalize
    M = M / sum(M, axis=0, keepdim=True)   # col normalize
return M  # ~doubly stochastic, used as H_res
```

术语一般如何实现？如何使用？

广泛用于最优传输中求解熵正则化问题。在 mHC 中前向和反向均实现为单一 GPU kernel——反向在片上重计算整个迭代过程而非保存 20 次迭代的中间矩阵。20 次迭代已产生近双随机矩阵（行列和接近 1），复合映射 Amax Gain Magnitude 仅 ~1.6。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---
