## Residual Stream Expansion (n-stream Residual)

术语是什么？

将 Transformer 标准一维残差流 $\mathbf{x}_l \in \mathbb{R}^{C}$ 扩展为 n 个并行流 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$。动机：残差流信息容量受限于 C，而 C 与 FLOPs 强相关——扩展 n 在不增加每层 FLOPs 前提下提升信息容量，提供独立于模型尺寸/数据量之外的第三条扩展路径。

从算法pipeline角度拆解：

```
# layer_in: aggregate n streams, layer_out: standard F, x_next: update with mixing
layer_in = sum_i(H_pre[i] * x_l[i, :])            # (C,)  n→1
layer_out = F(layer_in, W_l)                       # (C,)  standard
x_next = H_res @ x_l + H_post.T * layer_out        # (n, C)  1→n+ mix
```

在 mHC 中 $\mathcal{H}^{\text{res}}$ 双随机 → 每个新 stream 是旧 stream 的凸组合（$\sum_j H^{\text{res}}_{ij} = 1, H^{\text{res}}_{ij} \geq 0$），在保持信号均值的同时实现信息交换。

术语一般如何实现？如何使用？

n 通常取 2-4。系统开销：显存 I/O 增加约 n 倍、pipeline 通信增加 n 倍、中间激活增加 n 倍——需通过 kernel fusion、recomputing 和通信重叠缓解。mHC 中 n=4 时额外时间开销仅 6.7%。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections
