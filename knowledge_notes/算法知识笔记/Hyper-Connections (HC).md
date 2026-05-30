## Hyper-Connections (HC)

术语是什么？

Hyper-Connections (HC) 是一种对标准残差连接的扩展范式。标准残差连接公式为 $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$，残差流宽度固定为 C（模型隐藏维度）。HC 将残差流宽度扩展 n 倍（称为 n-stream residual），输入变为 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$，引入三个可学习线性映射：
- $\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$：将 n-stream 特征聚合为 C 维层输入
- $\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$：将层输出映射回 n-stream
- $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$：在残差流内混合 n 个 stream 之间的特征

完整前向：$\mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post}^\top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l)$。每个映射由输入依赖的动态映射（线性投影+tanh+gating factor）和全局静态映射（可学习 bias）组成。扩展率 n（如 4）远小于 C，FLOPs 开销可忽略。核心缺陷：$\mathcal{H}_l^{\text{res}}$ 无约束，跨层复合映射可能信号爆炸/消失（Amax Gain Magnitude 可达 ~3000 vs 理想值 1），导致训练不稳定。同时 n-stream 导致显存 I/O 和 pipeline 通信开销增大约 n 倍。

从算法pipeline角度拆解：

```
def HC_forward(x_l):  # x_l: (n, C)
    x_norm = RMSNorm(x_l)
    H_pre  = alpha_pre  * tanh(x_norm @ theta_pre)  + b_pre   # (1, n)
    H_post = alpha_post * tanh(x_norm @ theta_post) + b_post  # (1, n)
    H_res  = alpha_res  * tanh(x_norm @ theta_res)  + b_res   # (n, n)
    layer_in = H_pre @ x_l                          # (C,)
    layer_out = F(layer_in, W_l)                    # (C,)
    x_next = H_res @ x_l + H_post.T * layer_out     # (n, C)
    return x_next
```

术语一般如何实现？如何使用？

HC 插入 Transformer 每层（Attention 子层和 FFN 子层），n 通常取 2-4。参数总量约 $C \times (n^2 + 2n) + (n^2 + 2n)$ 每层。动态映射需读取完整 $\mathbf{x}_l$，n-stream 显存 I/O 约为标准残差的 $(5n+1)$ 倍读和 $(3n+1)$ 倍写。需配合 kernel fusion、recomputing 等系统优化才能实用部署。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---
