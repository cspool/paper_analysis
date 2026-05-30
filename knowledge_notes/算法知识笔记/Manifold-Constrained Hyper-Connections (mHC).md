## Manifold-Constrained Hyper-Connections (mHC)

术语是什么？

mHC 是 HC 的改进框架，将 HC 的 $\mathcal{H}_l^{\text{res}}$ 通过 Sinkhorn-Knopp 算法约束到 Birkhoff polytope（双随机矩阵流形），恢复 identity mapping 稳定性，同时保留 HC 的多流特征混合能力。三个关键约束：(1) $\mathcal{H}_l^{\text{res}}$ 双随机（行和=列和=1，元素 ≥ 0），使 $\mathcal{H}_l^{\text{res}} \mathbf{x}_l$ 成为 n 个 stream 的凸组合，谱范数 ≤ 1 非膨胀，乘法封闭性保证跨层稳定；(2) $\mathcal{H}_l^{\text{pre}}$ 经 Sigmoid，$\mathcal{H}_l^{\text{post}}$ 经 $2\sigma(\cdot)$ 约束为非负，防止正负系数抵消；(3) n=1 时退化为标量 1，完全恢复标准残差连接。与 HC 相比，复合映射 Amax Gain Magnitude 从 ~3000 降至 ~1.6（降低 3 个数量级）。

从算法pipeline角度拆解：

```
def mHC_forward(x_l):  # x_l: (n, C)
    x_flat = flatten(x_l); x_norm = RMSNorm(x_flat)    # (1, nC)
    H_pre_raw  = alpha_pre  * (x_norm @ phi_pre)  + b_pre   # (1, n)
    H_post_raw = alpha_post * (x_norm @ phi_post) + b_post  # (1, n)
    H_res_raw  = alpha_res  * reshape(x_norm @ phi_res, (n, n)) + b_res  # (n, n)
    # Manifold projection (key difference from HC)
    H_pre  = sigmoid(H_pre_raw); H_post = 2 * sigmoid(H_post_raw)
    H_res  = SinkhornKnopp(H_res_raw, t_max=20)  # doubly stochastic
    layer_in = H_pre @ x_l; layer_out = F(layer_in, W_l)
    x_next = H_res @ x_l + H_post.T * layer_out  # (n, C)
    return x_next
```

术语一般如何实现？如何使用？

需配合大量系统优化：5 个融合 kernel、选择性重计算、DualPipe 通信重叠。n=4 时额外训练时间仅 6.7%。flatten 操作展平 $\mathbf{x}_l$ 为 $\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$ 保留完整上下文。gating factor $\alpha$ 初始化为 0.01。基于 DeepSeek-V3 MoE 架构验证（3B/9B/27B 参数），8 个下游 benchmark 全面超越 baseline 和 HC。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---
