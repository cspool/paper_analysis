## Selective Recomputing / Gradient Checkpointing for mHC

术语是什么？

Selective Recomputing（选择性重计算）是在训练反向传播中，丢弃前向 pass 的中间激活并在反向 pass 中重新计算它们的策略，以显存换计算。mHC 由于 n-stream 设计导致中间激活量约为标准残差连接的 n 倍（每层额外存储 nC 元素的 stream + C 元素的层输入），选择性重计算将这些中间激活丢弃并在反向需要时重新执行 mHC kernel（不含沉重的层函数 $\mathcal{F}$）。

mHC 的重计算策略优化：对于 $L_r$ 连续层，仅需持久化首层输入 $\mathbf{x}_{l_0}$（nC 元素），中间层的 stream 和映射系数在反向中重计算。最优块大小：$L_r^* \approx \sqrt{nL/(n+2)}$，在实践中与 pipeline stage 中的层数对齐。总持久化存储：$\lceil L/L_r \rceil$ 个 $\mathbf{x}_{l_0}$，瞬态峰值：$(n+2)C \times L_r$。

从kernel调度角度拆解：

```
# Forward pass storage strategy
for each block of L_r consecutive layers:
    store x_{l_0}       # first layer input, nC elements, persistent
    store F_outputs     # every layer's F output, C elements each
    # DISCARD: intermediate x_l (nC), H_pre*x_l (C), RMSNorm result (C)
    
# Backward pass recomputation
for each block (reverse order):
    load x_{l_0}
    for l in l_0 .. l_0+L_r-1:
        # Re-execute mHC kernels (without heavy F)
        x_flat = flatten(x_l); r = norm(x_flat)/sqrt(nC)
        H_tilde = x_flat @ phi
        H_pre, H_post, H_res = manifold_project(H_tilde, r, alpha, bias)
        layer_in = H_pre @ x_l
        # Now compute gradients using stored F_output and recomputed intermediates
        ...
        x_l = H_res @ x_l + H_post.T * F_output  # recover next x_l
```

术语一般如何实现？如何使用？

PyTorch 中通过 `torch.utils.checkpoint` 实现，但 mHC 的自定义重计算策略更精细：重计算边界与 pipeline stage 对齐；仅重计算 mHC kernel 不含 $\mathcal{F}$（因为 $\mathcal{F}$ 的输出已存储）；重计算过程与 pipeline 通信解耦（首层激活已在本地缓存）。在 DualPipe schedule 中进一步将重计算与通信重叠。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections
