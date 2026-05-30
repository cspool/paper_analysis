## Recomputation (in Fused Attention Kernels)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Recomputation 是 FlashAttention 系列的核心内存优化技术——反向传播时不存储前向的中间矩阵（S, P ∈ R^{n×n}），而是根据紧凑统计量重新计算它们。以额外 FLOPs 换取内存从 O(n²) 降至 O(n)。AdaSplash 继承这一策略，但需额外存储 O^(2) ∈ R^{n×d}（替代 softmax 存储 O）和 τ ∈ R^n。

从kernel调度角度拆解。

```
// 前向: 计算并存储 compact state (仅 O(n))
for each block:
    S = Q@K^T; P = [(α-1)S-τ]_+^{1/(α-1)}  // compute & discard
    O += P@V
Store: O, τ, O^(2)  // O^(2) = (ΣU·V)/||U||

// 反向: Recomputation from compact state
for each block:
    S = Q@K^T; P = [(α-1)S-τ]_+^{1/(α-1)}  // recompute!
    U = P^{2-α}                              // recompute!
    dS = U ⊙ (dP - δ)
    // accumulate dQ, dK, dV
```

术语一般如何实现？如何使用？

FlashAttention-1/2 在 CUDA 中存储 O + lse，反向 recompute S、P。AdaSplash 在 Triton 中存储 O + τ + O^(2)，反向 recompute S、P、U。核心权衡：2-3× 额外 GEMM FLOPs vs. n²→n 内存节省。GPU 上内存带宽常是 bottleneck，因此多做的 GEMM 往往比 HBM 写入更快。

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention

---
