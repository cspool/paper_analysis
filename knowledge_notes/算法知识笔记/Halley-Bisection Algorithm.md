## Halley-Bisection Algorithm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Halley-Bisection 是 AdaSplash 提出的混合求根算法，快速求解 α-entmax 中的归一化阈值 τ。结合 Halley 方法的三次收敛与 Bisection 的保证收敛：$$\tau_H = \tau - \frac{2 f(\tau) f'(\tau)}{2 f'(\tau)^2 - f(\tau) f''(\tau)}$$ 其中 f/f'/f'' 为 α-entmax 阈值方程及其一/二阶导。Fail-safe：当 Halley 更新超出 [τ_lo, τ_hi] 回退 bisection。结果：3 次迭代到 machine precision（vs. bisection 23 次），~15× runtime 加速（2.38ms vs 36.67ms at n=8192），1.75× 内存节省。

从算法pipeline角度拆解术语。

```
Input: s ∈ R^n, α, T
1. τ_lo = max(s)-1, τ_hi = max(s)-n^{1-α}, τ = (τ_lo+τ_hi)/2
2. for t=1..T:
     // Block-wise 累积 f,f',f'' (GPU SRAM)
     for each block j: S_blk = Q_i@K_j^T
         f   += Σ [(α-1)S_blk-τ]_+^{1/(α-1)} - 1
         f'  += -1/(α-1) Σ [(α-1)S_blk-τ]_+^{1/(α-1)-1}
         f'' += (2-α)/(α-1)² Σ [(α-1)S_blk-τ]_+^{1/(α-1)-2}
     if f<0: τ_lo=τ else: τ_hi=τ
     τ_H = τ - 2·f·f'/(2·(f')²-f·f'')
     τ = τ_H if τ_H∈[τ_lo, τ_hi] else (τ_lo+τ_hi)/2
3. Return τ
```

术语一般如何实现？如何使用？

Triton kernel 将 f/f'/f'' 累积分布到多个 K block，SRAM 中增量累加不写入 HBM。每 Q_i block 独立运行 Halley-Bisection。适用于任何需快速求解 α-entmax τ 的场景。

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention
