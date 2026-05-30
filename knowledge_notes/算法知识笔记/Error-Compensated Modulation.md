## Error-Compensated Modulation

术语是什么？
Error-Compensated Modulation（误差补偿调制）是MoDiff框架的第二个核心组件，解决标准调制方法中量化误差在时间步间累积的问题。通过引入中间变量 $\hat{\mathbf{a}}_t$ 存储经过量化后的近似激活，使当前时间步的量化误差 $\mathbf{e}_t = \mathbf{a}_t - \hat{\mathbf{a}}_t$ 被显式追踪，并在下一时间步通过输入反馈被算子补偿。

具体机制：$\hat{\mathbf{a}}_t = Q(\mathbf{a}_t - \hat{\mathbf{a}}_{t+1}) + \hat{\mathbf{a}}_{t+1}$。当前步的量化误差被保留在 $\hat{\mathbf{a}}_t$ 中（因为 $\hat{\mathbf{a}}_t = \mathbf{a}_t - \mathbf{e}_t'$），下一步计算的差分基是 $\hat{\mathbf{a}}_{t+1}$ 而非 $\mathbf{a}_{t+1}$，从而将上一步遗漏的 $A(\mathbf{e}_{t+1})$ 重新纳入计算。

从算法pipeline角度拆解术语：
```
// 标准调制（无误差补偿）——误差指数增长
// õ_t = A(Q(a_t - a_{t+1})) + õ_{t+1}
// Theorem 4.4: 误差界 ∝ Σ 2^{T-k-1} × c ∥A∥² ∥d_k∥²  (指数增长)

// MoDiff误差补偿调制——误差指数衰减
// â_t = Q(a_t - â_{t+1}) + â_{t+1}
// ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}
// Theorem 4.4: 误差界 ∝ Σ (2c)^{T-k-1} × ∥A∥² ∥d_k∥²  (指数衰减, 当c<1/2)

// 误差追踪 (Eq.18):
e_t = (a_t - â_{t+1}) - Q(a_t - â_{t+1})    // 当前步量化误差
    = (a_t - â_{t+1}) - (â_t - â_{t+1})     // 代入Eq.13: â_t = Q(a_t - â_{t+1}) + â_{t+1}
    = a_t - â_t                              // 简化

// 误差补偿在下一步生效：
// 下一步输入 = a_{t-1} - â_t = a_{t-1} - (a_t - e_t) = (a_{t-1} - a_t) + e_t
// 算子输出 = A((a_{t-1} - a_t) + e_t) ≈ A(a_{t-1} - a_t) + A(e_t)
// 其中 A(e_t) 补偿了上一步遗漏的误差分量
```

关键数学保证：Theorem 4.4证明当量化误差系数 $c < 1/2$ 时（Corollary A.3表明通过选择足够高的位宽 $\hat{b} \ge \log_2(\sqrt{4d/c} + 1)$ 可达），标准调制的误差以 $2^{T-k-1}$ 速率指数增长（误差被逐步放大），而误差补偿调制的误差以 $(2c)^{T-k-1}$ 速率指数衰减——c<1/2保证每一步误差均被缩小而非放大。

术语一般如何实现？如何使用？
论文在PyTorch中实现：为每个被MoDiff改造的线性层维护两个额外中间变量 $\hat{\mathbf{a}}_t$ 和 $\hat{\mathbf{o}}_t$ 在时间步间传递。Abalation验证（Table 4）：W8A4时标准调制（w/o EC）FID=25.42，误差补偿调制FID=4.38，差距6×。Figure 3显示误差补偿使relative ℓ₂ distance在3-bit时保持接近0（vs w/o EC持续增长到40%）。额外内存开销：单张CIFAR-10上W8A4时仅~4MB（Table 6）。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

---
