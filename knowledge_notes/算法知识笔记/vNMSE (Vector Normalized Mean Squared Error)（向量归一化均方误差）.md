## vNMSE (Vector Normalized Mean Squared Error)（向量归一化均方误差）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
vNMSE 是 ASQ 研究中的标准评估指标，定义为 vNMSE = E[‖X - X̂‖₂²] / ‖X‖₂²，即量化后向量与原始向量之间的期望 MSE 除以原始向量的 ℓ₂ 范数平方。通过归一化，vNMSE 消除了向量维度和幅值/方差的影响，使不同 d、不同分布下的量化质量可比。vNMSE 越低表示量化精度越高。在 QUIVER 论文中，vNMSE 用于衡量随 s（量化值个数）、d（维度）和分布类型变化时的量化精度趋势。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
vNMSE 计算流程：

```
输入: 原始向量 X, 量化值集合 Q = {q₁,...,q_s}

// 对每个 x ∈ X 做随机量化得到 x̂
// 实际实验中用多次独立随机量化求平均（QUIVER 论文用 5 seeds）
E_sq_error = 0
for seed in 1..5:
    对每个 x ∈ X, 以概率 (q_{j+1}-x)/(q_{j+1}-q_j) 取 q_j
    计算 sq_error_seed = Σ (x̂_i - x_i)²
    E_sq_error += sq_error_seed / 5

vNMSE = E_sq_error / ‖X‖₂²
```

vNMSE 与 ASQ 的理论关联：
- 精确 ASQ 最小化 Σ (b_x-x)(x-a_x)，这正是 MSE 的期望值
- vNMSE 将该 MSE 归一化，便于不同尺度的向量比较
- Apx. QUIVER 的近似保证：vNMSE_{2s-2} ≤ vNMSE_opt_s + d/(2m²)
  （使用 2s-2 个量化值的近似解的 vNMSE 不超过 s 个量化值最优解的 vNMSE + d/(2m²) 的附加项）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
vNMSE 在梯度压缩和量化方法评估中广泛使用 [Vargaftik et al., NeurIPS 2021]。实现简单：对量化输出向量 X̂ 和原始 X 计算 ℓ₂ 距离平方后除以 ‖X‖₂²。典型 vNMSE 取值范围：优秀 1-bit 量化（s=2, d=10⁶, LogNormal 分布）在 0.1-0.3 量级；4-bit（s=16）可降至 10⁻³ 以下。该指标与下游任务性能（如分布式学习收敛速度）有强相关性。

涉及论文标题：
- Optimal and Approximate Adaptive Stochastic Quantization
