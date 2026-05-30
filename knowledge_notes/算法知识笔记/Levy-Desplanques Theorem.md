## Levy-Desplanques Theorem

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Levy-Desplanques Theorem 是线性代数中的一个经典定理，由 Naimark & Zeheb (1997) 进行了扩展。定理的核心陈述：所有严格对角占优矩阵（Strictly Diagonally Dominant Matrix）都是可逆的（非奇异的）。严格对角占优的定义为：对矩阵 A 的每一行 i，对角线元素的绝对值严格大于该行其他所有元素绝对值之和：|a_ii| > Σ_{j≠i} |a_ij| for all i。定理直观含义：如果矩阵的对角线元素在各自行中占据足够主导的地位，则该矩阵必然满秩。在 AffineQuant 中，该定理被用作保证仿射变换矩阵 A 在优化过程中始终保持可逆的理论基础——作者将 A 初始化为对角矩阵（天然严格对角占优、必然可逆），并通过 Gradual Mask 在优化过程中抑制非对角线元素的幅度和更新速率，使 A 持续满足严格对角占优条件，从而保证每一步的 A⁻¹ 计算有效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 AffineQuant 优化过程中，Levy-Desplanques 定理的应用链条为：
```
初始化：A₀ = diag(s₁,...,s_d) → 对任意行 i: |a_ii| = |s_i| > 0 = Σ_{j≠i} |a_ij|
       → A₀ 严格对角占优 → Levy-Desplanques → A₀ 可逆 ✓

优化中（epoch=e）：
  前向: A* = A_e ∘ GM → α<1 缩小非对角线 → |a*_ii| > Σ α|a*_ij| 更易满足
  反向: A_{e+1} = A_e + η·GM·∂L/∂A* → 非对角线更新被 α 抑制

理论保证（Theorem 1, Appendix A.2）：
  若 N_e 严格对角占优，且 α < |n_ii^0+ηΣ∂L/∂n_ii*| / (η·Σ|Σ∂L/∂n_ij*|)
  则 N_{e+1} 也严格对角占优 → Levy-Desplanques → N_{e+1} 可逆 ✓
```
即：只要 α 足够小，严格对角占优性质从初始状态通过整个优化过程向前传播，矩阵可逆性始终得到理论保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AffineQuant 的具体实现中，Levy-Desplanques 定理主要通过 Gradual Mask 隐式使用——不显式检查对角占优条件。因为：(1) 对角初始化自动满足条件；(2) GM 逐步释放对角线附近元素，α<1 持续抑制非对角线幅度；(3) 论文的理论证明（Appendix A.2）提供了严格的 α 上界，实践中选择 α∈[1e-4, 1] 的经验值即可。该定理在更广泛的数值计算中常用于：保证 Jacobi/Gauss-Seidel 迭代收敛、分析矩阵条件数上界、偏微分方程有限差分解的稳定性分析。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models

---
