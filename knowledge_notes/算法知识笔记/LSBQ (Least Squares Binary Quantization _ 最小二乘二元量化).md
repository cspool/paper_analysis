## LSBQ (Least Squares Binary Quantization / 最小二乘二元量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LSBQ（Least Squares Binary Quantization，最小二乘二元量化）是 Pouransari et al.（2020, CVPRW）提出的量化值估计方法。目标：将向量 u ∈ R^d 近似为 w_i = Σ_{j=1}^n v_j s_j(u_i)，其中 v_j 为正递减标量，s_j: R → {-1, 1} 为二进制函数，最小化 ‖u - w‖²。量化值集合 Q = {±v_1 ± v_2 ± ... ± v_n}，共 2^n 个可能值。n=1 时闭式解 v_1 = ‖u‖₁/d, s_1(u_i)=sign(u_i)；n>2 时通过 greedy foldable representation 求解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
n=2 (2-bit/ternary): 
  求解 v₁, v₂ → Q = {±(v₁+v₂), ±(v₁-v₂)} (4值 或 ternary 3值若 v₁=v₂)
n>2: greedy foldable
  for j=1..n:
      residual = u - Σ_{ℓ=1}^{j-1} v_ℓ s_ℓ(u)  // 累积重建残差
      s_j(u_i) = sign(residual_i)                // 二进制残差方向
      v_j = mean(|residual|)                     // 最优残差幅度
```
PARQ 在每轮迭代从 u^{t+1} 在线估计 Q^{t+1}，值从随机初始小值→早期膨胀→后期缓慢收缩。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LSBQ 产生 foldable (scaled binary) 量化——量化值间距由数据分布决定。在 PARQ 中使用 Q 直接作为 prox_PARQ 的 flat segment 位置。避免手动预设量化值 {q_k} 和正则化超参数 λ、{a_k}。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization
