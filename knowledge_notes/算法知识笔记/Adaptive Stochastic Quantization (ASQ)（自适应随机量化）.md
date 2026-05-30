## Adaptive Stochastic Quantization (ASQ)（自适应随机量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Stochastic Quantization (ASQ) 是一种根据输入向量 X 的具体分布自适应选择量化值集合 Q（而非使用固定量化值）来最小化随机量化均方误差（MSE）的技术。形式化地，给定排序向量 X ∈ R^d 和量化值个数 s，ASQ 寻找 Q ⊆ X（即最优量化值必定是输入的某个子集 [Zhang et al., ZipML 2017]），|Q| = s，使得 MSE = Σ_{x∈X} (b_x - x)(x - a_x) 最小化，其中 a_x, b_x 是包围 x 的连续两个量化值。与分布无关（distribution-agnostic）方法不同（如 QSGD 仅使用向量范数确定量化值，NUQSGD 使用全局 min/max），ASQ 针对每个特定输入向量优化 Q，可显著降低量化误差。然而，ASQ 问题是 non-convex 的（即使 s=4 即 2-bit 量化也不凸），排除了梯度下降等常规方法。ZipML 首次提出用动态规划（DP）在多项式时间内求精确解，但时间 O(s·d²) 和空间 O(d²) 使其在大向量上（d > 10⁵）不可行。QUIVER 通过预处理 + Quadrangle Inequality + SMAWK 将复杂度降至 O(s·d) 时间和 O(s·d) 空间，使 ASQ 在大规模 ML 场景中实用化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ASQ 在分布式学习的 gradient compression pipeline 中的执行流程：

```
输入: 梯度向量 X ∈ R^d, 量化值个数 s

// === Step 1: 自适应选择量化值（QUIVER 精确解）===
1. 排序 X（若非已排序，O(d log d)；GPU 上可并行排序）
2. 预处理 (O(d)):
   β_j = Σ_{i=1}^j x_i        // 累积和
   γ_j = Σ_{i=1}^j x_i²       // 累积平方和
   // 使 C[k,j] = -x_j·x_k·(j-k) + (x_j+x_k)·(β_j-β_k) - (γ_j-γ_k) 可 O(1) 求值
3. DP + SMAWK (O(s·d)):
   MSE[2, j] = C[1, j]  ∀j  // 初始化
   for i = 3 to s:
       K[i,·] = SMAWK(隐式矩阵 A where A[k,j] = MSE[i-1,k] + C[k,j])
       MSE[i, j] = MSE[i-1, K[i,j]] + C[K[i,j], j]
4. 回溯 (O(s)):
   Q = {x_1, x_d}, j = d
   for i = s to 3:
       j = K[i, j]; Q = Q ∪ {x_j}
   // 输出最优量化值集合 Q ⊆ X, |Q|=s

// === Step 2: 随机量化（Stochastic Quantization）===
for each x ∈ X:
    找到 a_x = max{q∈Q | q≤x}, b_x = min{q∈Q | q≥x}
    以概率 p_a = (b_x-x)/(b_x-a_x) 输出 x̂ = a_x
    以概率 p_b = (x-a_x)/(b_x-a_x) 输出 x̂ = b_x
    // 性质: E[x̂] = x (无偏), Var[x̂] = (b_x-x)(x-a_x)
```

核心数学洞察：C[k,j] 满足 quadrangle inequality，使得 DP 矩阵 A[k,j] = MSE[i-1,k] + C[k,j] 成为 totally monotone matrix，从而可用 SMAWK 算法在 O(d) 时间内找到每列的行最小值索引，替代原生 O(d²) 的逐列枚举。近似变体 Apx. QUIVER 将候选量化值离散化为均匀网格上的 m 个点，使用直方图预处理实现 O(d + m·s) 复杂度，并提供严格近似保证：AQ_{X,2s-2} ≤ opt_{X,s} + d·(x_d-x_1)²/(4m²)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：QUIVER 的 C++ 参考实现开源在 https://github.com/ranbenbasat/QUIVER。核心依赖 SMAWK 算法（1986 年提出，已有多种开源实现如 Python recipe by David Eppstein）。预处理部分需要 O(d) 额外空间存储 β, γ 数组。API 用法：输入已排序向量和 s → 输出最优 Q → 对每个 x 按概率做随机量化。

典型使用场景：（1）分布式/联邦学习中梯度压缩——sender 端先排序 → QUIVER 求 Q → 随机量化每个梯度分量 → 发送 Q + 每分量的比特编码 → receiver 端解码并反量化。1M 维梯度向量可在 ~1 秒内完成最优 4-bit 量化。（2）数据集量化——QUIVER 可扩展为 weighted variant，通过 weight 参数支持经验分布的 ASQ 求解，仅比非加权版慢 10-20%。（3）模型后训练量化——ASQ 可用于 weight/activation/KV cache 的无偏量化，避免偏置方法在分布式场景中误差不随 n 衰减的问题。

局限性：QUIVER 非 GPU-friendly（依赖 SMAWK 的 sequential recursion），Apx. QUIVER 不需要排序（O(d+m·s) 对未排序输入同样有效）。精确解要求输入已排序，否则需额外 O(d log d) 排序开销。

涉及论文标题：
- Optimal and Approximate Adaptive Stochastic Quantization
