## Quantization Levels for Normal Distribution (正态分布最优量化级别)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Levels (QLs) for Normal Distribution 是 DANUQ 通过数值优化预计算的一组最优离散值，用于最小化标准正态分布下的期望量化误差。给定 B-bit 表示和 QLs 集合 Q = {q_0, q_1, ..., q_R}（其中 R = 2^{B-1} - 1，q_0 = 0），量化边界为 u_r = (q_{r-1} + q_r)/2（1 ≤ r ≤ R），量化规则 x ∈ [u_r, u_{r+1}) → q_r。最优 QLs 通过最小化 E[(x-q)^2] = Σ_{r=0}^R ∫_{u_r}^{u_{r+1}} (x - q_r)^2 φ(x) dx 求解（φ 为标准正态 PDF）。因 closed-form 解不可得（含 erf 和高斯积分交互项），DANUQ 采用暴力搜索在离散化搜索空间中枚举所有满足排序约束的 QL 组合。预计算结果：1-bit[-0.798, 0.798]；2-bit[-1.224, 0, 0.765, 1.724]；4-bit[16个非均匀值]。QLs 非均匀分布反映了正态分布的高密度区域（均值附近）需要更细粒度量化的原则。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QLs 的数值优化过程：

```python
# Offline QL optimization (run once before FL training)
def optimize_QLs(B, search_range=(-3, 3), grid_points=1000):
    R = 2**(B-1) - 1
    candidates = linspace(search_range[0], search_range[1], grid_points)
    
    best_QLs = None
    best_error = inf
    
    # Exhaustive search over discretized QL space
    # Constraint: 0 = q_0 < q_1 < q_2 < ... < q_R
    for q_combo in combinations_with_order(candidates, R):
        QLs = [0.0] + list(q_combo)   # q_0 = 0 fixed (except 1-bit)
        # Build boundaries
        u = [0.0]  # u_0 = q_0
        for r in range(1, R+1):
            u.append((QLs[r-1] + QLs[r]) / 2)
        u.append(float('inf'))  # u_{R+1}
        
        # Evaluate expected error via Eq.(10)
        error = compute_expected_error(QLs, u)
        if error < best_error:
            best_error = error
            best_QLs = QLs
    
    return best_QLs
# For 1-bit special case: omit q_0=0, search symmetric pair directly
```

**Annotations**: search_range 限制在 [-3, 3] 因为 N(0,1) 的 99.7% 概率质量在 [-3σ, 3σ] 内。grid_points=1000 足够细粒度。计算量在 B≤4 时可控（4-bit: C(1000,15) 太大，实际用启发式 + 并行加速）。1-bit 特例：只有 2 个 QLs，无 q_0=0 约束，直接搜索对称对。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QLs 预计算后以 lookup table 形式存入代码常量。Client 端量化：对归一化后的值 x，查找最近 QL：|x - q_r| 最小 → 返回 B-bit 索引 r。Server 端反量化：通过 索引 r → QLs[r] → 乘 scale 恢复。DANUQ 的 QLs 设计的优势：(1) 固定 QLs 无需学习或传输，零额外通信；(2) 基于正态分布先验，在 LMPU 真实分布匹配假设时达到近似最优；(3) 与 UQ 的等距间隔相比，DANUQ 的 1-bit QLs (±0.798) 在实际正态分布下可压缩 MSE 约 18%-25%（FedWSQ Table 2 数据估计）。对比现有 NUQ 方法：NF (NormalFloat) 对各 bit-width 使用通用公式，DANUQ 为每个 B 独立优化；FP (Floating Point) 的 exponent/mantissa 分配不适合 1/2-bit 极端场景。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
