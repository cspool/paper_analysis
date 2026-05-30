## Surrogate Maximum (X-Row Tile Maximum) for Tensorized Softmax

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Surrogate Maximum（替代最大值，记作m̂[i]）是FlashAttention-T (PPoPP'26) 在Tensorized Online Softmax算法中提出的关键概念。在fused attention的online softmax中，attention output rescaling操作 `O = exp(m_old - m)·O` 要求scaling factor `exp(m_old[i]-m[i])` 对每行独立计算。然而，repurposed tensor MMA scaling instruction要求scaling factor α在所有行上uniform（X行共享同一个α值，X=16 for HMMA.1688, X=64 for HGMMA.64x8x8）。为满足这一约束，Surrogate Maximum定义为attention logit矩阵S的第i行所在X-row tile的最大值：
$$hat{m}[i] = \max({S[i',j']: j' \in [0,s), i' \in [X \cdot \lfloor i/X \rfloor, X \cdot \lfloor i/X \rfloor + X)})$$

m̂[i]在X行内保持uniform，使scaling factor `exp(m̂_old - m̂)` 满足tensor MMA uniform scaling constraint。

Numerical safety guarantees:
1. **No overflow**（严格保证）：m̂[i] ≥ m[i]（tile max ≥ row max），故 exp(S[i,j]-m̂[i]) ≤ exp(S[i,j]-m[i]) ≤ 1，永不超F_max
2. **Negligible all-underflow probability**（高概率）：更大m̂增大single exponent underflow概率，但整行所有exponent同时underflow的联合概率在典型分布（Gaussian等）下asymptotically small
3. **Fallback机制**：极端分布触发all-underflow时，选择性fallback到vectorized rescaling（跳过surrogate，保持其他primitives tensorized）

与FlashDecoding++的static maximum不同，X-row tile surrogate动态适应局部行分布。

从算法pipeline角度拆解术语：

Tensorized Online Softmax (Algorithm 1)的核心流程：
```
// Input: S∈R^{n×s}, O∈R^{n×d}, m_old∈R^n, l∈R^n, surrogate tile size X
// Step 1: Compute X-row tile maxima (warp REDUX, 2 instructions)
m̂ ← tilemax(S, X)                    // m̂ ∈ R^{⌈n/X⌉}
// Step 2: Get old surrogate maximums (broadcast)
m̂_old ← m_old[X·i] for i ∈ [0, ⌈n/X⌉)
// Step 3: Tensorized O rescaling
//   scaling factor exp(m̂_old[k] - m̂[k]) uniform ∀ rows in tile k
//   → satisfies tensor MMA uniform scaling constraint ✓
for k in 0..⌈n/X⌉-1:
    O[kX:(k+1)X,:] ← exp(m̂_old[k] - m̂[k]) · O[kX:(k+1)X,:]
// Step 4: Assign surrogate to per-row m
m[i] ← m̂[⌊i/X⌋] for i ∈ [0, n)
// Step 5: Tensorized S rescaling (constant log₂(e) → always uniform)
Z ← log₂(e) · S - (log₂(e) · m)
// Step 6: Vector exp₂
P̃ ← exp₂(Z)                          // MUFU.EX2, stay vectorized
// Step 7: Tensorized row-sum reduction
l ← exp(m_old - m)·l + rowsum(P̃)
// Step 8: return P̃, O, m, l
```

对比Baseline per-row maximum:
```
// Baseline: m[i] = max(S[i,:]) — per-row, non-uniform
// O[i,:] ← exp(m_old[i] - m[i]) · O[i,:]
//   scaling factor varies per row → CANNOT use tensor MMA scaling ✗

// FlashAttention-T: m̂[k] = max(16/64 consecutive rows) — tile-uniform
// O[kX:(k+1)X,:] ← exp(m̂_old[k] - m̂[k]) · O[kX:(k+1)X,:]
//   scaling factor uniform in tile → CAN use tensor MMA scaling ✓
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Surrogate maximum的实现：(1) warp all-reduce REDUX in 2 instructions（vs baseline逐行SHFL-based max需多次warp shuffle）；(2) FA2+Max16 ablation（仅加surrogate maximum，无tensorization）即带来1-3% speedup，因REDUX > SHFL in throughput；(3) Hopper TLP实现中surrogate未被使用（仅tensorize P̃ row-summation无需scaling factor），数值稳定性与baseline一致。surrogate maximum概念可推广到任何需在特定粒度approximate per-element operations以对齐hardware-aligned computation的场景（如block-level normalization with uniform statistics）。

涉及论文标题：
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
