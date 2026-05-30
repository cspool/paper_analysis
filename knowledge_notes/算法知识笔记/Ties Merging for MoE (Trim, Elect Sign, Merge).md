## Ties Merging for MoE (Trim, Elect Sign, Merge)

术语是什么？
Ties（Trim, Elect Sign, and Merge, Yadav et al. 2024）通过三步解决参数干扰：(1) **Trim**：每个 task vector 中 drop bottom (100-p)% 最小 magnitude 参数（消除冗余）；(2) **Elect Sign**：每个参数位置确定总 magnitude 更大的符号方向为"主导符号"（解决 sign conflict）；(3) **Disjoint Merge**：仅累加与主导符号方向相同的 task vector 值。MergeME 首次将 Ties 应用于 MoE 合并，设置 p=80%, λ=1/3。Table 1: Ties avg 12.52 vs BTX 11.72 (+6.94%)。

从算法pipeline角度拆解术语：
```
// Step 1: Trim — threshold = magnitude_percentile(|τᵢ|, 100-p); 重置小值为 0
// Step 2: Elect Sign — for each j: dom[j] = ±1 based on total magnitude
// Step 3: Disjoint Merge — τ_m[j] += τᵢ[j] only if sign(τᵢ[j]) == dom[j]
θ_m = θ_b + λ · τ_m
```

术语一般如何实现？如何使用？
- 开源：mergekit 库提供 `ties` 方法，参数 `density`(=p/100) 和 `weight`(=λ)。
- Ties vs Dare: Ties 显式解决 sign conflict；Dare 处理 magnitude disparity 更轻量。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---
