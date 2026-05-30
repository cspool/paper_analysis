## BOF4 / BOF4-S（4-bit Block-Wise Optimal Float / 4-bit 逐块最优浮点量化码本）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BOF4（4-bit Block-wise Optimal Float）是 Blumenberg et al. (2025) 提出的 block-wise absmax 量化最优码本家族。核心创新：将 Lloyd's algorithm 的 centroid 更新修改为考虑 block maximum 分布权重——最小化端到端 `MSE(W,Q(W))` 或 `MAE(W,Q(W))`，而非仅归一化权重的量化误差。MSE centroid = 归一化权重的 `w_b^max` 平方加权平均（Eq. 6），MAE centroid = `w_b^max` 加权中位数（Eq. 8）。大 `w_b^max` block 的归一化权重在 centroid 更新中贡献更大（解码时误差被放大）。BOF4 固定 3 个 level（-1, 0, 1），BOF4-S 配合 signed normalization 仅固定 2 个（0, 1）。码本可在高斯权重假设下通过数值积分（Eq. 5/7）或 MC 采样（Eq. 6/8）计算，两者等价（MSE ≈ -56 dB）。BOF4-S(MSE)+OPQ 在 Llama-3.1 8B 上 WikiText-2 PPL=8.43，优于 NF4 (8.53) 和 AF4 (8.51)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BOF4 EM Algorithm (modified Lloyd's algorithm)
# Input: p_W = N(0,1), block size I
# Output: codebook x̂[1..16]

# Step 1: Initialize (e.g., from NF4 levels)
x̂ = [-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
       0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0]
fixed = {1, 8, 16}  # BOF4-S: {8, 16}

# Step 2: Sample W ~ N(0,1) shape [B, I], normalize x=b,i = w[b,i]/w_max[b]

repeat until convergence:
    # 3a. Partition (nearest neighbor): assign each x to nearest x̂[j]
    
    # 3b. Centroid update (MODIFIED):
    for ℓ in 2..15 where ℓ not in fixed:
        # MSE (Eq. 6): weighted mean by w_max²
        x̂[ℓ] = Σ_k (w_max[k]² * x_k) / Σ_k (w_max[k]²)
        # MAE (Eq. 8): weighted median by w_max
        # x̂[ℓ] = weighted_median({x_k}, weights={w_max[k]})
```
关键区别 vs 标准 Lloyd：centroid 从均匀加权均值变为 block max 平方加权均值。源于 `MSE(W,Q(W)) = (1/B)·Σ_b (w_b^max)²·MSE(X_b, Q̃(X_b))` 分解（Eq. 60）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/ifnspaml/bof4。码本以 Python 数组提供（BOF4-S MSE, I ∈ {32, 64, 128, 256}），也可 MC 方法按需生成。使用时替换 NF4/AF4 码本，解码逻辑相同。码本硬编码，不随具体模型权重调整，不受 OPQ 影响。

涉及论文标题：
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)

---
