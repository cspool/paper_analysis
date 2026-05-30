## Recentralized Quantization (再中心化量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recentralized Quantization 是 Focused Quantization 的核心子机制。传统 shift quantization 假定权重以零为中心分布（量化层级最密集在零附近），但剪枝后 CNN 的权重分布"中空"——非零权重集中在远离零的特定值区域。Recentralized Quantization 解决这一问题：先对每层非零权重拟合 GMM 找到高概率聚类的均值 μ_c 和标准差 σ_c，然后对每个聚类独立做"再中心化"——即对归属于聚类 c 的权重 θ，先做 (θ - μ_c) / σ_c 将其映射为零均值、单位方差的分布，在此归一化空间内做 shift quantization（此时量化层级密集在零附近且恰好对应权重高概率区域），最后再反归一化回到原始尺度。

数学表达：`Q_c^{rec}[θ] = Q^{shift}_{n,b}[(θ - μ_c)/σ_c] * σ_c + μ_c`。其中 `Q^{shift}_{n,b}` 是 n-bit shift quantization，`μ_c` 经额外汇总量化为 2 的幂次值以保持硬件效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 ResNet-50 block3f/conv1 层为例（Figure 2 in paper）：

```
# 该层权重分布呈现双峰: 正权重簇在 ~0.10, 负权重簇在 ~-0.10

# Step 1: GMM fitting
# EM 算法收敛后:
μ_- ≈ -0.10, σ_- ≈ 0.02  (负值聚类)
μ_+ ≈ +0.10, σ_+ ≈ 0.02  (正值聚类)

# Step 2: 分量分配 (per weight)
# Example: θ = 0.095
p(m_θ = +) = λ_+ * N(0.095|0.10, 0.02) / q_mix(0.095) ≈ 0.92
p(m_θ = -) = λ_- * N(0.095|-0.10, 0.02) / q_mix(0.095) ≈ 0.08
→ m_θ = + (正分量)

# Step 3: Recentralization (将 θ 映射到零附近)
θ_norm = (θ - μ_+) / σ_+ = (0.095 - 0.10) / 0.02 = -0.25
# θ_norm ≈ 0, 位于 shift quantization 层级密集区

# Step 4: Shift quantization (利用零附近的高精度层级)
# 3-bit shift quant: v = s * 2^{e-b}
# θ_norm ≈ -0.25 → nearest power-of-two: -0.25 = -2^{-2}
θ_hat_norm = -2^{-2} = -0.25  (exact match, no error)

# Step 5: De-normalization
θ_hat = α * (θ_hat_norm * σ_+ + μ_+)
      = α * (-0.25 * 0.02 + 0.10) = α * 0.095
# 完美恢复原始值 (量化前后无损)
```

**Annotations**: 若不用 recentralization 而直接用 shift quantization，θ=0.095 的最近 2 的幂次值为 0.0625 (2^{-4}) 或 0.125 (2^{-3})，量化误差大得多。Recentralization 通过将每个聚类独立零中心化，使每个聚类内的量化精度最大化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Recentralized Quantization 不单独使用，而是作为 FQ 框架的内部组件。在实现中，GMM 的 EM 拟合为逐层离线计算（fine-tuning 期间每 k 个 epoch 更新一次超参数）；硬件实现中 μ_c 额外量化为 2 的幂次值，σ_+ = σ_- 约束为相等以融入逐层 α。Mayo 框架 (https://github.com/deep-fry/mayo) 包含完整实现。

涉及论文标题：
- Focused Quantization for Sparse CNNs
