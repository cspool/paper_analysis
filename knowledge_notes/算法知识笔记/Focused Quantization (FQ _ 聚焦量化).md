## Focused Quantization (FQ / 聚焦量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Focused Quantization (FQ) 是专为稀疏 CNN 设计的混合量化策略，由 Cambridge/SIAT/UMacau 团队于 2019 年提出。FQ 的核心思想是：**将量化 effort（即量化层级资源）从权重分布稀疏的零附近重新集中到剪枝后权重实际分布的高概率区域**。FQ 包含两个子策略：

1. **Recentralized Quantization**：对逐层非零权重拟合 2-分量高斯混合模型（GMM，使用 EM 算法求 MLE），找到两个高概率密度聚类（Cluster+ 和 Cluster-），分别对每个聚类以各自的均值和标准差做零中心化（减均值、除标准差），再在归一化后的小范围内做 shift quantization，最后反变换回去。公式为 `Q_c^{rec}[θ] = Q^{shift}_{n,b}[(θ-μ_c)/σ_c] * σ_c + μ_c`。
2. **Shift Quantization（退化模式）**：当两个高斯分量高度重叠时（2-Wasserstein 距离 < 阈值 w_sep=2.0），Recentralized Quantization 退化为普通 shift quantization，且因为不需要 component selection bit，精度等效提升 1 bit。

FQ 的 5-bit 实现实际使用：1-bit sign + 1-bit component selection + 3-bit unsigned shift value = 5 bits total。硬件实现中，μ_c 额外量化为 2 的幂次值，σ_+ 和 σ_- 约束相等并融合到 α 缩放因子中，α 再融入 BN，消除推理时的所有乘法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FQ 完整 pipeline 伪代码（逐层处理）：

```
# ===== Preprocessing: EM for GMM fitting =====
θ_nonzero = {w ∈ W | w ≠ 0}

# Initialize
μ_- = mean({θ < 0}), σ_- = std({θ < 0})
μ_+ = mean({θ > 0}), σ_+ = std({θ > 0})
λ_- = λ_+ = 0.5

# EM Algorithm (repeat until convergence)
repeat:
    # E-step: Compute responsibilities
    for each θ:
        γ_c(θ) = λ_c * N(θ|μ_c, σ_c) / Σ_j λ_j * N(θ|μ_j, σ_j)

    # M-step: Update parameters
    for each component c:
        N_c = Σ_θ γ_c(θ)
        μ_c = Σ_θ γ_c(θ) * θ / N_c
        σ_c^2 = Σ_θ γ_c(θ) * (θ - μ_c)^2 / N_c
        λ_c = N_c / |θ|

# ===== Wasserstein Separation Check =====
σ²_global = Var(θ_nonzero)
W = ((μ_+ - μ_-)² + (σ_+ - σ_-)²) / σ²_global

if W < w_sep (2.0):
    # Use plain Shift Quantization (higher precision by 1 bit)
    for each θ:
        θ_hat = Q^{shift}_{n,b}(θ)  # n-bit, no component bit needed
else:
    # Use Recentralized Quantization
    # Quantize μ_c to nearest power-of-two
    for each θ:
        m_θ = argmax_c λ_c * N(θ|μ_c, σ_c)  # component assignment
        θ_norm = (θ - μ_{m_θ}) / σ_{m_θ}    # recentralize
        θ_hat_norm = Q^{shift}_{n-1,b}(θ_norm)  # (n-1)-bit shift quant
        θ_hat = α * (θ_hat_norm * σ_{m_θ} + μ_{m_θ})

# ===== Fine-tuning with INQ schedule =====
# Gradually increase quantized proportion: 25% → 50% → 75% → 87.5% → 100%
# Fine-tune at each step, update GMM hyperparams every k epochs
```

**Annotations**: GMM fitting 依赖剪枝后权重分布；EM 收敛到局部最优；Wasserstein 阈值 w_sep=2.0 是通过 CIFAR-10 上 9 层 CNN 的 grid search（1.0~3.5, step=0.1）确定的；INQ schedule 中每步 3 epochs（最后一步 10 epochs）, LR=0.001, 每 3 epochs 衰减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FQ 作为压缩 pipeline（称为 Focused Compression, FC）的一部分使用：Dynamic Network Surgery 细粒度剪枝 → FQ 量化 → INQ 增量 fine-tune → Huffman 编码。FQ 的逐层自适应特性使其特别适用于剪枝后稀疏度不均匀的 CNN（ResNet、MobileNet 等）。开源实现见 Mayo 框架 (https://github.com/deep-fry/mayo)。FQ 最适合的场景是：同时追求高压缩率（18× CR on ResNet-50）和极低硬件开销（5-bit FQ 仅需 275.6M 逻辑门，与 3-bit shift quant 相当），且对精度损失容忍度低（Top-5 损失 ≤0.24%）。

涉及论文标题：
- Focused Quantization for Sparse CNNs
