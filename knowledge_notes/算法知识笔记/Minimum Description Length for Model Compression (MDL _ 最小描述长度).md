## Minimum Description Length for Model Compression (MDL / 最小描述长度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Minimum Description Length (MDL) 是信息论中的模型选择原则，由 Hinton & van Camp (1993) 和 Graves (2011) 引入神经网络压缩。MDL 在模型压缩中将优化问题表述为在给定数据集 D 下，找到权重 θ 和超参数 φ 的最优编码，使得描述模型的总代价最小：`L(θ, φ) = L_E + L_C`，其中：

- `L_E = E_{θ~q_φ(θ)}[-log p(y|x, θ)]` 是**误差代价**（Error Cost），即量化模型在数据集上的交叉熵损失。
- `L_C = KL(q_φ(θ) || p(θ|D))` 是**复杂度代价**（Complexity Cost），即量化权重分布 q_φ(θ) 与原始后验分布 p(θ|D) 之间的 KL 散度。

直观上：`L_E` 惩罚精度损失，`L_C` 惩罚模型复杂度。FQ 将这一框架应用于量化：`L_E` 通过 SGD fine-tuning 优化，`L_C` 通过最小化 GMM 分布与原始权重分布的 KL 散度来近似——而拟合 GMM 的 MLE 恰好等价于最小化该 KL 散度。因此 FQ 的 EM+GMM 步骤和 fine-tuning 步骤交替进行，对应 MDL 目标中 `L_C` 和 `L_E` 的交替优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MDL 双目标优化在 FQ pipeline 中的对应关系：

```
# MDL Objective:
#   min L(θ, α, φ) = L_E + L_C
#
#   L_E = E_{θ~q}[ -log p(y|x, α, θ_hat) ]  ← Cross-entropy loss
#   L_C = KL( q_φ(θ) || p(θ|D) )            ← Distribution matching

# Optimization Strategy (alternating):

# Phase 1: Optimize L_C (Complexity Cost)
# → GMM fitting via EM (MLE):
#   这一步找到 q_φ^mix(θ) ≈ p(θ|D)
#   等价于 min KL( q_φ^mix(θ) || p(θ|D) )
for each layer:
    μ_c, σ_c, λ_c = EM_fit(θ_nonzero)  # Find best GMM

# Phase 2: Optimize L_E (Error Cost)
# → Fine-tuning with SGD:
#   固定量化超参数 φ，优化权重 θ 和缩放 α
#   Forward: θ_hat = Q[θ; φ]  (quantized)
#   Loss: CE(θ_hat(x), y)
#   Backward: STE through Q[·]
for epoch in range(epochs):
    train_one_epoch()  # Standard SGD

# Repeat: alternate between Phase 1 and Phase 2
# (FQ paper: update φ every k epochs, k increasing exponentially)
```

**Annotations**: MDL 框架解释了为什么 FQ 需要交替进行 GMM fitting 和 fine-tuning——二者分别优化 L_C 和 L_E；单纯的 post-training quantization 只优化 L_C 而忽略 L_E，会导致较大精度损失；而仅 fine-tuning 不调整量化参数则 L_C 无法改善。MDL 为 FQ 的设计提供了理论支撑。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MDL 在模型压缩中主要作为理论框架（而非具体实现代码）使用。实际实现中不显式计算 KL 散度，而是通过 MLE（EM 算法）隐式最小化 L_C，通过 SGD 最小化 L_E。Hinton & van Camp (1993) 的原始工作使用 Gaussian 近似，Graves (2011) 将其推广到更一般的变分推断。FQ 使用 GMM 作为 q_φ 的参数化形式，比单一高斯更具表达力，能捕获剪枝后权重的多峰分布。

涉及论文标题：
- Focused Quantization for Sparse CNNs
