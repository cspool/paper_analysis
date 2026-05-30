## Optimal Scaling Factor for Binary Weights（二值权重最优缩放因子）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
二值权重最优缩放因子（Optimal Scaling Factor for Binary Weights）是 PB-LLM 在 QAT 框架下提出的解析推导方法。核心问题是：给定 FP 权重列向量 w_F ∈ R^n，将其二值化为 w̄_B = sign(w_F) ∈ {−1, +1}^n 后，如何选择标量缩放因子 α 使得重构误差最小？PB-LLM 证明当 w̄_B = sign(w_F) 时，存在闭式解：α* = ||w_F||_1 / n = mean(|w_F|)，即列向量的 L1 范数平均值。推导过程：最小化 L2 误差 J(α) = ||w_F − α w̄_B||²₂，展开得 J(α) = α²(w̄_B^T w̄_B) − 2α(w_F^T w̄_B) + (w_F^T w_F)。由于 w̄_B = sign(w_F), w̄_B^T w̄_B = n, w_F^T w̄_B = Σ|w_F,i| = ||w_F||₁，求 ∂J/∂α = 0 得 α* = (w_F^T w̄_B)/n = ||w_F||₁/n。这与 XNOR-Net 的 channel-wise L1 norm scaling 在形式上一致，但 PB-LLM 的创新在于将其集成到部分二值化的 QAT 框架中，与 Salient Weights Frozen 协同工作，使训练更高效——无需像 AWQ 那样通过 grid search 寻找最优 scaling factor。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B 某 Linear 层的一个权重列为例（QAT 前向传播）：
```
# 输入: FP latent weight column w_F ∈ R^n (unsalient 权重)
#       salient weights 已 frozen 为 INT8

# Step 1: 二值化
w_bar_B = sign(w_F)  # ∈ {−1, +1}^n

# Step 2: 最优缩放因子（闭式解，无需搜索）
alpha_star = sum(|w_F|) / n  # = ||w_F||₁ / n

# Step 3: 重构二值化权重
w_hat = alpha_star * w_bar_B  # 最优 L2 逼近

# Step 4: 混合精度前向传播
y = W_salient @ x + alpha_star * sign(W_F_unsalient) @ x
# salient 部分: INT8 × FP16 → FP16
# unsalient 部分: (α × sign) × FP16 → FP16 GEMM
```
注意：该最优解仅在 w̄_B = sign(w_F) 的条件下成立——如果使用其他二值化策略（如随机二值化），则需要重新求解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PB-LLM 的 QAT 实现中，该缩放因子按 column-wise 粒度计算——每个输出 channel 的权重列计算独立的 α*。这是基于以下考量：LLM 中不同 channel 的权重幅度差异很大（channel-wise variance），per-column scaling 比 per-tensor scaling 能更精确地近似原始权重分布。与 AWQ 的 per-channel scaling 不同，AWQ 的 scaling 应用于量化前（作为预处理），而 PB-LLM 的 scaling 是二值化后的重构步骤（作为 post-binarization correction）。训练时 α* 每步前向重新计算（因为 w_F 随训练更新），推理时 α* 固定存储（每个列一个 FP16 标量，开销可忽略）。

涉及论文标题：
- PB-LLM Partially Binarized Large Language Models

---
