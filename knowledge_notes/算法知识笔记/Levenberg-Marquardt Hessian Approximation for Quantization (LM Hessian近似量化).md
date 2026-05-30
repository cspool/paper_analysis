## Levenberg-Marquardt Hessian Approximation for Quantization (LM Hessian近似量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Levenberg-Marquardt (LM) Hessian 近似是量化领域中将权重 W 的量化误差期望 E[||XW^T - X(W+Δ)^T||²] 近似为 Δ(X^T X)Δ^T 的数学方法。其推导路径：(1) 将权重量化误差 Δ = Ŵ - W 代入 MSE 展开：E[||X W^T - X Ŵ^T||²] = E[||X Δ^T||²]；(2) 使用二阶 Taylor 展开（在 Δ=0 处）：≈ Δ g^X + ½ Δ H^X Δ^T，其中 g^X 为梯度、H^X 为 Hessian 矩阵；(3) 对已训练模型 g^X = 0（最优性条件），且 H^X = E[2 X^T X]（Levenberg-Marquardt 近似，用一阶 Jacobian 的外积替代二阶 Hessian）；(4) 最终简化为 E[Δ (X^T X) Δ^T]，即量化误差由输入激活 X 的自相关矩阵 X^T X 加权。LM 近似的关键优势是计算高效——仅需一步矩阵乘法 X^T @ X，无需完整的二阶导数计算（后者在 LLM/V-DM 规模上不可行）。该近似最早由 Optimal Brain Compression (Frantar & Alistarh, NeurIPS 2022) 和 GPTQ (Frantar et al., 2022) 引入量化领域，S²Q-VDiT 进一步将其用于构造量化敏感度指标 C_quant = ||x_t^T x_t||_2 作为校准数据选择的一个维度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LM Hessian 近似在 SDS 中的应用
# 输入: 候选校准样本的隐变量 x_t ∈ R^{n×d}

# Step 1: 计算 LM 近似 Hessian
# H_X ≈ 2 * X^T X,  X ∈ R^{n×d}  (batch 维度展开)
H_approx = x_t.T @ x_t  # H_approx ∈ R^{d×d}, O(n·d²)

# Step 2: 提取量化敏感度（Hessian 的 L2 范数）
C_quant = ||H_approx||_2  # 即 ||x_t^T x_t||_2
# C_quant 越大 → 样本对量化扰动越敏感 → 更应在校准中被覆盖

# Step 3: 归一化并与其他指标联合
C_quant_norm = (C_quant - C_quant_min) / (C_quant_max - C_quant_min)
```

LM 近似在量化中的另一核心用途是 GPTQ 的逐列误差补偿：H_inv = Cholesky((H_approx + λI)^(-1))，利用 H_approx 的 Cholesky 分解引导量化误差沿未量化列进行补偿分配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LM Hessian 近似在 PyTorch 中的实现极为简洁：`H_approx = x_t.T @ x_t`（一行代码）。对于大规模模型，通常使用 group-wise 或 block-wise 方式分批计算以控制显存。S²Q-VDiT 中计算 LM 近似的额外开销极小——CogVideoX-2B 的 SDS 构造仅增加 0.009 分钟（7.708 → 7.717 min），HunyuanVideo-13B 增加 0.003 分钟（19.505 → 19.508 min）。计算完成后丢弃 H_approx，仅保留逐样本的标量 C_quant 得分用于排序选择。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---
