## Bias-Corrected Output Stabilization (BCOS / Channel-wise Affine Compensation for VQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BCOS（Bias-Corrected Output Stabilization）是 KBVQ-MoE 框架的后处理模块，校正 VQ 量化后 MoE expert 输出的 distributional shift。问题背景：量化误差在各层累积导致 layer output 的 per-channel mean/variance 偏离 FP16 基线。MoE 架构中多个 expert 的 biased outputs 通过 gating weights 加权求和后被聚合放大，distributional drift 比 dense LLM 更严重（Fig. 3: Direct VQ 后 mean/variance 显著偏离 FP）。BCOS 以 channel-wise affine compensation 校正：`y_corr = (1+s) ⊙ ŷ + b`，其中 ŷ = W_VQ x。s 和 b 基于 MMSE 闭式解：`s_j ≈ σ_{y_j}/σ_{ŷ_j} - 1`, `b_j = μ_{y_j} - (1+s_j)μ_{ŷ_j}`。推导（Appendix A.4）：将校正化为 `min_{s,b} E[||y - ((1+s)⊙ŷ + b)||²]`，对第 j 个 channel 等价于一元线性回归，闭式解为 `α_j = Cov(y_j,ŷ_j)/Var(ŷ_j)`, `b_j = μ_{y_j} - α_j μ_{ŷ_j}`。因 y_j 和 ŷ_j 高度相关（仅差量化噪声），近似 `Cov(y_j,ŷ_j) ≈ σ_{y_j} σ_{ŷ_j}` 得 `s_j = α_j - 1 ≈ σ_{y_j}/σ_{ŷ_j} - 1`。该近似在高相关条件下等价于 MMSE-optimal 估计，非启发式调整。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BCOS 参数估计 + 推理流程
# 离线: 估计 s,b (per expert, per layer)
输入: 原始权重 W, 量化权重 Ŵ = W_share + W_quant,VQ
      校准激活 X

# 收集 per-channel 统计量
y = W @ X                                      # 原始输出 oc×B
ŷ = Ŵ @ X                                      # 量化输出 oc×B

for j in 1..oc:
    μ_y[j] = mean(y[j, :])
    σ_y[j] = std(y[j, :])
    μ_ŷ[j] = mean(ŷ[j, :])
    σ_ŷ[j] = std(ŷ[j, :])

# 计算校正参数 (MMSE 闭式解)
for j in 1..oc:
    s[j] = σ_y[j] / σ_ŷ[j] - 1                 # scale: 对齐 variance
    b[j] = μ_y[j] - (1 + s[j]) * μ_ŷ[j]        # bias: 对齐 mean

# 推理: channel-wise affine
y_corr = (1 + s) ⊙ (Ŵ @ x) + b                 # 逐 channel 乘加
```

存储开销：每层 2·oc 个 FP16 参数（s 和 b 各 oc 个），对 Qwen1.5-MoE-A2.7B gate_proj (oc=5632) 仅 ~22KB/layer。推理计算：仅 element-wise multiply-add，<0.1% expert forward FLOPs。消融（Table 15: BCOS 内部分解）：variance-only 贡献更大（PPL 11.03→10.38），mean-only 贡献小（11.03→11.01），mean+variance 组合最佳（9.61）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BCOS 配合 IDRE 使用效果最佳（IDRE 先消除冗余降低量化难度，BCOS 再校正残余误差）。IDRE+BCOS 组合在 Qwen3-30B-A3B 3-bit 下 PPL 从 18.72（无处理）降至 9.26（-50.5%）。BCOS 为通用模块：可作为 plugin 集成到任何 MoE VQ pipeline 中，在 GPTVQ 和 VPTQ 上均有验证（Table 5: GPTVQ+IDRE+BCOS vs GPTVQ only 在 Qwen1.5-MoE-A2.7B 2-bit 下 PPL 从 12.88 降至 9.43）。局限：BCOS 的 scale 近似 `Cov(y_j,ŷ_j) ≈ σ_{y_j} σ_{ŷ_j}` 在极低比特下（如 1-bit）相关度下降时精度可能降低，论文未测试该场景。

涉及论文标题：
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

---
