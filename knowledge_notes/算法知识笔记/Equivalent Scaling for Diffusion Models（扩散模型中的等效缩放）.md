## Equivalent Scaling for Diffusion Models（扩散模型中的等效缩放）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Equivalent Scaling（等效缩放）是一种通过插入通道级缩放因子 τ 来双向重分布激活与权重的量化难度的方法。数学上，线性层 Y = XW 被改写为 Y = (X/τ)(τ^T ⊙ W)，其中 / 为通道级除法，⊙ 为通道级乘法。τ ∈ R^{Cin} 将某些通道的激活缩小（降低激活量化难度）的同时将对应通道的权重放大（增加权重量化难度），或反之。因整体输出 Y 不变，这种变换是数学等效的。该方法最早由 SmoothQuant（Xiao et al., 2023）引入 LLM 量化，使用手工启发式 τ_c = (max|X_c|)^β / (max|W_c|)^{1-β}。DMQ 论文发现 SmoothQuant 直接迁移到扩散模型会严重失败：因为扩散模型中激活幅度远大于权重幅度，手动 τ 极大 → 权重量化范围被大幅扩展 → 权重量化误差暴增（FFHQ W4A8: Weight Quant. Error 从 0.0060 飙升至 0.0694，FID 从 36.08 飙升至 454.16）。DMQ 提出 Learned Equivalent Scaling (LES)，通过梯度下降直接学习 τ 以最小化量化输出的 MSE（L_i = ||X_iW - Q(X̂_i) Q(Ŵ)||²），避免手动启发式的不准确。LES 的 τ 通过融合策略（τ^T ⊙ W 预计算，τ ⊙ s^X 融合到激活 scale）实现零推理开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 DMQ 的 LES 学习流程为例（W4A8, LDM-4 FFHQ 256×256）：

```
# 输入: FP32 权重 W ∈ R^{Cin×Cout}, 校准集 {(X_i, t_i)}
# 初始化 τ = 1 (所有通道不缩放)
# 累积损失 Λ_t 初始化为 0

for iteration in range(6000):
    batch = sample_calibration_batch(B=32)
    for each (X_i, t_i) in batch:
        # Eq.5: 等效缩放变换
        X_hat = X_i / τ           # 激活缩放 (channel-wise)
        W_hat = τ^T ⊙ W           # 权重缩放 (channel-wise)
        
        # MinMax 量化
        X_q = MinMaxQuant_8bit(X_hat)   # per-tensor
        W_q = MinMaxQuant_4bit(W_hat)   # per-channel
        
        # Eq.6: Layer-wise MSE loss
        L_i = ||X_i @ W - dequant(X_q) @ dequant(W_q)||^2
        
        # Eq.8: Adaptive timestep weight
        lambda_ti = (1 - Λ_{ti} / sum(Λ)) ^ α  # α=20
        loss += lambda_ti * L_i
    
    # 梯度下降更新 τ
    τ = τ - lr * ∇_τ loss
    
    # Eq.9: 更新累积损失 (EMA)
    for each t:
        Λ_t = 0.95 * Λ_t + 0.05 * mean(L_i for i where t_i=t)
```

**融合到推理（零开销）**：
- 权重侧：W_fused = τ^T ⊙ W（预计算存储，替换原 FP32 权重后再量化）
- 激活侧：s_X_fused = τ ⊙ s_X（预计算融合 scale，量化时使用）
- 推理时：X_q = round(X / s_X_fused)，W_q 已含 τ 信息

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 LLM 中，SmoothQuant 的 τ 通过手动公式计算后融合到前一层（LayerNorm weight 或前一层 Linear weight）。但扩散模型中 U-Net 包含非线性的 Swish/SiLU 激活（位于矩阵乘法之前），阻止了 τ 融合到前一层。DMQ 改用 τ 融合到激活 scale（s_X_fused = τ ⊙ s_X），利用静态量化的量化 scale 不变特性回避这个问题。实现时可复用现有量化框架（BRECQ/AdaRound 用于后续的权重量化精炼）。LES 因素仅需几千次迭代优化，远超于 QAT 的完整训练开销，适合 PTQ 的高效部署场景。开源：https://github.com/LeeDongYeun/dmq。

涉及论文标题：
- DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

---
