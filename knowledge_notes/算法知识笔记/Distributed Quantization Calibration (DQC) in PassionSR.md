## Distributed Quantization Calibration (DQC) in PassionSR

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distributed Quantization Calibration (DQC) 是 PassionSR 提出的两阶段 PTQ 标定策略，旨在解决同时训练 LBQ 的边界参数和 LET 的等效变换参数时的训练不稳定问题。DQC 将整个标定过程拆分为两个阶段：(1) **Stage 1**：冻结 LBQ（使用初始边界），仅训练 LET 的 scale factors s 和 offsets δ。此阶段 LET 将激活分布调整为对量化友好的形态。(2) **Stage 2**：在 LET 更新后的变换向量上重新初始化 LBQ，然后联合训练 LBQ 和 LET。重新初始化的原因在于：LET 的变换改变了各层的激活/权重分布，旧的 LBQ 边界已不再适用。DQC 的核心效果是：标定时间从 3.87h（LBQ+LET 联合训练）降至 1.07h，GPU 显存从 40GB 降至 28GB，显著加速收敛（Fig. 5 展示了 w/ 和 w/o DQC 的 loss 曲线对比）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PassionSR 中 DQC 的执行流程：
```
# DQC Stage 1: 仅训练 LET
for param in LBQ_parameters: param.requires_grad = False
for param in LET_parameters: param.requires_grad = True
for epoch in range(2):
    for X_lr, X_hr in calib_loader:
        Y_fp = fp_model(X_lr)         # FP32 教师输出
        Y_q = quantized_model(X_lr)    # LBQ(frozen) + LET(trainable) 量化输出
        loss = block_wise_mse(Y_q, Y_fp)
        loss.backward()                # 梯度仅流向 LET 的 s, δ

# DQC Stage 2: 重新初始化 LBQ + 联合训练
reinitialize_LBQ_boundaries()          # 基于 LET 变换后的分布重新初始化 B_l, B_u
for param in LBQ_parameters: param.requires_grad = True
for param in LET_parameters: param.requires_grad = True
for epoch in range(2):
    for X_lr, X_hr in calib_loader:
        Y_fp = fp_model(X_lr)
        Y_q = quantized_model(X_lr)    # LBQ+LET 均可训练
        # 模块级逐层损失
        loss_UNet = ||I(Z_lq, ε_q) - I(Z_l, ε_fp)||_2   # latent space MSE
        loss_VAE_enc = ||V_qe(X_fp) - V_fpe(X_fp)||_2
        loss_VAE_dec = ||V_qd(X_q) - V_fpd(X_fp)||_2
        total_loss = loss_UNet + loss_VAE_enc + loss_VAE_dec
        total_loss.backward()          # 梯度流向 LBQ 和 LET
```
UNet 损失函数中的 I(Z_l, ε) 是 OSD 模型特有的从输入 latent Z_l 和预测噪声 ε 到输出 latent Z_h 的变换函数：Z_h = sqrt(1/α̂) · Z_l - sqrt((1-α̂)/α̂) · ε(Z_l)，利用 OSD 模型时间步为常数的特性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DQC 适用于以下场景：(1) 量化器包含可训练参数且与等效变换参数耦合导致训练不稳定时；(2) 需要降低标定 GPU 显存和时间成本时。实现要点：Stage 1 用较小的学习率稳定搜索 LET 的最优解空间；Stage 2 必须重新初始化 LBQ（否则新分布上的旧边界导致性能退化）。PassionSR 开源代码中在 `ptq_quantize_single.py` 实现了 DQC 的完整流程，其中 epoch 数、学习率（1e-5）、标定 batch 均可配置。注意：PassionSR 的 DQC 与 2DQuant 的 DQC（Distillation-based Quantization Calibration）虽然缩写相同但机制不同——PassionSR 的两阶段解耦训练 vs 2DQuant 的知识蒸馏微调 clip bounds。

涉及论文标题：
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

---
