## PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PassionSR 是面向 one-step diffusion (OSD) 图像超分模型的 post-training quantization (PTQ) 方法。核心包含三个设计：(1) **模型简化**——将 OSEDiff 的 DAPE-CLIPEncoder 分支替换为基于空字符串的常数 embedding，得到仅包含 UNet 和 VAE 的 PassionSR-FP（参数减少 27.13%，操作减少 6.25%，性能几乎无损）；(2) **Learnable Boundary Quantizer (LBQ)**——使用可训练的上下界参数 B_l, B_u 替代传统 fixed-range 量化器，通过 fake quantization 模拟量化误差，量化与反量化公式为 X_c = Clip(X, B_l, B_u), α = (B_u-B_l)/(2^N-1), β = B_l, X_I = round((X_c-β)/α), X_q = α·X_I+β；(3) **Learnable Equivalent Transformation (LET)**——在线性层、卷积层和注意力矩阵乘法中引入逐通道可学习 scale factor s 和 offset δ，通过等效变换（Linear: W̃ = s⊙W, X̃ = (X-δ)⊘s, B̃ = B+δW；Attention: Q̃ = Q⊘s, K̃ = s⊙K）重新分布激活值以抑制离群值，s 和 δ 可融入前序层或权重/偏置中，无额外推理开销；(4) **Distributed Quantization Calibration (DQC)**——将标定过程分为两阶段：Stage 1 仅训练 LET 的 scale/offset，Stage 2 重新初始化 LBQ 并联合训练，从而稳定训练、加速收敛并降低 GPU 显存。
  - 实验比较：PassionSR (W8A8/W6A6) vs **MaxMin** (传统 min-max 量化)、**LSQ** (learned step size QAT)、**Q-Diffusion** (多步扩散量化 PTQ)、**EfficientDM** (QALoRA-based 量化微调)。所有对比方法均基于 PassionSR-FP backbone 进行量化。在 RealSR、DRealSR、DIV2K val 三个数据集上评估 PSNR/SSIM/LPIPS/DISTS（参考 IQA）和 NIQE/MUSIQ/MANIQA/CLIP-IQA（非参考 IQA）。消融实验验证 LBQ、LET、DQC 各组件的贡献。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明 GPU 型号。消融实验表 4 显示 GPU 显存占用为 28-40 GB，推测使用 NVIDIA A100 (40GB) 或类似高端 GPU。软件环境：CUDA 11.8 + PyTorch 2.0.1。标定训练使用单卡（CUDA_VISIBLE_DEVICES="0"）。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：以 OSEDiff (NeurIPS 2024) 为 backbone，简化后得到 PassionSR-FP（仅含 UNet + VAE，FP32 下参数 949M、操作 4,240G）。量化设置：PassionSR-U（仅 UNet 量化）和 PassionSR-UV（UNet+VAE 量化），W8A8 和 W6A6 两种精度。
  - **标定数据集**：从 DIV2K train 中随机裁剪 500 对 128×128 LR-HR 图像。
  - **测试数据集**：RealSR、DRealSR、DIV2K val。全尺寸图像评估。
  - **评估指标**：PSNR、SSIM、LPIPS、DISTS（参考 IQA）；NIQE、MUSIQ、ManIQA、CLIP-IQA（非参考 IQA）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/libozhu03/PassionSR（CVPR 2025），含 PTQ 标定脚本 `ptq_quantize_single.py`、推理脚本 `inference_single.py`、评估脚本 `measure.py`、YAML 配置及预训练模型链接（Google Drive）。
  - PassionSR 量化 pipeline（以 OSEDiff 的 UNet + VAE 为例，标定 W8A8）：
    ```
    # === Step 1: 模型简化 ===
    # 将 OSEDiff 的 text embedding 分支替换为空字符串的 ClipEncoder 输出常数
    # 删除 DAPE 模块，仅保留 UNet + VAE → PassionSR-FP

    # === Step 2: 构建 LBQ 量化器 ===
    # 对每层权重 W 和激活 X，定义可训练上下界 B_l, B_u
    # 前向: X_c = clamp(X, B_l, B_u)
    #       alpha = (B_u - B_l) / (2^bit - 1)
    #       X_q = alpha * round((X_c - B_l) / alpha) + B_l
    # STE 反向传播梯度

    # === Step 3: LET 等效变换 ===
    # 对 Linear 层 (X ∈ R^{N×C_in}, W ∈ R^{C_in×C_out}):
    #   s = exp(param_s), delta = param_delta  # 可学习参数
    #   W_tilde = s ⊙ W                          # 按元素乘
    #   X_tilde = (X - delta) ⊘ s               # 按元素除
    #   B_tilde = B + delta @ W
    # 对 Conv 层: 沿 channel 维度应用相同变换
    # 对 Attention (Q,K,V 矩阵乘法):
    #   Q_tilde = Q ⊘ s, K_tilde = s ⊙ K
    #   P_q = Softmax(Q_a1(Q_tilde) · Q_a2(K_tilde^T))
    # 变换后 s,δ 合并入前层/权重，无额外推理开销

    # === Step 4: DQC 两阶段标定 ===
    # Stage 1: 冻结 LBQ，仅训练 LET 的 s 和 δ
    #   for epoch in range(2):
    #       for X_lr, X_hr in calib_loader:
    #           Y_q = quantized_forward(X_lr)   # LBQ(fixed) + LET(trainable)
    #           Y_fp = fp_forward(X_lr)
    #           loss = MSE(Y_q, Y_fp)           # 模块级逐层标定
    #           loss.backward()                 # 仅更新 LET 参数
    # Stage 2: 重新初始化 LBQ，联合训练 LBQ + LET
    #   for epoch in range(2):
    #       for X_lr, X_hr in calib_loader:
    #           Y_q = quantized_forward(X_lr)   # LBQ+LET 均可训练
    #           loss_unet = ||I(Z_lq, ε_q) - I(Z_l, ε_fp)||_2  # latent space MSE
    #           loss_vae_e = ||V_qe(X_fp) - V_fpe(X_fp)||_2
    #           loss_vae_d = ||V_qd(X_q) - V_fpd(X_fp)||_2
    #           loss.backward()                 # 更新 LBQ 和 LET 参数

    # 标定完成，scale/offset 合并入权重，得到 INT8 推理模型
    # 推理: HR ≈ VAE_decoder(UNet_int8(VAE_encoder_int8(LR)))
    ```
