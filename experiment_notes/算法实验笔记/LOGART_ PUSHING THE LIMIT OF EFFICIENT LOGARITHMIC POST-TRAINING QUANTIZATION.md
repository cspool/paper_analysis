## LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

- 属于算法pipeline的实现是什么？实验比较什么？
  - **LogART**：首个将对数域可学习舍入（Learnable Logarithmic Rounding, LLR）集成到 PTQ 中的方案。核心组件：
    1. **LLR（Learnable Logarithmic Rounding）**：在基-2对数域中将 RTN 替换为 floor + 可学习变量 R，通过 sigmoid 函数 σ(R) 使每个 weight 的舍入决策 soft 化。损失函数 = 逐层/逐块重建误差 (Frobenius 范数) + 正则化项（鼓励 σ(R) → 0 或 1）。梯度链：∂L/∂W̃ → ∂W̃/∂Q_W → ∂Q_W/∂R，其中包含指数项 2^{-Q_W} 和对数缩放因子 s·ln2。
    2. **OHS（Optimized Hyperparameter Search）**：三级搜索策略 —— (a) ABS: tensor-wise 非对称边界搜索（无需校准数据），基于 max/min weight 分配不同数量的正负码字；(b) SFS: block-wise 缩放因子搜索（通过最小化块级重建误差搜索最优 s_of 抵御 outlier）；(c) DBS: block-wise 动态基搜索（自适应分配 n₁:n₂ 的 base-√2 和 base-2 比例）。
    3. **Dynamic Base Quantizer**：大值用 base-√2（细粒度），小值用 base-2（粗糙但硬件友好），比例由 DBS 按分布感知方式搜索。量化码本含 n₁ 个 base-√2 码字和 n₂ 个 base-2 码字。
    4. **Asymmetric Quantizer**：首次为对数域设计非对称量化，通过自适应边界 l_a 为正值和负值分配不同码字数，解决 LLM 中常见的非对称 weight 分布。
    5. **Outlier-Resilient Quantizer**：引入可搜索超参数 s_of 替代 max(|W|) 来确定量化范围，实现自适应极值裁剪。
    6. **HAF（Hardware Approximation Function）**：用 K-term Signed Dyadic Expansion (SDE) 近似 √2（如 √2 ≈ 2⁰ + 2⁻¹），将乘 √2 替换为 shift-add 操作。HAF 嵌入 LLR 前向传播，近似误差在优化过程中被吸收为噪声。
  - 实验比较的 baselines：
    - LLM: GPTQ（linear/RTN）、BRECQ（linear/optimization）、AffineQuant（linear）、aespa（linear）
    - CNN: AdaRound（linear/learnable rounding）、BRECQ（linear）、FlexRound（linear）、LogNet（log/RTN）、SLogII（log）
    - Vision Transformer: BRECQ、APHQ（linear）、AdaLog（linear weight + log activation）、LogNet、SLogII

- 硬件平台是什么，配置是什么。
  - 单块 NVIDIA RTX 5090D GPU（32 GB）用于所有量化实验。
  - AE 硬件评估：Synopsys Design Compiler，28nm UMC 工艺，250 MHz，0.9V。

- 模型是什么。数据集和bench分别是什么。
  - LLM: OPT-125M, OPT-1.3B, OPT-6.7B, LLaMA2-7B, LLaMA3-8B；评估数据集 WikiText-2 (PPL) 和 C4 (PPL)
  - CNN: ResNet18, ResNet50, MobileNetV2；评估数据集 ImageNet (Top-1 Accuracy)
  - Vision Transformer: ViT-Small, ViT-Base, DeiT-Tiny, DeiT-Base；评估数据集 ImageNet (Top-1 Accuracy)
  - 校准数据：LLM 从 WikiText-2 或 C4 随机采样 32 段 2048 token；Vision 从 ImageNet 随机采样 2048 张无标签图片

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/logart-lab/logart
  - 算法 pipeline 伪代码：
  ```
  # LogART 量化流程（per-channel weight quantization）
  # 输入: FP16 weight W, calibration data X, bitwidth N
  
  # Step 1: OHS - 搜索最优超参数
  For each weight channel:
      # ABS: 基于 max/min 计算非对称边界 l_a（无需校准数据）
      w_h = max(w_max, -w_min); w_l = min(w_max, -w_min)
      l_a = floor(d_a / 2)  # d_a 由 w_h, w_l, threshold t 决定
  
  For each block (e.g., attention module):
      # SFS + DBS: 联合搜索 s_of 和 n₁:n₂ 以最小化块级重建误差
      # 校准集上的前向传播，搜索使 ||ΔW·X||_F² 最小的配置
      argmin_{s_of, n₁, n₂} E[||(W - W̃)X||_F²]
  
  # Step 2: LLR - 可学习对数舍入
  Initialize R = 0  # 每元素一个可学习变量
  For iter in 1..max_iters (LLM: 500, Vision: 2000):
      # Soft quantize (Eq. 12, 17):
      Q_W = clamp( floor(-log_B(|W| / (s_of * S))) + σ(R), 0, U )
      W̃ = S * sign(W) ⊙ B^{-Q_W}  # B ∈ {2, √2} per-element
  
      # HAF: 硬件近似的 √2 用 SDE 替代
      If B == √2:  # 仅对 base-√2 的奇数 Q_W 元素
          W̃' = W̃ * (1 + (γ - 1) * M)  # M = (Q_W mod 2) ⊙ [B == √2]
  
      # Loss:
      L_recon = tr(ΔW · E[XX^T] · ΔW^T)  # 逐层重建
      L_reg = λ * Σ(1 - |2σ(R) - 1|^β)   # 鼓励 hard rounding
      L = L_recon + L_reg
  
      # Gradient descent on R (Eq. 29):
      ∂L/∂R = 2s·ln2 · M_c ⊙ 2^{-Q_W} ⊙ sign(W) ⊙ [(WX - W̃X)X^T] ⊙ σ'(R)
             + λ · ∂f_reg/∂R
      R = R - lr * ∂L/∂R
  
  # Step 3: Hard rounding
  Q_W_hard = clamp( floor(-log_B(|W| / (s_of * S))) + round(σ(R)), 0, U )
  ```
  - 量化位宽：3-bit 和 4-bit 权重量化（weight-only）；支持与任意激活量化方法（SmoothQuant, AdaLog, QuaRot 等）组合
  - 优化器：Adam，CosineAnnealingLR scheduler，lr 从 0.05 衰减到 0.015，rounding loss weight λ=1
