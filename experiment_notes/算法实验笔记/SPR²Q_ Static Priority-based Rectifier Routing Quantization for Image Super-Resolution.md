## SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 SPR²Q，一种针对 Mamba 架构图像超分辨率模型的低比特后训练量化（PTQ）方法。包含两个核心组件：(1) **Pre-Quantization Fine-tuning with Fused Rectifier (PQFR)**：在量化前将低秩 rectifier 模块（ΔW = BA，A∈ℝ^{r×d_in}，B∈ℝ^{d_out×r}）学习到的权重增量融合到骨干网络，注入补偿信息，联合优化 rectifier 参数 (A,B) 和量化器参数 (a,b)；(2) **Static Priority-Based Rectifier Routing (SPR²)**：构建 N=4 个 rectifier 组成的 rectifier group，通过动态门控网络 g_i 加权聚合训练后，离线校准得到静态路由表（SPR²Q Table），推理时每个模块从路由表中检索最优增量并融合，不引入额外推理开销。
  - 实验比较：(a) 与 SOTA Mamba 量化方法对比（PTQ4VM, Quamba, MambaQuant）在 4-bit 和 2-bit 精度下；(b) 与 SwinIR Transformer 量化方法对比（2DQuant, FIMA-Q, APHQ-ViT）验证跨架构泛化性；(c) 消融实验：组件消融（PQFR→+RGT→+OSRC 逐步增益）、rectifier rank（r=2/4/8/16）、rectifier group size（N=2/4/8）；(d) 极端 1-bit 量化评估；(e) 实际效率：模型尺寸压缩（4-bit:2.51×, 2-bit:2.81×）和 FLOPs 加速（4-bit:3.44×, 2-bit:4.15×）。

- 硬件平台是什么，配置是什么。
  - NVIDIA RTX 4090 GPU。基于 PaddlePaddle 深度学习框架实现。

- 模型是什么。数据集和bench分别是什么。
  - 主模型：MambaIRv2-light（Mamba-based SR backbone）。跨架构泛化模型：SwinIR-light（Transformer-based）。
  - 训练集：DF2K（DIV2K + Flickr2K）。
  - 评估 Benchmark：Set5（5张）、Set14（14张）、B100（100张）、Urban100（100张）、Manga109（109张）。
  - 评估指标：PSNR 和 SSIM（在 YCbCr 空间的 Y 通道上测量）。
  - 缩放因子：×2 和 ×4。量化精度：4-bit、2-bit（主实验）、1-bit（极端实验）。
  - 对比 Baseline：PTQ4VM、Quamba、MambaQuant（Mamba 量化 SOTA）；2DQuant、FIMA-Q、APHQ-ViT（Transformer 量化 SOTA）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：论文未明确提供开源代码链接。论文声明基于 PaddlePaddle 框架实现。
  - 算法 pipeline 张量计算流程（SPR²Q, 4-bit, MambaIRv2-light, ×2 SR）：

  **阶段一 — Rectifier Group Training（动态路由训练, 12,000 iterations, batch=8）：**
  ```
  # 输入：预训练的 MambaIRv2-light 权重 W (frozen)
  # 初始化：N=4 个 rectifier {(A_i, B_i)}, A_i∈ℝ^{r×d_in}, B_i∈ℝ^{d_out×r}, r=8
  # 初始化：轻量门控网络 G（输出 N 维权值 g_i），量化器裁剪界 (a, b)
  # 优化器：Adam, lr=1e-2, Cosine Annealing

  for each training iteration:
      X_lr = sample_batch(DF2K)  # 低分辨率输入图像
      
      # 前向传播
      for each Mamba module l in model:
          # Step 1: 门控计算
          g = G(X_lq)  # g ∈ ℝ^N, softmax 归一化
          
          # Step 2: rectifier 加权聚合
          ΔW_fused = Σ_{i=1}^{N} g_i · (B_i @ A_i)  # 融合后增量
          
          # Step 3: 权重更新 + 伪量化
          W' = W + ΔW_fused                     # 融合 rectifier 补偿
          Ŵ' = clip(W', a, b)                   # 裁剪到 [a, b]
          s = (b - a) / (2^n - 1)               # 量化步长, n=4
          W_q' = round((Ŵ' - a) / s) · s + a    # 量化-反量化
          
          # Step 4: 量化权重前向计算
          Y = X_q @ W_q'                         # 线性变换输出
      
      # Step 5: Loss 计算
      L_pixel = || f_q(x) - y_FP ||_1           # 像素级重建 loss
      L_feature = Σ_{l=1}^{L} || φ_l(f_q(x)) - φ_l(f_FP(x)) ||_2²  # 逐块特征对齐 loss
      L = L_pixel + λ · L_feature               # 混合损失
      
      # Step 6: 反向传播 (STE 梯度估计)
      ∂L/∂A_i = g_i · B_i^T @ ∂L/∂W'           # STE 近似通过 round()
      ∂L/∂B_i = g_i · ∂L/∂W' @ A_i^T
      ∂L/∂(a,b) = ∂L/∂W_q' · ∂W_q'/∂(a,b)      # 裁剪界梯度 (Eq. 8)
      update(A_i, B_i, a, b, G) via Adam
  ```

  **阶段二 — Offline Static Routing Calibration（500 iterations）：**
  ```
  # 输入：训练好的 rectifier 组 {(A_i, B_i)} 和预训练权重 W（均 frozen）
  # 目标：为每个模块学习最优静态门控权重 ĝ

  for each Mamba module l:
      # 优化 ĝ ∈ ℝ^N（本文用梯度下降法, Eq. 12）
      ĝ = argmin_g L(f(X, Q_{a,b}(W + Σ g_i · (B_i@A_i))))
      # 收集 ĝ 并构建 SPR²Q Table
      SPR2Q_Table[l] = Σ ĝ_i · (B_i @ A_i)  # 预计算最优增量
  ```

  **阶段三 — 推理（零额外交付）：**
  ```
  for each Mamba module l:
      # 从 SPR²Q Table 检索该模块的最优增量
      ΔW_opt = SPR2Q_Table[l]
      
      # 权重融合（offline，实际推理前完成）
      W_final = W + ΔW_opt                    # 补偿后的权重
      
      # 量化（offline 完成）
      W_q_final = Q_{a,b}(W_final)            # 4-bit/2-bit 量化权重
      
      # 推理时直接加载量化权重进行前向计算
      Y = X @ W_q_final                       # 无额外门控、无动态路由
  ```

  **关键数值结果（4-bit, ×2）：** Set5 PSNR=37.72（vs PTQ4VM 37.17, MambaQuant 36.67）；Urban100 PSNR=31.53（vs PTQ4VM 30.47, MambaQuant 28.08）。MambaIRv2-light 从 3.01MB→1.20MB(4-bit, 2.51×), 1.07MB(2-bit, 2.81×)；FLOPs 从 75.6G→22.0G(4-bit), 18.2G(2-bit)。
  - 推理阶段无额外计算开销：所有 rectifier 参数离线融合，SPR²Q Table 在推理前已固化，模型结构与原始 MambaIRv2-light 完全一致。
