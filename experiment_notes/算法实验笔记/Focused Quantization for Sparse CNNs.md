## Focused Quantization for Sparse CNNs

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 **Focused Quantization (FQ)**，一种针对稀疏 CNN 的混合量化策略，将 shift quantization（权重量化为 2 的幂次值 `{0,±1,±2,±4,...}`）与 recentralized quantization 结合。核心创新：**(1) Recentralized Quantization**：对稀疏层的权重分布拟合高斯混合模型（GMM，2 个分量），用 EM 算法找到高概率密度区域，独立对每个区域做 shift quantization（先减均值除以标准差，shift quantize 后再反变换），使量化层级集中匹配权重分布。**(2) Wasserstein 分离判定**：用 2-Wasserstein 距离衡量两个高斯分量的分离程度，当 `W(c₁,c₂) < w_sep`（默认 2.0）时退化为普通 shift quantization，自适应选择量化策略。**(3) MDL 视角优化**：将量化建模为最小描述长度（MDL）优化，误差代价 `L_E` 为交叉熵，复杂度代价 `L_C` 为 KL 散度。**(4) 完整压缩流水线**：Dynamic Network Surgery 细粒度剪枝 → FQ 量化 → INQ（增量量化，逐步增加量化比例 25%→50%→75%→87.5%→100%，每步 fine-tune 3 epochs, LR=0.001，最后一步 10 epochs）→ Huffman 编码。

  实验对比：
  - Baselines：TTQ（三元量化）、INQ（shift quantization, 2/3/5 bit）、ADMM（极低比特）、ABC-Net（5 bases 二值卷积）、LQ-Net（可学习量化, 2 bit）、D&Q（蒸馏+量化）、Coreset-Based Compression、ThiNet（filter 剪枝）、Clip-Q（剪枝+量化+权重共享）
  - 评估配置：5-bit FQ、7-bit FQ
  - 评估指标：Top-1/Top-5 准确率、压缩率 CR（×）、模型大小（MB）、Sparsity（%）、logic gate 数量（硬件效率）
  - 消融实验：Wasserstein 分离阈值 w_sep 从 1.0 到 3.5 以 0.1 递增（CIFAR-10 9层 CNN，每个值训练 100 次）
  - 渐进量化消融：逐步量化 weights（5-bit FQ）→ activations（8-bit integer）→ BN parameters（16-bit integer）

- **硬件平台是什么，配置是什么。**
  训练平台论文未详细说明。硬件资源评估针对自定义加速器，使用 3×3 卷积、padding=1、8×8×100 输入激活、8×8×100 输出，估算双输入逻辑门数下界（unrolled architecture, same throughput）。FPGA 加速器生成参见配套工作 [24] "Automatic generation of multi-precision multi-arithmetic CNN accelerators for FPGAs"（ICFPT 2019）。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：ResNet-18、ResNet-50、MobileNet-V1、MobileNet-V2；CIFAR-10 快速分类器（9 层 CNN，用于 w_sep 消融）
  - 数据集/Benchmark：ImageNet（ILSVRC 2012）用于主要评估；CIFAR-10 用于超参数消融
  - 剪枝方法：Dynamic Network Surgery [6]

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/deep-fry/mayo（Mayo 框架）。

  **FQ 算法pipeline核心流程（逐层处理）**：

  ```
  # ===== 阶段1: 剪枝后权重预处理 =====
  # 输入: W ∈ R^{Cout, Cin, Kh, Kw}  (已由 Dynamic Network Surgery 剪枝)
  # z_θ ∈ {0,1}: pruning mask

  # ===== 阶段2: FQ Recentralized Quantization (逐层) =====
  # Step 2.1: 拟合高斯混合模型
  θ_nonzero = {w ∈ W | w ≠ 0}  # 非零权重集合
  初始化: μ_- = mean({θ < 0}), σ_- = std({θ < 0})
          μ_+ = mean({θ > 0}), σ_+ = std({θ > 0})
          λ_- = λ_+ = 0.5

  # EM 算法求 MLE
  repeat until convergence:
      # E-step: 计算每个权重属于各分量的后验概率
      γ_c(θ) = λ_c * N(θ|μ_c, σ_c) / Σ_j λ_j * N(θ|μ_j, σ_j)
      # M-step: 更新参数
      N_c = Σ_θ γ_c(θ)
      μ_c = Σ_θ γ_c(θ) * θ / N_c
      σ_c^2 = Σ_θ γ_c(θ) * (θ-μ_c)^2 / N_c
      λ_c = N_c / |θ|

  # Step 2.2: 分量分配与量化
  for each weight θ:
      m_θ = argmax_c λ_c * N(θ|μ_c, σ_c)  # 选择最可能的分量
      # Recentralize: 归一化到零均值
      θ_norm = (θ - μ_{m_θ}) / σ_{m_θ}
      # Shift quantize: 量化为 2 的幂
      θ_hat_norm = Q^{shift}_{n,b}(θ_norm)  # n-bit shift quantization
      # De-normalize
      Q[θ] = z_θ * α * (θ_hat_norm * σ_{m_θ} + μ_{m_θ})

  # Step 2.3: Wasserstein 分离判定
  # 归一化方差
  σ²_global = Var(θ_nonzero)
  W(c₁,c₂) = ((μ₊-μ₋)² + (σ₊-σ₋)²) / σ²_global
  if W(c₁,c₂) < w_sep (default 2.0):
      退化为 shift quantization (精度高 1 bit，因不需要 m_θ bit)
  ```
  
  最终量化后值形式：`Q_c^{rec}[θ] = Q^{shift}_{n,b}[(θ-μ_c)/σ_c] * σ_c + μ_c`
  
  其中 `Q^{shift}_{n,b}[v] = s * 2^{e-b}`（s ∈ {-1,0,1}, e ∈ [0,2^k-1], b 为逐层 bias）。
  
  乘法被替换为 bit-shift：`x * (s * 2^{e-b}) = s * (x << (e-b))`（或 `>>` 当 e<b 时）。

  **硬件实现优化**：
  - μ₊, μ₋ 量化为最近的 2 的幂次值
  - σ₊ 和 σ₋ 约束为相等，可融合到逐层缩放因子 α 中
  - α 可融入 BN 融合，消除推理时乘法
  - 5-bit FQ 内部使用 3-bit 无符号 shift quantization（1 bit sign + 1 bit component selection + 3 bit shift value = 5 bit total）
  - 最终 dot-product 仅含 bit-shift 和整数加法，无浮点乘法
