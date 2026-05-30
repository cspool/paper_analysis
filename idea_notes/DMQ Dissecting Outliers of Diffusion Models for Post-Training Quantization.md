## DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization

- baseline方法是什么？
  Baseline 是 TFMQ-DM (CVPR 2024)，结合基本 PTQ 量化（MinMax quantizer + BRECQ/Adaround）和跨时间步校准策略。具体流程：(1) 使用 Q-Diffusion 的均匀时间步采样策略生成校准集（N=5120，每步 n=256，20 步 DDIM）；(2) 逐层使用 MinMax 量化器确定 activation per-tensor scale 和 weight per-channel scale；(3) 使用 BRECQ 进行 block-wise 权重量化重建（Adaround 自适应舍入）；(4) TFMQ-DM 额外引入时间步特定的量化参数来适配跨时间步的激活分布变化。也对比了 SmoothQuant（将 LLM 的等效缩放直接迁移到扩散模型），效果极差（W4A8 FFHQ: FID=454.16 vs Baseline=36.08）。
  
  Baseline 全栈执行例子（LDM-4 FFHQ 256×256 W4A8, 20 步 DDIM 采样）：
  - 算法pipeline：加载 FP32 预训练 LDM-4 U-Net → 校准数据集采样（20 步 DDIM, 256 样本/步 = 5120 校准点）→ per-tensor 激活 scale s^X = (max|X|)/(2^{b-1}-1) → per-channel 权重 scale s^W → BRECQ block-wise 量化重建 → 量化推理：x_t 输入 → 所有 Linear/Conv 层权重为 INT4、激活为 INT8 → QKV/FFN 计算 → 输出 ε̂_θ(x_t,t) → DDIM 更新 x_{t-1} → 重复 T 步。SmoothQuant 直接套用时：τ = (max(|X_c|)^β/max(|W_c|)^{1-β})^(1/2)，因扩散模型中激活 >> 权重，τ 极大 → 权重量化范围被显著扩展 → 权重量化误差骤增（Weight Quant. Error: 0.0694 vs Baseline 0.0060）。
  - 系统框架：PyTorch + LDM（latent-diffusion, https://github.com/CompVis/stable-diffusion）。评估使用 guided-diffusion 的 ADM TensorFlow 评估器。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（标准 PyTorch 量化推理，无自定义 kernel）。
  - 硬件架构：论文未明确说明（GPU 上执行 PyTorch 推理）。

  **Baseline 的核心缺陷：**
  1. **通道级异常值未处理**：扩散模型的 activation 存在显著的通道间方差，尤其 skip connection 层（ResBlock skip connection 的 activation 通道间方差远大于 Transformer 层，Fig. 5）。标准 per-tensor 量化中，一个 outlier 通道的极端值拉伸了整层量化范围，使非 outlier 通道的量化精度严重下降。
  2. **SmoothQuant 迁移失败**：SmoothQuant 的缩放因子 τ = (max|X_c|/max|W_c|)^{1/2} 基于最大幅值比，但扩散模型中激活 >> 权重，导致 τ 极大 → 权重量化范围扩展 → 权重量化误差放大（权重在每步都使用，误差累积严重）。Baseline + SmoothQuant 的 Weight Quant. Error 从 0.0060 飙升至 0.0694。
  3. **等效缩放仅重分布不消除异常值**：即使正确应用等效缩放，它只是将激活的量化难度转移到权重（或反之），无法根本消除极端层中的异常值（如 skip connection 中某些通道的激活值远大于其他通道）。
  4. **时间步均匀加权次优**：早期去噪步（大 t）的量化误差虽小，但因在迭代过程前期引入，其影响会在后续步中累积放大；后期步的量化误差大但对最终质量的影响并非线性对应。均匀加权忽略了这一不对称性。
  5. **小校准集下 PTS 因子选择不可靠**：直接最小化校准集上的量化 MSE 选择 δ 因子会过拟合，在未见数据上性能退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 DMQ，通过两个核心设计解决缺陷：

  **(1) Learned Equivalent Scaling (LES) 解决异常值重分布问题（对应缺陷 1-2）**
  SmoothQuant 手动计算 τ 失败的本质原因是忽略了扩散模型的特殊性（激活 >> 权重）。DMQ 改为**学习** τ：以 layer-wise block reconstruction MSE 为目标（L_i = ||X_i W - Q(X̂_i) Q(Ŵ)||²），通过梯度下降直接优化 τ。不再依赖最大幅值比的启发式，而是利用校准数据找到最小化整体量化误差的 τ。
  - 引入 **Adaptive Timestep Weighting**（对应缺陷 4）：损失权重 λ_t = (1-Λ_t/ΣΛ_{t'})^α，Λ_t 为时间步 t 的累积损失（指数移动平均，ξ=0.95）。低误差的早期步得到更高权重，高误差的后期步仍有足够优化信号。避免了均匀权重偏向后期高误差步的问题，也避免了固定线性/二次权重忽略各层差异的问题（Fig. 4 右：各层误差趋势随 t 变化不同）。
  - **零推理开销融合**：τ 融合入权重和激活 scale（τ^T ⊙ W 预计算，τ ⊙ s^X 预计算），推理时无需额外操作。

  **(2) Power-of-Two Scaling (PTS) 解决极端异常值根本消除问题（对应缺陷 3、5）**
  等效缩放本质是双向转移量化难度，不能消除异常值。PTS 直接对 activation 施加通道级 2^δ 缩放：
  - 数学形式：X̃ = clamp(⌊X / (2^δ ⊙ τ ⊙ s^X)⌉, l, u)，输出时 Y ≈ s^X s^W · Σ X̃ · (W̃ ≪ δ)
  - **Bit-shift 高效实现**（对应缺陷 3）：2 的幂次缩放等价于整数 bit-shift，在 kernel 加载权重后立即执行 Ŵ^{shifted} = Ŵ ≪ δ，不需要乘法。仅应用于 skip connection 层（高通道间方差的层），总体开销极小。
  - **Voting Algorithm**（对应缺陷 5）：对每个校准样本和通道评估候选 δ ∈ {0,...,D}，选择最优 δ*_{i,k}；对每个通道计算众数 δ_k^{mode} 和一致性 r_k；仅当 r_k > κ(=0.85) 时采用 δ_k^{mode}，否则 δ_k=0（不缩放）。这种保守策略避免了小校准集下的过拟合，仅对确有统计共识的通道应用 PTS。
  - **Selective application**（对应缺陷 5 延伸）：消融实验（Tab. 7）证实仅对 skip connection 层应用 PTS 优于全层应用（FID: 30.37 vs 31.91），因为只有 skip connection 层存在严重通道间方差。

  论文方法全栈执行例子（LDM-4 FFHQ 256×256 W4A8, 20 步 DDIM）：
  - 算法pipeline：FP32 LDM-4 U-Net → **离线阶段**：① 校准数据收集（20 步 DDIM, 256/步 = 5120 校准点）→ ② LES 逐层学习 τ（4000-6000 iter, B=32, α=20, L = Σ λ_{t_i}||X_iW - Q(X_i/τ)Q(τ^TW)||²）→ ③ BRECQ 权重量化重建 → ④ PTS 因子投票（仅 skip connection 层，D=3, κ=0.85）→ ⑤ 融合：τ^TW 预计算存储，τ⊙s^X 预计算 → **推理**：⑥ 量化前向 X̃ = MinMaxQ_8bit(X/(2^δ⊙τ⊙s^X)) → ⑦ CUDA kernel：加载 W̃（INT4）→ Ŵ^{shifted} = W̃ ≪ δ → INT8@INT32 GEMM → 反量化 Y = s^X·s^W·C → 输出 → ⑧ DDIM 更新 → 重复 T 步。结果：W4A8 FID=30.37（Baseline=36.08, ↓15.8%），W4A6 FID=26.38（Baseline TFMQ-DM=29.76, ↓11.4%）。W4A6 下 Stable Diffusion 的 LPIPS=0.537（TFMQ-DM=0.691, ↓22.3%），CLIP=30.67（TFMQ-DM=25.32, ↑21.1%）。
  - 系统框架：PyTorch + LDM（https://github.com/CompVis/stable-diffusion）+ 自定义 CUDA kernel（W4A8 GEMM with bit-shift）。评估使用 guided-diffusion 的 ADM TensorFlow 评估器（50K 样本）。
  - 编译框架：论文未明确说明。
  - kernel调度（Section E）：自定义 CUDA kernel 将量化 + bit-shift + GEMM + 反量化融合为单 kernel，在 M=3072 时 vs FP32 GEMM 达到 5.17× 加速。bit-shift 在权重加载时执行（不影响 multiply-accumulate 路径），PTS 仅影响 skip connection 层（网络子集），整体延迟增长极小。
  - 硬件架构：论文未明确说明（GPU 上执行 PyTorch + CUDA kernel）。

  关键设计动机映射：
  - SmoothQuant 手动 τ 导致权重量化误差暴增 → LES 通过梯度下降学习最小化输出 MSE 的 τ，避免手动启发式
  - 通道间异常值仅靠等效缩放无法根除 → PTS 用 2 的幂次缩放直接压缩超大激活值，完全不同的机制
  - 等效缩放转移负担不消除 → PTS + bit-shift 在硬件层面以极低成本消除异常值影响
  - 均匀/固定时间步加权忽略早期步积累效应 → Adaptive Timestep Weighting 动态优先低误差高影响步
  - 小校准集下直接 MSE 选择 δ 过拟合 → Voting Algorithm 基于统计共识选择 δ，仅 r_k>0.85 的通道生效
  - Skip connection 层的 extreme outlier 是主要瓶颈 → PTS 仅针对性应用于 skip connection 层（消融验证优于全层应用）
