## PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models

- baseline方法是什么？
  Baseline 是已有的通用 PTQ 量化方法，包括：(1) 训练无关 scaling 方法——**SmoothQuant**（per-channel 平均对齐激活和权重 range）、**OS+**（对齐所有激活通道到共同中心 + 缩放）、**RepQ***（重参数化统一激活 range）；(2) 训练依赖方法——**OmniQuant**（反向传播优化 scaling factor 和 weight clipping，需数小时训练）；(3) 旋转变换方法——**QuaRot**（随机 Hadamard 旋转抑制 outlier，在 ARVG 中因 AdaLN 不保持旋转不变性需在线计算，引入严重开销）；(4) 低秩分解方法——**SVDQuant**（低秩分解隔离 outlier，需自定义 CUDA kernel，在 ARVG 中效果不佳）。

  这些 baseline 在 ARVG 模型上存在三个核心缺陷：
  (a) **无法处理 channel-wise outlier**：ARVG 中 AdaLN 模块调整后的激活存在严重的 channel-wise outlier（激活 range 跨通道差异极大），SmoothQuant/OS+ 等经验设计的 scaling 方法缺乏理论保证，次优且无法保证有效性；OmniQuant 需昂贵训练且不稳定。
  (b) **无法高效处理 token-wise 动态激活**：ARVG 中 AdaLN 输入沿 token 维度高度动态（含位置嵌入信息），线性层存在 sink token（首 token 含条件信息，分布显著不同于其他 token）。LLM 的动态 per-token 量化（如 LLM.int8）引入在线 min-max 校准开销（0.5× speedup loss）且精度下降（VAR 上 FID 降 15.3）。
  (c) **样本间分布不匹配导致校准偏差**：ARVG 中网络激活跨样本高度相似（尤其无条件样本），样本级冗余导致量化参数校准不匹配。现有校准策略（如 EDA-DM 的时序校准）针对扩散模型的时间步维度，无法处理 ARVG 的样本级冗余。

  Baseline 全栈执行例子（以 SmoothQuant 量化 RAR-B 到 W6A6 为例）：
  - **算法层**：加载 RAR-B 预训练权重 → 对 qkv 和 fc1 层计算 per-channel 激活和权重 range → 使用默认平滑因子 α=0.5 做等效缩放：激活除以 s_i = max(|X_i|)^α / max(|W_i|)^{1-α}，权重乘以 s_i → 缩放因子融合到 AdaLN 权重中（离线）→ 校准 128 张随机采样 ImageNet 图像，layer-wise min-max 确定激活量化 range，channel-wise min-max 确定权重量化 range → uniform quantizer：W_int = round(W/δ_W) + z_W, X_int = round(X/δ_X) + z_X → 推理时 INT 矩阵乘法 + 反量化。
  - **系统框架层**：PyTorch fake quantization，GPU 上完成校准（论文未明确指定 GPU 型号，baseline 中 OmniQuant 用 A100-80G）。使用 ADM's TensorFlow evaluation suite 做 FID/IS 评估。生成 50K ImageNet 图像。
  - **编译框架/kernel调度/硬件架构层**：论文未明确说明。使用标准 PyTorch fake quantization，无自定义 CUDA kernel。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **PTQ4ARVG**，一个 training-free 且 hardware-friendly 的 PTQ 框架，通过三个组件分别解决 baseline 的三个缺陷：

  **(1) GPS (Gain-Projected Scaling) → 解决 channel-wise outlier**：不同于 SmoothQuant 等经验设计的 scaling，GPS 首次基于数学优化推导 scaling factor。将量化损失做 Taylor 展开，定义 scaling gain g(s) = g_x − g_W（激活量化损失减少 − 权重量化损失增加），通过 ∂g/∂s = 0 求得闭式最优解 s_i = s_k · √(Σ|ΔW_{i,j}·x_i|) / √(Σ|W_{i,j}·Δx_i|)。无需训练（vs OmniQuant 需数小时训练），无在线计算开销（vs QuaRot 的在线 Hadamard 旋转），零推理开销（scaling factor 离线融合到权重中）。

  **(2) STWQ (Static Token-Wise Quantization) → 解决 token-wise 动态激活**：利用 ARVG 的两大独有特性——固定 token 序列长度和跨样本位置不变分布——将 per-token 量化参数离线静态设定。对 AdaLN 沿 token 序列分配独立量化参数；对线性层将 sink token 与 normal token 分开量化。使用 percentile 校准保证精度。相比 LLM 的动态 token-wise 量化（LLM.int8），STWQ 无在线校准开销（speedup 保持 2.92× vs DTWQ 的 2.46×），且精度更高（FID 10.41 vs DTWQ 30.14）。

  **(3) DGC (Distribution-Guided Calibration) → 解决样本间分布不匹配**：基于 Mahalanobis 距离 ρ(x) = √((x-u)^T S^{-1} (x-u)) 量化每样本对整体分布熵的贡献，选择 top 50% 高熵样本构成校准集。相比 random/uniform 采样，DGC 在所有指标上一致提升，且随校准集增大保持鲁棒。

  PTQ4ARVG 全栈执行例子（以 RAR-B W6A6 量化为例）：
  - **算法层**：
    - 离线校准阶段：加载 RAR-B 预训练权重 → DGC：从校准池计算 Mahalanobis 距离，选 top 50% 高熵样本（128 张 ImageNet）→ GPS：对每个 block 的 qkv 和 fc1 层，量化当前权重和激活计算 ΔW 和 ΔX → 找到 activation range 最大的通道 k → s_k = √(R_x^k/R_W^k) → 对每个通道 i≠k，闭式求解 s_i = s_k·√(Σ|ΔW_{i,j}·x_i|)/√(Σ|W_{i,j}·Δx_i|) → 应用等效缩放 X'=X⊘s, W'=s⊙W，离线融合到 AdaLN 权重 → STWQ：对 AdaLN 输入逐 token 做 percentile 校准设定 δ[t]；对线性层输入分离首 token（sink）和其余 token，分别 percentile 校准 → 存储所有静态量化参数（δ, z, bit-width）。
    - 推理阶段：输入条件信息（类别标签 + 位置编码）→ AdaLN 生成 shift/scale 参数 → 每步 token 生成：使用预设的静态 per-token 量化参数对激活做 INT 量化 → INT 矩阵乘法（权重已事先量化并融合 scaling）→ 反量化 → 输出 token → 最终生成 50K ImageNet 图像评估 FID/IS/Precision。
  - **系统框架层**：PyTorch fake quantization 用于校准和精度评估；RTX 3090 GPU 实际部署 8-bit 量化模型测试延迟/内存。标准 CUDA kernel，无自定义 kernel（与 SVDQuant 不同）。使用 ADM's TensorFlow guided-diffusion 评估套件。128 张校准图像（DGC 选择），50K 生成图像评估。
  - **编译框架/kernel调度/硬件架构层**：论文未明确说明。PTQ4ARVG 使用标准 CUDA kernel 部署，无自定义 kernel 或硬件修改。PTQ4ARVG 的 STWQ 兼容标准 CUDA kernel（论文明确论证了这一点）。
