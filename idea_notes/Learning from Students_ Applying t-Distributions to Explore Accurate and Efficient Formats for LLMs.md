## Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

- baseline方法是什么？
  Baseline 是 **NF4（Normal Float 4-bit）** [Dettmers et al. 2023] 和 **INT4** 量化。NF4 假设权重服从正态分布 N(0,σ²)，使用 Gaussian 分位数函数等概率划分 16 个量化层级。INT4 为均匀量化，在 [-8, 7] 范围内等间距分布。

  **Baseline 全栈执行例子**（LLaMA2-7B, weight-only PTQ with NF4, block size 128）：
  - **算法pipeline**：加载预训练 FP32/FP16 权重 → 按 block size 128 分块 → 每块用 absmax（w_max = max|w_i|）归一化到 [-1,1] → NF4 码本查表量化（16 个固定值基于 Gaussian 分位数）→ 存储 4-bit index + per-block w_max（FP16）。推理时：Ŵ = w_max × NF4_table[index]。
  - **系统框架**：bitsandbytes 库（HuggingFace PEFT/QLoRA 集成），基于 PyTorch + HuggingFace Transformers。推理时使用 CUDA fused dequantization kernel。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：bitsandbytes 4-bit CUDA kernel（dequantize → FP16 GEMM），无专用硬件 kernel。
  - **硬件架构**：NVIDIA GPU（无自定义硬件）。但论文同时评估了各数据类型的 MAC 单元面积和功耗（SystemVerilog + Design Compiler + TSMC 28nm），INT4 MAC = 160.7 µm²、48.5 µW。

  **Baseline 的核心缺陷**：
  1. **正态分布假设错误**：论文对 30+ DNN 的 weight/activation 做大规模 profiling，用 Kolmogorov-Smirnov 检验证明大多数 DNN 分布由 Student's t-distribution（ν≈5）最优近似，而非正态分布。正态分布无法同时拟合分布的尖峰（peak）和厚尾（tail）（Figure 2：Mistral-7B 的 Q-Q plot 中 t-distribution 呈直线，normal 显著偏离）。NF4 基于错误分布假设，其量化层级在概率空间的分布与实际 weight 分布不匹配。
  2. **INT4 均匀量化忽略分布结构**：INT4 在 [-8,7] 区间等间距分布，绝大多数量化层级落在权重稀少的边缘区域，而对权重密集的中心区域仅分配少量层级，导致对典型值的量化精度不足。
  3. **E2M1 FP4 浪费位数空间**：因存在正负零的浮点表示冗余，E2M1 仅使用 15/16 = 93.75% 的位数空间。在 4-bit 仅有 16 个可能值的极端受限条件下，6.25% 的浪费显著。
  4. **缺少质量-效率联合优化视角**：数据类型选择通常在精度和硬件效率之间各说各话，没有系统的 Pareto 权衡分析。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过四个层次的设计逐一解决 baseline 缺陷：

  **(1) SF4（Student Float）基于 t-distribution 导出最优查找表（解决缺陷 1）**
  NF4 用 Gaussian 分位数划分，SF4 改用 Student's t-distribution (ν=5) 分位数。Algorithm 1 将概率质量均匀分 16 份，经 t-distribution 分位数函数 Q_S(p; ν=5) 映射到量化值空间。这确保了每个量化层级的使用频率大致相等（等概率原则），量化直方图近似平坦。实验证实 SF4 比 NF4 在各 LLM 上持续提升精度（如 LLaMA2-7B LAMBADA: NF4=71.98%, SF4=72.54%），且 ν 的选择基于 profiling 结果（最频 ν≈5），非任意参数。

  **(2) Supernormal Support 回收 E2M1 的浪费位数（解决缺陷 3）**
  将 E2M1 的负零位重映射为额外超常值，提出两种变体：
  - **Super-range (SR)**：将负零 → 8.0，扩展动态范围。精度提升有限，因额外点位于分布边缘很稀疏的区域。
  - **Super-precision (SP)**：将负零 → 5.0，在分布内部增加一个层级。精度提升更显著（如 Phi-2 W4A4: E2M1 平均准确率降 -8.41%, E2M1+SP 降至 -7.25%），因额外层级位于高概率密度区域。SP 的硬件开销（MAC 面积 +27.9%，系统 +3.6%）高于 SR（MAC +12.3%，系统 +1.9%），但在精度-面积 Pareto 上提供更高精度选项。

  **(3) t-distribution 洞察解释 E2M1 为何优于 INT4（解决缺陷 2 的根源）**
  论文发现 E2M1 的形状分段逼近 SF4：E2M1 对分布中心的小值区域分配更密集的层级（0, 0.5, 1, 1.5, 2），而对边缘的极大值分配稀疏层级（3, 4, 6）。这恰好匹配 t-distribution 的尖峰厚尾特征——中心概率密度高需要更细粒度量化，尾部概率密度低可以用粗粒度。这就从理论上解释了为何 FP4 优于 INT4：不是因为浮点格式本身，而是因为 E2M1 的形状隐含地匹配了 t-distribution 的形状。

  **(4) 质量-效率 Pareto 曲线系统化设计指导（解决缺陷 4）**
  论文首次将 11 种 4-bit 数据类型在模型精度（LAMBADA+HellaSwag+Winogrande+PIQA+BoolQ+ARC-c 平均准确率损失）和芯片面积（MAC 单元面积 + 系统级开销估算）两个维度上绘制 Pareto 曲线，揭示：
  - Pareto frontier: INT4（最低面积/精度）→ E2M1（0.6% 系统开销，精度损失降低 7.34%）→ E2M1+SP（3.6% 系统开销，最高精度）
  - E2M1-I 和 E2M1-B 属于严格劣化点（面积更大且精度更低），应从实际部署中排除
  - APoT4 精度接近 E2M1 但需要额外格式转换逻辑，实用性有限

  **论文方法全栈执行例子**（以 LLaMA2-7B weight-only PTQ with E2M1+SP 为例）：
  - **算法pipeline**：
    1. 离线 t-distribution profiling（可选，ν=5 已固定）→ 确定 SF4/E2M1+SP 量化层级
    2. 加载预训练权重 → 按 block size 128 分块 → 可选 MSE clipping calibration 优化裁剪阈值
    3. 对每 block：归一化 → E2M1+SP 码本查表量化（16 个层级，含 SP 值 5.0）→ 存储 4-bit index + per-block FP16 scale
    4. 推理：dequantize → FP16 GEMM（标准流程，与 NF4/INT4 完全相同）
  - **系统框架**：修改版 Intel Neural Compressor 库（添加 lookup-based quantization for SF4/NF4/E2M1 variants + supernormal support）。基于 PyTorch + HuggingFace Transformers。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：与 NF4 相同的查表解码 + FP16 GEMM（bitsandbytes 风格）。论文未涉及自定义 kernel。
  - **硬件架构**：
    - E2M1 MAC 单元：17-bit accumulator, 总面积 170.4 µm², 功耗 49.6 µW
    - E2M1+SP MAC 单元：19-bit accumulator（需更大累加器容纳 SP 引入的额外值 5.0），总面积 218.0 µm², 功耗 54.6 µW
    - 系统开销：E2M1 0.6%，E2M1+SP 3.6%（假设 MAC 占芯片 10%、存储 60%）
    - 设计决策：SP 的 MAC 面积比 SR 大（27.9% vs 12.3%），但精度提升也更显著，在精度敏感场景值得额外面积投入
