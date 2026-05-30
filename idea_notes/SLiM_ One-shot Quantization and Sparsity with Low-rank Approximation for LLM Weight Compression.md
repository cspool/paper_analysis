## SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

- baseline方法是什么？
  Baseline 方法分为三类：(1) **单独剪枝**：SparseGPT（基于逐层 Hessian 逆的稀疏回归）、Wanda（基于 weight × activation 幅度的简单剪枝）、Magnitude Pruning；(2) **单独量化**：OPTQ（基于 OBS 的逐层量化）、AWQ（激活感知权重量化，scale 显著 channel）、OmniQuant（可学习 clipping + channel scaling）、AffineQuant（等价仿射变换）、Group AbsMax；(3) **联合剪枝+量化**：JSQ（仅支持 8-bit，低位宽精度差）、L²QER（仅量化的一-shot 低秩适配，与稀疏结合时精度显著下降）。

  **Baseline 全栈执行例子（以 Wanda + Group AbsMax 为例）：**
  - **算法 Pipeline**：Wanda 逐行计算 weight × activation norm 重要性分数 → 保留 top 50% 权重（2:4 模式每 4 个保留 2 个）→ Group AbsMax 以 group size 128 对剩余权重做 4-bit 量化 → 输出稀疏量化模型。两种误差 E_Q 和 E_S 各自独立累积，不做联合补偿。
  - **Serving 框架**：论文未明确说明（实验使用 Sparse Marlin + vLLM 仅用于 SLiM 自身的加速比评估，未修改框架调度逻辑）。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：Sparse Marlin CUDA kernel 实现 2:4 稀疏 × 4-bit 量化矩阵乘法（数千行 CUDA，仅支持有限 GPU 架构）；无自定义 kernel 调度优化。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  Baseline 的核心缺陷：(a) 量化误差和稀疏误差独立累积，无联合补偿机制；(b) 均匀量化 scaling factor 选择使用 AbsMax 对 outlier 敏感，Grid Search 次优且昂贵；(c) 低秩适配（如 L²QER）初始化基于 weight norm 而非对模型输出的实际影响，需要昂贵重训练；(d) 联合稀疏+量化时精度显著下降，尤其是 4-bit + 2:4 稀疏场景；(e) Group Quantization 增加反量化开销和 GPU kernel 实现复杂度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SLiM 通过三个协同设计的组件解决 baseline 缺陷：

  1. **SLiM-Quant**：将非凸量化 MSE 优化问题通过概率化重表述为凸问题，在权重直方图上做数值积分 + 多网格搜索，高效找到全局最优 uniform scaling factor α*。避免了 Grid Search 的次优性，且 uniform quantization 比 group quantization 去除了反量化开销（实测 6% 加速）。激活感知变体 SLiM-Quant^O 对 1% 最高显著性 channel 做 weight-activation scaling 互换，结合联合显著性 saliency = |diag(x_mean) × W|，进一步降低输出误差。

  2. **Wanda 剪枝** 在量化权重上施加 2:4 半结构化稀疏，直接产生硬件可加速的稀疏模式。

  3. **SLiM-LoRA**：提出满足可逆性和可加性的 saliency 函数 F(W) = diag(x)W（x 为校准集输入平均绝对值），使得低秩适配器可通过 SVD 数学推导闭式解，无需迭代训练。关键创新：(a) 可加性允许将适配器的显著性从压缩误差中隔离——F(-(E_Q + E_S)) = F(W^C - W)，SVD 分解后通过逆变换直接得到 L, R；(b) 显著性加权确保适配器优先修正对输出影响最大的权重通道，而非均匀最小化 Frobenius 范数（对比 Naive-LoRA）；(c) 无需重训练，one-shot 完成，比 L²QER 更好地处理联合稀疏+量化误差。

  **SLiM 全栈执行例子：**
  - **算法 Pipeline**：加载预训练权重 W → SLiM-Quant：构建权重直方图 f_abs → 多网格搜索 α* 最小化 E_quant + E_clip → W^Q = round(clip(W/α*)) × 2^{q-1} → Wanda 在 W^Q 上施加 2:4 稀疏 → 计算 E_C = W^C - W → 构建显著性矩阵 S_C = diag(x_mean)E_C → SVD(S_C) 取 rank r=0.1d → 逆显著性变换 L = diag(1/x_mean)L̃, R = R̃ → 可选：对 L, R 做 AbsMax group quantization（group size 128, 4-bit）→ 可选：在 C4 (300K tokens) 上 PEFT 微调（冻结 W^C，仅更新 L, R，使用 STE 处理量化适配器）。
  - **Serving 框架**：论文未明确说明（推理加速使用 Sparse Marlin kernel + vLLM，但未修改调度逻辑）。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：量化稀疏矩阵乘法使用 Sparse Marlin CUDA kernel；低秩适配器乘法使用 Dense Quantized Marlin（适配器量化时）或标准 PyTorch kernel；PEFT 微调阶段使用 Triton 自定义量化/反量化 kernel 降低 STE 开销。无自定义 kernel 调度优化。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  对比 Baseline，SLiM 的核心突破在于：(a) 将量化+稀疏+低秩适配三个组件通过统一显著性函数协同优化，而非各自独立处理；(b) SLiM-LoRA 的可逆可加显著性设计使低秩适配器获得闭式解，消除重训练需求；(c) SLiM-Quant 的概率化 uniform quantization 在保持硬件友好性的同时达到 group quantization 精度；(d) 进一步量化适配器 + PEFT 微调形成完整压缩-补偿-精调闭环。
