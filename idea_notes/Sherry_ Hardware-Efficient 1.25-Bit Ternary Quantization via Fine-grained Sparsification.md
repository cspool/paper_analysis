## Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification

- baseline方法是什么？
  Baseline 方法为标准的 1.58-bit/1.67-bit 三值量化（TWN、BitNet、TernaryLLM、LLM-QAT、ParetoQ、Spectra、Tequila），将权重限制为 {+1, 0, -1} 三值集合。推理时通过查表引擎（BitNet.cpp, T-MAC）将浮点乘法转换为硬件高效的整数加法。Baseline 在存储和 SIMD 对齐方面有两种策略：(1) **2-bit 打包**（如 BitNet I2_S）：每权重用 2 bits 存储（4 值需要 2 bits），简单但对齐付出了 0.42 bit/weight 的浪费（实际信息只有 1.58 bits）；(2) **1.67-bit 打包**（如 Tequila TL2）：3 个三值权重打包为 5 bits（3³=27 < 2⁵=32），虽节省存储但引入 3-way 不规则 pattern，与 SIMD 的 4/8/16-way 向量通道不对齐，需昂贵的 bit shuffle 操作。训练时标准三值 QAT 存在 weight trapping 问题：处于 deadzone [-Δ, Δ] 内的权重因 STE 接收无信息梯度而停滞，梯度同质化导致表示坍缩。

  **Baseline 全栈执行例子（以 BitNet 1.58-bit 在 LLaMA-3.2 1B 上的推理为例）：**
  - **算法 Pipeline**：权重矩阵 W 经三值量化 Q(W) = α·sign(W)·I[|W| ≥ Δ] → 每权重存储为 2 bits（浪费 0.42 bits）或以 1.67-bit 打包（3-权重 5-bit → SIMD 不友好）→ 推理时查表执行 X·Q(W) 替代浮点 MUL → BF16 激活保持全精度。
  - **Serving 框架**：论文未明确说明（推理使用 llama.cpp 或其衍生引擎，加载 GGUF 格式，无特定 serving 调度优化）。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：BitNet.cpp 的 SIMD kernel 使用 2-bit 打包 → 4 权重/8 bits → 4-way 对齐 SIMD（128-bit: 4×FP16 = 64 bits 激活，256-bit: 4×FP32 = 128 bits 激活），但存储效率只有 1.58/2 = 79%。若使用 1.67-bit 不规则打包（如 Tequila），3-way pattern 导致 SIMD vector lane 不对齐，需 bit-level shuffle → 额外的位操作开销和缓存线碎片化 → 推理速度变慢。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  Baseline 的核心缺陷：(a) **存储-SIMD 对齐矛盾**：2-bit 打包浪费存储（~21% 开销），1.67-bit 打包破坏 SIMD 对齐（3-way 不规则 pattern 导致 kernel 需 bit-level 操作，抵消低位宽的推理加速收益）；(b) **训练时 weight trapping**：标准三值 QAT 中，梯度通过 STE 量化函数时，deadzone 内权重梯度为零信号 → 权重停滞无法逃离 deadzone → 同质化梯度导致表示坍缩（低秩，模型精度显著下降）；(c) 现有三值方法难以在 1.25-bit 极低位宽下同时保持精度和硬件效率。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 Sherry，通过两个核心创新解决 baseline 缺陷：

  **创新一：3:4 细粒度结构化稀疏**
  打破 "2-bit 浪费 vs 1.67-bit 不对齐" 的矛盾——强制每 4 个权重中恰好 3 个非零和 1 个为零，4 个权重恰好有 C(4,3)×2³ = 32 种状态，完美饱和 5-bit 索引 → 等效 1.25 bit/weight（比 1.67-bit 节省 25% 存储），同时 4-way pattern 天然对齐 128/256/512-bit SIMD 向量通道 → 零 bit-level shuffle 开销。

  **创新二：Arenas（Annealing Residual Synapse）**
  解决 weight trapping 问题——训练时注入全秩残差 Y = X·Q(W) + λ_t·X·W（λ_t 从初值退火至零），为 deadzone 内权重提供连续梯度信号，防止 ∂L/∂X 坍缩为低秩。λ_t → 0 后 Arenas 路径融合入静态参数，推理零耗。

  **论文方法全栈执行例子（Sherry 1.25-bit 在 LLaMA-3.2 1B 上的推理）：**
  - **算法 Pipeline**（训练时）：
    1. 每个连续 4 权重打包为一组，argmin|w_i| 权重置零，其余 ±1 量化 → 32 种排列 → 5-bit 索引
    2. Arenas 路径 Y = X·Q(W) + λ_t·X·W 并行注入异构梯度，λ_t 退火到 0
    3. 梯度 ∂L/∂X = ... 含全秩分量（来自 Arenas），避免低秩坍缩
    4. 训练完成：Arenas 融合入 bias，模型仅有 1.25-bit 权重 + BF16 激活
  - **Kernel 调度**：5-bit 打包 → 4-way SIMD 对齐 → 128-bit SIMD 处理 4 个 FP16 激活 × 4 个三值权重（1 组完美对应），256-bit SIMD 处理 4 个 FP32（1 组完美对应），512-bit SIMD 处理 8 个 FP16（2 组整除）→ 零 bit shuffle → 查表引擎（BitNet.cpp/T-MAC）将 MUL→ADD
  - **编译框架**：论文未明确说明。
  - **Serving 框架**：论文未明确说明。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  方法如何解决 Baseline 缺陷：
  - 缺陷 (a) 存储-SIMD 矛盾：3:4 稀疏的 5-bit 4-way 打包同时实现 1.25 bit/weight 存储效率（优于 2-bit 的 1.58-bit 实际和 1.67-bit 的存储）和 SIMD 友好对齐（4-way pattern 无 bit shuffle），在 Intel i7-14700HX 上实现 148.27 t/s（0.7B）和 45.55 t/s（3B），相比 BitNet I2_S 分别快 12% 和 9%，模型大小减少 ~20%。
  - 缺陷 (b) weight trapping：Arenas 的残差路径 λ_t·X·W 为 deadzone 内权重提供 heterogeneous 梯度信号，防止同质化和低秩坍缩，使 Sherry 在 1.25-bit（比 BitNet 少 25% bits）下平均基准精度反而更高（1B 模型 0.519 vs BitNet 0.483; 3B 模型 0.567 vs BitNet 0.527）。
  - 缺陷 (c) 极低位宽精度保持：Sherry 在 1.25-bit 下仅比 BF16 基线低 3.9%（1B）和 6.9%（3B），与 1.67-bit Tequila 持平甚至略高的同时使用更少 bits。这证明 3:4 稀疏 + Arenas 的组合在 ~1.25 bit 处找到了硬件效率和模型精度的"甜点"。
