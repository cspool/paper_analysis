## PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- baseline方法是什么？
  Baseline 是传统后训练 KV Cache 量化方法（KIVI、RotateKV、MiKV、SKVQ 等），这些方法为短上下文场景（<8K tokens）设计，直接应用于长 CoT LLM 会导致严重性能退化。

  **Baseline 全栈执行例子（以 KIVI 为例，DeepSeek-R1-Distill-Qwen-7B，2-bit KV Cache）：**

  - 模型推理算法层：每步解码时直接用非对称分组量化将 KV Cache 压缩到 2-bit。Key Cache 使用 per-channel 量化，Value Cache 使用 per-token group-wise 量化（group size=128），保留首 token 和最近 token 为高精度。标定数据使用短序列（512 tokens），通道重参数化因子 λ_i 在短序列上标定，无法捕获 RoPE 低频通道（周期 > 54000 tokens）在长上下文下的完整数据分布。
  - 系统框架层：论文未明确说明（使用标准 HuggingFace Transformers 推理流程，未修改推理引擎或 Serving 框架）。
  - 编译框架层：论文未明确说明（未涉及编译器修改）。
  - kernel调度层：论文未明确说明（fake quantization 实验，非真实部署 kernel）。
  - 硬件架构层：论文未明确说明（纯软件层量化，不涉及硬件修改）。

  Baseline 的两个核心痛点：
  1. **大累积误差**：每步直接量化到目标 2-bit，内存预算未被充分利用（前期存在大量空闲内存），但量化误差随生成长度线性累积，长 CoT 场景（最大 32K tokens）下精度严重退化。
  2. **短标定数据无法反映长上下文分布**：RoPE 低频通道周期长达 54K tokens，512-token 标定无法覆盖这些通道在长序列下的正弦分布特征，导致通道重参数化因子不准确。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PM-KVQ 提出三项技术分别解决 Baseline 的两个缺陷。

  **PM-KVQ 全栈执行例子（DeepSeek-R1-Distill-Qwen-7B，Fbit=2-bit）：**

  - 模型推理算法层：
    1. **Progressive Quantization（解决累积误差）**：不直接量化到 2-bit，而是按 16→8→4→2 bit 逐步降低。初期以 16-bit 高精度存储，当内存预算耗尽时执行 Equivalent Right Shift：X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b。该策略等价于先反量化到浮点再重新量化，在长 CoT 前期（内存未满时）保持零量化误差，后期再有损压缩早期 token。
    2. **Block-wise Memory Allocation（解决内存利用不均）**：不采用统一位宽，而根据各 block 对量化的敏感度（一阶泰勒近似 s_{i,b}）分配不同位宽，建模为整数规划问题并用 CVXPY 求解（几秒内完成）。敏感 block（深层 + 第一层）分配高位宽，不敏感 block 分配低位宽，在相同总内存预算下最大化精度。
    3. **Calibration with Positional Interpolation（解决短标定问题）**：在 RoPE 中对位置索引 m 乘以缩放因子 s（实验中 s=4），使 2048-token 标定数据携带 8192-token 的位置信息，覆盖 RoPE 低频通道的更完整周期分布，从而准确标定通道重参数化因子 λ_i。

  - 系统框架层：论文未明确说明（使用 HuggingFace Transformers + fake quantization 进行评测，未修改推理引擎）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明（为 fake quantization 实验，但论文指出实际推理时 Equivalent Right Shift 可通过整数加法和移位高效实现，无需浮点反量化）。
  - 硬件架构层：论文未明确说明。

  **关键性能对比（Baseline vs PM-KVQ）：**
  - DeepSeek-Qwen-7B (2-bit) AIME-2024 pass@1：KIVI 32.08% → PM-KVQ 40.00%（+7.92%）
  - DeepSeek-LLaMA-8B (4-bit) AIME-2024 pass@1：KIVI 41.25% → PM-KVQ (BS=6, block-wise) 47.71%（+6.46%，超 16-bit 的 44.17%）
  - DeepSeek-LLaMA-70B (2-bit) AIME-2024 pass@1：KIVI 51.88% → PM-KVQ 64.79%（+12.91%）
  - Voting accuracy 提升更显著：DeepSeek-Qwen-7B voting KIVI 43.33% → PM-KVQ (BS=32) 66.67%（+23.34%）

  **设计思路映射（缺陷→方法）：**
  - 累积误差大 → Progressive Quantization：用时间换精度，前期高精度存储，内存满后再逐渐降位宽
  - 内存仍浪费（块间敏感度不均） → Block-wise Memory Allocation：敏感块多分配内存，不敏感块少分配
  - 短标定不能反映长上下文 RoPE 分布 → Positional Interpolation in Calibration：在短序列 RoPE 中嵌入长位置信息
