## Focused Quantization for Sparse CNNs

- **属于硬件架构的实现是什么？实验比较什么？**
  设计了一个基于 FQ 量化权重的自定义硬件加速器的卷积 dot-product 实现（Figure 4）。该硬件架构利用 FQ 量化的权重为 2 的幂次值，将卷积乘法替换为 bit-shift 和整数加法，无需浮点乘法单元。评估了实现 3×3 卷积（padding=1，8×8×100 输入，8×8×100 输出）所需的最少逻辑门数，并与 ABC-Net、LQ-Net、标准 shift quantization 对比。

  实验比较：
  - Baselines：ABC-Net（5 bases/binary convolutions）、LQ-Net（2 bits）、shift quantization（3-bit unsigned）
  - FQ 配置：5-bit FQ、5-bit FQ + Huffman 编码
  - 评价指标：双输入逻辑门数（#Gates）下界、Ratio（相对 shift quantization 的门数比）
  - 结果：FQ (5-bit) 仅需 275.6M gates，与 3-bit shift quantization (275.2M) 相当，远低于 ABC-Net (806.1M, 2.93×) 和 LQ-Net (314.4M, 1.14×)

- **模拟器名，模拟器链接（web search），或论文修改的模拟器。**
  论文未使用外部模拟器。硬件资源评估基于自定义设计的逻辑门数估算，假设 unrolled architecture 和相同吞吐量。逻辑门数估算针对双输入逻辑门（two-input logic gates）下界。论文引用的 FPGA 加速器生成工作 [24] 使用自动生成工具（Mayo 框架，https://github.com/deep-fry/mayo），但本论文中硬件评估为手动设计估算。

- **模拟器模拟什么的性能，修改了什么。**
  评估自定义 CNN 推理加速器的资源利用率（逻辑门数）。对比不同量化方案在实现相同卷积层时的硬件开销。FQ 的硬件效率优势在于：
  1. 所有乘法被 bit-shift 取代，无需乘法器阵列
  2. σ₊ 和 σ₋ 约束为相等，可融合到逐层 α 中，仅需一个最终缩放
  3. α 可进一步融入 BN 融合，消除推理时所有乘法
  4. ABC-Net 和 LQ-Net 需要 N 路并行二值卷积 + 高精度乘积累加（O(MN) 额外开销），而 FQ 无需此开销

- **开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。**
  开源代码：https://github.com/deep-fry/mayo（Mayo 框架，包含自动多精度多算术 CNN 加速器生成工具，参见 ICFPT 2019）。

  硬件评估原理：以 Figure 4 的 dot-product 数据路径为基础，手工推导每种量化方案在实现同一卷积层时所需的最小双输入逻辑门数。不涉及仿真或综合工具，属于 paper-level 的硬件复杂度估算。方法：
  1. **输入**：量化后的权重参数和卷积层规格（3×3 kernel, 8×8×100 input/output activations）
  2. **数据路径**：Input activations（整数）→ 与 shift-quantized weights 做 dot-product（bit-shift + integer add）→ Accumulate → Scale by α → Output
  3. **门数估算**：对每个 bit-shift、整数加法、累加器和缩放乘法，统计所需的基础逻辑门数（AND/OR/NOT/XOR/MUX 等）
  4. **输出**：所有方案在同一吞吐量假设下的 #Gates 下界对比表
