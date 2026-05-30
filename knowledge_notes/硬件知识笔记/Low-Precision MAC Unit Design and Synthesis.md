## Low-Precision MAC Unit Design and Synthesis

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
低精度 MAC（Multiply-Accumulate）单元是 DNN 加速器的核心计算单元，执行乘法和累加操作：output = Σ(Aᵢ × Bᵢ) for i=1..N（N 为 dot-product 长度，如 256）。对于 4-bit 数据类型，MAC 单元包含：(1) 4-bit multiplier——执行两个 4-bit 数的乘法；(2) N-bit accumulator——累加 N 次乘法的中间结果，位宽需保证无损累加（无溢出/无下溢）。低精度 MAC 的特殊性在于：(a) multiplier 的复杂度随位宽平方增长（4-bit multiplier 远小于 8/16/32-bit）；(b) accumulator 位宽取决于数据类型的动态范围——浮点格式（如 E2M1）的动态范围通常大于整数格式（INT4），需要更宽的 accumulator；(c) 在 4-bit 精度下，accumulator 的面积可能超过 multiplier（如 E2M1 accumulator=90.7 µm² vs multiplier=79.7 µm²），与 8-bit+ 下 multiplier 面积主导的模式不同。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
各数据类型的 MAC 单元组成（TSMC 28nm 综合结果）：

```
Datatype    | Accum. Bits | Mult. Area | Accum. Area | Total MAC | Power
INT4        | 16          | 75.3 µm²   | 85.4 µm²    | 160.7 µm² | 48.5 µW
INT5        | 18          | 106.6      | 97.0        | 203.6     | 59.8
E2M1        | 17          | 79.7       | 90.7        | 170.4     | 49.6
E2M1+SR     | 18          | 96.8       | 94.5        | 191.3     | 53.5
E2M1+SP     | 19          | 121.5      | 96.5        | 218.0     | 54.6
E2M1-I      | 20          | 119.1      | 109.1       | 228.2     | 59.7
E2M1-B      | 23          | 137.9      | 131.0       | 268.9     | 67.9
E3M0        | 22          | 98.0       | 119.7       | 217.7     | 59.5
APoT4       | 16          | 96.2       | 85.4        | 181.6     | 47.2
APoT4+SP    | 16          | 99.7       | 85.4        | 185.1     | 45.5
```

Accumulator 位宽确定原理：对于每个格式，计算最大可能 dot-product 值 max((max|W|)×(max|A|)×256)，确保 accumulator 的整数表示范围覆盖该值。INT4 范围最小（7×7×256=12544 → 16-bit signed），E2M1-B 范围最大（12×12×256=36864 → 23-bit signed）。

系统级芯片开销估算方法：
- 假设 MAC 单元占芯片面积 10%、存储系统占 60%（参考 Eyeriss v2, TPUv4 的设计比例）
- Rel. Chip Overhead = (MAC_Area_ratio - 1) × 10% / (10%+60%)
- 例：E2M1 MAC=170.4 vs INT4=160.7 → 面积比=1.060 → 系统开销=0.060×10%/70%=0.6%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MAC 单元通过 SystemVerilog RTL 描述，经 Synopsys Design Compiler 在 TSMC 28nm 标准单元库上综合。每个 MAC 含 multiplier + accumulator 两级数据路径：(1) multiplier 采用标准 Booth/Wallace 架构（整数乘法）或浮点解码器+乘法器（浮点格式）；(2) accumulator 为宽位加法器（支持累加 256 次不溢出）；(3) 每种数据类型独立设计专用 MAC（非统一的通用 MAC），因为各格式的 decode 逻辑、范围特性和 accumulator 需求不同。论文的硬件评估目的不是为单一格式设计完整加速器，而是为系统设计者提供量化证据：在给定的面积/功耗预算下，哪些格式是 Pareto 最优的——例如 E2M1 仅需 0.6% 系统开销即可大幅提升精度，E2M1+SP 需 3.6% 系统开销提供最高精度选项。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs
