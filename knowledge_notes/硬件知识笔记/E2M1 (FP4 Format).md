## E2M1 (FP4 Format)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
E2M1 是一种 4-bit 浮点数据格式，位分配为：1-bit sign (S) + 2-bit exponent (E) + 1-bit mantissa (M)，共 4 bits。E2M1 的数值表示为 (-1)^S × 2^E × (1 + M/2)。E=0 时为 subnormal 区域（值的间距固定为 2^{-1}×1/2=0.25），E>0 时为 normal 区域。正值集合（标准配置，无 subnormal 偏置）：{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}——仅 8 个正值、7 个可用负值加上正负零冗余（总共 2^{4}=16 个编码中仅 15 个有效值，因正零和负零是两个不同的编码）。存在多种变体：Intel 变体 (E2M1-I) subnormal 值更集中于零附近（0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0），bitsandbytes 变体 (E2M1-B) 动态范围更大（0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
E2M1 MAC 单元在硬件中的计算流程：

```
输入: A (E2M1 × 1), B (E2M1 × 1)
1. Decode A:
   sign_A = A[3]
   exp_A  = A[2:1]
   mant_A = A[0]
   if exp_A == 0:  # subnormal
       value_A = (-1)^{sign_A} × 2^{0} × mant_A/2  # 即 0 或 0.5
   else:
       value_A = (-1)^{sign_A} × 2^{exp_A} × (1 + mant_A/2)

2. Decode B (同理)

3. Multiply: product = value_A × value_B  # FP4 × FP4 → 更高精度中间结果

4. Accumulate: accum += product  # 与之前 255 项的 dot-product 累加
   # accumulator 位宽需无损容纳 256 项累加，决定 area/power
   # E2M1 最大值约 6.0，256×6×6=9216 → 需约 17-bit signed accumulator
```

在 MAC 面积分解中（Table 10）：E2M1 multiplier 面积 79.7 µm²，accumulator 面积 90.7 µm²，总 MAC 170.4 µm²，功耗 49.6 µW（TSMC 28nm）。与其他格式对比：INT4 MAC=160.7 µm²（accumulator=85.4 µm²），E2M1+SP=218.0 µm²。关键观察：低比特下 accumulator 面积可超过 multiplier（E2M1 accumulator 比 multiplier 大 13.8%），这与高精度下 multiplier 面积平方增长占主导不同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
E2M1 的 RTL 设计使用 SystemVerilog 描述 MAC 数据路径，通过 Synopsys Design Compiler 在 TSMC 28nm 工艺下综合得到面积和功耗。论文比较了三种 E2M1 变体的硬件效率：(1) E2M1（标准）：accumulator 17 bit，MAC 170.4 µm²，系统级开销仅 0.6%（vs INT4），精度损失降低 7.34%——属于 Pareto 最优；(2) E2M1-I（Intel）：因 subnormal 值极度集中导致动态范围增大，accumulator 20 bit，MAC 228.2 µm²，系统开销 4.2%，精度反而不如标准 E2M1——属于严格劣化；(3) E2M1-B（bitsandbytes）：accumulator 23 bit（最大动态范围），MAC 268.9 µm²，系统开销 6.7%，精度最差。这揭示了格式内部设计（subnormal 位置）对硬件效率的影响与精度影响一致——更集中的 subnormal 值不仅降低精度还增大 accumulator 需求。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs
