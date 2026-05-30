## Supernormal Support (SR/SP for Low-Precision Formats)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Supernormal support 是本论文提出的技术，通过回收 E2M1 浮点格式中正负零冗余位，将其重新分配为额外的有效量化值。在标准 IEEE 754 浮点中，sign bit 导致正零和负零是两个不同的编码——在 8-bit（256 个值中浪费 1 个即 0.4%）可忽略，但在 4-bit（仅 16 个值）浪费 6.25% 的位数空间。Supernormal 将负零编码重映射为一个有用的超常值（supernormal），区别于正常的 subnormal 值。论文提出两种变体：(a) Super-range (SR)：将负零 → 8.0（在分布边缘增加一个值扩展动态范围）；(b) Super-precision (SP)：将负零 → 5.0（在分布内部增加一个值提升精度）。SP 在精度上通常优于 SR（因额外值位于高概率密度区域），但 MAC 面积开销更大（+27.9% vs +12.3%，27.9% 是因为 SP 值 5.0 落地在中间区域需 19-bit accumulator，而 SR 值 8.0 需 18-bit accumulator）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Supernormal support 的硬件实现变更：

```
# 标准 E2M1 decode（15 个有效值，负零 = 冗余）
value_table_positive = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
# 编码 1000 (sign=1, exp=00, mant=0) = 负零 → 无对应正值

# E2M1 + SP decode（16 个有效值）
value_table_positive = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0}
# 编码 1000 → 5.0（super-precision value）
# 正零 0000 → 0.0; 负零编码 → 映射到 5.0（将负号忽略或特殊解码）

# E2M1 + SR decode
value_table_positive = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0}
# 编码 1000 → 8.0（super-range value）
```

硬件层面的修改：MAC 单元中的 decode 逻辑需增加一个特殊 case 判断——当检测到负零编码时（sign=1, exp=00, mant=0），将其映射到 SR 或 SP 值而非 0。这增加了一个多路选择器（MUX）——逻辑开销可忽略。但 accumulator 位宽需要增加：SR 最大值 8.0 → accumulator 需 18 bit（256×8×8=16384），SP 最大值 6.0 → accumulator 需 19 bit（因 SP 值与 6.0 不同组合产生的最大累积值略大于标准 E2M1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 SystemVerilog MAC 设计中实现 supernormal decode：在浮点解码器后增加一级 MUX，当 (sign=1, exp=00, mant=0) 时将输出映射到 SR 或 SP 值而非 0。该修改在 RTL 中仅增加一行组合逻辑，综合开销可忽略。主要的硬件成本来自 accumulator 位宽增加：SR=18 bit（+1 bit vs 标准 E2M1），SP=19 bit（+2 bit）。系统级芯片开销估算：SR 增加 1.9%（vs INT4 0%）、SP 增加 3.6%（vs INT4 0%）。这些"微小"的开销在质量-效率 Pareto 曲线上提供了重要的精度提升选项：如 E2M1+SP 在 Phi-2 上提升准确率至多 2.19%（系统开销仅 1.22%）。论文也将 super-precision 概念扩展到 APoT4 格式（利用 APoT 的冗余编码位增加一个额外的求和组合值）。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs
