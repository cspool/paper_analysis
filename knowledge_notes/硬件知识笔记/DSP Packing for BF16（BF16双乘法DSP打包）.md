## DSP Packing for BF16（BF16双乘法DSP打包）

术语是什么？通过联网搜索让回答具体和精准。
DSP Packing是一种FPGA DSP资源优化技术，利用BF16浮点格式mantissa仅7 bits的特性，将两个独立BF16×BF16乘法映射到单个DSP48/E单元并行执行。传统单DSP支持一个乘法（如A×B），DSP packing通过将两乘法的mantissa分配到DSP不同bit段并行计算，使每DSP等效计算吞吐翻倍。DFVG在PE微架构中采用此技术，使V80的10848个DSP48等效支持21696个BF16 MAC/cycle。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DFVG中DSP packing的工作原理（单DSP内双BF16×BF16乘法）：
```
// BF16: 1b sign | 8b exponent | 7b mantissa
// DSP48: 27-bit × 18-bit 乘法器

// 双乘法packing:
// 乘法1: A1.mantissa[6:0] × B.mantissa[6:0] → P1[15:0]
// 乘法2: A2.mantissa[6:0] × B.mantissa[6:0] → P2[31:16]
// 约束: B操作数相同（在multi-branch drafting中自然满足）

// PE内使用:
for each PE in systolic array:
    A1 = weight_buffer_1[addr]  // branch 1 weight
    A2 = weight_buffer_2[addr]  // branch 2 weight
    B  = activation[addr]       // shared activation
    (P1, P2) = DSP.packed_multiply(A1, A2, B)
    acc1 += BF16_assemble(A1.sign, A1.exp, P1)
    acc2 += BF16_assemble(A2.sign, A2.exp, P2)
```
关键约束：两乘法的B操作数须相同。在speculative drafting中因多branch共享prefix而自然满足。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Xilinx DSP48E2原语手动实例化或HLS directive，将操作数分配到DSP不同bit段→后处理分离乘积→各自完成exponent调整。无法pack的乘法回退到标准单乘法模式。DFVG在Verilog中直接实例化DSP48配置bit-splitting逻辑。该技术已在一系列OPU设计中验证。开源：https://github.com/ShaoqiangLu/DFVG。

涉及论文标题：
- DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

