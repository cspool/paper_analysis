## BRU 与 LPU（Blind-rotation Unit / LWE Processing Unit）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BRU 与 LPU 是 FlashTFHE 加速器中每个 compute core 内的两个专用功能单元，分工对应 multi-bit TFHE 的两类运算：(1) BRU（Blind-rotation Unit）= 面向盲旋转/外部乘积的单片宽深流水线，执行多项式运算（FFT/IFFT + VecMAC），是吞吐核心——单 core MAC 吞吐 512 coefficients/cycle，采用 48-bit 定点复数数据通路（48-bit 是论文在 TFHE-rs 与 Concrete Optimizer 全部参数集上测出的最优位宽，保证正确性）；两个 BRU 共享一个 I-FFT 单元（FFT:I-FFT 操作数比约为 l_b:1）；(2) LPU（LWE Processing Unit）= 面向 LWE 密文的向量引擎，执行不含多项式运算的操作——key-switching、sample extraction 与 LWE-native 元素级加/乘，64-bit 宽度匹配 2^64 torus modulus，8 个独立 lane（每 lane 32 个并行 64-bit 值，可独立寻址、独立时钟门控），含向量加/乘单元、decomposer 与 rotator。
- 两者通过批级交错（Figure 12）协同：BRU 跑 PBS（盲旋转）时 LPU 同时跑 key-switching/sample extraction/native 运算，跨 batch 重叠以隐藏延迟。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 一个 PBS 在 core 内流程：LPU 先做 key-switching-first 的 KS（8 lane 按需用）→ BRU 的 FFT-A/FFT-B 集群对 GLWE 做 FFT，VecMAC 与 BSK chunk 做 tiled 乘累加进 9.2MB Acc buffer → (k+1)·l_b 次累计后共享 I-FFT → sample extraction 回 LPU 域 → LPU 做 native 运算或交给下一 PBS。每 core 组面积/功耗：BRU 11.01mm²/22.29W、I-FFT 4.25mm²/12.59W、Acc buffer（9.2MB）9.83mm²/3.11W、LPU 1.32mm²/0.61W（16nm、1GHz），4 core 组成 108.08mm²/132.98W 全芯片。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Chisel HDL 编写，RTL 正确性在 Xilinx Virtex VU47P FPGA（Beethoven 框架）验证，Synopsys Design Compiler + TSMC N16 综合至 1GHz；accumulator/GLWE/LWE buffer 用 Arm Artisan physical IP 建模。使用：编译器把指令流按 batch 调度到 BRU/LPU；LPU 的 lane partitioning 支持 adaptive batching（见该条目）。论文无开源链接（联网未找到公开仓库）。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
