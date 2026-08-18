## DSP48E2（Xilinx UltraScale+ DSP Slice）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DSP48E2 是 Xilinx/AMD UltraScale+（含 Alveo U55c 的 xcu55c）FPGA 中硬化的 DSP 原语，核心是 27-bit A 口 × 18-bit B 口的整数乘法器（W_mul=27+18=45 bits），配套预加（pre-adder）、后加（post-adder/ALU）、内部流水寄存器与级联（cascade）逻辑。乘法器是主要算术资源，预加/后加/流水逻辑结构更简单、面积与功耗显著低于乘法器本身（Xilinx UG579）。操作位宽利用率定义：U_DSP=(w_a+w_b)/W_mul，w_a/w_b 为两操作数有效位宽——这一定义（源自 M4BRAM [9] 的 operand-bit 模型）是评估 DSP 利用效率的基准。XtraMAC 的评估即以此量化：AMD Xilinx Floating-Point Operator 在低精度 workload 下平均 DSP 位利用仅 32.4%，空间复制 26.7%，TATAA 时间共享在 BF16 下仅 8.9%，而 XtraMAC 通过位级打包把 DSP 利用率提到接近 100%（FP4/FP8 达 4 lane/DSP、BF16 2 lane）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DSP48E2 在 MAC 架构中的运转：作为"不透明单 lane 乘法器"时，两个操作数放最低有效位，一次只算一个乘积——低位宽操作数浪费高位位空间；作为"可分区的位空间"时（XtraMAC 的做法），把 P 个低精度操作数的尾数/幅值按非重叠位偏移 (a_i<<s_i, b_j<<t_j) 打包进 A/B 口（Eq.9），DSP 一次宽乘 P_DSP=A_DSP·B_DSP=Σ(a_i·b_j)<<(s_i+t_j)（Eq.10），后计算用固定 shift-and-mask (P_DSP>>(s_i+t_j))&(2^S-1)（Eq.11）提取各 lane 乘积，stride S≥W_lane+G（G≈1 位 guard 吸收进位）保证无跨 lane 干扰。并行度受 DSP 输入宽度约束：P≤min(⌊L_A/S⌋,⌊L_B/S⌋)（Eq.12，L_A=27、L_B=18）。XtraMAC 配置 DSP 内部流水寄存器关闭，使乘法成为 Stage1/Stage2 寄存器间的纯组合块，配合固定四阶段流水线实现恒定 4 cycle 时延与 II=1。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Verilog 例化 DSP48E2 原语（XtraMAC 用 common/dsp_usage.v 封装），Vivado 综合目标 xcu55c-fsvh2892-2L-e；iverilog 仿真需行为级 DSP stub（iverilog 无 DSP48E2 模型），Vivado xsim 用原生模型（-L unisims_ver）。使用：作为 FPGA 混合精度 MAC 的核心乘法资源——XtraMAC 的 49 个固定数据类型 MAC + 4 个运行时双模 MAC 每设计恰好用 1 个 DSP48E2（尾数打包在 A 口）；综合输出每设计 DSP=1、WNS 与 Fmax（synth_all.csv，6c 变体 ~500 MHz），全部配置 >400 MHz。

涉及论文标题：
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA
