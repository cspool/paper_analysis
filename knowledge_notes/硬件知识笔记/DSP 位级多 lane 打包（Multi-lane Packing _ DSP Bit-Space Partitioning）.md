## DSP 位级多 lane 打包（Multi-lane Packing / DSP Bit-Space Partitioning）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把单个 DSP48E2 乘法器当作可分区的位空间，将多个低精度操作数 lane 打包进其 27-bit A 口与 18-bit B 口的非重叠位区间，一次宽整数乘法同时算出多个乘积的微架构技术（XtraMAC 的核心，源自 HiPACK [8]/HiKonv [24]/INT8 打包 [14] 的整数打包思想，扩展到浮点与混合精度）。前提是"所有 MAC 数据类型都能约化为整数尾数乘积"：INT/FP/INT×FP 乘法经符号/指数轻量处理后，DSP 只做纯整数乘法（Eq.1/Eq.4），因此多个 lane 的尾数/幅值可并行打包。打包公式：A_DSP=Σ(a_i<<s_i)、B_DSP=Σ(b_j<<t_j)，其中 s_i/t_j 为按 lane 数据位宽与 stride 选择的非重叠偏移；乘积 P_DSP=Σ_{i,j}(a_i·b_j)<<(s_i+t_j) 是多项式结构，保证 lane 隔离；提取用固定 shift-and-mask，stride S≥W_lane+G（G≈1 bit guard）防进位串扰。并行度 P≤min(⌊27/S⌋,⌊18/S⌋)（Eq.12），与数据类型位宽成反比：FP4（E2M1）与 FP8（E4M3）达 4 lane/DSP、BF16 2 lane、INT8 2 lane。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 XtraMAC 四阶段流水线中的运转：Stage1（操作数解释/位映射）并行实例化 N 个数据类型映射子模块（浮点子模块提取 sign/exponent/mantissa、恢复隐式前导 1、把尾数打包进 DSP 口位置并转发指数/符号元数据；整数子模块做二补码解码、打包幅值、转发符号位），datatype 信号选出一对 bit-packed DSP 口操作数与元数据；Stage2 DSP 一次整数乘法 + 逐 lane 后处理（浮点 lane 做 LZC 归一化与指数更新、整数 lane 只做幅值提取与符号 XOR），P 条相同 lane 流水并行重建；Stage3 分离式累加（整数 bank 二补码加法器、浮点 bank 指数对齐+尾数加减+重归一化，每 cycle 算 P 个 lane）；Stage4 把 P 个 lane 结果拼成打包输出字（如 4 个 FP8 lane 或 2 个 BF16 lane 组成 32-bit）。效果：INT4×BF16 下每 DSP 每 cycle 2 个 MAC，1920 个 XtraMAC 在 U55c 跑 300 MHz，GEMV 时延 0.0246 ms 反超 CUTLASS H100（0.0294 ms）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Verilog 位映射逻辑 + DSP48E2 原语（common/dsp_usage.v 封装），Vivado 综合/实现（xcu55c）；仓库 mac_configs/ 按 1lane/2lane/4lane 分目录，mac_cores/ 提供参数化原语（fp32/fp16/bf16/fp8e4m3/fp8e5m2/int32/mixed_precision），User_spec/generate_mac_bundle.py 按 spec（如 "FP8e4m3*FP8e4m3+FP16"）生成 rtl/tb/manifest.json bundle。使用：混合精度 MAC 的资源压缩——XtraMAC 相对 AMD FP Operator IP 平均降 DSP 50.0%、LUT 30.0%、FF 47.9%，计算密度 1.4–2.0×；设计准则：sub-8-bit 量化（INT4/FP4）优先以获得最高每 DSP 并行度；lane 打包效率与操作数位宽成反比。

涉及论文标题：
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA
