## GDDR6-AiM（SK Hynix Accelerator-in-Memory，GDDR6 基存内计算芯片）

术语解释
SK Hynix 2022 年发布的首款 GDDR6 接口的存内计算（PIM）商用芯片，在 DRAM die 内集成 MAC 阵列。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GDDR6-AiM 是 SK Hynix 在 ISSCC 2022 发布的 GDDR6 基 Accelerator-in-Memory 芯片（IEEE Xplore 9731711）：1ynm 工艺、1.25V 工作电压（标准 GDDR6 为 1.35V）、8Gb 容量、16Gb/s/pin，die 内集成 MAC 阵列，峰值 1 TFLOPS MAC 算力并支持多种激活函数；用扩展 DRAM 命令集控制内部 PU，与标准 GDDR6 接口兼容，配套 FPGA 参考平台 + FMC AiM 扩展卡的软件栈（Hot Chips 2022，IEEE Xplore 9895629）。属"银行级 PIM"：PU 嵌入 DRAM die、贴近 bank，利用全部 bank 并行操作获得的内部高带宽做 GEMM（RNN/MLP 等访存受限负载），SK hynix 宣称相对常规 DRAM 最高 16× 加速、功耗降 80%。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
FlexQ-NDP 以 GDDR6-AiM 为硬件模板扩展出"低比特 FP 导向 NDP 架构"：32 颗芯片 × 2 通道 × 16 bank = 1024 个 PU、32 GB、12 Gbps/pin 聚合 1.5 TB/s 内部带宽，主机经 PCIe 通信；每 bank 一个多精度 PU（0.4 GHz）+ 5Kb SRAM（20×32B，可配为 value/scale/partial-sum 缓冲），PU 操作数位宽 256b 对齐 GDDR6 读写位宽。执行流程：GEMM 算子被划分到各 bank → PU 从本 bank DRAM 流式读出值做 MAC、与 SRAM 内驻留操作数点积 → 部分和写回缓冲/DRAM。硬件约束决定编译目标：DRAM 行切换（precharge/activate）贵、SRAM 只有 KB 级、PU 频率低——数据布局与缓冲分配直接决定性能。容量约束的直接证据：32 GB 装不下 FP16 的 LLaMA2-34B 参数（~68GB），需运行时权重重写；W4A4S8 量化把参数压到容量内、免重写即获得 7.17× decode 加速——这就是"低比特 FP 缓解 NDP 容量压力"论点的来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：商用 GDDR6-AiM 已流片（SK hynix + SAPEON 合作），配套 FPGA 参考平台与软件栈供开放研究；学术研究中多以模拟器建模——FlexQ-NDP 用修改版 UniNDP + DRAMSim3 时序（tCK=0.66ns、tRCD/tRP=24、tCCDL=4、BL=16、tCL=24）做 cycle-accurate 仿真。使用方式：把 GEMM（MVM/MM）映射到各 bank 的 PU 执行，非线性算子查表/近似或回主机。限制：PU 频率低（商用 1 GHz、研究配置 0.4 GHz）、每 bank SRAM 仅 KB 级、bank 间跨芯片通信受限——编译期必须精细规划划分/tiling/布局。

涉及论文标题：
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
