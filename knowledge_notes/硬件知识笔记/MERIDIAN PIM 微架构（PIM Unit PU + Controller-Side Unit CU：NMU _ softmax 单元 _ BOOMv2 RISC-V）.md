## MERIDIAN PIM 微架构（PIM Unit PU + Controller-Side Unit CU：NMU / softmax 单元 / BOOMv2 RISC-V）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MERIDIAN（ISCA'26）LPDDR5X-PIM 设备的内部计算组织，分两层：**PIM Unit（PU）**——每个 DRAM bank 旁放置一个，采用 All-Bank-Mode 设计（DRAM 命令广播到所有 bank 同地址，利用 bank 级并行最大化内部带宽，连续 column 命令间隔受 t_CCD_L 限制）；每 PU 消费 256-bit（16 个 FP16）输入，含 16 个 FP16 比较器（非线性区间选择）+ 16 个 FP16 乘法器 + 16 个 FP16 加法器（可重构 reduction/elementwise-add）+ 4 个双缓冲 4KB buffer，breakpoint 与非线性参数存 DRAM array。**Controller-Side Unit（CU）**——每 PIM 控制器每 channel 一个 Near-Memory Unit（NMU）：加法单元（channel 内 reduction 与逐元素累加）+ 专用 softmax 单元（完整 softmax 流水、高吞吐、数值稳定）；CXL 控制器内 8 个 BOOMv2 RISC-V 核做跨 channel/跨设备聚合与轻量控制/计算（softmax 等内存内难实现的算子）。每设备含 8 个 LPDDR-PIM package、每 package 由 8-channel PIM 控制器（128-bit 总宽、每 channel 4 个 16-bit DRAM die）管理。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件运转流程（一次 decode 步的文档注意力 GEMV + 非线性）：host 经 CXL.mem 发 PIM 命令（PIM_MAC 乘累加 / PIM_CMP 比较 / PIM_EW_MULT 逐元素乘 / PIM_EW_ADD 逐元素加 / PIM_ACT 全 bank 行激活 / PIM_WR_PB / PIM_RD_PB 写/读 PU buffer）→ 设备控制器广播到相关 channel 与 PU → PU 在 All-Bank-Mode 下从各 bank 取 16 个 FP16 就地乘加（GEMV/skinny-GEMM），非线性（GeLU/Swish）用分段线性 y=ax+b：比较器按输入区间选 breakpoint、取 LUT 系数、复用乘法/加法器单次乘加；softmax 与 channel 内归约交给 NMU（加法单元做 reduction、softmax 单元做完整数值稳定 softmax）→ 跨设备聚合由 BOOMv2 RISC-V 核完成（也承载融合等轻量任务）。面积：10nm DRAM 工艺每 PU 0.15mm²（算术 50.5%/buffer 34.9%/控制 14.6%），16 个 PU/die 共 2.41mm²，仅占 47.53mm² LPDDR5X die 的 5.07%；7nm 逻辑下 16-lane FP16 加法单元 0.02mm²、softmax 单元 1.38mm²、每 BOOMv2 核 2.94mm²。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PU 是 bank 级 PIM 单元（与 HBM-PIM/AiM 的 bank MAC 同类），CU 是控制器侧近存单元（类似 CHIME 的 rank PU/softmax 单元，但集成 softmax 专用硬件）；BOOMv2 是开源 Berkeley Out-of-Order Machine RISC-V 核（github.com/riscv-boom）。仿真/综合：PU/CU 算术单元用 Verilog + Synopsys Design Compiler 28nm 综合、按 10nm-class DRAM 工艺缩放并 10× 面积/功耗膨胀（逻辑 vs DRAM 工艺效率差），controller 侧按 7nm 建模，SRAM buffer 按 AttAcc 估计，DRAM 能量用 Micron LPDDR5/LPDDR5X datasheet + DRAMPower。使用：与文档注意力分解协同——DAC/CEC 两 cluster 微架构同构，允许动态负载迁移（CEC 参数静态复制到 DAC，空闲时协助上下文计算）与动态资源重分配。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
