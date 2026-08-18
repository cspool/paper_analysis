## PNM（Processing-Near-Memory，近内存处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PNM 是 Processing-In-Memory（PIM）的一种形态：在内存设备/内存控制器侧放置有限算力，让数据在离内存最近处被处理，减少数据搬运。与 PIM 的广义概念相比，PNM 强调处理单元"靠近"内存（如 CXL Type 3 设备上的 PNM 引擎、内存扩展器内的近数据处理 NDP 单元），计算能力受限（低频、少缓存）但可复用内存带宽。AXLE 论文中的 PNM 指 CCM 设备内的计算资源：FPGA 硬核 PFL（MAC/ACC/CMP）或通用核（Cortex-A72），用于执行卸载的 kernel。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
AXLE 中的 PNM 执行流程：主机把 kernel 描述符写到设备内存 → CCM 调度器把任务切块 → 每个 µthread 处理固定大小输入向量（如 KNN 的一个查询向量对数据库分块求距离）→ PNM 单元读设备本地 DDR5（16 通道）执行乘加/累加/比较 → 结果写回设备内存待反流。性能权衡：PNM 单元慢（2GHz vs 主机 3GHz、16 处理单元 vs 32），但省掉跨 CXL 链路的数据搬运；对数据搬运占比高的负载（图分析 47.77%）收益大，对主机计算占比高的负载收益小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：ASIC 硬核 PFL IP（MAC/ACC/CMP 等基础函数逻辑，面向数值/字符串过滤、向量距离等固定功能）、FPGA 可编程逻辑、或嵌入式通用核（ARM/RISC-V）。M²NDP 用扩展 RISC-V 向量指令的细粒度多线程核实现通用 PNM。使用方式：部分卸载内存密集操作；配合不同卸载机制（RP/BS/异步背流）。局限：算力有限、与主机需要通信协议协调（AXLE 的核心问题）。

CompAir 的补充视角（ISCA'26，arXiv:2509.13710）：在可扩展 DRAM-PIM 系统 CENT 中，PNM 具体指 CXL 控制器内集中的 CPU + 大规模专用 NLU（Non-Linear Unit，非线性单元，7nm 4.4mm²，约 4× 一个 32MB DRAM bank 的面积）。NLU 承担 Softmax/SiLU/RMSNorm 等非线性算子与 RoPE 的标量重排。缺陷：所有 channel 共享同一 NLU → 大量数据在 NLU 与各 bank 间往返；CompAir 实测 4K 上下文时非线性通信+计算占 block 时间 >20%、长上下文可达 25%。给每 bank 分布式 NLU 又因面积不可行（4.4mm²/bank）。CompAir 的解法：把 NLU 降级为 NoC 路由器内的 Curry ALU（仅为 router 面积 2.94%），以在途计算去中心化非线性与集合通信；设备控制器只保留指令下发、不再含非线性执行单元。

FlexQ-NDP 的银行级 NDP 视角（ISCA'26）：该论文语境中的 NDP 指"处理单元（PU）放进 DRAM 芯片内部、贴近每个 bank"的银行级存内计算，而非 CXL 设备侧 PNM——PU 与内存单元紧耦合，NDP 性能同时取决于 workload 到 PU 的映射和 DRAM 数据布局（不同布局造成不同 row-buffer miss 率）。其目标架构取 Hynix GDDR6-AiM 的形态：32 颗 GDDR6 芯片 × 2 通道 × 16 bank = 1024 PU，32 GB，12 Gbps/pin、1.5 TB/s 聚合内部带宽，主机经 PCIe 通信；每 bank 配一个多精度 PU（0.4 GHz）+ 5Kb SRAM（20×32B 缓冲）。与 CXL 侧 PNM 的差异：NDP 直接利用 bank 级内部高带宽做 GEMM，但 PU 频率低、缓冲是 KB 级，编译期必须精细规划算子划分/循环 tiling/DRAM 行列映射与缓冲分配——这正是 FlexQ-NDP 编译器要解决的问题。启发：NDP 架构评估维度 = 行缓冲命中率（行切换开销）、SRAM 缓冲容量与分配、PU 算力/频率、数据布局。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
