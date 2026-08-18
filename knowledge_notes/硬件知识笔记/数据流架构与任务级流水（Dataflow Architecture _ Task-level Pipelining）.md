## 数据流架构与任务级流水（Dataflow Architecture / Task-level Pipelining）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 数据流架构是一种由数据可用性驱动运算的计算组织方式：任务（函数/循环）之间通过 FIFO 或缓冲通道连接，生产者产出数据即触发消费者执行，不同函数/循环经任务级流水（task-level pipelining）重叠执行而非串行；任务间用高效片上通信取代频繁外部内存访问，从而低延迟处理大量数据搬移与运算（CODO 引言动机：引用 [5][14][20][42][44]）。
- 术语边界（CODO 明确界定）：本文的数据流架构 ≠ 动态调度数据流电路（如 [21]）≠ input/output stationary 等数据流映射策略（如 [11]），三者概念正交。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- FPGA 上的数据流实现：每个任务综合为独立硬件模块，模块间用片上缓冲互连；HLS dataflow pragma 让综合器自动生成通道与握手。执行时间线（Fig. 2 Padding→Conv2D→ReLU 例子）：ping-pong 数据流中 Conv2D 须等 Padding 写满整块才启动（长间隔、重叠少）；理想 FIFO 数据流中 Conv2D 在 Padding 产出第一个所需元素后即启动，K1/K2/K3 各迭代色块重叠、总体 latency 最短。
- 性能三要素（CODO 归纳）：correctness（代码满足约束才能流式执行）、communication（缓冲类型与写时机决定吞吐）、parallelism（tiling/unroll/pipeline 平衡各任务延迟）；FIFO 型数据流中三者耦合——任何一项的变换都可能破坏其他项，因此需要联合 co-optimization。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 商用实现：Vitis HLS `#pragma HLS dataflow`（自动通道 + 任务级流水，但要求代码满足单产单消等严格约束）。CODO 把任意 C/C++ 或 PyTorch 模型自动变换为满足约束的形式，配 FIFO 优先缓冲与自动调度，输出可上板的数据流加速器（Alveo U280 上 DNN overall speedup 9.6×–127.5×）。资源特性：FIFO 只存 in-flight 数据、BRAM 开销小；ping-pong 需双份缓冲。

- M100 补充视角（ISCA'26，车规 NPU 的 orchestrated dataflow）：M100 把数据流架构从 FPGA/HLS 任务级流水扩展到量产 AI 推理 NPU：以 tensor（大或小）为指令粒度，编译器/固件（"orchestration"）编排数据在计算单元与片上/片外存储间的流动；几乎完全消除多级 cache（以软件管理的数据流 + 硬件同步计数替代 cache 一致性），计算由"数据就绪"触发而非指令流顺序。与 CODO（HLS 任务级 FIFO 流水）的差异：M100 的并行粒度是 tensor 级指令 + TPB 内功能单元间数据流（2MB HBSM 共享 + 生产者-消费者同步计数），指令流松散有序（同一功能单元保序、跨单元可乱序完成，同步依赖交软件），硬件无需 cache 一致性/register file/全局执行顺序。
涉及论文标题：
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
