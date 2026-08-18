## 超图划分与 PaToH（Hypergraph Partitioning）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 超图划分把超图（hypergraph，边可连接多个顶点的"超边"）顶点切分成 K 个规模平衡的集合，同时最小化被切开的超边（割）代价——比普通图划分更适合表达多顶点共享资源/通信（如多路 fanout 信号），是 VLSI 布局、并行计算任务映射、稀疏矩阵并行化的标准工具。PaToH（Partitioning Tool for Hypergraphs）是经典的**多级**（multilevel）超图划分工具（Çatalyürek & Aykanat，Bilkent 大学）：粗化（coarsening，17 种算法，默认基于 absorption 度量的聚类）→ 初始划分（11 种算法取最优，Greedy Hypergraph Growing）→ 细化（uncoarsening + KL/FM 迭代改进，12 种算法）；递归二分实现任意 K 路划分，支持固定顶点与多约束。Web 证据：官方手册（https://faculty.cc.gatech.edu/~umit/PaToH/manual.pdf ）、多级三阶段综述（https://hal-lirmm.ccsd.cnrs.fr/lirmm-00809529/document ）。
- 在 Lotus（ISCA'26）中：编译器用 PaToH 做**分层**（hierarchical）超图划分映射任务：先一轮把任务分到 FPGA，再一轮在每 FPGA 内把任务分到 tile，以平衡负载并最小化跨芯片通信。论文加入两条映射约束：①同周期边（wire）通信的任务必须同 FPGA（避免高跨芯片延迟上关键路径）；②经内存通信的任务必须同 tile（架构无跨 tile 一致性）。评估：Hier 比随机映射快 3.3×–10.3×；Multicore 上从 flat 到 hierarchical 仅把跨设备边比例从 0.57% 降到 0.49% 就带来 1.7× 加速（跨设备通信极其昂贵）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（Lotus 映射 pass）：①以任务为顶点、任务间通信边（按边数/流量赋权）为超边构建超图；②第一轮 PaToH K=8 划分：任务→8 FPGA，附加强制约束（同周期边任务同 FPGA）；③第二轮 PaToH 每 FPGA 内任务→tile，附加强制约束（内存通信任务同 tile）；④输出静态任务-位置映射，写入任务单元配置（每个任务的 tile 归属、函数指针、输出 token 目的地）。目标函数同时照顾：最小化跨 tile/跨 FPGA 通信量、平衡各 tile 工作量、避免通信延迟落上关键路径。
- 与 flat（单轮划分到 tile）和随机映射对比（Fig.16）：Flat/Hier 显著优于 Rand（跨设备边比例从随机的大多数降为个位数百分比）；Hier 比 Flat 在 NTT、Multicore 上额外显著提速（两轮划分把跨设备边进一步压缩）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：直接调用 PaToH 库/工具（C 库 libpatoh.a，递归二分多级划分）；Lotus 同时用于 CPU 后端的 RepCut 风格线程划分。通用实现还包括 hMETIS 等其他超图划分工具，PaToH 以速度快、稳定著称。
- 使用：任务→多核/多芯片映射、电路网表划分（仿真器、布局）、稀疏矩阵行/列/细粒度划分（SpMV 并行化）、FPGA 分区。在 Lotus 中是"多 FPGA 扩展性"的关键使能：论文显示性能对跨设备通信占比极其敏感，好的分层划分是 8 FPGA 有效扩展的前提。

涉及论文标题：
- Lotus A Multi-FPGA Task Dataflow Architecture to Accelerate Cycle-Level Simulation
