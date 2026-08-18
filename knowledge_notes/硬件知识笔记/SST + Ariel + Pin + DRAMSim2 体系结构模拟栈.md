## SST + Ariel + Pin + DRAMSim2 体系结构模拟栈

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SST（Structural Simulation Toolkit，Sandia 主导，BSD-3 开源，github.com/sstsimulator，https://sst-simulator.org）是组件化并行体系结构模拟框架；Ariel 是其中基于动态应用 trace 的多核 CPU 模型——前端用 Pin（Intel 动态二进制插桩）或 PEBIL 采集访存序列，CPU 模型只详细模拟访存、非访存指令按单周期处理、不追踪指令依赖，经 memHierarchy（cache/总线/内存控制器模型）把 memEvent 发往下游；DRAMSim2 是外部 DRAM 时序模拟器，作内存后端接入 SST（https://sst-simulator.org/sst-docs/docs/elements/ariel/intro；OSTI ISCA 教程 https://www.osti.gov/servlets/purl/1257684）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Dorado 的模拟流程：1024 线程在 48 核 SkyLake 服务器（单 NUMA 域）上运行，Pin 采集全部指令与数据访问 trace（含同步点；不采集 OS 指令）；SST 中每核用 Ariel 乱序核模型执行自己线程的指令、按周期推进，访存逐级经 L1/L2/TD+ED 目录/PointerSpace/2D mesh（簇内 5 cycles/hop、跨簇 60 cycles RT）→ DRAMSim2 输出主存时序；同步点保证互斥区单线程、barrier 同时离开。作者对 Ariel 的改造：高精度 TSO（投机 load、write-exclusive prefetch、SB 有序 drain）。精度交叉验证：对微服务/FaaS/KV 负载另跑 QEMU 全系统 64 核仿真对比，结论相近故 1024 核只用 Pin trace。配套 RTL 侧用 OpenROAD + Verilator 评估目录面积/功耗（见论文）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与使用：SST 组件经 Python 配置脚本互连，memHierarchy 连接处理器组件（Ariel/MacSim/gem5 前端）与内存后端（DRAMSim2/Ramulator/HybridSim/NVDimmSim/HBM DRAMSim2）；SST v8 兼容 Pin 2.14 与 DRAMSim2 2.2.2。使用要点：trace 驱动仿真须处理 warmup/measure 标记（Dorado：50M 指令/核预热、500M 指令/核测量）与多线程同步的保真重建；Pin 不采 OS 指令是方法局限，需全系统仿真补证。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
