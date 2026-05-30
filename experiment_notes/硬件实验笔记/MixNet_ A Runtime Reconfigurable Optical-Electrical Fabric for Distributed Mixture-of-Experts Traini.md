## MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现核心：**区域可重构高带宽域 (Regionally Reconfigurable High-Bandwidth Domain)** 设计——在现有静态电气互连（EPS）的基础上，使用毫秒级可重构 OCS（Optical Circuit Switching）在 scale-up（NVSwitch）和 scale-out（EPS）网络边界构建可动态重构拓扑的光交换域，专门承载 MoE 训练中 Expert Parallelism (EP) 的动态 all-to-all 通信。每台服务器将部分 NIC 连接 EPS（处理 DP、PP），部分 NIC 连接 OCS（处理 EP 的 sparse all-to-all）。OCS 按 MoE block 的通信局部性划分为多个隔离的 region，各 region 由去中心化 topology controller 独立管理。
  - 实验比较：
    - **原型测试（32 A100 GPU）**：MixNet（3 OCS NIC + 1 EPS NIC per server）vs 4×100G EPS 全电气 baseline（4 NIC per server），训练 Mixtral 8×7B、LLaMA-MoE、Qwen-MoE。
    - **大规模仿真（128 servers, 1024 GPU）**：MixNet vs Fat-tree（1:1 non-blocking）、OverSub Fat-tree（3:1）、Rail-optimized、TopoOpt，训练 Mixtral 8×7B/8×22B、Qwen-MoE、DeepSeek-R1。评估指标：training iteration time、cost efficiency (performance per dollar)、scalability（up to 32768 GPU）、failure resiliency。
    - **前瞻仿真（NVL72 scale-up）**：MixNet (co-packaged optical I/O) vs NVL72，训练 DeepSeek-V3，2048 GPU cluster。

- 模拟器名，模拟器链接，或论文修改的模拟器。
  - **大规模仿真**：基于 **FlexFlow**（github.com/flexflow/FlexFlow）+ **htsim**（github.com/nets-cs-pub-ro/NDP/wiki/NDP-Simulator）构建两阶段 packet-level simulator。第一阶段使用扩展的 FlexFlow（增加 pipeline parallelism 支持 + 修正 profiler 与 testbed 实际运行时间对齐）将 MoE 模型、micro-batch size、并行策略转化为 task DAG。第二阶段使用事件驱动的 htsim packet-level simulator 模拟 GPU 间数据包通信。
  - **原型**：无仿真器，使用真实硬件搭建 testbed（32 A100 GPU + Polatis OCS + Mellanox ConnectX-6 NIC + NVIDIA SN3700 Ethernet switch）。
  - **生产环境测量研究**：Certified NVIDIA DGX SuperPOD，128 H800 GPU，128 ConnectX-7 400Gbps NIC，Rail-optimized 拓扑。

- 模拟器模拟什么的性能，修改了什么。
  - htsim 模拟 packet-based GPU 间通信性能，link propagation delay 设为 1 μs。每 server 8 GPU（NVSwitch 900 GB/s 互联）+ 8 NIC（每 NIC 带宽 B）。仿真跨越多个 training iteration。
  - FlexFlow 扩展：添加 pipeline parallelism 支持；修正 profiler 确保 profiled computation time 与 testbed 实际 runtime 对齐。
  - 仿真配置：MixNet 中每 server 2 NIC 连接 EPS fat-tree，6 NIC 连接 OCS fabric（默认）。OCS reconfiguration 时间设为 25 ms（FP 第一个 all-to-all 阻塞网络，后续 all-to-all 在计算期间隐藏）。
  - 模拟的 interconnect fabrics：Fat-tree（1:1 non-blocking）、OverSub Fat-tree（3:1）、Rail-optimized、TopoOpt（all NIC 经大型 flat optical patch panel）、MixNet。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 开源：https://mixnet-project.github.io/（项目网站，代码仓库链接论文未明确给出具体 GitHub 路径，需根据网站确认）
  - 原型测试（硬件架构全栈执行流程，以 Mixtral 8×7B 训练为例）：
    1. **硬件层**：4 台 server，每台 8×NVIDIA A100 GPU + 4×Mellanox ConnectX-6 100G NICs。3 NIC 连接 Polatis 576×576 端口 OCS（支持 10-25ms 重配置），1 NIC 连接 NVIDIA SN3700 Ethernet switch。使用 100 Gbps QSFP28 光模块 + duplex LC 光纤。所有 NIC 运行在 RoCEv2 模式。每 server 内部 4 条 NVLink 连接相邻 GPU 对。
    2. **拓扑配置**：每台 server 在 OCS 侧有 3 个 100G 端口（共 12 optical ports），在 EPS 侧有 1 个 100G 端口（共 4 electrical ports）。
    3. **OCS 重配置流程**：Control server 通过 TL1 commands over Ethernet 向 Polatis OCS 发送重配置指令（平均 41-46ms，99th percentile <70ms）→ 光模块和 NIC 初始化物理链路和网络设备（NIC 激活时间平均 ~5.67s，受限于 commodity 光模块未针对快速重配置优化——论文排除此时间计算实际训练时间）。
    4. **训练执行**：使用 Megatron-LM + MixNet 自定义 collective communication runtime（Python 集成）。通信原语暴露为 mixnet.all_to_all 和 mixnet.all_reduce。DP/PP 走 EPS + NCCL；EP all-to-all 走 MixNet topology-aware routing（优先 OCS 直接链路，其次 EPS fallback）。
    5. **性能测量**：End-to-end training iteration time，对比 MixNet（12 optical + 4 electrical ports）vs EPS baseline（16 electrical ports）。MixNet 达到与 4×100G EPS baseline 相当的性能。
  - 大规模仿真流程：
    1. **输入**：MoE 模型配置（#MoE blocks, #experts, EP/TP/PP degree, seq len, micro-batch size）、cluster 配置（#servers, #GPUs/server, NIC bandwidth）、并行策略。
    2. **Phase 1 — FlexFlow task DAG 生成**：将前述输入转化为描述 cluster 中 computation 和 communication 任务的 DAG。
    3. **Phase 2 — htsim packet-level simulation**：基于 DAG，模拟 GPU 间 packet-based 通信。对 MixNet 的模拟包括：OCS topology reconfiguration（block 25ms for first FP all-to-all）、topology-aware EP routing（优先 OCS 直接电路，次选 EPS）、DP hierarchical all-reduce（intra-host NVSwitch + inter-host ring all-reduce via EPS）。
    4. **输出**：Training iteration time → 计算 speedup、cost efficiency（performance per dollar = 1/iteration_time / networking_cost）。
