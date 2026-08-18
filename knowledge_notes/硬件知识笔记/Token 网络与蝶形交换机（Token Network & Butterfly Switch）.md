## Token 网络与蝶形交换机（Token Network & Butterfly Switch）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Token 网络是连接任务数据流系统中所有 tile 的片上通信网络，负责在生产者与消费者任务之间传递输出 token（携带 (cycleId, taskId) 与可选数据值）。蝶形交换机（butterfly switch）是多级互连网络（如 Omega/Butterfly 拓扑）的交换机实现：多个小规模交叉开关级联，以对数量级级数连接大量端口，天然支持多路径、流水化。
- 在 Lotus（ISCA'26）中：每个 FPGA 内用 25×25 两阶段（2-stage）蝶形交换机连接 68 个 tile 并为跨 FPGA 通信提供 8 个端口；每 4 个 tile 组成一组共享 1 个内存通道（HBM2）与 1 个交换机端口（concentration），换取可实现的 tile 密度（68 tile/FPGA）。token 是变长的（最多 2 个值 + 元数据），交换机按需路由到目标 tile 的任务单元输入单元。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：①任务执行完 → 输出单元为每条输出边产生 token（复制输出值）→ 注入本 tile 的交换机端口；②交换机在 2 级内把 token 路由到目标 tile（或跨 FPGA 出口端口）；③目标 tile 输入单元按 (cycleId, taskId) 索引输入内存。同 tile 内任务间的 token 也经交换机（或本地短路径）送达，通信模式是任务间数据流而非一致性协议流量。
- 设计取舍：concentration（4 tile 组共享端口）在带宽充足的前提下显著降低交换机与内存互连成本；token 带宽与内存带宽分离（token 走交换机、数据走 L1/L2/HBM），避免两者争用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：25×25 两阶段蝶形交换机（每端口 4 tile 集中），每个 FPGA 8 个端口接 Aurora shim 用于跨 FPGA（8 FPGA 全互联见芯片设计层 Aurora 条目）；跨 FPGA token 经 Aurora 64B/66B 光链路（每 64-bit lane 串行化、滑动窗口重传+流控）。
- 使用：是 Lotus"通信与计算解耦"的物理载体——token 异步到达、任务单元缓冲输入，使长跨芯片延迟（200ns）不阻塞本地核执行；NTT 每周期 51200 条边的高流量场景下交换机接近饱和（NTT 32% idle 周期来源之一），是论文调整配置（更少更大的 tile）以降低片内通信的动因。

涉及论文标题：
- Lotus A Multi-FPGA Task Dataflow Architecture to Accelerate Cycle-Level Simulation
