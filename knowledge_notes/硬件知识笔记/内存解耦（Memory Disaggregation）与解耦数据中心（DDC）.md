## 内存解耦（Memory Disaggregation）与解耦数据中心（DDC）

术语解释
将传统服务器机箱内原本共置（co-located）的 CPU、内存、存储等硬件资源拆开，成为独立管理的资源池（compute/memory/storage pool），通过数据中心级或机架级网络互联访问的架构范式；采用该架构的数据中心称为 Disaggregated Datacenter（DDC）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 传统服务器中 CPU 与内存/存储通过处理器总线、内存总线、PCIe 等本地互连直接访问，延迟极低但利用率耦合（一台服务器内存多则浪费、少则不足）。解耦的核心动机是提升资源利用率、降低运营成本：把内存从计算节点剥离成共享池，任意 compute 节点可按需挂载远端内存。
- 代价是"本地互连"被替换为"外部网络"：系统软件访问远端资源必须走网络，产生 disaggregation traffic——由系统软件生成、用于资源访问与管理的关键流量，要求超低延迟（sub-10µs）与高带宽（40–400 Gbps）。
- 为满足该要求，DDC 通常部署一条独立于应用流量、专用且 RDMA-capable 的高带宽全连接网络（如 fat-tree + 400 GbE 交换机和 NIC）。论文量化（Table I）：512 节点系统网络占系统成本 23.1%、功率 21.3%；2048 节点占 23.0%/25.7%。
- 代表系统：Infiniswap（one-sided RDMA 远端内存）、LegoOS（RDMA 分布式 OS）、FastSwap、Hermit、Clio（硬件软件协同解耦内存）、Pond（CXL 内存池化）、AIFM、SMART 等。R2D2 本身针对的是解耦网络本身（interconnect+runtime），与 RDMA 及内存节点侧技术正交互补。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 R2D2 硬件架构中，compute 与 memory 节点分布在机架/行内，通过 R2D2 fabric 互联。一次内存访问流程：compute 节点 NIC → R2D2 MoR 单元的 patch panel 受体 → 机器人预先建立好的被动光纤链路 → 目标 memory 节点 NIC → 内存控制器读写。链路由机器人按需建立（单跳直连），稳态无交换、无逐包转发。
- 与之对比的 baseline 硬件：compute NIC → ToR/leaf switch（逐包转发，500ns/跳）→ spine → 目标 memory 节点，多跳共享带宽、incast 拥塞。解耦决定了流量形态（compute↔memory 配对、稀疏且稳定），这正是 R2D2 硬件按需直连设计的前提。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 业界实现路径：(1) 网络侧——专用 RDMA fat-tree/OCS 网络（Google、AWS 等）或 CXL 池化（Pond/Intel Rack Scale）；(2) 主机侧——one-sided RDMA 库（libibverbs/rdma-core）、远内存 OS/运行时（Infiniswap、LegoOS、Semeru）；(3) 本文 R2D2——用机器人重构物理光纤拓扑，按需提供 compute-memory 直连，消除全连接网络的空间/时间过配。评估中用 Gao et al.（OSDI'16）与 Shoal 的内存流量 trace、Protean VM 分配 trace 驱动。

涉及论文标题：
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters
