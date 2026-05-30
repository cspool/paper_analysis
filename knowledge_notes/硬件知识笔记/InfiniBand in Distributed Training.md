## InfiniBand in Distributed Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
InfiniBand (IB) 是一种高性能计算机网络通信标准，广泛用于 HPC 和 AI 训练集群的节点间（inter-node）互联。相比以太网，InfiniBand 提供更低的延迟（~1 μs）、更高的带宽（HDR 200 Gb/s、NDR 400 Gb/s），并支持 RDMA（Remote Direct Memory Access）——允许 GPU 直接读写远程节点的内存而无需 CPU 介入。在 MoE 分布式训练中，InfiniBand 承载了跨节点的 All-to-All 通信流量，但其带宽远低于节点内 NVLink（如 NetMoE 集群中：IB 100 GB/s vs NVLink 400 GB/s），因此跨节点通信常成为训练瓶颈。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 NetMoE 的 4 节点 A800 集群中：
- 每节点内有 8 张 A800 GPU（NVLink 400 GB/s 全互联）
- 节点间通过 InfiniBand 互联（100 GB/s）
- IB 通过 PCIe 连接到 GPU（需经过 PCIe switch），引入额外延迟
- All-to-All 通信中，inter-node 通信量取决于 routing 分布：`S_inter = {(i, e) | Node(SmpDev(i)) ≠ Node(ExpDev(e))}`，即 sample 需要发给不同 node 上的 expert 的 token 集合
- NetMoE 通过动态样本放置减少 `s_inter`（跨节点通信量），在 2 nodes/16 GPUs 下 MoE-GPT-S 减少 39.10%
- NCCL 在 InfiniBand 上使用 Ring 或 Tree 算法实现 All-to-All，具体选择取决于消息大小和拓扑

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- InfiniBand 需要专用 HCA（Host Channel Adapter）网卡和 IB 交换机，典型部署如 NVIDIA Quantum-2 NDR 交换机（400 Gb/s per port）
- NCCL 通过 `ncclIbInit` 初始化 IB 通信，使用 GPUDirect RDMA 绕过 CPU 内存，实现 GPU-to-GPU 直接数据传输
- 在 MoE 训练中，IB 带宽通常为 NVLink 的 1/4 到 1/8，是 All-to-All 通信的主要瓶颈——这也是 NetMoE 优化 inter-node 通信的根本动机
- 调优建议：(1) 使用多个 IB HCA 增加聚合带宽；(2) 使用 SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）在交换机上做 in-network reduction；(3) 控制 EP 规模使 A2A 不跨越 IB（如 MoE Parallel Folding 中的策略）

涉及论文标题：
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement
