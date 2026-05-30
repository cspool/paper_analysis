## RoCE / InfiniBand / Inter-node GPU Interconnect

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

RoCE (RDMA over Converged Ethernet) 和 InfiniBand 是两种主流的高性能 GPU 集群跨节点互联技术，均支持 RDMA（Remote Direct Memory Access），允许一台机器的 NIC 直接读写远程机器的 GPU 或 CPU 内存。FUSCO 的评估使用 400 Gbps Mellanox ConnectX-7 NIC 通过 RoCE 进行跨节点通信，每节点 10 张 NIC 提供约 50 GB/s 的聚合跨节点带宽。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 FUSCO 的 dComm 中，跨节点传输（Node-Level Forwarding）使用 GPUDirect RDMA over RoCE：

```
# Sender 节点 GPU → NIC → RoCE fabric → NIC → Receiver 节点 GPU
# 路径:
#   GPU HBM → PCIe Bridge → ConnectX-7 NIC (GPUDirect RDMA read)
#   → 400Gbps RoCE link → Remote ConnectX-7 NIC
#   → PCIe Bridge → Remote GPU HBM (RDMA write)

# dComm pipelined workflow:
# Time →
# GPU: |== Slice₀ gather to ring buf ==|== Slice₁ gather ==|== Slice₂ ==|
# NIC:                                |== RDMA Slice₀ =======|== RDMA Slice₁ ===|
#                                        ↑ NIC 直接在 GPU mem 中读 slice
```

FUSCO 利用每节点 10 张 NIC 的并行能力：Online Load Balancer 将 GPUs 分为 M 个 communication group，每组通过独立的物理 NIC channel 并行执行跨节点通信，最大化聚合带宽利用率。在 load-imbalanced 场景下，Balancer 将高负载 GPU 分散到不同 group，缓解单 NIC channel 的热点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- InfiniBand (IB) 和 RoCE 均提供 RDMA 能力，但 IB 是专用协议（需要 IB 交换机），RoCE 运行在标准以太网上（需要支持 RoCE 的交换机和 NIC）
- Mellanox ConnectX-7 支持 400 Gbps 线速，FUSCO 评估的 10 NIC × 400 Gbps = 4 Tbps 聚合带宽（理论值），但实际约 50 GB/s（因为 PCIe 瓶颈和多 NIC 调度 overhead）
- 节点内 NVLink（480 GB/s）和节点间 RoCE（50 GB/s）的 9.6× 带宽比是 FUSCO Hierarchical Routing 设计的核心硬件依据
- NCCL transparently supports both RoCE and InfiniBand，通过 IB Verbs 接口访问 RDMA 能力
- DeepEP 紧密依赖 InfiniBand 的 IBGDA 特性，而 FUSCO 通过 NCCL transport layer 保持了 RoCE/IB/TCP 的透明兼容

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
