## GPUDirect P2P / GPUDirect RDMA

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPUDirect 是 NVIDIA 的一系列技术，使 GPU 和相关设备（NIC、存储、其他 GPU）之间可以直接进行数据传输而无需经过 CPU 内存作为中转。**GPUDirect P2P**（Peer-to-Peer）允许同一节点内不同 GPU 直接访问彼此的内存（通过 NVLink 或 PCIe），省去 CPU 侧的 staging buffer。**GPUDirect RDMA** 允许 NIC（网卡）直接通过 RDMA 从 GPU 内存读取/写入数据，无需 CPU 参与，是跨节点 GPU 通信的关键使能技术。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FUSCO 的 dComm 引擎分别在不同传输路径使用这两项技术：

```
# Intra-node: GPUDirect P2P over NVLink
# Sender GPU → Receiver GPU (同节点)
cudaMemcpyPeer(dst_ptr, dst_device, src_ptr, src_device, size);
# 数据传输路径: GPU₀ HBM → NVLink → GPU₁ HBM
# 无需 CPU staging buffer, dComm 在此路径中集成 descriptor 解释

# Inter-node: GPUDirect RDMA
# GPU memory → NIC (RDMA write) → remote NIC → remote GPU memory
# NIC 直接通过 PCIe 读取 GPU ring buffer 中的数据
# 数据传输路径: GPU HBM → PCIe bridge → ConnectX-7 NIC → RoCE fabric
```

在 dComm 的 pipelined workflow 中：
- GPU Producer Kernel 通过 GPUDirect P2P（intra-node）或 local copy to ring buffer（inter-node pre-RDMA）完成 descriptor-driven gather
- NIC Consumer 通过 GPUDirect RDMA 从 ring buffer 直接读取 slice 并发送
- 两个操作在不同硬件单元上并行执行，GPU memory copy 被 NIC 传输时间完全掩盖

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 需要 CUDA 驱动支持，启用 P2P access：`cudaDeviceEnablePeerAccess(peer_device, 0)`
- NVLink 和 PCIe 均可支持 GPUDirect P2P，但 NVLink 提供高得多的带宽（480 GB/s vs ~64 GB/s for PCIe 5.0）
- GPUDirect RDMA 需要 NIC 支持（如 Mellanox ConnectX 系列 + InfiniBand 或 RoCE），且 GPU 内存需注册为 RDMA-accessible（NCCL 在初始化时自动完成）
- NCCL 自动检测 GPUDirect RDMA 可用性（环境变量 `NCCL_NET_GDR_LEVEL`），FUSCO 通过复用 NCCL transport 层继承此能力

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- LongCat-Flash Technical Report
