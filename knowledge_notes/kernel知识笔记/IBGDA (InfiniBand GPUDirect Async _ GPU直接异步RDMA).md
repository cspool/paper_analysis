## IBGDA (InfiniBand GPUDirect Async / GPU直接异步RDMA)

术语是什么？
IBGDA（InfiniBand GPUDirect Async）是 NVIDIA 提供的一种技术，允许 GPU threads 直接向 InfiniBand RDMA NIC 提交网络操作（RDMA write/send/atomic），完全绕过 CPU。GPU SM 通过写入 NIC 的 MMIO doorbell/register 接口直接将 work requests 提交到 NIC 硬件队列，NIC 从 GPU memory 直接 DMA 读取数据并通过网络发送。

从kernel调度角度拆解术语：
IBGDA 使 GPU kernel 可以直接发起和管理网络传输，无需 CPU 参与：
```
// 传统 CPU-initiated RDMA 路径:
// GPU compute → cudaMemcpy(→CPU) → CPU post WR → NIC DMA → network
// 延迟: GPU compute + PCIe read + CPU post + NIC DMA

// IBGDA GPU-initiated RDMA 路径:
// GPU kernel 直接写 NIC MMIO doorbell → NIC DMA from GPU memory → network
// 延迟: GPU compute + NIC DMA (消除 CPU 和一次 PCIe 穿越)

// 在 DeepEP 中的使用:
// GPU SM thread:
//   1. 构造 RDMA work request (addr, length, dest_qp, ...)
//   2. 写入 NIC doorbell register (MMIO write)
//   3. NIC 从 GPU memory 直接读取 token activation data
//   4. NIC 打包 RDMA 包并发送
```

术语一般如何实现？如何使用？
IBGDA 要求：(a) InfiniBand-capable NIC（如 NVIDIA ConnectX-7）；(b) GPU 驱动暴露 NIC MMIO 接口（BAR1 mapping）；(c) GPU 和 NIC 之间基于 PCIe 的直接通信路径。NVIDIA 通过 NVSHMEM 库提供 IBGDA 接口。**核心可移植性问题**：IBGDA 要求 GPU 直接操作 NIC 的特定 MMIO 寄存器，每一个 (GPU vendor, NIC vendor) 组合都需要独立编写和维护集成代码。假设 m 种 GPU、n 种 NIC，需 O(m×n) 开发工作量。DeepEP 官方仅支持 NVIDIA GPU + NVIDIA NICs 组合，无法在 AWS EFA 或 Broadcom NIC 上运行。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
