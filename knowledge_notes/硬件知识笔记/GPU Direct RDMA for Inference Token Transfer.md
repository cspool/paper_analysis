## GPU Direct RDMA for Inference Token Transfer

术语是什么？

GPU Direct RDMA 是 NVIDIA 提供的允许 GPU 显存数据直接通过 RDMA 网络传输到远程 GPU 或节点的技术，无需经过 CPU 内存做中间拷贝。在 LMM serving 场景下，ModServe 使用 GPU Direct RDMA 实现 Image Instance 和 Text Instance 之间的 image token 传输——image encoder 的输出（GPU memory 中的 image tokens tensor）直接通过 InfiniBand RDMA 被远程 Text Instance 的 GPU pull 过去，绕过 CPU 和 host memory。ModServe 论文测量：使用 InfiniBand RDMA 时 P99 transfer latency = 5ms（<0.5% TTFT for CroAttn, <0.3% for DecOnly），TCP over Ethernet 则为 P50 100ms, P99 180ms。

从硬件架构角度拆解术语：

GPU Direct RDMA 的数据路径（ModServe 场景）：
```
Image Instance GPU (Server A):
  GPU memory → PCIe → NIC (InfiniBand HCA) → InfiniBand fabric
  → NIC (InfiniBand HCA) → PCIe → Text Instance GPU (Server B)
  
  vs 非GPU Direct路径:
  GPU memory → PCIe → CPU DRAM (copy 1)
  → NIC via PCIe (copy 2) → InfiniBand → NIC → CPU DRAM (copy 3)
  → GPU memory via PCIe (copy 4)
```

关键硬件要求：(1) 支持 GPUDirect RDMA 的 GPU（A100/H100 等 Data Center GPU）；(2) InfiniBand HCA（如 Mellanox ConnectX-6/7）；(3) GPU 和 NIC 在同一 PCIe root complex 下（避免跨 CPU socket 的额外延迟）；(4) nvidia-peermem kernel module 加载（使 NIC 可直接访问 GPU BAR 空间）。

术语一般如何实现？如何使用？

ModServe 通过 PyTorch distributed communication + NCCL backend 使用 GPU Direct RDMA。Pull-based 设计：Image Instance 完成 encoding 后，通过 `torch.distributed` 注册 RDMA 地址，Text Instance 的 NCCL backend 发起 RDMA read 从远程 GPU memory 直接 pull tokens。当 Image 和 Text Instance co-locate 在同一 8-GPU server 时，使用 NVLINK 3.0（600 GB/s）而非 RDMA，完全避免网络传输。论文发现 TCP 仍可用（因 image token 体积小——6K-25K tokens × 2 bytes BF16 × d_model 维度 ≈ MB 级），但 InfiniBand RDMA 提供最优延迟。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

---
