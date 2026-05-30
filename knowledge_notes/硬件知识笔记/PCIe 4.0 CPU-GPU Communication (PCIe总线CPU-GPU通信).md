## PCIe 4.0 CPU-GPU Communication (PCIe总线CPU-GPU通信)

术语是什么？
PCIe (Peripheral Component Interconnect Express) 4.0 是 CPU 与 GPU 之间的标准数据传输总线。在 ES-MoE 中，PCIe 4.0 承载 expert 参数的 CPU↔GPU 双向传输——训练 forward pass 中从 CPU 上传 experts（CPU→GPU），backward pass 中下载 gradients（GPU→CPU）。PCIe 4.0 x16 单向理论带宽约 32 GB/s，远低于 NVLink (600 GB/s) 但通过 pipelined overlap（在 token permutation/next expert compute 期间传输）和 GPU 计算隐藏。ES-MoE 的关键设计——将 permutation window 用于上传第一个 expert——正是基于 PCIe 带宽与 NVLink 带宽的设计权衡。

从硬件架构角度拆解：
ES-MoE 硬件平台的通信路径：

```
┌──────────┐         ┌──────────┐         ┌──────────┐
│  GPU 0   │←──NV──→│  GPU 1   │←──NV──→│  GPU 2   │  ...
│ A100 40G │ 600GB/s │ A100 40G │ 600GB/s │ A100 40G │
└────┬─────┘         └────┬─────┘         └────┬─────┘
     │ PCIe 4.0 x16       │ PCIe 4.0 x16       │ PCIe 4.0 x16
     │ ~32 GB/s           │ ~32 GB/s           │ ~32 GB/s
     └──────────┬──────────┴──────────┬─────────┘
                │                     │
           ┌────┴─────────────────────┴────┐
           │     CPU + Host Memory          │
           │ AMD EPYC 7543 + 512 GB DDR4    │
           │ Expert Params + Opt States     │
           └────────────────────────────────┘
```

Expert 传输的带宽需求分析：
- 单个 MoE-L expert (d_model=1536, d_ff=6144): ~75M 参数 × 2 bytes (fp16) ≈ 150 MB
- PCIe 4.0 x16 传输时间: 150 MB / 32 GB/s ≈ 4.7 ms
- Expert FFN 计算时间（约 500 tokens）: ~2-10 ms
- Overlap 可行性: 当 TC > TU 时传输被完全隐藏

术语一般如何实现？如何使用？
- **Pinned Memory**: DMA 传输要求源/目标内存在物理上连续且 pin 在 RAM 中（不可换出），通过 `cudaHostAlloc()` 或 `cudaMallocHost()` 分配
- **PCIe 代际**: 3.0 (~16 GB/s) → 4.0 (~32 GB/s) → 5.0 (~64 GB/s) → 6.0 (~128 GB/s, 2024 spec)，带宽每代翻倍
- **CXL 替代方案**: Compute Express Link (CXL) 提供 cache-coherent 的 CPU-GPU 内存共享，可能进一步降低 offload 延迟
- 局限：PCIe 带宽在 MoE expert offloading 中是主要瓶颈——当 expert compute time 很短（少量 tokens）时，传输无法被完全隐藏

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
