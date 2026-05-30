## NVLink (GPU-to-GPU Interconnect / 高速GPU直连)

术语是什么？
NVLink 是 NVIDIA 开发的高带宽 GPU 间直连互连技术，允许多个 GPU 以远高于 PCIe 的带宽直接通信，无需经过 CPU/host memory。在 MoE 训练中（如 ES-MoE 使用 4× A100 配置），NVLink 提供 600 GB/s 的 GPU-to-GPU 双向带宽，用于 token permutation 阶段的 all-to-all token 交换——这是 expert parallelism 中通信密集但计算前的关键操作。NVLink 的带宽直接决定了 token 交换的延迟和 MoE layer 的通信效率。

从硬件架构角度拆解：
ES-MoE 配置中 NVLink 的使用：
- **硬件**: 4× NVIDIA A100 40GB，通过 NVLink 互联，双向带宽 600 GB/s
- **通信模式**: All-to-All token scatter + gather：每个 GPU 的 tokens 按 gating 决策发送到持有对应 expert 的 GPU，计算完成后结果返回原位
- **与 PCIe 的关系**: NVLink 处理 GPU-GPU token 交换（高带宽、低延迟），PCIe 4.0 处理 CPU↔GPU expert 参数传输（带宽较低但通过 overlap 隐藏）。两者并行工作——NVLink all-to-all 期间，PCIe 同步上传第一个 expert

NVIDIA A100 的 NVLink 拓扑：
```
A100 SXM4 (4-way NVLink):
  GPU0 ←→ GPU1 (600 GB/s bi-directional)
  GPU0 ←→ GPU2 (600 GB/s bi-directional, via NVSwitch)
  GPU0 ←→ GPU3 (600 GB/s bi-directional, via NVSwitch)
  ...

All-to-All Bandwidth (4 GPUs):
  每个 GPU 同时与其他 3 个 GPU 通信
  聚合带宽 = 3 × 600 = 1800 GB/s (bi-directional, full-duplex)
```

术语一般如何实现？如何使用？
- **多 GPU 服务器**: A100 (NVLink 3.0, 600 GB/s), H100 (NVLink 4.0, 900 GB/s), B200 (NVLink 5.0, 1800 GB/s)
- **编程接口**: NCCL (NVIDIA Collective Communications Library) 自动利用 NVLink 拓扑优化 all-to-all/all-reduce
- **MoE 关键依赖**: Expert parallelism 的 token exchange 严重依赖 NVLink 带宽；若无 NVLink（消费级 GPU 如 RTX 4090），token 交换需通过 PCIe + host memory 中转，带宽大幅下降
- 与 NVSwitch 的关系：NVSwitch 是 NVLink 的交换芯片，实现 GPU 间全互联（非 ring/hierarchical），消除带宽瓶颈

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
