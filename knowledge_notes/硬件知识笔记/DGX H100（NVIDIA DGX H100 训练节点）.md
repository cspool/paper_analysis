## DGX H100（NVIDIA DGX H100 训练节点）

术语是什么？
DGX H100 是 NVIDIA 的 8-GPU AI 训练服务器，每节点配备 8 个 NVIDIA H100 GPU（各 80 GB HBM3，峰值 BF16 989.5 TFLOPS），通过 NVSwitch 和 NVLink 4th Gen 实现全互联，通过 InfiniBand/NVLink Network 实现多节点扩展。MoE Parallel Folding 论文的所有实验均在 DGX H100 组成的 Eos 集群上进行。

从硬件架构角度拆解术语：
DGX H100 节点的关键硬件规格：
- 8 × H100 SXM5 GPU（各 80 GB HBM3，峰值 BF16: 989.5 TFLOPS）
- 2 × Intel Sapphire Rapids CPU（各 56 核）
- NVLink 4th Gen：GPU-to-GPU 单向带宽 450 GB/s（通过 NVSwitch 全互联）
- InfiniBand NDR400：每节点 8 个 ConnectX-7 NIC，每 NIC 400 Gbps
- 节点内总聚合 NVLink 带宽：7200 GB/s（双向）

在 MoE Parallel Folding 中的作用：
- 节点内 8 GPU = EP group 最大有效大小（EP × TP × CP ≤ 8 时通信在 NVLink 内）
- 论文实验：128 GPU = 16 DGX H100 节点，256 GPU = 32 节点，最多 1024 GPU = 128 节点

术语一般如何实现？如何使用？
- NVIDIA Eos 集群由 576 个 DGX H100 节点组成（共 4608 H100 GPU）
- 节点内通信走 NVLink/NVSwitch，节点间走 InfiniBand
- MoE 训练中，通过 Megatron-Core 配置并行策略使 EP 通信尽量在节点内完成

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
