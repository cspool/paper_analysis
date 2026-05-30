## Frontier Supercomputer (OLCF Frontier 超级计算机)

术语是什么？

Frontier 是美国橡树岭国家实验室（ORNL）的百亿亿次级（Exascale）超级计算机，基于 HPE Cray EX 架构，由 9408 个计算节点组成，每个节点配备 1 个 AMD EPYC CPU 和 4 个 AMD MI250X GPU（共 8 GCD）。采用 Dragonfly 网络拓扑，节点内 GPU 通过 Infinity Fabric 互联（最高 200 GB/s），节点间通过 Slingshot-11 互联（25 GB/s per NIC，每节点 4 NIC）。截至 2025 年，Frontier 是首个公开达到 ExaFLOPs 计算能力的超级计算机。

从硬件架构角度拆解：

Frontier 的层次化拓扑结构与 X-MoE 优化策略的对应：
- **Intra-GPU（GCD 0 ↔ GCD 1）**：Infinity Fabric 200 GB/s → RBD 的 GCD 级 token 共享
- **Intra-Node（同节点不同 MI250X）**：Infinity Fabric 50-100 GB/s → RBD Stage 2 intra-node alltoall
- **Inter-Node（同 Rack ≤256 GCD）**：Slingshot 25 GB/s → RBD Stage 1 inter-node (pilot only)
- **Cross-Rack（>256 GCD）**：alltoall 延迟急剧升高（>10× + outlier >500ms）→ EP 限制 ≤256

X-MoE 在 Frontier 上的规模限制因素：
- 256 GCD 以内：性能稳定扩展
- 256-1024 GCD：跨 rack 通信延迟成为主导瓶颈，出现频繁的 alltoall outlier（>500ms per collective）
- X-MoE 通过限制 EP ≤256 来避免跨 rack alltoall 的性能退化

术语一般如何实现？

X-MoE 通过 AWS-OFI-RCCL plugin 实现节点间通信（映射 RCCL 到 libfabric），使用 libfabric 1.20.1。环境配置：CUDA_DEVICE_MAX_CONNECTIONS=1, NCCL_NET_GDR_LEVEL=3。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
