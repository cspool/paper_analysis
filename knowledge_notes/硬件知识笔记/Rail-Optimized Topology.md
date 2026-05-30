## Rail-Optimized Topology

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Rail-Optimized Topology（Rail 优化拓扑）是 NVIDIA 为 DGX SuperPOD/AI 集群推荐的一种 GPU 互连拓扑优化策略。核心思想：将具有相同 intra-server GPU index（rank）的 GPU 连接到同一台 leaf/ToR 交换机。例如，所有 server 的 GPU 0 连接到 leaf switch 0，所有 GPU 1 连接到 leaf switch 1，以此类推。这使得同 rail 内的 GPU 之间可以通过同一 leaf 交换机直接通信（无需经过 spine 交换机），实现超低延迟（等效于单跳交换延迟）。对于 all-reduce 等集体通信操作（按 rank 通信），Rail-optimized 将大量 traffic 限制在 leaf 交换机内部，减少 spine 交换机的带宽压力。NVIDIA NCCL 2.12+ 专门针对 Rail-optimized 拓扑优化了 all-to-all 性能（doubling all-to-all performance）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 MixNet 的仿真和测量中，Rail-optimized 是重要的对比 baseline：
- MixNet 生产测量使用 Certified NVIDIA DGX SuperPOD（128 H800 GPUs, Rail-optimized 拓扑）。
- MixNet 仿真中，Rail-optimized 的性能等同于 Fat-tree（两者在大多数 workload 下性能接近，如图 12a 所示），但 Rail-optimized 的 networking cost 与 Fat-tree 相当。
- MixNet 的 cost-efficiency 优越性 vs Rail-optimized（400 Gbps）：1.9×-2.6×（不同 MoE 模型）。这是因为 Rail-optimized 仍需要完整的 spine/leaf 交换机层，而 MixNet 的 OCS 替代了 spine 层的功能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 部署：DGX SuperPOD 参考架构中，每 DGX server 8 GPU + 8 NIC。8 台 leaf 交换机（每台连接所有 server 的同 rank NIC），spine 交换机连接所有 leaf。总计：k=8 时 8 leaf + 8 spine = 16 交换机。
- NCCL 支持：NCCL 能感知 Rail-optimized 拓扑并通过 NVLSTree 算法优化 collective 通信路径。
- Rail-only 设计（Wang et al. 2024）：进一步激进——完全移除 spine 层，仅保留 leaf 交换机，但代价是跨 rail 通信必须多跳转发（性能下降）。
- 与 MixNet 的关系：MixNet 不是替代 Rail-optimized，而是在其基础上增加 OCS 域。Rail-optimized 的 EPS leaf 交换机仍用于 DP/PP 全局通信。

涉及论文标题：
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

---
