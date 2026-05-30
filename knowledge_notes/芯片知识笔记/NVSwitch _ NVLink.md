## NVSwitch / NVLink

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVLink 和 NVSwitch 是 NVIDIA 专有的 GPU 互联技术，构成 GPU 集群的 scale-up 高带宽域。NVLink 是 GPU-to-GPU 直连链路——每个 GPU 通过多条 NVLink lane 连接到相邻 GPU，提供远高于 PCIe 的带宽（A100: 600 GB/s, H100: 900 GB/s, B200: 1.8 TB/s per GPU）。NVSwitch 是 NVLink 的交换芯片——将同一 server/node 内的多个 GPU 通过全交叉 crossbar 互联（fully-connected），使任意 GPU 对之间都能实现 NVLink 线速通信，无需经过 PCIe 或 NIC。NVIDIA DGX 系统使用 NVSwitch 实现 8 GPU 全互联（A100/H100），NVIDIA NVL72 系统将 72 GPU 通过 NVSwitch 互联为单一 scale-up domain。NVLink/NVSwitch 是 GPU 集群中带宽最高、延迟最低的通信域，用于 Tensor Parallelism 的 all-gather/reduce-scatter 和 MoE expert 的 intra-node all-to-all 通信。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
NVLink 的芯片级实现：
- 物理层：NVLink 使用差分信号对（differential pair）通过 PCB trace 或 NVLink bridge 连接 GPU die。每 lane 提供 50 GB/s（双向 per lane for NVLink 4.0）。A100 有 12 lanes × 50 = 600 GB/s，H100 有 18 lanes × 50 = 900 GB/s。
- NVSwitch 芯片：集成了多个 NVLink port，内部为 shared memory crossbar switch。A100 的 NVSwitch 有 64 个 NVLink port，每个 port 连接 1 条 GPU NVLink lane。DGX A100 使用 6 个 NVSwitch 芯片实现 8 GPU 的全互联。
- GPU die 集成：NVLink 控制器集成在 GPU die 内部，通过 NVLink I/O 模块直接访问 GPU 的 L2 cache 和 HBM memory（绕过 PCIe root complex），实现 GPU-to-GPU 的 load/store 和 P2P memory access。
- 拓扑：传统 NVSwitch 为 crossbar（每个 GPU 直接连接所有 NVSwitch，每对 GPU 间有独立路径）。NVL72 扩展为两级 NVSwitch（intra-tray 4×NVSwitch + inter-tray 18×NVSwitch），实现 72 GPU 全互联。

在 MixNet 中的角色：
- Scale-up 域：NVSwitch 负责 TP 的 all-gather/reduce-scatter（intra-node，最高带宽）。
- Intra-host data movement：MixNet 的 topology-aware EP routing 在每 server 内部使用 NVSwitch 进行 intra-host gather/scatter/all-to-all（步骤 2,4,5）。
- MixNet 的定位：OCS 作为 "augmented scale-up" 域——将 NVSwitch 的高带宽电路交换概念扩展到跨 server 的 EP group 范围，填补 NVSwitch（intra-node）和 EPS（global）之间的中等规模高带宽域空白。
- 前瞻（§8）：当 co-packaged optics 直接连接到 GPU chip 时，MixNet 的 OCS 与 NVSwitch 可以在 chip-level 联合工作——NVSwitch 提供 intra-tray 高带宽，OCS 提供 inter-tray 可重构高带宽。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- NVLink 代际演进：NVLink 1.0 (P100, 160 GB/s) → 2.0 (V100, 300 GB/s) → 3.0 (A100, 600 GB/s) → 4.0 (H100, 900 GB/s) → 5.0 (B200, 1.8 TB/s)。
- NVSwitch 代际：NVSwitch 1.0 (A100, 64 ports per chip) → NVSwitch 2.0 (H100, 64 ports) → NVSwitch 3.0 (B200/NVL72, 144 ports per tray switch + 18 inter-tray switches)。
- 在 MoE 训练中的 bottleneck：NVSwitch 仅限 intra-node（8 GPU），expert parallelism 的 all-to-all 经常跨 node（特别是大 EP degree 如 64），必须走 scale-out 网络。MixNet 的 OCS 高带宽域正是为了解决 NVSwitch 无法覆盖跨 node EP 通信的问题。
- NCCL 感知：NCCL 能感知 NVSwitch 拓扑并自动选择最优 P2P 通信路径（NVLink 直连 vs NVSwitch 中继）。
- 与 PCIe 的对比：PCIe 5.0 ×16 = 128 GB/s（双向），NVLink 4.0 = 900 GB/s（双向）——约 7× 带宽差距。
- MixServe 中的使用：H20 集群使用 NVLink 4.0（900 GB/s）作为 intra-node 高带宽域，用于 TP group 的 RS/AG 通信和 MoE block 的 intra-node TP 切分。与 inter-node InfiniBand 400 Gbps（≈50 GB/s）形成约 18× 带宽差距，这是 hybrid TP-EP 将 TP 严格限制在 intra-node 的硬件基础。

涉及论文标题：
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm
