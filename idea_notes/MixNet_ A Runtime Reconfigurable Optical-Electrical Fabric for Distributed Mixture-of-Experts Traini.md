## MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

- baseline方法是什么？
  Baseline 为当前 GPU 集群中广泛使用的**静态电气互连架构**——scale-up 域使用 NVSwitch 全交叉（fully-connected crossbar），scale-out 域使用 Clos-style Fat-tree（或 Rail-optimized）拓扑。以典型 1024 GPU cluster（128 servers × 8 GPU）训练 Mixtral 8×22B（EP=8, TP=8, PP=8）为例的全栈执行路径：
  - **算法层（MoE 训练）**：每层 MoE block 包含 attention → gate unit → parallel expert FFN。Gate unit 做 per-token top-K routing 选择激活的 experts。EP 将不同 expert 分配至不同 GPU，每 training iteration 执行 4 次 all-to-all 通信（FP 2 次 dispatch+collect，BP 2 次）。
  - **系统框架层**：Megatron-LM 3D 并行（DP + TP + EP + PP）。DP 做全局 all-reduce 同步梯度，TP 做 intra-server 高带宽通信（NVSwitch），EP 做跨 server all-to-all 通信。
  - **编译框架层**：论文未明确说明（标准 PyTorch + NCCL）。
  - **kernel 调度层**：NCCL collective communication library 处理 all-to-all、all-reduce 等原语。通信拓扑固定，不支持训练过程中动态重配置。
  - **硬件架构层**：intra-server NVSwitch（900 GB/s 或 NVLink），inter-server Fat-tree/Rail-optimized EPS fabric（100G/200G/400G/800G Ethernet 或 InfiniBand）。全网使用 uniform full bisection bandwidth，拓扑在整个训练过程中保持静态不变。
  - **核心缺陷**：MoE 的 EP 通信具有**时间非确定性**（token-specific expert activation 导致通信矩阵在 iterations 间变化）和**空间非均匀性**（sparse all-to-all，仅部分 GPU 对间有大量通信），且存在**强局部性**（只有同一 MoE block 内的 expert 层需要 all-to-all）。但现有 Fat-tree/Rail-optimized 等静态拓扑使用过配的 full bisection bandwidth 来容纳这些变化，导致宝贵的网络带宽大部分时间处于 under-utilized 状态。OCS-based 方案（如 TopoOpt、Google Lightwave Fabrics）仅支持训练前的一次性重配置（one-shot），无法在训练过程中随 traffic pattern 变化动态调整拓扑，因此在 MoE 训练场景下性能显著劣于 Fat-tree。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MixNet 通过**区域可重构高带宽 OCS 域 + 混合光电 fabric + 训练中拓扑重配置**三层次设计解决上述缺陷。全栈执行路径（以 Mixtral 8×22B，1024 GPU cluster，400 Gbps links 为例）：
  - **算法层（MoE 训练不变）**：MixNet 不改变 MoE 的并行策略和训练算法（EP/TP/PP/DP 照旧），仅优化底层数据传输路径。MoE 训练 accuracy 不受影响。
  - **系统框架层 — Custom Collective Communication Runtime**：
    1. **Traffic Demand Characterization（§5.1）**：利用 MoE block 内 4 次 all-to-all 的对称性（2 次 FP + 2 次 BP 的 traffic matrix 相同或转置），配合 MixNet-Copilot 预测算法（SLSQP 估计条件概率转移矩阵），提前预测 traffic pattern 以支持主动重配置。预测准确度（Top-K accuracy）显著高于 random/uniform 方案。
    2. **Topology Reconfiguration（§5.2）**：Greedy Algorithm 1 迭代识别 bottleneck server pairs（完成时间最长），优先为这些 pairs 分配直接 OCS 电路。在 OCS NIC 端口用尽后停止。重配置是去中心化的——各 region 独立运行 topology controller，无需全局控制面。
    3. **Topology-Aware EP Routing（§5.3）**：5 步流程——(1) topology lookup 确定 delegation GPU → (2) intra-host gather via NVSwitch → (3) inter-host all-to-all via OCS 直连（优先）+ EPS fallback → (4) intra-host all-to-all via NVSwitch → (5) intra-host scatter。步骤 (3) 和 (4) overlap 执行。
    4. **DP Hierarchical All-Reduce**：intra-host NVSwitch reduction → inter-host EPS ring all-reduce → intra-host NVSwitch broadcast。多 EPS NIC 时 multi-ring 并行。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层 — RDMA-based Data Transfer**：
    - 自定义 collective communication runtime（C++ ~6K LoC）基于 FuseLink raw ibverbs library 实现 RDMA 高速传输。
    - 通信原语暴露为 Python 接口（mixnet.all_to_all, mixnet.all_reduce），集成入 Megatron-LM。
    - EPS 通信复用 NCCL 高性能原语，OCS 通信走 RDMA over RoCEv2。
  - **硬件架构层 — Regionally Reconfigurable OCS + EPS Hybrid**：
    1. **架构设计思想（§4）**：利用 MoE 通信的强局部性（同一 MoE block 内的 expert 才需要 all-to-all），将 OCS 划分为多个隔离的 region（而非 global OCS），各 region 独立处理局部 traffic。突破 OCS 技术中 reconfiguration speed 与 port count 的根本 trade-off——毫秒级 OCS（如 Polatis, MEMS）port count 仅数百，但每个 EP group 最多 64-128 GPU，足以被单一 OCS 覆盖。
    2. **每 server NIC 分配**：2 NIC → EPS Fat-tree（处理 DP、PP 全局通信），6 NIC → OCS（处理 EP 局部 all-to-all）。OCS 仅需连接同一 EP group 内的 GPU（最大 64 GPU），可由 500-port 级 commodity OCS 轻松支持。
    3. **重配置时机**：FP 第一个 all-to-all 阻塞网络等待 OCS 重配置（25ms），后续 BP 的 2 个 all-to-all 在 attention/expert computation 期间隐藏重配置延迟。总重配置次数：每 MoE layer 2 次（FP 一次 + BP 一次）。
  - 对比 baseline 的改进映射：
    - **静态 full bisection bandwidth → 按需分配 OCS 直连电路**：Fat-tree 对全部 GPU 对提供均等带宽，但 MoE 的 EP 通信是 sparse 且动态的 → MixNet 的 greedy algorithm 识别 traffic-intensive GPU 对并分配专用 OCS 电路，其他 pairs 走 EPS fallback。用更少的网络硬件（更低的 cost）达到与 Fat-tree 相当的性能。1024 GPU + 400 Gbps 下 cost-efficiency 提升 1.9×-2.3× vs Fat-tree。
    - **训练前一次性重配置（TopoOpt/Google Lightwave）→ 训练中动态重配置**：TopoOpt 假设 traffic pattern 在训练全程不变 → MixNet 每 iteration 根据实时 traffic demand 调整拓扑。仿真结果：MixNet 比 TopoOpt 快 1.3×-1.5×（因 TopoOpt 的静态拓扑无法适应 MoE 的动态 all-to-all）。
    - **Global OCS 的 scalability-reconfiguration trade-off → 区域可重构**：Global OCS 需要上千端口（scalability）但重配置慢（分钟级）→ MixNet 的 regional design 将 OCS 限制在 EP group 范围（<128 GPU），使毫秒级重配置成为可能。仿真验证 MixNet 可扩展至 32768 GPU（4096 servers），通过多个 region 并行工作。
    - **NVSwitch/NVLink 仅限 intra-server → OCS 扩展 scale-up 域边界**：前瞻分析（§8）显示，当 OCS 通过 co-packaged optics 直接连接到 GPU chip 时，MixNet 将 scale-up 域从 NVL72 的 72 GPU 扩展到整个 EP group，训练 DeepSeek-V3 时比 NVL72 快 1.3×。
