## Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Hecate 在 kernel 调度层面的核心实现：
    1. **Sparse Collective Communication Scheduling**：两个新稀疏通信原语 SparseAllGather 和 SparseReduceScatter 的 NCCL 实现与调度。利用 NCCL group calls 同时调度一系列 Broadcast（SparseAllGather）和 Reduce（SparseReduceScatter）操作，每个 Broadcast/Reduce 操作对应一个 chunk（即一个 expert）到一组 target devices 的通信。
    2. **Communication-Computation Overlap Scheduling**：将 SparseAllGather 和 SparseReduceScatter 的通信与 Attention layer 的前向/后向计算重叠。Forward pass 中，SparseAllGather 的延迟 ≤ Attention forward 时间即可完全隐藏；Backward pass 中，SparseReduceScatter（当前层梯度 reduce）+ SparseAllGather（下一层 re-materialize）同时重叠于 Attention backward（后者耗时约为 forward 2×）。
    3. **Topology-Aware Dispatching**：Dispatcher 在 token dispatching（All-to-All）阶段优先 intra-node 通信，仅当 source node 内无 expert replica 时才跨 node dispatching。均匀分配 tokens 到同一 expert 的多个 replica devices。
    4. **Calibration Stage**：在 MoE gate 输出真实 token assignment 后，重新运行 Algorithm 1，用实际负载和剩余 memory capacity 判断是否追加一次 on-critical-path 的 SparseAllGather 来进一步减少 imbalance。
    5. **Re-materialization Scheduling**：Hecate-RM 在 backward pass 释放已用 expert 参数后，重新调度 SparseAllGather 来物化下一层的 expert 参数，形成 "release → re-materialize" 的流水线。
  - 实验比较：(1) FSSDP vs EP 的 layer-wise speedup（2.8-18.8×，geo-mean 11.87×）；(2) All-to-All 通信时间对比（Hecate 减少 12.3× vs EP）；(3) 各系统 critical path 分解（FasterMoE 的 FusedKernel、FlexMoE 的 Rearr overhead、Hecate 的 SpAG/SpRS overhead）；(4) Hecate-RM re-materialization overhead（3.6× 增加 sparse collectives 通信但仍优于 baseline 1.4×）。

- 后端平台是什么，配置是什么。
  - **Cluster A**：4× AWS p3dn.24xlarge nodes，每 node 8× NVIDIA V100-32G GPU（NVLink 300 GB/s intra-node），node 间 100 Gbps 网络。
  - **Cluster B**：4× AWS p4d.24xlarge nodes，每 node 8× NVIDIA A100-40G GPU（NVSwitch 600 GB/s intra-node），node 间 400 Gbps 网络。
  - 在网络带宽较低的 Cluster A (100 Gbps) 上 All-to-All straggler 效应更显著，Hecate 加速效果更明显。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **PyTorch** + **NCCL** 实现。使用 **Megatron-LM** 作为训练框架，baseline systems 仅优化 MoE layer 训练。
  - 修改的内容：
    - 在 NCCL 之上实现了 SparseAllGather 和 SparseReduceScatter 两个稀疏通信原语，通过 `ncclGroupStart/End` 包装一组 Broadcast（spAG）或 Reduce（spRS）操作。
    - 实现 Communicator 组件：维护通信任务队列，调度执行稀疏 collectives 和 token dispatching All-to-All。
    - 实现 Scheduler 组件：基于 expert load 分布估计（滑动窗口平均，w=5）和 overlap degree / memory capacity 约束，生成 placement plan 并驱动稀疏 collectives 调度。
    - 实现 Dispatcher 组件：拓扑感知的 token 路由决策。
    - Hecate 不实现 expert execution 与 All-to-All 的 overlap（认为正交），稀疏 collectives 仅与 Attention computation 重叠。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未公开 Hecate 代码。作为 prototype system，稀疏 collectives 用 NCCL group calls 实现，作者指出更高效的稀疏 collective 算法（利用数据稀疏性和网络拓扑）留作 future work。
  - SparseAllGather 的 NCCL 实现原理与执行全过程：

```
┌── SparseAllGather 执行模型 ──────────────────────────────┐
│ 逻辑输入：expert parameters 划分为 equal-sized chunks     │
│ C = {C_0, C_1, ...}（每个 chunk = 一个 expert 的参数）   │
│ Pre-condition P_0: 每个 chunk 唯一归属于某 source device │
│ Post-condition P_1: P_0 ⊆ P_1（物化目标 placement）      │
│                                                           │
│ NCCL 实现：                                               │
│   ncclGroupStart()                                        │
│   for each (c, d_target) in P_1 \ P_0:                   │
│       // 对每个需要物化的 (expert, target_device) 对      │
│       d_src = 唯一持有 chunk c 的 device (from P_0)       │
│       ncclBroadcast(chunk_c_data,                        │
│                      root=d_src,                          │
│                      comm=sub_group_containing_d_target)  │
│   ncclGroupEnd()                                          │
│                                                           │
│ 通信量分析：                                              │
│ - expert 参数大小 = expert_size bytes                     │
│ - 需物化的 expert set Ĉ（|Ĉ| ≤ |C|）                      │
│ - 稀疏度 λ = |Ĉ| / |C|                                   │
│ - 每个物化 expert 以 Broadcast 发送到 target devices     │
│ - 最坏情况：某 device 需接收所有 Ĉ 中的 chunks            │
│ - 通信量上界：O(λ · S)，其中 S = |C| × expert_size      │
│ - 相比 FSDP AllGather 的 O(S)，当 λ << 1 时显著降低     │
└───────────────────────────────────────────────────────────┘

┌── SparseReduceScatter 执行模型 ──────────────────────────┐
│ Pre-condition P_0: gradients 分布在多个 device 上        │
│ Post-condition P_1: 每个 chunk 的 reduce 结果在唯一      │
│                      source device (P_1 surjective)      │
│                                                           │
│ NCCL 实现：                                               │
│   ncclGroupStart()                                        │
│   for each (c, d_src) in P_1:                            │
│       // 对每个需 reduce 到 source 的 chunk              │
│       ncclReduce(chunk_c_grad_data,                      │
│                   root=d_src,                             │
│                   comm=sub_group_with_replica_of_c)       │
│   ncclGroupEnd()                                          │
│                                                           │
│ 与 spAG 对称：spRS(P', P) 的通信量上界 = O(λS)          │
│ 与 rearrangement 系统 AllReduce 等价：                   │
│   Vol(AllReduces) ≈ Σ_i 2(|D_i|-1)/|D_i| · S/|C|        │
│   ≈ O(2λS) ≈ Vol(spAG) + Vol(spRS)                      │
└───────────────────────────────────────────────────────────┘

┌── 通信-计算重叠调度时序 ─────────────────────────────────┐
│ Forward:                                                  │
│   [Attention Forward]                                     │
│   ├── SparseAllGather (overlap with Attn Fwd) ──┤        │
│   [MoE Gate + Token Dispatch + Expert Comp]               │
│                                                           │
│ Backward:                                                 │
│   [Attention Backward]  ← 耗时约 2× Forward               │
│   ├── SparseReduceScatter (layer l gradients) ──┤        │
│   ├── SparseAllGather (layer l+1 re-materialize) ┤        │
│   [MoE Gate Bwd + Expert Bwd]                             │
│                                                           │
│ 约束条件：                                                │
│ - t = T_non-MoE · bw / expert_size                        │
│   (overlap degree: 可在 attention 时间内隐藏通信的        │
│    最大 expert 数)                                        │
│ - 拓扑感知：bw 异构时使用 inter-node bandwidth            │
│             同构时使用 uniform inter-device bandwidth     │
└───────────────────────────────────────────────────────────┘
```
