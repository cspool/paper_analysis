## All-to-All Communication in MoE Training（MoE训练中的全交换通信）

术语是什么？
All-to-All Communication 是 MoE 分布式训练中 expert parallelism 的核心通信原语。在 expert parallelism 中，不同 expert 分布在不同 accelerator 上，每个 accelerator 上的 token 需要被发送到持有对应 expert 的 accelerator 处理。这产生两个方向的 All-to-All：(1) Dispatching：将 token 从 token 所在 accelerator 发送到 expert 所在 accelerator；(2) Combining：将 expert 处理后的 token 结果从 expert 所在 accelerator 发送回 token 原始所在 accelerator。EfficientMoE 论文通过 profiling 发现，在 32 Ascend 910 集群上，纯通信时间占 MoE 训练总时间的 75%，严重浪费 accelerator 计算资源。

从系统架构角度拆解术语：
MoE 训练中一个 iteration 的 All-to-All 通信流程：

1. **Token Grouping**：每个 accelerator 根据 gate 输出的路由结果，将本地 token 按目标 expert 所在的 accelerator 分组。
2. **All-to-All Dispatch (Scatter)**：各 accelerator 同时向其他所有 accelerator 发送 token，节点间通过 RoCE (Ascend 910: 100 GB/s) 或 InfiniBand 传输。
3. **Expert Compute**：各 accelerator 在收到的 token 上执行对应 expert 的 FFN (SwiGLU MLP)。
4. **All-to-All Combine (Gather)**：各 accelerator 将 expert 输出按 token 原始 accelerator 分组，反向 All-to-All 送回。
5. **Token Reorder**：各 accelerator 按原始 token 顺序重新排列返回的 expert 输出。

EfficientMoE 的优化策略：通过 hot expert replica 调度，将原本需跨节点的 token All-to-All 通信转化为本地 replica 处理，仅同步 replica 参数更新（volume 远小于 token 传输量），使通信时间减少约 12%。

术语一般如何实现？如何使用？
All-to-All 在主流框架中的实现：NCCL `ncclAllToAll`（NVIDIA GPU）、HCCL `hcclAllToAll`（Ascend）、MPI `MPI_Alltoallv`（变长 All-to-All，用于非均匀 token 分布）。优化方向：通信计算重叠（Lina, Janus）、数据流重定向（Janus: All-to-All→AllReduce）、层级通信（DeepSpeed-MoE: 节点内高带宽+节点间压缩）、参数化替代（EfficientMoE: 用 expert 参数同步替代 token 传输）。

涉及论文标题：
- EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
- Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

Pro-Prophet (NUDT) 使用 Tutel 的高效 P2P 实现 A2A 通信，并基于 performance model 将其建模为 T_A2A(R) = max_i(R_i * size(input)) / B_bar。Pro-Prophet scheduler 通过将 Plan 原语（placement search）调度到前迭代的 A2A 通信中执行，使得 A2A 通信期间同时完成未来迭代的 load balancing 策略搜索。
