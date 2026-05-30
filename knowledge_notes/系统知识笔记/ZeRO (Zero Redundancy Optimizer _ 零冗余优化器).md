## ZeRO (Zero Redundancy Optimizer / 零冗余优化器)

术语解释
ZeRO 是 DeepSpeed 提出的内存优化技术系列（ZeRO-1/2/3），通过将 optimizer state、gradients 和 model parameters 分片（shard）到 data parallel group 的各个 GPU 上，消除数据并行中的内存冗余，使数百B参数级模型训练成为可能。

术语是什么？
在标准数据并行中，每个 GPU 持有完整的 optimizer state、gradients 和 parameters 副本，导致严重的内存冗余（N 个 GPU 各有完整副本 = N× 内存）。ZeRO 分三个阶段消除冗余：
- **ZeRO-1 (Pos)**: optimizer state 分片（如 Adam fp32 param + momentum + variance = 12B/param），每 GPU 仅持有 1/N
- **ZeRO-2 (Pos+g)**: optimizer state + gradients 分片
- **ZeRO-3 (Pos+g+p)**: optimizer state + gradients + parameters 分片

在 MoE 训练中，ZeRO-1 通常应用于 expert 的 optimizer state：每个 expert 的 optimizer 在持有该 expert replica 的 EDP group 内均匀分片。

从系统架构角度拆解术语：
SYMI 对 ZeRO-1 的 MoE 应用做了关键修改：
```
# 传统 DeepSpeed ZeRO-1 for MoE:
# Expert e_i 的 optimizer state 仅分片在持有 e_i 实例的 r 个节点上
optimizer_partition[e_i] = shard(optimizer[e_i], r_ways)  # r = replicas per expert

# SYMI 修改后的 ZeRO-1:
# Expert e_i 的 optimizer state 均匀分片到 ALL N 个节点
optimizer_partition[e_i] = shard(optimizer[e_i], N_ways)  # N = total nodes
```
两种分片方式的总内存占用量相同（M = E×O），但 SYMI 的分片使 optimizer placement 完全独立于 expert placement，支持无开销的 expert rebalancing。

术语一般如何实现？如何使用？
- ZeRO 通过 PyTorch FSDP 或 DeepSpeed 引擎实现，训练脚本中配置 ZeRO stage
- 在 MoE 场景，DeepSpeed-MoE 默认对 expert components 使用 ZeRO-1
- SYMI 的关键修改：将 optimizer 分片域从 EDP group（r 个节点）扩展到全体节点（N 个节点），配合 batch point-to-point 通信替代原有 collective
- ZeRO-Offload 进一步将 optimizer state 卸载到 CPU host memory，SYMI 也采用此策略以物理隔离静态/动态组件

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)
- Aria An Open Multimodal Native Mixture-of-Experts Model（ARIA 使用 ZeRO-1 将 MoE decoder 的 optimizer states 分片到 data parallel group；与 expert parallelism 组合后无需 tensor parallelism 即可高效训练 24.9B 模型）
- Continual Pre-training of MoEs How robust is your router（CPT 实验使用 64×A100 GPU + data parallelism + ZeRO-1。未使用 tensor/pipeline/expert parallelism，仅使用纯数据并行 + ZeRO-1 分片 optimizer states。MoE 模型（2B total）的 optimizer step 约 100ms，dense 模型（570M）约 39ms——MoE 因更多参数导致 optimizer 开销更大）

---
