## All-to-All Communication in MoE Training

术语是什么？
All-to-All 是 MoE 训练中 Expert Parallelism 的核心通信原语。在 MoE 的每一层，Router（门控网络）为每个 token 分配目标 expert 后，需要通过 All-to-All 将 tokens 从当前设备 dispatch 到拥有对应 expert 的设备，并在 expert 计算完成后通过第二次 All-to-All 将处理后的 tokens collect 回原设备。这一过程涉及两个 All-to-All 阶段：(1) Dispatch：将 tokens 按路由决策分散到各 GPU（T_I → T_DI）；(2) Combine：将 expert 输出收集回原设备（T_DO → T_O）。NCCL 没有原生 All-to-All 原语，PyTorch 通过 `torch.distributed.all_to_all_single` 配合 `input_split_sizes`/`output_split_sizes` 参数实现，底层由 NCCL 的 P2P send/recv 组合实现。

从kernel调度角度拆解术语：
MPMoE 中的 All-to-All 沿 batch 维度切分为多个 micro All-to-All（每个 micro-batch 一次独立 All-to-All），而非 FasterMoE 的沿 node 维度切分（降级为多组 P2P 通信）。这两种方式的区别：

```
// FasterMoE方式: 按 device 维度切分 (Figure 5a)
将 N 个 devices 分为 m 个 groups
for each group g (size G):
    for each partition p in {1..m}:
        在 group g 内部执行 P2P dispatch/recover
// 问题: (m-1) 次 NCCL group calls，退化为 P2P

// MPMoE方式: 按 batch 维度切分 (Figure 5b)
将 B 个 tokens 切分为 n 个 micro-batch，每个大小 B/n
for each micro-batch i in {1..n}:
    全局 All-to-All dispatch(T_I[i])  // 保留 NCCL 优化
    全局 All-to-All collect(T_DO[i])
// pipeline 交替调度 S 和 R stage 以增强内存局部性
```

MPMoE 方式的优势：(1) 保留 NCCL 对 All-to-All 的 ring/tree topology 聚合优化；(2) pipeline granularity n 不受 device 数 N 限制（batch size B >> N）；(3) 异构带宽下不会因同步等待浪费资源。

术语一般如何实现？如何使用？
- PyTorch 实现：`torch.distributed.all_to_all_single(output, input, output_split_sizes, input_split_sizes)`
- 底层 NCCL 实现路径：基于 `ncclSend`/`ncclRecv` 的非对称 P2P 通信，通过预设的 send/recv counts 协调。对于 MoE 的不均匀 token 分布，需先交换 metadata（各 rank 的 send/recv counts），再执行实际数据传输。
- 性能关键点：(a) 消息大小：token 数 × hidden_dim × sizeof(fp16)，小消息时 latency-bound，大消息时 bandwidth-bound；(b) 网络拓扑：NVLink 节点内延迟 ~10μs，InfiniBand 跨节点延迟 ~1-2μs + 带宽共享；(c) 不均匀 token 分布导致某些 link 成为 bottleneck。
- MPMoE 的通信效率验证：micro-benchmark（Figure 13）显示 MPMoE 的 dispatch/recovery 时间明显低于 FasterMoE（因避免 P2P 拆解的 kernel launch 开销和组同步等待）。

MixNet 从通信拓扑角度揭示了 EP all-to-all 的三个关键动态特性（基于生产环境测量）：
- **时间非确定性**：每个 training iteration 中 token routing 的结果不同，导致 all-to-all 通信矩阵在 iterations 间变化。即使 load balancing loss 使 overall volume 收敛，traffic matrix 的 sparsity 仍然持续。
- **空间非均匀性（sparse all-to-all）**：每个 traffic matrix 中仅有少数 GPU 对之间有大量通信，大部分 pairs 之间流量很小或为零。这种稀疏性源自 MoE 的 sparse activation（每个 token 仅激活 top-k expert）。
- **强局部性**：仅同一 MoE block 内的 expert 层需要 all-to-all，不同 PP stage 的 expert 层不直接通信。

基于这些特性，MixNet 设计了 topology-aware EP routing：优先将通信密集型 GPU 对通过 OCS 直连电路传输（专用高带宽、无排队），其余 pairs 走 EPS fallback。路由依赖 5 步流程：(1) topology lookup 确定 delegation GPU → (2) intra-host gather via NVSwitch → (3) inter-host all-to-all via OCS（优先）或 EPS → (4) intra-host all-to-all via NVSwitch → (5) intra-host scatter。步骤 (3) 和 (4) 通过 CUDA stream overlap 并行执行以减少总完成时间。OCS 重配置利用 all-to-all 通信之间的 computation phase（>100ms）隐藏延迟（~25ms）。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

---
