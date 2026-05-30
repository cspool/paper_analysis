## Hierarchical All-to-All (1DH-A2A / 2DH-A2A)

术语是什么？
Hierarchical All-to-All 是一类利用 GPU 集群异构拓扑（intra-node 高带宽 vs inter-node 低带宽）的 A2A 集合通信算法。基本思想是将全局 A2A 分解为两阶段：第一阶段在节点内执行局部 A2A（利用高带宽 intra-node 连接），第二阶段在节点间执行全局通信（仅传输必要数据到跨节点 GPU），从而减少跨节点通信轮次和利用快速 intra-node 连接。

- **1DH-A2A (1D-Hierarchical A2A)**：由 HetuMoE [31] 提出。Phase 1: intra-node A2A（每个节点内各 GPU 交换数据）；Phase 2: inter-node A2A（节点间交换跨节点目标的数据）。减少的通信轮次：相比完全扁平化的 P×(P-1) 个 Send/Recv 对，1DH-A2A 将跨节点通信轮次从 P×(P-M) 降至更少的层次化轮次。
- **2DH-A2A (2D-Hierarchical A2A)**：由 DeepSpeed-MoE [36] 和 Tutel [16] 提出。在 1DH 基础上进一步优化：Phase 1: local token permutation（节点内按 expert 重排）；Phase 2: local expert computation of shared experts；Phase 3: global cross-node exchange of data for remote experts。2DH-A2A 比 1DH-A2A 更细致地利用 intra-node 带宽。

从kernel调度角度拆解术语：
以 8 GPU (2 node × 4 GPU) 和 2DH-A2A 为例：

```
// 2DH-A2A 在 MoE dispatch 中的执行流程
// 输入: 每个 GPU 持有 B×L tokens，gating 确定每个 token 的 target expert

// Phase 1: Intra-node scatter (node 内)
// GPU 0 持有 tokens for experts on GPU 0,1,2,3,4,5,6,7
// 在 node 0 内部 (GPU 0-3):
for gpu in [0,1,2,3]:
    // PCIe/NVLink 高速传输: GPU i → GPU j (i,j 同节点)
    send_tokens_to_intra_node_target(gpu)

// Phase 2: Inter-node exchange
// 每个 node 将所有跨节点的 token 聚合到一个代表 GPU
// node 0 的 GPU 0 持有所有需要发往 node 1 的 token
// node 0 → node 1: InfiniBand 传输 (仅一次跨节点 A2A per node pair)
ncclGroupStart()
for node_pair in cross_node_pairs:
    ncclSend(node_rep_gpu, tokens_for_other_node, ...)
ncclGroupEnd()

// Phase 3: Intra-node gather (node 内)
// 各 node 内将收到的跨节点 token 分发到目标 GPU
// 再次利用高速 intra-node 连接
```

但 ScheMoE 指出，1DH-A2A 和 2DH-A2A 的共同局限是 Phase 1 和 Phase 2 必须顺序执行——intra-node 和 inter-node 带宽无法同时利用。Pipe-A2A 通过双 stream 并发执行消除了这一瓶颈。

术语一般如何实现？如何使用？
1DH-A2A 在 Hetu 框架（https://github.com/Hsword/Hetu）中实现，2DH-A2A 在 Tutel（https://github.com/microsoft/tutel）和 DeepSpeed-MoE 中实现。ScheMoE 的 AbsAlltoAll 抽象接口同时支持这些算法作为可插拔实现，用户可在初始化时选择 NCCL-A2A、1DH-A2A、2DH-A2A 或 Pipe-A2A。

涉及论文标题：
- ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling
- HetuMoE: An Efficient Trillion-scale Mixture-of-Expert Distributed Training System
- Tutel: Adaptive Mixture-of-Experts at Scale
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
