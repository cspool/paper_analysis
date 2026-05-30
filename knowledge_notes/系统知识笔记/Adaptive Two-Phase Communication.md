## Adaptive Two-Phase Communication

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Adaptive Two-Phase Communication 是 JANUS 针对解耦式 MoE 推理中 attention-MoE 跨子集群通信设计的通信机制。其核心思想是用少量的大块数据传输替代大量的小消息传输，通过两阶段聚合适配非对称通信模式（attention 实例数 m ≠ MoE 实例数 n）。

Phase 1 (Intra-Node Aggregation): 同一物理节点上的多个 attention/MoE 实例通过 NVLink collectives (NCCL) 聚合中间激活/结果为更大的 payload。
Phase 2 (Inter-Node Transfer): 聚合后的 payload 通过 GPUDirect RDMA (NVSHMEM one-sided put) 传输到目标节点。

根据资源配置和流量负载自适应选择两种传输模式：
- Case-1 (直接传输): 每个 attention 节点需要向少量 MoE 节点发送数据时，聚合后直接 RDMA 到目标节点
- Case-2 (中继传输): 目标数多或数据量大时，每个 attention 节点发送到指定 MoE relay 节点，relay 节点通过 NVLink multicast 分发给本地 MoE 实例

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
Strawman (O(m×n) pairwise):
  Attention GPU i → MoE GPU j (for all i,j)
  Total transfers: m × n per direction per layer
  Problem: 大量小消息在 critical path 上串行化

JANUS 2PC (O(m + n) bulk transfers):

Case-1 (Direct, 目标数少):
  Attention Node 0 (3 attn instances):
    Phase 1: [A0, A1, A2] --NCCL AllGather--> aggregated payload P0
    Phase 2: P0 --NVSHMEM put--> MoE Node 0, MoE Node 1
  Result: 3×(small) intra-node + 2 (large) inter-node transfers

Case-2 (Relay, 目标数多/数据量大):
  Attention Node 0:
    Phase 1: [A0, A1, A2] --NCCL AllGather--> P0
    Phase 2: P0 --NVSHMEM put--> MoE Relay GPU R0
              R0 --NVLink multicast--> [E0, E1, E2, E3] (local)
  Result: 最少的 inter-node transfers (1 per attention node)

Reverse direction (MoE → Attention):
  Phase 1: MoE instances intra-node NCCL all-reduce → aggregated result
  Phase 2: Aggregated result --NVSHMEM put--> attention nodes
```

通信优化要点：
- Gating 放在 MoE 侧：发送完整 activation（而非 per-expert packed tensors），避免 attention 侧 packing 开销、routing metadata 传输
- NVSHMEM putmem_signal: 打包 layer index + token count 到 64-bit signal value，CPU 侧仅首层 unpack
- Shared Expert 放 attention 侧，在等待 MoE 结果期间计算以 overlap

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现依赖：
- NVSHMEM one-sided putmem_signal/signal_wait (跨节点 GPUDirect RDMA)
- NCCL AllGather/AllReduce (节点内 NVLink collectives)
- NVSHMEM 参数调优: IBGDA transport、request-batching threshold、per-peer RC queue count

适用场景：任何需要非对称、弹性跨 GPU pool 通信的解耦式推理系统。

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
