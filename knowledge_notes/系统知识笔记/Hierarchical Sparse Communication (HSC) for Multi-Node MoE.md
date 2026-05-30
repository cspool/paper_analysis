## Hierarchical Sparse Communication (HSC) for Multi-Node MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Sparse Communication (HSC) 是 GRACE-MoE 替代 flat global All-to-All 的多节点通信方案。HSC 分两阶段：Stage 1 跨节点路由——所有 GPU 在单一 global communication group 中通过 zero-padded sparse P2P transfers 交换数据，利用 global collective 的 implicit barrier 实现 soft synchronization；Stage 2 节点内重分发——通过 NVLink P2P 将 tokens 分发到 expert-hosting GPU。关键优化：(a) token deduplication——同一 destination node 的多 token 聚合为单次跨节点发送；(b) fine-grained pipelining——跨节点通信与节点内 routing computation 重叠。HSC 单独降低 All-to-All time 35.19%、cross-node traffic 35.64%、GPU idle 49.88%（vs Occult baseline）。

从系统架构角度拆解：

```
Stage 1: Cross-node (global group, logically sparse)
  Each GPU aggregates tokens by dest node → single zero-padded send
  Non-destination ranks receive zero-padded (participate but no payload)
  Implicit barrier: all ranks sync at collective completion
Stage 2: Intra-node redistribution (NVLink, overlapped)
  GPU-to-GPU P2P via NVLink (50 GB/s × 12 links)
  Overlapped w/ Stage 1 routing computation (fine-grained pipelining)

vs Flat All-to-All: deduplication eliminates cross-node duplicate sends
```

HSC 的 deduplication 效果取决于 expert affinity grouping 质量——grouping 越好，同 node 内多 expert 命中越高，dedup 节省越大。

涉及论文标题：
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference
