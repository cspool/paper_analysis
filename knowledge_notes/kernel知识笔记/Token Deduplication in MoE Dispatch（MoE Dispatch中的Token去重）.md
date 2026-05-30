## Token Deduplication in MoE Dispatch（MoE Dispatch中的Token去重）

术语是什么？
Token deduplication 是 GPU-initiated token-level EP 通信的优化技术：当 MoE router 将同一 token 分配给位于同一节点的多个 experts 时，通信库仅跨网络（RDMA）发送该 token 的 activation 一次，利用节点内高速互联（NVLink/xGMI）转发给同节点内的其他 expert GPUs，避免多次跨节点重复传输相同数据。

从kernel调度角度拆解术语：
```
// 无 dedup: 每个 (token, expert) pair 独立发送 (N=top_k 次 RDMA)
//   token_A → expert_0 (GPU_0, node_0): RDMA
//   token_A → expert_3 (GPU_3, node_0): RDMA ← 同一节点, 浪费带宽
//   token_A → expert_5 (GPU_4, node_1): RDMA

// 有 dedup (DeepEP/UCCL-EP HT mode GPU kernel):
//   GPU kernel 在提交 TransferCmd 前执行:
//     1. 按 dest_node 分组 topk_indices:
//        {node_0: [expert_0, expert_3], node_1: [expert_5]}
//     2. 每个 distinct dest_node 仅提交 1 个 Write TransferCmd:
//        token_A → node_0 (1 RDMA write)
//        token_A → node_1 (1 RDMA write)
//     3. 目标 node 接收后, intra-node forwarding:
//        GPU_0 → GPU_3 via NVLink ring buffer
//   结果: 2 次 RDMA (vs 3 次无 dedup), NVLink 带宽远高于 RDMA
```

术语一般如何实现？如何使用？
DeepEP 和 UCCL-EP 的 HT mode kernel 实现。GPU SM thread 在构造 TransferCmd 前检查 routing table 去重，将同一 token 去往同一节点的多个 expert destinations 合并为一条消息。UCCL-EP 此功能通过 CPU proxy 透明支持：GPU kernel 仍按去重后的策略提交命令，CPU proxy 按标准流程执行 RDMA 即可。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
