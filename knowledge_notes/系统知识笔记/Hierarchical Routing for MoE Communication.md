## Hierarchical Routing for MoE Communication

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Routing 是 FUSCO 中 Communication Planner 的核心机制，通过将跨设备数据传输分解为拓扑对齐的两层跳转（topology-aligned hops），利用 GPU 集群的层次化带宽结构（intra-node NVLink 480 GB/s >> inter-node RoCE 50 GB/s）来消除冗余跨节点通信。具体地，对每个 destination node，发送端指定一个 forwarder GPU 作为该节点的唯一跨节点接收端点。如果一个 token 被路由到同一节点的多个 expert，发送端仅通过 inter-node 网络发送一份拷贝给 forwarder，forwarder 再利用 intra-node 高带宽链路分发给各 expert GPU。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
# 传统 NCCL A2A: 每个 token 对每个 expert 独立发送
# Token t₀ routed to E₁(on GPU₂, Node₀) and E₃(on GPU₃, Node₀)
# → 两次跨 Node₀ 传输 (通过 RoCE)，完全相同的数据发两次

# FUSCO Hierarchical Routing (两级 descriptor):
# Level 1 - Node-Level Forwarding:
#   sender_a(Node₁) → forwarder_b(Node₀): 仅发送一份 t₀ 拷贝
#   descriptor: {send: t₀.addr → recv: forwarder_b.buf[t₀.offset]}
#
# Level 2 - Expert-Level Distribution:
#   forwarder_b → GPU₂(E₁): NVLink P2P copy
#   forwarder_b → GPU₃(E₃): NVLink P2P copy
#   descriptor: {send: forwarder_b.buf[t₀.offset] → recv: expert_activations[E₁][t₀.pos]}
```

关键洞察：intra-node NVLink 带宽（480 GB/s）远大于 inter-node RoCE 带宽（50 GB/s），将 top-k fan-out 导致的重复跨节点传输（k 倍）替换为一次跨节点 + k 次节点内传输，极大减少 inter-node 链路竞争。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Communication Planner 根据 token-node 矩阵 B（从 token-expert 矩阵 A 和 expert placement 导出）构建 Node-Level descriptor：对 B[t] 中的每个唯一 node n，仅分配一个 send descriptor → 自动 deduplication
- 基于 token-expert 矩阵 A 构建 Expert-Level descriptor：将 forwarder 上的 token local address 映射到 expert GPU 上 expert activation tensor 的精确偏移
- Forwarder 由 Online Load Balancer 选定（贪心 circular-shift 分组）
- 在 single-node routed 场景（所有 expert 在同一节点），deduplication 效果最显著：FUSCO 比 DeepEP 快 1.95-2.01×

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

HierMoE 的 HierD-AlltoAll 将 FUSCO 的两级分层路由思想推广到 D 维（D≤4），并在每层执行 token 去重。与 FUSCO 需要指定 forwarder GPU 不同，HierD-AlltoAll 利用 NCCL AlltoAll collective 的固有 group 结构（Inter-level-i AlltoAll 自然将 experts 划分为 U[i] 组），无需手动管理 forwarder。此外，HierD-ES 在去重后进一步通过 expert swap 平衡各 group 的负载，解决了 "去重后负载可能更不均衡" 的问题——而 FUSCO 依赖独立的 Online Load Balancer 处理负载均衡。
