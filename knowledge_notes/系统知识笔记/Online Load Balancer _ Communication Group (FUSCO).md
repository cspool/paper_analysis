## Online Load Balancer / Communication Group (FUSCO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FUSCO 的 Online Load Balancer 通过构建 Communication Group 来解决专家级负载不均衡导致的跨节点通信热点问题。Communication Group 是每节点选一个 GPU 组成的组，组内 GPU 互为 forwarding endpoints。Balancer 的目标是在给定各节点各 GPU 的跨节点流量负载 L 的情况下，通过分区 GPU 使各组的最大 load 最小化（combinatorial max-min optimization），从而避免某个 GPU 因承担过多 forwarder 角色而成为网络瓶颈。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
# Algorithm 1: Greedy Group-Balanced Assignment
# Input: 每 GPU 的跨节点流量 load L[n][g] (n: node index, g: GPU index)
# Output: 分组 assignments G

for each node n:
    P[n] = sort_local_gpus_by_descending_load(L[n])  # O(M*log(M))
for each node n:
    S[n] = circular_shift(P[n], shift=n)  # 按 node index 循环移位
for group_id = 0 to M-1:                   # M = GPUs_per_node
    for each node n:
        G[group_id].add(S[n][group_id])    # 每节点取一位

# 效果：每节点最高负载的 GPU 因不同移位落到不同 group
#   高负载 GPU 分布到不同 group，利用独立物理 channel 并行执行
```

例如 4 节点 × 4 GPU：
- Node₀ 负载: [G₀:0.9, G₁:0.6, G₂:0.3, G₃:0.1] (desc)
- Shift₀: [G₀, G₁, G₂, G₃]; Shift₁: [G₁, G₂, G₃, G₀]; Shift₂: [G₂, G₃, G₀, G₁]; Shift₃: [G₃, G₀, G₁, G₂]
- Group₀: {Node₀:G₀(0.9), Node₁:G₁(0.6), Node₂:G₂(0.3), Node₃:G₃(0.1)} → 负载分散

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 算法复杂度：O(M log M)（局部排序）+ O(N × M)（group construction）——M 通常为 4-8，极快
- 每个节点独立执行，无跨节点协调，无 centralized bottleneck
- 由于时间约束，当前实现仅平衡发送端负载（sender-side only），使用 greedy heuristic 而非 optimal solution（exhaustive search 为 O((M!)^N)，不可行）
- 消融实验显示 Balancer 贡献 3.2%（single-node routed）至 16.6%（load-imbalanced）的性能提升

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
