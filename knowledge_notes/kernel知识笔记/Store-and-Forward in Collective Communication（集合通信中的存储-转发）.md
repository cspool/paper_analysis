## Store-and-Forward in Collective Communication（集合通信中的存储-转发）

术语是什么？
Store-and-Forward 是集合通信中一种逐跳数据传输机制。与硬件路由的多跳直连传输不同，Store-and-Forward 将每个多跳传输分解为多个单跳步骤：中间节点在接收到数据后暂存于本地端点缓冲区，再转发到下一跳。这种细粒度编排方式允许显式控制每步的链路分配，从而完全消除网络拥塞。

从kernel调度角度拆解术语：
以 4 节点环上 Ring 算法 Stage 3（3 跳传输）为例：
```
// 节点 1 发送紫色数据块到节点 4（逆时针 3 跳 = 顺时针 1 跳，但 Ring 固定双向发送）
// 顺时针方向（1→2→3→4，3 跳）:
Sub-stage 3-1: node1 → node2 (Fwd)  // 节点 1 转发给节点 2
Sub-stage 3-2: node2 → node3 (Fwd)  // 节点 2 存储后转发给节点 3
Sub-stage 3-3: node3 → node4 (Fwd)  // 节点 3 存储后转发给节点 4
// 每个子阶段：所有活跃节点对该跳的单跳传输同时执行；无链路共享→零拥塞
```

术语一般如何实现？如何使用？
在本文中，HalfRing/FoldedRing/MATE 均基于 Store-and-Forward 构建通信时间表——算法生成器离线计算每个阶段/子阶段的所有单跳传输对，运行时逐子阶段执行。实现中，store 操作将数据写入接收节点的中间缓冲区，forward 操作从该缓冲区读取并发送到下一跳。相比硬件路由（依赖交换机/路由器处理拥塞控制），Store-and-Forward 在 torus 等直连拓扑上可避免多跳传输的链路争用问题。代价是中间节点的存储开销和额外的转发延迟（每跳增加 α 传播延迟）。

涉及论文标题：
- Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks
