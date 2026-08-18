## AllToAll（全对全通信，Expert Parallelism 的 token 路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AllToAll 是集体通信（CC）操作之一：每个 rank 向所有其它 rank 发送/接收一个 tensor 的不同分片，实现数据的完全重新分布（full redistribution），是三类典型 CC（AllGather、AllReduce、AllToAll）中唯一"每个 rank 都既是发送方又是接收方且分片各不相同"的操作。在 MoE（专家并行）中，AllToAll 承担 token 路由：gate/router 决定每个 token 去哪个（可能位于其它 rank 的）专家，AllToAll 把各 rank 的 token 子集分发给对应专家，专家算完后反向 AllToAll 把结果送回。RoCC 论文把 AllToAll 分解为最简单的 primitive 序列（send → recv，仅 2 阶段）并支持在 ROP 上执行；AllToAll 无归约计算，本质是"数据搬运"，与 AllGather/AllReduce 一样是网络/内存 bound。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RoCC 论文中 AllToAll 在 ROP 上的执行（Table I，4-GPU ring 分解为 2 primitive）：send 展开为 ReadDoorbell→RingDoorbell（读本地分片、门铃转发给目标 rank），recv 展开为 ReadDoorbell→Write（收上一 rank 的分片、写本地）。伪代码（k/V 层 AllToAll dispatch 概念）：
```
for e in experts:                         # 每个专家一个目标分片
    tokens_to_e = gate(tokens, e)        # 路由决策：哪些 token 去专家 e
    if e on remote rank r:
        send(tokens_to_e, rank_r)        # 发给远端专家（RoCC: RingDoorbell）
for r in ranks:
    recv(tokens_from_r)                  # 收各 rank 发来的 token（RoCC: ReadDoorbell→Write）
    expert_gemm(tokens_from_r)           # 本地/远端专家计算
```
在本文评估中，AllToAll 场景模拟专家并行压力测试：每个专家与所有其它专家交换 token。CC-only 延迟比较中 RoCC 对 AllToAll 达 25% 加速（大消息，因 ROP 近内存避免 NoC 与 L1 miss 开销）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NCCL/RCCL 库提供 AllToAll 原语（如 ncclSend/ncclRecv 或 ncclAllToAll），底层按拓扑组织多阶段点到点传输；RoCC 将其编译为 ROP primitive（send/recv）执行。使用场景：MoE 训练/推理的 token dispatch 与 combine、序列并行（Sequence Parallelism）的激活交换、专家并行梯度同步。与相关概念区分：AllGather 是"每 rank 广播自己的分片、拼接全员数据"（RoCC 分解 4 阶段 recvCopySend），AllReduce 是"归约后全员可见"（RoCC 分解 7 阶段），AllToAll 只做分片交换无归约无拼接（RoCC 分解仅 2 阶段）。

- STAGE 补充视角（ISCA'26）：STAGE 的 Collective Communication Matcher 把 AllToAll 作为"Pull+Push"组合的特例系统化匹配：producer 分布 [B/dp,S,H@1/tp] 与 consumer 分布 [B,S/dp,H@1/tp] 之间（dp 作用于 batch、sp 作用于 seq），匹配结果即 AllToAll；还识别出此前被忽略的组合模式，如 [B/dp,S,H@1/tp]→[B/tp,S,H/dp] 匹配 ReduceScatter+AllToAll、[B/dp,S,H@1/tp]→[B,S,H] 匹配 AllReduce+AllGather（Table IV）。MoE 的 EP 层 AllToAll 通信量在 STAGE 中按张量尺寸精确计算，用于 DeepSeek-R1 推理与 MoE 训练的通信验证（Table VII 中 Mixtral 8x7 TP4-EP8-PP4 的 Send/Recv 误差 2.755%）。

涉及论文标题：
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
