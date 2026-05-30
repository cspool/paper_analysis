## Hierarchical AlltoAll Communication (层次化AlltoAll通信 / Resource-Aware Communication)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical AlltoAll 是 MoESys 针对 MoE 模型中 Expert Parallelism 的 AlltoAll 通信提出的两阶段网络拓扑感知通信优化。传统 AlltoAll 中，不同 rank 的 GPU 跨节点通信时经过 spine switch，造成路由冲突和带宽浪费。Hierarchical AlltoAll 将一次全交换拆为两个阶段：(1) **Intra-node AlltoAll**——利用 NVSwitch/NVLink（900GB/s）在单节点 8 GPU 间完成数据重排，将跨 rank 的数据搬运到同节点内同 rank 的 GPU；(2) **Inter-node AlltoAll**——将各节点中同一 rank（如所有 node 的 GPU0 组成一组、所有 node 的 GPU7 组成另一组）的 GPU 分组做跨节点 AlltoAll，同 rank GPU 的 NIC 接入同一 leaf switch，不经过 spine switch。

从kernel调度角度拆解术语：
Hierarchical AlltoAll 的通信调度流程（以 2 nodes, 16 GPUs 为例，单节点 8 GPU，目标是从 GPU0 Node1 发送数据到 GPU7 Node2）：
```
# Baseline AlltoAll 路径:
GPU0(Node1) → NIC1(rank0) → LE1 → SPq → LE1 → NICn(rank7) → GPU7(Node2)
# 经过 spine switch SPq，高延迟 + 带宽竞争

# Hierarchical AlltoAll 路径:
# Phase 1: Intra-node via NVSwitch
GPU0(Node1) --NVSwitch 900GB/s--> GPU7(Node1)  # 数据搬运到同 rank
# Phase 2: Inter-node via NIC grouped by rank
GPU7(Node1) → NIC7(rank7) → LE7 → NIC7(rank7) → GPU7(Node2)
# 仅经过 leaf switch LE7，不经过 spine switch
```

通信调度伪代码：
```
function HierarchicalAlltoAll(tokens_per_expert):
    # Phase 1: Intra-node
    for each GPU g in node:
        data_to_rank_r = tokens destined for expert on GPU with rank r
        NVSwitch_AlltoAll(data_to_rank_r)  # 在本节点内按 rank 重排
    
    # Phase 2: Inter-node (grouped by rank)
    for each rank r in 0..7:
        comm_group = all GPUs with rank r across nodes
        NIC_AlltoAll(comm_group, data_for_rank_r)  # 同 rank 组跨节点通信
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现的硬件前提：单节点内 GPU 通过 NVSwitch 全互联（900GB/s per GPU），节点间通过 NIC（100G/200G/400G Mellanox ConnectX）+ leaf/spine 交换机互联。
- 与 DeepSpeed 的 AlltoAll 优化的区别：DeepSpeed 通过 tensor fusion 将小 packet 合并为大 packet 解决 per-port 通信量小的问题，是通信 payload 层面的优化；Hierarchical AlltoAll 是利用网络拓扑的物理层次做路径选择优化，是通信 routing 层面的优化。两者正交互补。
- 性能提升：peer-to-peer 通信效率提升 p 倍（p=单节点 GPU 数）；80.7B model / 4 nodes 32 GPUs 下通信阶段加速 15.5%，端到端训练加速 10.3%。
- 局限性：该方案与网络拓扑强耦合——如果 cluster 改用 rail-optimized 或 fat-tree 拓扑，方案需要重新设计。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
