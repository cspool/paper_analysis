## Communication Group Pre-Registration (通信组预注册)

术语解释
Communication Group Pre-Registration 是 SYMI 避免运行时 NCCL communication group 创建开销的优化技术。由于 SYMI 每 iteration 的 expert placement 变化，各 expert class 的 all-reduce 通信组也会变化。在训练初始化时预创建所有可能的 contiguous-rank 通信组，训练期间仅需查表选择，避免每 iteration 动态创建 NCCL group 的巨大开销。

术语是什么？
NCCL 的 communication group 创建（`ncclCommInitAll` 或 `torch.distributed.new_group`）是阻塞式、单线程同步操作，在大集群（N=2048）中单次创建耗时可能超过 1000 秒（MegaScale 论文数据）。若 SYMI 每 iteration 动态创建新的 expert 通信组，开销完全不可接受。

从kernel调度角度拆解术语：
```
# 初始化阶段（training startup, 一次性开销）
def pre_register_groups(N_ranks):
    groups = {}
    # 仅注册 contiguous-rank groups（Expert Placement Scheduler 
    # 保证 expert 按 contiguous 方式分配）
    for start in range(N_ranks):
        for end in range(start + 1, N_ranks + 1):
            rank_list = list(range(start, end))
            groups[(start, end)] = torch.distributed.new_group(
                ranks=rank_list, 
                backend='nccl'
            )
    return groups  # O(N²/2) groups, 跨 expert 和 layer 复用

# 训练阶段（per-iteration, O(1) 查表）
def get_comm_group(expert_id, placement):
    ranks_with_expert = sorted(find_ranks_with_expert(expert_id, placement))
    # 因为 contiguous assignment, 这些 rank 是连续的
    return pre_registered_groups[(ranks_with_expert[0], ranks_with_expert[-1] + 1)]
```

术语一般如何实现？如何使用？
- 关键前提：Expert Placement Scheduler 的 contiguous assignment 策略——使 expert 实例始终分配到连续的 rank 集合上
- 组数量：O(N²/2)，例如 N=2048 → 约 2M groups，每个 group 仅包含 ranks 列表的 metadata（内存可管理）
- 组复用：同一 group 可被不同 expert、不同 layer、不同 iteration 共享（只要 rank 集合相同）
- 替代方案：使用 NCCL 的 dynamic group 或 P2P 通信，但 batch point-to-point 在 SYMI 中用于梯度收集和权重分发

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---
