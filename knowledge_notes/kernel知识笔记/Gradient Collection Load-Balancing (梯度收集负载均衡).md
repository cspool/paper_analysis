## Gradient Collection Load-Balancing (梯度收集负载均衡)

术语解释
Gradient Collection Load-Balancing 是 SYMI Optimizer 中用于高效收集梯度 shards 到 optimizer partition 的算法（Algorithm 2）。由于 optimizer 均匀分片在 N 个节点上，每个 optimizer partition 需要从持有对应 expert instance 的 rank 收集梯度 shard。该算法优先使用本地传输（零网络开销），远程传输则 round-robin 分配以避免网络热点。

术语是什么？
在 SYMI 中，每个 expert e_i 的梯度需要从 r_i 个 replica instance 收集到 N 个 optimizer partition。`get_source(expert_id, dst_rank)` 决定哪个 source rank 为指定的 (expert, optimizer_dst) 对提供梯度：
- 如果 dst_rank 本地持有该 expert 的 instance → 直接本地 PCIe 传输（无网络开销）
- 否则 → 从 r_i 个 remote replicas 中 round-robin 选择一个，确保梯度负载均匀分布

从kernel调度角度拆解术语：
```
# SYMI Algorithm 2: Gradient Collection
def get_source(exp_id, dst_rank):
    if dst_rank in exp_to_rank_map[exp_id]:
        return dst_rank  # local transfer preferred
    candidates = sorted(exp_to_rank_map[exp_id])
    idx = dst_rank % len(candidates)  # round-robin
    return candidates[idx]

def collect_grads():
    recv_tuples = {}  # (src_rank, dst_rank)
    send_tuples = {}  # (dst_rank, partition_idx)
    
    for exp_id in all_experts:
        # Each optimizer partition determines its source
        for dst_rank in range(N):
            src = get_source(exp_id, dst_rank)
            recv_tuples[(exp_id, dst_rank)] = (src, dst_rank)
    
    for slot, exp_id in local_expert_map.items():
        for dst_rank in range(N):
            if get_source(exp_id, dst_rank) == local_rank:
                send_tuples[(exp_id, dst_rank)] = dst_rank
    
    # Batch point-to-point: single batch_isend_irecv for all pairs
    batch_isend_irecv(send_tuples, recv_tuples)
```

术语一般如何实现？如何使用？
- 使用 PyTorch distributed 的 batch_isend_irecv（point-to-point communication）
- Round-robin 策略确保 hotspot free——任何单个 expert instance 不会成为多个 optimizer partition 的梯度源
- 本地优先策略减少约 E/(sN) 比例的网络通信（当 expert 本地有 replica 时）
- 与 weight distribution phase 对称：updated weights 从 optimizer partition 反向发送到 expert slots

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)
