## Intra+Inter Rank All-Reduce (秩内+秩间全规约)

术语解释
Intra+Inter Rank All-Reduce 是 SYMI 提出的新型 all-reduce 实现，支持同一 rank 内多个 GPU slots 持有同一 expert class 的 replica（intra-rank expert data parallelism），同时保持跨 rank 的梯度同步。传统 NCCL all-reduce 仅支持跨 rank 同步，不支持同 rank 内多 replica 场景，限制了 expert placement 灵活性。

术语是什么？
传统 expert gradient all-reduce：每个 expert class 的 r 个 replica 分布在 r 个不同 rank 上，执行 NCCL all-reduce 同步梯度。限制是 expert 最多只能被复制 N 次（每个 rank 最多 1 个 instance），导致 sub-optimal placement 和 up to 20% extra token drops。

SYMI 的三步 all-reduce：
- Step 1 (Intra-rank): 每个 rank 内选举一个 slot representative，其他 slot 将 gradient 累加到 representative
- Step 2 (Inter-rank): 仅在 representative 间执行 all-reduce（跨 rank）
- Step 3 (Intra-rank broadcast): representative 归一化后将结果广播回同 rank 其他 slot

从kernel调度角度拆解术语：
```
# SYMI Intra+Inter Rank All-Reduce（per expert class）
def syMI_allreduce_expert_grads(expert_id, grads_per_slot):
    # grads_per_slot: dict {slot_idx: gradient_tensor} for local slots of this expert
    
    # Step 1: Intra-rank sum (local GPU computation, no network)
    rep_slot = min(grads_per_slot.keys())  # elect representative
    for slot, grad in grads_per_slot.items():
        if slot != rep_slot:
            grads_per_slot[rep_slot] += grad  # accumulate to rep
    
    # Step 2: Inter-rank all-reduce (NCCL, only on representatives)
    rep_grads = [grads_per_slot.get(rep_slot, zeros_like(...)) for each rank]
    allreduced_rep_grad = allreduce(rep_grads[local_rank])  # NCCL
    
    # Step 3: Intra-rank broadcast (local GPU, copy)
    num_local_replicas = len(grads_per_slot)
    normalized_grad = allreduced_rep_grad / num_local_replicas
    for slot in grads_per_slot:
        grads_per_slot[slot] = normalized_grad
    
    return grads_per_slot
```

术语一般如何实现？如何使用？
- 需要配合 Expert Placement Scheduler 的 contiguous assignment——优先将同 expert class 的 replica 放在同 rank 内
- 优势：减少 inter-node 网络流量（同 rank 内通信为本地 GPU 操作，无网络开销）
- NCCL 限制：传统 NCCL all-reduce 要求每个 rank 恰好一个参与 tensor，SYMI 的 intra-rank sum 步骤绕过此限制
- SYMI 论文实测：此 all-reduce 实现比传统实现更高效（配合 locality-enhanced placement）

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---
