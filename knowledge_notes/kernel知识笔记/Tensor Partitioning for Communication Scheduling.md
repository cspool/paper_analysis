## Tensor Partitioning for Communication Scheduling

术语解释
Tensor Partitioning 是将大型通信张量（如 All-to-All 和 Allreduce 的 gradient/activation tensors）分割为统一大小的小块（chunk/micro-op），以便调度器精确控制每个通信原语的发射时机和带宽分配。Lina 首次将其应用于 MoE 训练的 All-to-All 优先级调度。

术语是什么？
在分布式 MoE 训练中，Lina 将每个 gradient tensor 沿 token 维度分割为固定大小（如 30MB）的 micro-ops。每个 micro-op 作为一个独立的通信单元进入 priority queue，由调度器按优先级发射。关键设计：
- 不跨 gradient 混合 chunk（保持 concat 简洁）
- 使用 LibTorch 内置 `chunk` 和 `cat` API
- Partition overhead: preprocessing+concatenation 平均 1.02% step time

从kernel调度角度拆解术语。
```
# Lina Tensor Partitioning 伪代码
def partition_gradient_for_scheduling(grad, partition_size):
    """将 gradient tensor 分为 micro-ops"""
    # grad: 梯度张量，沿 token 维度
    # partition_size: 固定 micro-op 大小（如 30MB）
    num_chunks = ceil(grad.numel() * grad.element_size() / partition_size)
    micro_ops = torch.chunk(grad, num_chunks, dim=0)  # 沿 token 维度分割
    return micro_ops  # 每个 micro-op 大小均匀

# 在 backward pass 中使用
micro_ops = partition_gradient_for_scheduling(grad, 30*1024*1024)
for op in micro_ops:
    if op.type == ALLTOALL:
        priority_queue.push(op, priority=HIGH)
    else:
        priority_queue.push(op, priority=LOW)

# 调度循环
while not priority_queue.empty():
    if priority_queue.has_priority(HIGH):
        op = priority_queue.pop(HIGH)
        launch_nccl_alltoall(op)
    else:
        op = priority_queue.pop(LOW)
        launch_nccl_allreduce(op)
```

术语一般如何实现？如何使用？
- LibTorch `tensor.chunk(chunks, dim)` 沿指定维度分割
- 分割后逐个包装为 micro-op 入队
- Priority queue 按类型 (All-to-All > Allreduce) 优先级出队
- 分区大小需要平衡：小于 10MB 导致每 micro-op 传输 overhead 过大，大于 50MB 粒度不够精细
- 最优分区大小取决于模型和集群配置（Lina 默认 30MB）

涉及论文标题：
- Accelerating Distributed MoE Training and Inference with Lina

---
