## Micro-Op Communication Scheduling

术语解释
Micro-Op Communication Scheduling 是 Lina 提出的训练端通信调度优化。将 All-to-All 和 Allreduce 通信分解为统一的 micro-ops，通过 priority queue 保证 All-to-All 优先获得满带宽，而 Allreduce micro-ops 仅在无 All-to-All 待处理时发射，同时 All-to-All micro-ops 与 Expert FFN 计算组成 pipeline。

术语是什么？
Lina 的 Micro-Op Scheduler 运行在每个 device 上的单线程中，维护一个 priority queue：
- All-to-All micro-op: HIGH priority，始终优先发射
- Allreduce micro-op: LOW priority，仅当队列无 All-to-All 时发射
- Combine computation 阶段停止 Allreduce 发射（预示 All-to-All 即将到来）

配合 Expert Packing，使 FFN micro-op time 与 All-to-All micro-op time 对齐，最大化 pipeline efficiency。

从kernel调度角度拆解术语。
```
# Lina Micro-Op Scheduler
class MicroOpScheduler:
    def __init__(self, partition_size=30*1024*1024):
        self.pq = PriorityQueue()
        self.partition_size = partition_size
        self.in_combine_phase = False
    
    def enqueue_gradient(self, grad, op_type):
        micro_ops = torch.chunk(grad, 
            ceil(grad.numel()*grad.element_size() / self.partition_size), dim=0)
        priority = HIGH if op_type == ALLTOALL else LOW
        for op in micro_ops:
            self.pq.push(op, priority)
    
    def schedule_loop(self):
        while True:
            # 始终优先 all-to-all
            if self.pq.has(HIGH):
                op = self.pq.pop(HIGH)
                launch_alltoall(op)
                # 每 micro-op 完成后立即启动对应 FFN
                trigger_expert_ffn_for_tokens(op.tokens)
            elif self.pq.has(LOW) and not self.in_combine_phase:
                op = self.pq.pop(LOW)
                launch_allreduce(op)
            else:
                # idle: 等待下一 micro-op 或 combine 结束
                yield_cpu()
    
    def on_combine_phase_start(self):
        self.in_combine_phase = True
    
    def on_combine_phase_end(self):
        self.in_combine_phase = False
```

All-to-All Micro-Op Pipelining 时间线:
```
         All-to-All Micro-Op 1          All-to-All Micro-Op 2          All-to-All Micro-Op 3
Stream b |====A2A-chunk1====|           |====A2A-chunk2====|           |====A2A-chunk3====|
Stream a                    |FFN-tokens1|                   |FFN-tokens2|                   |FFN-tokens3|
                            <----------- Pipelining: computation hidden behind communication ----------->
```
FFN 在第一个 All-to-All micro-op 完成后立即启动，无需等待全部 All-to-All 完成。

术语一般如何实现？如何使用？
- 每 device 单线程 priority queue (C++ `std::priority_queue`)
- LibTorch `chunk`/`cat` API 做 tensor partition/concatenation
- NCCL 通信原语 (ncclAllToAll / ncclAllReduce)
- Expert Packing 配合：当 FFN micro-op << All-to-All micro-op 时，增加每 device expert 数使 FFN 时间对齐 All-to-All（pipeline efficiency 从 33% 提升至 86%）
- Overhead: micro-op 传输 overhead 平均 +1.7% vs 不分区

涉及论文标题：
- Accelerating Distributed MoE Training and Inference with Lina

---
