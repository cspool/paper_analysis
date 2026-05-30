## Tensor Chunk-Based Priority Scheduling for All-Reduce in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlowMoE 提出的反向传播梯度 all-reduce 调度策略。将每层的 all-reduce 梯度张量切成大小为 S_p 的 chunk，放入通信任务池，赋予低于 A2A 通信的优先级。运行时仅当 A2A 队列为空时才执行 AR chunk，使 all-reduce "见缝插针"地填充 A2A 通信间隙，实现全重叠。Theorem 1 证明在 A2A 任务间隙插入 AR chunk 可减少反向传播总时间；Theorem 2 证明理想无启动开销下 S_p→0 时 per-iteration time 最小化，实际 S_p 需平衡系统开销（NCCL kernel launch、小 chunk 低带宽利用率），由 BO 自动搜索。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 通信池管理器（后台线程）:
class CommPoolManager:
    A2AQueue: PriorityQueue    // 优先级: HIGH
    ARQueue: PriorityQueue     // 优先级: LOW

    def run():
        while training_active:
            if not A2AQueue.empty():
                execute(A2AQueue.pop())
            elif not ARQueue.empty():
                execute(ARQueue.pop())
            else:
                wait()

// All-Reduce 切分:
chunk_size = S_p  // BO 搜索, BERT-Large-MoE 上 ~2.5MB
for c in range(num_chunks):
    ARQueue.push(grad[c*S_p : (c+1)*S_p], priority=LOW)

// Timeline (反向):
// A2A: |C_1|    |D_2|    |C_2|    |D_1|
// AR:      |chk1|    |chk2|    |chk3|
// Comp: |AT_1'|AT_2'|E_1'|E_2'|
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 仅在反向传播期间激活（前向无 all-reduce 需求）
- S_p 最优值高度依赖硬件（GPU 型号、网络带宽、模型配置），不可跨集群复用
- BO 采样约 8 次（每次测 10 轮迭代平均时间），总开销 < 1% 训练迭代时间
- 硬件环境变化时需重新 profiling
- 与 DeAR (reduce-scatter + all-gather 两阶段) 正交，chunk 方法可应用于 DeAR 的两个阶段

涉及论文标题：
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training
