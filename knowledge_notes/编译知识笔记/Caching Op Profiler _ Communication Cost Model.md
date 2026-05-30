## Caching Op Profiler / Communication Cost Model

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Caching Op Profiler 和 Communication Cost Model 是 Lancet 编译器的性能建模子系统。Profiler 负责 profile 和缓存所有计算算子在各种 (partitioned) shape 下的 GPU 执行时间；Cost Model 负责预估通信算子（特别是 partitioned all-to-all）的执行时间。两者共同为 Lancet 的贪心调度和 DP 搜索提供代价预估。Cost model 预测精度在实验中达 96.17%（3.83% 误差）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Profiling 流程：

```
Phase 1: 一次性 Profiling（编译优化前，单 GPU）
  ├── Computation Op Profiling:
  │     for each operator type f:
  │       for each (partitioned) shape in model:
  │         warmup(3次) + measure(10次) → 取平均
  │         Cache[(f, shape, partition_count)] = exec_time
  │
  └── Communication Cost Model:
        for msg_size in [1KB, 2KB, 4KB, ..., max_possible]:
          profile NCCL Send/Recv at msg_size
          CostModel[msg_size] = exec_time
        // 未 profile 的 size 之间线性插值

Phase 2: 编译时查询
  ├── dW Scheduling: GetInstrExecTime(I) → 查 Cache
  └── PipelineScheduler:
        partitioned compute: 查 Cache[(f, shape/k, k)]
        partitioned all-to-all (n-partition, capacity C):
          查 CostModel[C/n]  // static-shape approximation
```

Static-Shape Approximation：n-partition 不规则 all-to-all 的时间用 uniform C/n 近似。实际时间取决于运行时 token 分布，但近似足够准确（图 14 验证），因为估值主要用于比较不同 partition range 的相对优劣而非精确预测。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Profiler 在编译优化开始时运行一次（`python run_exp_configs.py --lancet-profile`），cache 被后续 Pass 复用。Profiling 时间包含在总优化时间（<20 分钟）内。Cost model 的线性插值假设对 NCCL 通信足够准确（通信时间在合理范围内近似线性增长）。设计原则是"足够好而非完美精确"——优化目标是比较不同 partition range 的相对优劣，而非绝对时间预测。

涉及论文标题：
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
