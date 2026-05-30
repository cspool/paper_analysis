## Heterogeneous-Aware Expert Allocation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Heterogeneous-Aware Expert Allocation 是 HEXA-MoE 提出的在异构 GPU 设备间按计算能力比例分配 MoE 训练 workload 的方法。传统 expert parallelism 依赖同构硬件假设，无法利用异构设备的差异化计算能力。HEXA-MoE 用 tensor parallelism 替代 expert parallelism，使各设备 workload 由 batch size（data-centric）或 FFN intermediate sub-dimension（model-centric）精确决定，将异构调度转化为确定性的比例分配问题：先通过 proxy benchmark 测量各设备计算能力（t_i），再按 1/t_i 的反比分配 workload。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
┌── Heterogeneous-Aware Allocation 流程 ────────────────────┐
│                                                              │
│ Step 1: Benchmark 计算能力                                   │
│   proxy task: 1024 次 size=2048 的 torch.matmul             │
│   记录各设备延迟 t_i → 能力 ∝ 1/t_i                         │
│                                                              │
│ Step 2: 按比例分配 workload                                 │
│   Data-Centric:                                             │
│     B_i = (1/t_i) / Σ_j(1/t_j) · B_global                  │
│   Model-Centric:                                            │
│     h_i = (1/t_i) / Σ_j(1/t_j) · H                         │
│   取整保证 Σ B_i = B_global, Σ h_i = H                      │
│                                                              │
│ 实验结果 (TITAN RTX + RTX 2080 Ti):                         │
│   Data-Centric: optimal vs uniform                          │
│     100W/300W: -13.2% latency                               │
│     300W/100W: -25.3% latency                               │
│   Model-Centric: optimal vs uniform                         │
│     100W/300W: -6.3% latency                                │
│     300W/100W: -11.9% latency                               │
│   等能力 (300W/300W): optimal = uniform (验证公式正确性)     │
└──────────────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

训练前使用 proxy task 测量各设备 t_i，按公式分配 B_i 或 h_i，取整后微调保证求和约束。在 DataLoader 中为各设备构造不同大小的 mini-batch（data-centric）或在模型初始化时分配不同的 intermediate sub-dimension（model-centric）。完全静态，零运行时开销。可通过 `nvidia-smi -pl` 调整功率限制模拟不同计算能力的设备。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy
