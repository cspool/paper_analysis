## Expert Rearrangement (in MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Rearrangement 是 MoE 训练系统中用于缓解 Expert Parallelism (EP) straggler 效应的动态 expert placement 调整技术。在 EP 中，experts 被均匀分配到各 device，但因 MoE gate 训练的动态性，expert load（token 分布）频繁波动和不平衡，导致最重载 device 成为整个 MoE layer 的计算和通信瓶颈（straggler effect）。Expert rearrangement 系统通过在训练过程中动态修改 expert placement（哪些 expert 在哪些 device 上有副本/被迁移）来适应 load imbalance。

三类主要 rearrangement 策略：
1. **FasterMoE (PPoPP'22)**：选择性将最热门 expert 复制到所有 device（shadow expert），粗粒度管理（expert 要么在 1 个 GPU 要么在所有 GPU）。
2. **SmartMoE (USENIX ATC'23)**：通过交换 experts 在 devices 间的 position 来平衡 device loads（如将最多和最少 token 的 expert 放在同一 device），要求每 device 能容纳多个 expert。
3. **FlexMoE (SIGMOD'23)**：最全面的 rearrangement，通过 vExpert 抽象支持三种操作——Expand（为 overloaded expert 创建 replica）、Shrink（释放 underutilized replica）、Migrate（在 GPU 间交换 replica 以减少同步成本），使用 cost-model driven search 决定最优修改。

Hecate 揭示了 rearrangement 系统的两个核心挑战：(C1) Memory Challenge——更 balance 的 placement 需要更多内存来容纳 replica experts 及其 optimizer states（Adam mixed precision 下 optimizer states 至少 6× 参数量），预留内存不足会限制 placement 优化空间（FlexMoE 实验 4× 内存仅换 2.65× speedup）；(C2) Timeliness Challenge——rearrangement 频率的 trade-off（高频 → placement timely 但通信开销大，低频 → placement 过时），最优频率因训练场景而异无法统一确定（SmartMoE 每 10 steps vs 25 steps：non-rearrangement iteration 快 2.9% 但整体慢 10.2%）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Expert Rearrangement 在 FlexMoE 系统架构中的运转流程：

```
┌── FlexMoE Rearrangement Workflow ──────────────────────────┐
│                                                              │
│  1. Monitoring: Scheduler 实时监控各 device 的 expert load  │
│     - 统计每个 expert 的 token count                         │
│     - 计算 balance ratio = max_load / avg_load              │
│                                                              │
│  2. Trigger: 当 balance ratio 超过阈值 (β > threshold)      │
│     → Policy Maker 启动 rearrangement                       │
│                                                              │
│  3. Cost-Model Search:                                      │
│     - 枚举可能的 vExpert 操作 (Expand/Shrink/Migrate)       │
│     - 评估每个操作的 cost (通信时间 + 计算时间)              │
│     - 选择 cost-benefit 最优的操作组合                       │
│                                                              │
│  4. Execution (ON CRITICAL PATH):                           │
│     - Expert 参数 + Optimizer States 的 P2P 传输             │
│       (optimizer states 可达到 6× 参数量, Adam + mixed FP)  │
│     - 更新 placement map (expert → device 映射)              │
│     - 后续 iterations 使用新 placement                       │
│                                                              │
│  5. Gradient Sync (每 iteration 结束):                      │
│     - 对有 replica 的 expert, AllReduce 同步 gradients      │
│     - 通信量: Σ_i 2(|D_i|-1)/|D_i| · S/|C| ≈ O(2λS)        │
│                                                              │
│  问题:                                                       │
│  - 迁移 optimizer states 造成巨大通信开销                    │
│  - Rearrangement 在 critical path → 限制了频率               │
│  - 预留内存限制 placement 灵活性                             │
└──────────────────────────────────────────────────────────────┘
```

Hecate 的 FSSDP 范式从根本上重新设计了这一流程：不再在 iteration 间迁移 expert 状态，而是每次 iteration 从全局唯一的 MoE shards 用 SparseAllGather 从零构建临时 placement，用 SparseReduceScatter 同步 gradients。FSSDP 的通信量 O(2λS) 与 rearrangement 的 AllReduce 等价，但消除了迁移开销（timeliness challenge），且全局仅保留一份 optimizer states（memory challenge 解决）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Rearrangement 系统通常基于 PyTorch + Megatron-LM 等框架实现，在 MoE layer 的 training loop 中插入 rearrangement manager。
- FlexMoE 的 vExpert 抽象将 expert 视为可独立调度的最小单元，通过 expand/shrink/migrate 三个原语操作 vExpert → physical GPU 的映射。
- SmartMoE 的两阶段方法：offline 构建执行计划池（基于 workload-aware performance model），online 用贪心/混合 DP 算法每 ~10 iteration 选择和 refinement 计划（切换开销 ~20ms）。
- Rearrangement 频率是关键超参数，需要在 load timeliness 和通信 overhead 之间 trade-off，且最优值因 training scenario 而异。

涉及论文标题：
- FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models
- SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization
- FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
