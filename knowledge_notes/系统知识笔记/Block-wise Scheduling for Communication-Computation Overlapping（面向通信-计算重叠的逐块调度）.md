## Block-wise Scheduling for Communication-Computation Overlapping（面向通信-计算重叠的逐块调度）

术语是什么？
Block-wise Scheduling 是 Pro-Prophet 提出的 MoE 训练调度策略。将每个相邻的 MoE layer + non-MoE layer 组合为一个 MoE block，以 block 为单位进行 sub-operator 级调度。核心思路：将 load balancing 涉及的三个数据依赖操作——Plan（搜索 placement）、Trans（传输 expert parameters）、Agg（聚合 gradients）——拆分并调度到 computation 操作中并行执行，消除通信曝光。

从系统架构角度拆解术语：
在 MoE 训练的单个 iteration 中的调度流程：

```
// 操作类型标注
Plan: comp（仅计算，决定 load balancing 策略）
Trans: comm（传输 expert parameters）
A2A: comm（token dispatch/combine）
FEC/BEC: comp（expert 计算）
FNEC/BNEC: comp（non-MoE 层计算）
Agg: comm（聚合 gradients）

// Block-wise Scheduling（以 MoE block 为单位）
iteration j:
  FP:
    A2A dispatch || Plan_{j+1} (从 iteration j-1 的 distribution 预测并搜索)
    [FEC_i || Trans_{i+1} sub-op1] (Trans 拆分为 2 个子操作)
    [FNEC_i || Trans_{i+1} sub-op2]
  BP:
    [BEC_i || Agg_{i+1} sub-op1]
    [BNEC_i || Agg_{i+1} sub-op2]

// 关键约束：
// - Plan 最早可前移到前迭代的 A2A 通信中
// - Trans 限制在单 iteration 内（兼容 layer-by-layer 和 concentrated updating）
// - Agg 同理限制在单 iteration 内
// - FNEC/BNEC 的时间是静态的，可在训练前估计用于精确规划 split
```

术语一般如何实现？如何使用？
Pro-Prophet 在 PyTorch 上实现。Scheduler 通过建立 Scheduling Space（定义每个操作的类型和调度窗口）→ 执行 block-wise scheduling（以 MoE block 为单位 reorder）→ sub-operator 级调度（将 Trans/Agg 拆分为子操作与对应 computation 并行）。Planner 的 performance model 也集成了 Scheduler 的并行执行时间：T_PTrans = max(0, T_Trans - T_FEC - T_FNEC)，即被计算隐藏的通信部分不算入总时间。实验结果：Scheduler 单独贡献 1.01-1.14x speedup。

涉及论文标题：
- Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models
