## Operator Partition Pass (DP-based, for MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Operator Partition Pass 是 Lancet 编译器中优化前向传播的 IR 级 Pass，通过沿 batch 维度分区 non-MoE 计算算子并将其与 all-to-all + expert 组成 computation-communication pipeline，实现更大范围的重叠。由三个组件协同：Dynamic Programming（DP）搜索最优 partition range 和 partition count、Partition Axis Inferencer（CSP 求解分区轴）、Pipeline Scheduler（模拟 pipelined 时间线评估 P(i,n,k) 代价并反馈给 DP）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

DP 状态转移方程：

$$T(n) = \min_{1 < i < n-1} \{T(i) + \min_{1 < k < K} P(i,n,k)\}$$

其中 T(n) 为指令 1 到 n 的最优执行时间，P(i,n,k) 为指令 i 到 n 分 k 个 partition 后的端到端时间（考虑重叠）。若 range (i,n) 无法分区（如包含不支持分区的 gating），P(i,n,k)=∞。

```
DP 搜索流程:
  for n in 1..N_groups:       // N_groups: 指令组数
    T(n) = ∞
    for i in max(1, n-G)..n-1:  // G: 最大partition range
      for k in 1..min(K, batch_size):
        // (1) CSP求解分轴
        axes = PartitionAxisInferencer(I_groups[i:n], k)
        if axes == UNSAT: continue
        
        // (2) 模拟pipeline时间线
        P_cost = PipelineScheduler(I_groups[i:n], k, axes)
        
        T(n) = min(T(n), T(i) + P_cost)
```

复杂度：原始 O(N²K)，指令分组后 O(N'GK)，其中 N' 为指令组数（~5 groups per MoE layer），G 为最大 partition range。K 受 batch size 和 partition overhead 限制（实践中从未超过 4）。每个 MoE layer 形成独立 pipeline。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 Lancet 中，Operator Partition Pass 是优化时间的主要消耗者（占 <20 分钟中的绝大部分），因为每次 DP 候选都需要运行 CSP solver 和 Pipeline Scheduler。它依赖 Caching Op Profiler 提供执行时间、Communication Cost Model 提供预估通信时间。优化结果仅在单 GPU 上计算一次，生成的优化 IR 被所有 GPU 复制使用。用户无需手动指定分区策略——DP 自动根据硬件 profile 决策最优 (i,n,k)。三个可调超参通过环境变量设置。

涉及论文标题：
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
