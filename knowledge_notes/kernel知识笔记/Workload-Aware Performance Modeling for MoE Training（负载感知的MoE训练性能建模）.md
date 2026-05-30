## Workload-Aware Performance Modeling for MoE Training（负载感知的MoE训练性能建模）

术语是什么？
Workload-Aware Performance Modeling 是 SmartMoE 离线阶段使用的性能预测方法。与传统的 data-insensitive 性能模型（仅使用模型结构和硬件信息）不同，它利用 gating network 的设计语义估计训练时的 per-expert 负载分布——在实际训练之前预测执行计划的性能。核心思想：虽然实际 expert selection 在训练前不可获得，但 gating network 的超参数（capacity factor / topology-aware constraints）提供了负载的上界估计，该上界通常接近实际瓶颈，可用于准确预测。

从kernel调度角度拆解术语：
针对两类 gating network 分别估算：
```
类别1: Load-Balanced Gating (GShard gate)
  输入: capacity_factor (e.g., 1.2, 2.4, +∞)
  计算: max_tokens_per_expert = (capacity_factor × batch_tokens) / num_experts
         bottleneck_expert_load = min(max_tokens_per_expert, total_tokens)
  通信量估算: 使用 capacity_factor 控制下的路由分布估算 All-to-All dispatch 量
  
  例子: 4 GPUs, 16 experts, capacity=2.4, batch=1024 tokens
        max_tokens_per_expert = 2.4 × 1024 / 16 = 153.6 → 154 tokens
        最重 expert ≤ 154 tokens, 通信比例 ≤ 154/64 = 240%

类别2: Topology-Aware Gating (Faster Gate)
  输入: 硬件拓扑（node数, GPUs_per_node）+ hierarchical routing 算法
  计算: 按 Faster Gate 的两层路由算法模拟
        优先 intra-node routing → 估计 cross-node all-to-all 量
        max communication per device pair = f(hardware_topology, gate_algorithm)
  
性能模型输出:
  T_layer = T_compute(max_expert_load) + T_comm(max_cross_node_alltoall)
  其中 T_compute 和 T_comm 使用 FasterMoE 的基础性能模型
  R² > 0.5 for all evaluated configurations
```

术语一般如何实现？如何使用？
SmartMoE 在离线池搜索阶段使用该模型：对每个候选池（固定 DP/TP/PP + 可变 expert placement），用 workload-aware 模型评估所有 expert placement 变体的平均/最差性能 → 选性能最优的池。注意原始 FasterMoE 性能模型需要运行时负载数据作为输入——SmartMoE 的关键创新是用 gating 语义估算替代实际数据。该模型也可用于其他需要离线预测 MoE 性能的场景。

涉及论文标题：
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization
