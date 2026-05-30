## Enlarged Hybrid Parallelism Space for MoE（面向MoE的扩展混合并行空间）

术语是什么？
Enlarged Hybrid Parallelism Space 是 SmartMoE 提出的概念：在传统 MoE 混合并行空间（DP、TP、PP、EP 的组合）基础上，新增 **Expert Placement** 作为可搜索的并行策略维度。传统自动并行化系统（Alpa、Tofu）和 MoE 训练系统（FasterMoE、Tutel）将 expert 到 GPU 的映射视为固定的（按索引顺序放置），但 SmartMoE 发现 expert placement 的顺序直接影响负载均衡和性能——两个 expert placement 方案即使在"每个 GPU 上有相同数量的 expert"这一粗粒度下等价，其实际性能也可能因动态 token 分布而差异显著。

从kernel调度角度拆解术语：
以 4 experts, 2 GPUs 为例说明 placement 的影响：
```
Workload: E0=200t, E1=300t, E2=200t, E3=100t (极度不均)

方案 A (按索引顺序):         方案 B (按负载交错):
GPU_0: {E0(200), E1(300)}     GPU_0: {E1(300), E3(100)}
  load=500t                     load=400t
GPU_1: {E2(200), E3(100)}     GPU_1: {E0(200), E2(200)}
  load=300t                     load=400t
  Imbalance: 200t                Imbalance: 0t ✓

传统系统: 方案 A = 方案 B（每 GPU 均有 2 experts）
SmartMoE: 方案 B > 方案 A（考虑实际负载）
```

SmartMoE 的 Enlarged Space 将搜索空间从"选择何种 DP/TP/PP/EP 组合"扩展到还包括"expert 如何映射到 expert slot"，使用 expert slot 抽象统一表达。这使得系统能在更大范围内搜索最优并行配置。

术语一般如何实现？如何使用？
在 SmartMoE 中，Enlarged Space 被分解为两阶段处理：离线阶段搜索固定部分（DP/TP/PP/EP 组合），在线阶段动态调整可变部分（expert placement within pool）。实现基于 FastMoE 的 expert slot 抽象，使用 PyTorch。搜索策略：离线穷举候选池 + workload-aware 性能模型评估；在线 Greedy/DP/Hybrid 算法。

涉及论文标题：
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization
