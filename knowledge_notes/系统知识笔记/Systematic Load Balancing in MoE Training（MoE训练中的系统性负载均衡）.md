## Systematic Load Balancing in MoE Training（MoE训练中的系统性负载均衡）

术语是什么？
Systematic Load Balancing in MoE Training 是指在 MoE 分布式训练中，不修改模型 gating/routing 算法（不改变 loss function 或 gating 网络），而是通过调整 expert-to-device 映射和通信调度来平衡各 device 负载的系统级方法。与算法级方法（如 auxiliary load balancing loss、capacity factor）不同，系统级方法不影响模型收敛且对硬件友好。代表性工作包括 FasterMoE（dynamic shadowing：runtime 检测负载、将 heavy-load expert 复制到 light-load device）和 Pro-Prophet（planner + scheduler：lightweight expert placement + block-wise scheduling）。

从系统架构角度拆解术语：
在 MoE 训练中，系统级负载均衡的运转流程：

```
1. Profile: 在每个 MoE layer 收集各 expert 的 input distribution（token 数量）
2. Search: 基于 distribution 搜索最优 expert-to-device placement
3. Place: 根据 placement 传输 expert parameters 到目标 devices
4. Compute: 各 device 用 local inputs 计算持有的 experts
5. Reduce: 反向传播后将 gradients 聚合回 expert 原始 device

// Pro-Prophet 的优化（vs FasterMoE）：
// - Step 2-3-5 的通信开销占训练时间 29%-37%（FasterMoE）
// - Pro-Prophet 通过 lightweight placement（仅传输到必要 devices 子集）减少通信量
// - 通过 block-wise scheduling 将 Step 2/3/5 与 computation 重叠，消除通信曝光
```

术语一般如何实现？如何使用？
FasterMoE 基于 FastMoE 框架，使用 dynamic shadowing 策略。Pro-Prophet 基于 PyTorch 实现，包含 Planner（locality-based greedy algorithm + performance model）和 Scheduler（block-wise scheduling）。Pro-Prophet 在 NVIDIA 3090/2080Ti + InfiniBand/NVLink 集群上实现 1.01-2.66x speedup vs DeepSpeed-MoE 和 FasterMoE。核心 metric：RB（Ratio of Balance degree）= balance_degree_before / balance_degree_after（使用 input distribution 标准差），Pro-Prophet 最高 11.01x RB ratio vs FasterMoE。

涉及论文标题：
- Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models
