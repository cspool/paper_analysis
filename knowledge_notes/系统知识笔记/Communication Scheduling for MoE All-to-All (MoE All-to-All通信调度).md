## Communication Scheduling for MoE All-to-All (MoE All-to-All通信调度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Communication Scheduling for MoE All-to-All 是 Aurora 提出的通过确定 token 在 all-to-all 通信中的传输顺序来最小化通信时间的技术。核心思想：在 all-to-all 通信中，不同 GPU 向同一目标 GPU 同时发送 token 会导致带宽竞争（bandwidth contention），延长通信时间。通过合理安排传输顺序，确保任何时刻每个 GPU 只从单一源接收数据，可将通信时间压缩至理论下界。

Aurora 证明（Theorem 4.2）：最小通信时间 b_max = max(Σ_j d_ij, Σ_i d_ij) / B，由 bottleneck GPU（最大发送或接收流量的 GPU）决定。这与直觉相符：通信时间不可能少于最大负载 GPU 发送或接收所有数据所需的时间。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Aurora 的通信调度算法（Alg. 1）流程：

**输入**：All-to-All traffic matrix D（n×n，d_ij 表示 GPU i 发往 GPU j 的 token 量）
**输出**：每个 GPU 的 token 传输顺序 O

```
1. 找到 bottleneck GPU（总流量最大的 GPU）
2. 为 bottleneck GPU 随机确定传输顺序，加入 O
3. 从 D 中移除 bottleneck GPU 的流量
4. while D 非空:
5.   按剩余流量降序排列 GPU
6.   for 每个 GPU i:
7.     安排 token 传输顺序，避免与 O 中已有顺序冲突
        （确保不与其他 GPU 同时向同一目标发送）
8.     将 GPU i 的新顺序加入 O
9.     从 D 中移除 GPU i 的流量
```

**执行原理**：通过添加非负人工 traffic X 将原始 D 转化为每行/列和均为 b_max 的 D'，用 Farkas' Lemma 证明 X 存在 → 在 D' 中每个 GPU 可持续无间断地发送和接收 → 通信时间 = b_max → 因 D ≤ D'，D 的通信时间也不超过 b_max。

**实例**：假设 3 GPUs，GPU 1 向 GPU 2 发 2 units、向 GPU 3 发 1 unit，GPU 2 向 GPU 1 发 1 unit、向 GPU 3 发 1 unit。随机顺序（GPU 1: [2→3, 1→2] 重叠）→ 3 time units。Aurora 优化（GPU 2: [3, 1]，错开接收端）→ 2 time units。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在实际系统中，通信调度可通过在计算操作的 buffer 层调用 NCCL 集体通信库按指定顺序执行来实现。
- Aurora 的传输顺序优化针对的是**同步 all-to-all**场景（所有 GPU 等通信完成后才开始计算）。在 Exclusive+Homogeneous 场景中，SJF（最短作业优先）调度效果与随机调度相当，因为单纯优先发送小流量无法减少带宽竞争。
- Aurora 的通信调度在 Exclusive+Homogeneous 场景下实现 1.38× 加速（vs. SJF），且调度决策只需 traffic matrix 作为输入。

涉及论文标题：
- Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling
