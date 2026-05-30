## Expert Placement Scheduler (专家放置调度器)

术语解释
Expert Placement Scheduler 是 SYMI 中负责每 iteration 计算 expert-to-GPU-slot 映射的核心组件。它接收 global expert popularity（来自 forward pass 的 router 聚合），输出下一 iteration 的 expert placement 方案，决定每个 GPU slot 被分配哪个 expert class。

术语是什么？
Scheduler 运行在每个 rank 本地（deterministic algorithm），唯一需要跨 rank 协调的是输入 popularity 的 all-reduce。核心算法：
1. 归一化 popularity → 按比例分配 instance counts
2. 下限保护：每个 expert 至少 1 个 instance（保证可达性）
3. Rounding correction：floor 后调整使 sum = total_slots
4. Contiguous assignment：相同 expert class 的 instance 相邻放置（优先同 rank）

从系统架构角度拆解术语：
调度流程在一轮 training iteration 中的位置：
```
Forward Pass:
  Router → aggregate per-expert token counts → all-reduce popularity
  → Layer Metadata Store.set(popularity)     # 存储供 scheduler 使用

Optimizer Step:
  popularity[t] = Layer Metadata Store.get()  # 读取当前 iteration 的 popularity
  placement[t+1] = Scheduler.compute(popularity[t])  # 计算下轮 placement
  → Sync placement to router, runtime engine, SYMI Optimizer
  → SYMI Optimizer 按 placement[t+1] 分发 updated weights
```

术语一般如何实现？如何使用？
- 论文使用最简单的策略：placement 直接 mimic previous iteration popularity（有效且 overhead 极低）
- 调度器设计为可扩展：可接入预测模型、历史统计、或基于数据集的 static schedule
- 调度器 overhead 仅占 iteration time 的 <0.1%（纯本地计算，无通信）
- 与 NCCL communication group 预注册配合：contiguous assignment 保证 groups 可复用

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---
