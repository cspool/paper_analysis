## Adaptive Expert Allocation in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Adaptive Expert Allocation 是 Lazarus 提出的 MoE 训练系统策略：根据 expert 的运行时负载分布（gate network routing 历史），动态为每个 expert 分配不同数量的 replicas (GPUs)，而非传统 EP 的均等分配。核心公式（Eq. 1）：r_e = max{⌊t_e / Σ_{e'=e}^{E} t_{e'} × (N·c - Σ_{e'=1}^{e-1} r_{e'})⌋, f}，其中 t_e 为 routed to expert e 的 token 数，N 为节点数，c 为每节点 replica 槽位数（受 GPU memory 限制），f 为容错阈值。

该策略保证：(a) Σ_e r_e = N·c —— 所有 GPU 槽位被利用；(b) r_e ≥ f —— 每个 expert 至少有 f 个 replicas，支持 <f 个节点故障时 100% 恢复；(c) r_e/r_{e'} ≈ t_e/t_{e'} —— popular experts 获得更多 replicas 和计算资源。由于 expert load distribution 在不同层和训练阶段变化，Lazarus 每 200 steps 根据 agent 周期性收集的 routing history 重新平衡（rebalance）allocation。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Lazarus 的 Adaptive Expert Allocation 运转流程：

```
┌── Monitoring Phase (continuous) ──────────────────────┐
│ Agent (per GPU node) → periodically collect routing   │
│   history (gate network decisions) from Lazarus runtime│
│ → send to Load Monitor on Controller (CPU node)       │
└───────────────────────────────────────────────────────┘
                          ↓
┌── Allocation Phase (every 200 steps or on failure) ───┐
│ Load Monitor → compute t_e for each expert from       │
│   aggregated routing history across all layers        │
│                                                       │
│ Scheduler → apply Eq. 1:                              │
│   Sort experts by t_e ascending                       │
│   For e = 1 to E:                                     │
│     remaining_slots = N·c - sum of allocated slots    │
│     remaining_tokens = sum of t_{e'} for e' ≥ e       │
│     r_e = max(floor(t_e/remaining_tokens ×             │
│             remaining_slots), f)                      │
│                                                       │
│ Result: r_e for each expert, Σ r_e = N·c, r_e ≥ f     │
└───────────────────────────────────────────────────────┘
                          ↓
┌── Placement Phase ───────────────────────────────────┐
│ MRO algorithm: compute placement plan from r_e        │
│   → maximize recovery probability                    │
│   → compute per-layer independently (load varies)    │
└───────────────────────────────────────────────────────┘
                          ↓
┌── Deployment Phase ──────────────────────────────────┐
│ Controller → send placement plan to each Agent        │
│ Agent → relay to Lazarus runtime on worker process    │
│ Runtime → fill MoE layers with assigned expert       │
│   replicas according to plan                          │
└───────────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Lazarus 用 Python 实现 allocation 逻辑（在 Controller 上运行，<100ms 计算时间）。Controller 维护每个 worker 的 expert routing history 滑动窗口。Per-layer 独立计算（因为 expert load distribution 在不同层之间不同）。容错阈值 f 在 Lazarus 中设为 2（常见单节点故障场景），当剩余节点不足以维护 f 个 replicas 时自动放宽约束。与 MRO placement 和 flexible token dispatcher 协同工作，共同实现弹性+容错+高性能的目标。

相关概念：Expert Replication 在 FasterMoE 和 FlexMoE 中用于加速 MoE 训练，但仅针对固定集群大小；Lazarus 是首个在弹性环境（changing GPU membership）下进行 adaptive expert replication 的系统。

涉及论文标题：
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
