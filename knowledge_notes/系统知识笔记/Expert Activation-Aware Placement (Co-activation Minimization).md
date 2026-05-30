## Expert Activation-Aware Placement (Co-activation Minimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Activation-Aware Placement 是 JANUS 提出的 expert replica 放置策略，目标是 minimize MoE layer latency。核心观察：当两个 expert 经常被同一 batch 的 tokens 同时激活（即 co-activated），将它们放在同一 MoE GPU 上会增加该 GPU 的 distinct activated expert 计数 a_max，从而增加 MoE 执行延迟（因为 MoE latency ∝ a_max）。

JANUS 将 placement 形式化为 min-max 优化问题 (Eq. 7, Appendix B)：
$$
\min_{\{x_{e,g}\}} \max_{g} I(g) = \min_{\{x_{e,g}\}} \max_{g} \sum_{e,e' \in P(g)} a(e,e')
$$
其中 a(e,e') 是从 recent traces 估计的 expert e 和 e' 之间的 co-activation 频率。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Placement 流程 (Algorithm 3, Greedy Heuristic with Bounded Swap):

```
1. 确定 replica counts:
   - S = n_e · C total expert slots
   - 先给每个 logical expert 分配 1 个 replica (S 个 slots)
   - 剩余 S-E slots 按 per-replica activation load l(e) = c(e)/R(e) 降序分配
     (hot experts get more replicas, cold experts stay singleton)

2. Greedy placement:
   - 按 load 降序遍历 replicas
   - 对每个 replica i (expert e_i):
     a. 找到有 free slots 且不持有 expert e_i 的实例集合 G_i
     b. 若 G_i ≠ ∅: 选择 instance g* 使 incremental co-activation penalty minimal
        g* = argmin_{g∈G_i} Σ_{j∈P(g)} a(i,j)
     c. 若 G_i = ∅ (所有有 free slots 的实例已经持有该 expert):
        Bounded swap: 找到最优的 (g, j, h) 交换以最小化 co-activation load 增量
        从 instance g 移出 expert j → 放入 instance h
        instance g 接收 replica i
```

Replica Count 分配示例:
```
E=160 experts, n_e=8 instances, C=27 slots/instance
S = 8×27 = 216 total slots
每个 expert 1 个 base replica: 160 slots
剩余 216-160 = 56 slots 作为冗余
56 slots → 分配给 activation frequency 最高的 56 个 expert (各 +1 replica)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 从 recent activation traces (e.g., sliding window) 估计 co-activation 频率 a(e,e')
- Placement 在 scaling 决策后触发 (15min 间隔)，不在 per-layer critical path 上
- Greedy heuristic 近似 min-max NP-hard 问题 (等价于 unrelated-machines scheduling)
- 与 AEBS 互补：placement 决定静态 replica 布局，AEBS 在运行时动态选择 replicas

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
