## Expert Colocation for MoE Inference (MoE推理专家共置)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Colocation（专家共置）是将多个 MoE expert 放置在同一个 GPU 上的部署策略，目的是提高 GPU 利用率。Aurora 区分了两种共置模式：(1) **同模型共置**：将同一 MoE 模型的多个 expert 放在同一 GPU（如 Lina 将最热和最冷 expert 配对）；(2) **跨模型共置**：将来自**不同 MoE 模型**的 expert 放在同一 GPU（Aurora 的创新）。

同模型共置的缺陷在于：因同步 all-to-all 通信约束，同一模型的所有 expert 必须等待通信完成后才能开始 FFN 计算，即使用一 GPU 上有多个 expert，也无法交错利用计算和通信资源。跨模型共置则打破了此限制——两个不同模型的 expert 不共享同步 barrier，可以交替使用 GPU：Model a 做 FFN 计算时 Model b 进行 all-to-all 通信，反之亦然，实现计算和通信的完全交错。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 Aurora Colocating+Heterogeneous 场景为例（2 个 MoE 模型 a 和 b，各 8 expert，异构 GPU 集群）：

1. **Expert 配对决策（离线）**：基于历史 traffic matrix D_N^a 和 D_N^b，构建二分图——左侧为 Model a 的 expert，右侧为 Model b 的 expert，边权重为 max(a_i+b_j, a_{n+i}+b_{n+j})（即共置后的最大发送/接收流量）。求解 bottleneck matching problem 找到最小化最大边权重的完美匹配。
2. **GPU 分配**：配对后的 expert 对按 token 负载降序分配给性能降序的 GPU（Theorem 5.1）。
3. **推理执行**：在每层 MoE 中，两个模型交替使用资源：
   - t=0: Model a 开始 Gate + 第一个 All-to-All dispatch (N^a)，Model b 开始 Gate (G^b)
   - t=|G^b|: Model b Gate 完成，开始第一个 All-to-All dispatch (N^b)，此时 Model a 的 N^a 可能还在进行 — 两者在网络上重叠
   - t=max(G^b, N^a): Model a 开始 FFN (F^a)，使用 GPU 计算资源
   - Model b 的 N^b 继续利用网络，与 Model a 的 F^a 并行
   - 后续以此类推，两个模型的计算和通信完全交错

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Aurora 通过 bottleneck matching 算法（二分搜索 + Hopcroft-Karp，复杂度 O(n²√n log n)）找到最优 expert 配对。Case I（每 GPU 发送=接收流量）有更简单的解法：交替选择热门和冷门 expert（Theorem 6.2）。
- Colocating 场景中每 GPU 最多放 2 个 expert（来自不同模型）。超过 2 个会导致资源竞争——至少一个模型被迫等待。
- Aurora 评估显示：跨模型共置比 Lina 的同模型共置 GPU 利用率提升 1.28×-1.50×，推理时间加速 2.38×（同构）和 3.54×（异构）。
- 共置决策依赖历史 traffic 统计。Aurora 对输入噪声鲁棒：75% 不精确输入仅导致 15.8% 性能退化。

涉及论文标题：
- Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling
