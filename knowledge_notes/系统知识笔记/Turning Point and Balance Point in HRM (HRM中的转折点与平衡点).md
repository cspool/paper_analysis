## Turning Point and Balance Point in HRM (HRM中的转折点与平衡点)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 Hierarchical Roofline Model (HRM) 中，Turning Point 和 Balance Point 是分析异构系统性能瓶颈的关键概念。Turning Point P1 (Eq. 9)：当算子 x 在 CPU 上的性能 P_x^j ≥ B_peak^{j,i} × I_x^j 时，表示将数据从 CPU (level j) 传输到 GPU (level i) 进行计算的收益阈值——当 operational intensity I_x^j 低于此阈值时，不值得做跨层传输，应就地计算。Turning Point P2 (Eq. 10)：当 B_peak^{j,i} × I_x^j ≤ min(P_peak^i, B_peak^i × I_x^i) 时，表示系统吞吐受限于 CPU→GPU 带宽。Balance Point (Eq. 11)：B_peak^i × I_x^i = B_peak^{j,i} × I_x^j，此时 GPU 带宽与 PCIe 带宽达到平衡——进一步增加 batch size 不会提升吞吐。MoE-Lightning 的 policy optimizer 目标就是找到给定硬件约束下的最大 Balance Point。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 Mixtral 8x7B MoE FFN on L4 为例（Fig. 5）：(1) 当 batch size 对应的 I（operational intensity regarding CPU data）< P1 的 I 时 → 不值得把数据从 CPU 传到 GPU 做 FFN → 应选 CPU 计算（对应 latency-oriented 场景的静态权重放置策略）；(2) 当 P1 ≤ I < P2 时 → 计算受限于 PCIe 带宽 → 需要增大 batch 或部分 weights 常驻 GPU（两者都增加 operational intensity）；(3) 当 I = P2 时 → 达到 peak performance（受限于 GPU kernel 的 micro-batch size μ）；(4) 当达到 Balance Point 时 → GPU BW × I_gpu = PCIe BW × I_cpu → 无需继续增大 N（资源最大化利用）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- HRM 的 turning points 和 balance point 通过分析式推导（不需要 profiling），仅需硬件峰值参数（peak FLOPS, peak BW）和算子理论 FLOPs/bytes。
- MILP optimizer 以式 12 (T = max(comm, T_cpu, T_gpu)) 为优化目标，在满足 CPU/GPU memory 约束下搜索达到 Balance Point 的 (N, μ) 组合。
- 实践中，MoE-Lightning 在各 setting 下都能达到 GPU memory capacity 决定的 throughput 上限（即为 Balance Point 的特殊情况）。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
