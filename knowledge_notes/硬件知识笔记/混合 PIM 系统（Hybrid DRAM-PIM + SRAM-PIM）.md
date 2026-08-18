## 混合 PIM 系统（Hybrid DRAM-PIM + SRAM-PIM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把 DRAM-PIM（容量/带宽/向量并行）与 SRAM-PIM（低延迟矩阵/高能效）集成进同一个可扩展系统，按算子形态分派：memory-bound 的 GeMV→DRAM-PIM、compute-bound 的 GeMM→SRAM-PIM。三种集成粒度：inter-device（跨设备混合）、inter-channel（跨通道）、intra-channel（同通道内，CompAir 的选择）——因为 SRAM-PIM 的权重重载不可避免，需要 DRAM 内部级带宽喂它（AiM 单通道内部带宽 512GB/s vs 外部 I/O 32GB/s；128 输入 8 输出 INT8 SRAM-PIM 在 16ns 延迟下需要 64GB/s 才能满负荷）。CompAir 自称首个系统化探索"混合 PIM + in-network 计算"的工作（ISCA'26 / arXiv:2509.13710）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CompAir 的 intra-channel 组织：DRAM die 与逻辑 die 经 hybrid bonding 每 bank 1:1 配对（256 bonds/bank），逻辑 die 含 4×8KB SRAM-PIM + 4 router。执行例子（batch=64 decode）：Q/K/V 投影与 FFN 在 SRAM-PIM 以权重驻留算 GeMM；QK^T/SV 在 DRAM-PIM 算 GeMV（TP 沿 seqlen 切分）；Softmax/RoPE 由 NoC 在途完成。设计权衡：DRAM/SRAM 使用比例决定能耗——cross-die 通信使 hybrid 能耗略高于纯 DRAM-PIM，SRAM 过度使用能量代价高（需按 workload 优化比例）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
3D 集成路线：HB（CompAir，密度 10K–100K/mm²）、TSV/µbump（HBM-PIM 堆叠）、passive DRAM 堆叠（SRAM-PIM stacking DRAM 基线）。映射规则（CompAir 结论）：(1) SRAM-PIM for batched FC（性能）；(2) DRAM-PIM for attention（能效）；GQA 下 QK^T 是否用 SRAM-PIM 依 TP 与 seqlen 而定。编程：分层 ISA 统一 SIMD 行级接口与 MIMD 包级执行（见编译框架库）。评估：ramulator2.0 + Booksim + CENT 模拟器（开源 https://github.com/Man0xbfc00380/comp-air.git）。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
