## Litinski 风格编译器与 LLI/Timeline（FTQC 编译→调度注释流）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Litinski 风格编译器指基于 Daniel Litinski《A Game of Surface Codes》（Quantum 3:128, 2019, arXiv:1808.02892）的 surface code 编译方案：把表面码操作抽象为 tile-based game——每个逻辑量子比特是占据 tile 的 patch（虚线边=X 算子、实线边=Z 算子），操作 = patch 初始化/测量（0 代价）、multi-patch 测量（1 代价）、patch deformation（enlarge 1/shrink 0），时间步 = surface-code cycle（d 轮 syndrome 测量）；配套数据块布局（compact ~1.5n+3 / intermediate 2n+4 / fast 2n+√(8n)+1 tile）与蒸馏块（distillation block）架构。Triage 用它生成 LLI（Low-Level Instruction）流作为离线调度分析输入。Triage 的 LLI 指令集为三类：multi-patch measurement、patch rotation、idle。后续实用编译器：Watkins et al. 高吞吐 surface-code 编译器（Quantum 8:1354, 2024）、LeBlond et al.（ACM TQC 2023）——Triage 论文引用 [38] 即 Litinski、[62][63] 即这两者作为 benchmark 编译来源。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Triage 的离线编译→静态分析流水线（编译框架输入到输出的全过程）：
```
# 输入：高层逻辑电路（Clifford+T），如 QASMBench benchmark
1) Litinski 风格编译器（Compact Layout [38] / Standard Layout [44]）：
   - 电路 → PBC/Pauli rotation 序列 → tile 布局放置 patch
   - 生成 LLI 流：multi-patch measurement / patch rotation / idle（每条带 patch 坐标与类型）
2) 静态分析器（Triage 离线阶段，单趟）：
   对每个 LLI 单元构建 Timeline 条目：
   (i) 整数 layer 索引 t（单位=1 个 syndrome 测量周期）
   (ii) 空间坐标 (r, c)
   (iii) 操作标签
   (iv) 6-bit 即时邻居 mask（t−1, t+1, ↑, ↓, ←, →）
   (v) deadline = 到最近关键同步点（T 门）的层数（无则∞）
   (vi) 可能的 causal cone 引用
3) 输出：带注释的 Timeline（调度器在线阶段的决策依据）
```
关键设计：deadline 在编译期从 T 门位置前向传播算出；causal cone 引用采用 lazy 计算（在线按需 BFS + LRU 缓存）而非编译期全量展开。布局选择（Compact vs Standard）改变 patch 邻接 → 改变空间邻居掩码与 slice 约束图，因此同一 benchmark 有 CL/SL 两个变体（如 mult15_CL/mult15_SL、adder28_CL/adder28_SL）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Litinski 编译器本身无官方统一仓库（论文为方法与资源模型，社区实现如 sfc-aqua/gosc-graph-state-generation 基于其规则做 graph-state 编译；Watkins/LeBlond 有独立实现）；Triage 论文用 Python 3.9 自建编译器+静态分析器复现（未开源）。使用场景：FTQC 全栈仿真的前端——编译器输出 LLI，静态分析输出 Timeline，调度器（Triage）消费 Timeline 做实时解码调度；这也为"编译器与调度器协同设计"留下接口（论文未来工作：让编译器以经典资源感知优化电路）。评价：LLI/Timeline 是"编译产物直接驱动实时调度"的中间表示设计，类比传统编译器的 IR→调度器注释流。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
