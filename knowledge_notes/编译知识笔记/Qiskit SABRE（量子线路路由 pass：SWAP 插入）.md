## Qiskit SABRE（量子线路路由 pass：SWAP 插入）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SABRE（SWAP-based BidiREctional search，Li, Ding & Xie，ASPLOS 2019）是 Qiskit transpiler 的核心路由 pass：把逻辑线路映射到受限物理耦合图上，用 SWAP 门把两比特门涉及的 qubit 移到相邻位置。特点：正反向交替搜索（forward 用 heuristic 距离估计，backward 校准）、解耦的初始映射寻找与路由（与布线解耦的初始布局 + 路由分离）、启发式 cost（基于 qubit 距离求和）优化 SWAP 数；线性可扩展、对大规模线路可用。Qiskit 中可用 pass manager 配置（routing_method='sabre'）。
- 本文用途（Q4）：验证逻辑优化是否在受限连接硬件上保持收益——把原始/优化后线路经 Qiskit SABRE 映射到 2D 平面耦合图（square grid），报告 (i) 加权两比特门数（每个 SWAP 计为 3 个 CNOT）与 (ii) 物理线路深度。
- 关键发现：PhasePoly 的逻辑缩减在映射后常被放大（逻辑深度平均减 22.47% → 物理深度减 28.35%，大电路 22.47%→40.91%），因为更少两比特交互 → 更少 SWAP；跨 block 奇偶性复用减少多比特交互重建。反例：拓扑友好电路（gf2^4_mult 等）路由开销主导、逻辑收益被抵消；QAOA 类线路 CNOT 减 3% 但深度增 20%（并行性降低，可用领域特定映射器弥补）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（一次映射）：输入逻辑线路 + 耦合图（如 2D grid）→ ① 初始布局：启发式为每逻辑 qubit 找物理 qubit（与后续路由解耦的两阶段）；② 逐门处理：对每个两比特门，查当前映射下两 qubit 的物理距离（最短路径/启发式）；若不相邻，插入 SWAP 序列把控制/目标移到相邻，并更新逻辑-物理映射；③ 正反向交替：正向 pass 结束后反向重跑校准初始映射，迭代改善；④ 输出物理线路（含 SWAP）→ 计算加权 CNOT 数与物理深度。
- 在本文实验中的例子：MCX/Adder/HWB 大电路族映射后 PhasePoly 物理深度平均减 40.84%（2.68–15.13× 改善），Quartz 仅 2.7%、QUESO 15.25%（Fig.16b）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Qiskit（https://github.com/Qiskit/qiskit）transpiler 中 `transpile(circ, coupling_map=..., routing_method='sabre', optimization_level=...)`；SABRE 是 Qiskit 默认路由算法之一。论文按此流程评估逻辑优化对物理指标的影响。领域特定替代：QAOA 专用映射器（论文引 [45][46]，利用可交换门自由度进一步减门）。
- 与本文关系：SABRE 是评估工具而非被修改对象；它把 PhasePoly 的逻辑收益翻译为物理收益（SWAP 减少），支撑"gate-count 优化 → 硬件执行提升"的结论链。
- 补充（CANOPUS 论文）：CANOPUS 继承 SABRE 的骨架（DAG 分层、front layer F、extended set E、双向迭代 layout 与 routing），但在其启发式上做 ISA-aware 增强：① 把 SABRE 的绝对平均最短路径距离启发式 H_SABRE = Avg{dist}_F + k_E·Avg{dist}_E 替换为差分距离 ΔAvg{dist} × ISA 特定 SWAP 成本 c_swap，并新增 SWAP mirroring 边际成本 c_g 与深度增量 Δdepth（wire duration record D）两项，形成统一启发式 H = w_g·c_g + w_d·Δdepth + (ΔAvg{dist}_E + k_E·ΔAvg{dist}_E)·c_swap；② 因 c_g/Δdepth 提供精确硬件感知反馈，取消 SABRE 的经验衰减因子；③ 超参数 k_E=0.5 与 SABRE 一致，w_g=w_d=0.5，w_d 乘拓扑自适应因子 d̄/(2+d̄)。SABRE 在论文中是主要 baseline（Python QISKIT 实现，刻意不用 Rust 加速版保证公平对比），CANOPUS 相对它平均降 routing overhead 16.06%（Ccount）/26.44%（Cdepth）；CANOPUS 编译时间为 SABRE 的 1-2×（大规模 2-5×，多项式扩展）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
