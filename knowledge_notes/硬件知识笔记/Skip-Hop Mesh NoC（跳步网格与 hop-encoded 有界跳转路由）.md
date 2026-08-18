## Skip-Hop Mesh NoC（跳步网格与 hop-encoded 有界跳转路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Skip-Hop Mesh 是 MLX 的片上网络拓扑：在常规本地邻居转发的网格（4×4，可扩 8×8）之上，给每个 PE 增加固定距离直连链路（skip links），直接跨越折叠层的依赖半径。动机：层折叠把跨层依赖变成有界、规则的通信模式——BSMM/FFT 每层访问确定性的 stride-2^k 邻居（蝴蝶交换距离），这些跨层传输对全局内存流量不友好、却天然匹配拓扑感知网格 NoC。skip-hop 使大部分跨层传输只需 1-2 跳（stride=2/4 水平跳、stride=4/8 转 1-2 跳垂直 ±1/±2）。
- 实现该传输的最小硬件状态：hop-encoded 数据搬运原语——每条 xfer 指令只携带残差 hop 数、路由方向与目标寄存器；路由器无状态：hop 数为 0 时本地写入，否则消耗最大可接受步长（unit 或 skip）并转发数据包。该原语统一覆盖蝴蝶 stride、FFT 配对、稠密 MM 脉动运动与有界窗口交互（SWA），无需路由表、虚拟通道与动态路由计算。面积开销：skip-hop 使每 PE 增加约 6.2% 面积（PE 0.482 mm² 中含 skip-hop 开销 6.2%），12nm @1GHz 时序开销小，8×8 网格近线性扩展。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（BSMM 蝴蝶层在 skip-hop 网格上）：第 k 个蝴蝶层以 stride=2^k 做两两混合 → PE_x 的 XFER 流水发出 hop-encoded 包（如 stride=4 编码为 1 个 skip 跳）→ 无状态路由器沿 x 方向消耗最大可接受步长、残差 hop 数递减 → hop=0 时写消费 PE_{x+s} 的目标寄存器 → 多个 BSMM 层可并发执行而路由不冲突，形成严格分层片上流水（Fig.10(c)）。跨层依赖不再经全局内存往返：中间结果以部分和形式直接在阵列内流动，蝴蝶的 strided 交换变成有界跳数（≤2）的确定性传输。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog RTL 中每 PE 增加固定距离直连链路 + hop-encoded 路由器（无状态、无虚拟通道）；综合于 12nm @1GHz，skip-hop 面积开销约 6.2%（PE 面积 0.482 mm² vs 无 skip-hop 情形）。使用：编译器把 CDC 跨层传输按 (Δx,Δy) 路由类参数化 + 残差 hop 数编码进 xfer 指令；mesh 扩展时依赖半径增长、通信延迟上升，靠增大 active-layer window/块计算预算 C 保持覆盖（B_T·C ≥ T_load+T_xfer）。效果：蝴蝶算子 roofline 利用率 52%-84%（vs GPU 12%-31%）、8×8 网格近线性扩展（3.6×）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
