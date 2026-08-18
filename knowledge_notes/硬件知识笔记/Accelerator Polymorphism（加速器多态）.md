## Accelerator Polymorphism（加速器多态）

术语解释
- 一种加速器架构概念：单一物理衬底在运行时动态"变形"为多种专用执行模型，跨算法域复用同一套计算与存储资源，既不退回通用处理、也不像 DSA 一样为每个域单独流片。本文以 Morphatron 实例化：五种执行形态（morpha）——queue-centric SIMD、graph、tree、vector、systolic。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：机器人端到端 pipeline 跨多个算法域（图搜索、控制、优化、SLAM、神经网络），呈现四类访存模式与四类并行度、无单一形态主导（Amdahl 定律：只加速一种形态瓶颈必然转移到其余形态）→ 逐 kernel 的 DSA 碎片化、NRE 成本高、算法演进即需重新设计硬件 → CGRA/dataflow 需要复杂互连与空间编译（布局布线）→ 因此提出"加速器多态"：用轻量可配置互连 + power gating，把同一物理衬底重塑为多个指令驱动的独立加速器。本文的 Morphatron 是 32×4 Morpha Core 网格，仅需 4 种固定路由模式（每 switch 2 个配置位）即可覆盖全部五种 morphas，编译沿用常规加速器工具链。与 TRIPS（单一受限 EDGE 数据流模型）、MorphCore（OoO↔高度线程化 in-order 切换）的区别：本文首次让单一设计在多个"专用执行模型"之间运行时切换（论文自述，Web 未见第三方独立确认）。评估结果：vs Jetson Orin Nano 5.5× 加速/6.6× PPW，vs ARM A78 7.7×/7.7×，vs Xeon 1.9×/28.9× PPW，vs RTX 3090 0.9×/39.6× PPW，均 10 W TDP；多态切换仅占运行时 1.3%。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运行时形状切换流程（论文 Table III 给出各步延迟）：应用 phase 边界执行 SYNC 屏障 → Morphatron Controller 经 32-bit 控制网络广播 MORPHA_STATIC_CONFIG（指定 morpha 类型、目标核 XY 坐标、按 N/S/W/E/Local 顺序编码 crossbar 方向位）或 MORPHA_DYN_CONFIG（graph/tree：编程地址比较器的路由地址范围）→ 各 switch 以 2 位模式寄存器切换路由、power-gate 未用端口与逻辑 → 新 morpha 开始执行。延迟：Systolic/Queue 重配 165 cycle、Graph/Tree 288 cycle、collective 配置 129 cycle。资源复用例子：scratchpad sub-bank 在 queue morpha 存子队列、在 vector morpha 存 tensor 操作数、在 systolic morpha 作权重缓冲；执行 stage 行（ALU+sub-bank 经流水寄存器相连）结构上等价于 systolic 行，垂直列核直接构成 systolic 阵列；tree morpha 只留 1 个核计算、其余核 power-gate 逻辑仅保留 sub-bank 与 switch，把全片 16 MB 汇成"单核可见的大内存"。geomean 各 morpha 运行时占比：Vector 38.6%、Queue 19.8%、Tree 17.0%、Systolic 15.6%、Graph 7.7%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要素：(1) 解耦控制与存储——Queue Manager（队列管理逻辑）与 scratchpad sub-bank（数据存储）分离，使存储可在 morphas 间换角色；(2) 固定路由模式而非任意空间映射——互连仅 11% 面积、7% 能耗；(3) 指令驱动而非数据流图映射——每个 morpha 是独立 ISA 加速器，编译不用布局布线。使用方式：编译器按算法 phase 把负载映射到最匹配的 morpha；对照实验显示全部五种 morphas 都必要（固定分区变体均不如全多态，最好固定分区仅 0.83×）。Web 证据：未见 Morphatron 开源仓库（联网搜索无结果），概念对比可参考 TRIPS EDGE 架构与 MorphCore 论文（均只切换单一执行模型）。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
